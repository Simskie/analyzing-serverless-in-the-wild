import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import requests
class Dataset:

    def __init__(self, day_index, Mem_days, minute):
        self.minute = minute
        self.day_index = day_index
        self.Mem_days = Mem_days
        # Extract paths and file names
        self.data_name = 'azurefunctions-dataset2019'
        self.tar_file = self.data_name + ".tar.xz"
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.current_dir, self.data_name)
        self.data_uri = "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz"

        # Check if data is already downloaded and extracted
        if not os.path.exists(self.data_path):
            self.fetch_data()
            self.extract_data()

        self.parse_data() 
        

    def fetch_data(self):
        print(f"Downloading {self.tar_file}")
        try:
            response = requests.get(self.data_uri)
            response.raise_for_status()  # Check if successful
            with open(self.tar_file, 'wb') as file:
                file.write(response.content)  # Save tar.xz file
            print(f"Download complete")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")

    def extract_data(self):
        print(f"Extracting {self.tar_file}")
        # open and extract file
        file = tarfile.open(self.tar_file)
        try:
            file.extractall(self.data_path)
        except:
            pass
        file.close()
        print("Extraction complete")
        os.remove(self.tar_file)

    def parse_data(self):
        print(f'Parsing {self.data_name}, days: {self.day_index[-1]}, minutes: {self.minute}')
        list_invocations = []
        list_durations = []
        list_memory = []
        # Load data for each day
        for day in tqdm(self.day_index, desc="Parsing Data"):
            df_invocations = pd.read_csv(os.path.join(self.data_path, f'invocations_per_function_md.anon.d{day:02d}.csv'), delimiter=',')
            # Sum invocations starting from the 5th column (1 to 1440 are the invocation counts)
            additional_columns = ['HashApp', 'HashFunction', 'HashOwner', 'Trigger']
            df_invocations = df_invocations.loc[:,'1':str(self.minute)].join(df_invocations[additional_columns])
            df_invocations["total_invocations_per_function"] = df_invocations.loc[:,'1':str(self.minute)].sum(axis=1)
            df_invocations["day"] = day
            list_invocations.append(df_invocations)

            df_durations = pd.read_csv(os.path.join(self.data_path, f'function_durations_percentiles.anon.d{day:02d}.csv'), delimiter=',')
            list_durations.append(df_durations)

            df_memory = pd.read_csv(os.path.join(self.data_path, f'app_memory_percentiles.anon.d{day:02d}.csv'), delimiter=',')
            list_memory.append(df_memory)


        # Concatenate all DataFrames
        self.df_invocations_per_function = pd.concat(list_invocations, ignore_index=True)
        # Group data by HashApp (each app)
        self.df_grouped_by_app = self.df_invocations_per_function.groupby('HashApp')
        self.df_durations = pd.concat(list_durations, ignore_index=True)
        self.df_total_durations = self.df_durations[['HashApp', 'Average', 'Count']].copy()
        self.df_total_durations['Total'] = self.df_total_durations['Average'] * self.df_total_durations['Count']
        self.df_total_durations = self.df_total_durations.groupby('HashApp')[['Total']].sum().reset_index()
        self.df_memory = pd.concat(list_memory, ignore_index=True)



