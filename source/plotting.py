import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_cdf_coldstarts(df_list, save_file=False, file_name=""):

    fig, ax = plt.subplots(figsize=(7, 6))

    for data in df_list:
        df = data['DF']['ColdStartPercentage']
        label = data['Label']
        linestyle = data['Linestyle']
        color = data['Color']
        ax.ecdf(df, complementary=False, label=label, linestyle=linestyle, color=color)

    ax.set_xlabel('App Cold Start Percentage')
    ax.set_ylabel('CDF')
    ax.legend(loc='lower right')

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])

    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=0.5)

    for y in [0, 0.25, 0.50, 0.75, 1.0]:
        ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.5)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_ylim(0, 1.05)

    if save_file:
        if not file_name:
            raise ValueError("File name must be given if save_file is True")
        plt.savefig(file_name)

    # Display the plot
    plt.show()


def plot_pareto_front_cv(df_list, normalize_value, markers, colors, save_file=False, file_name=""):
    x_vals = []
    y_vals = []
    labels = []

    for idx, result in enumerate(df_list):
        df = result['DF']

        x_val = np.percentile(df['ColdStartPercentage'], 75)
        y_val = df['WastedMemoryRatio'].mean()

        x_vals.append(x_val)
        y_vals.append(y_val)
        labels.append(result['Label'])

    y_vals_normalized = [(y / normalize_value) * 100 for y in y_vals]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    handles_df_list = []
    labels_df_list = []
    for i in range(len(x_vals)):
        handle = ax.scatter(x_vals[i], y_vals_normalized[i], marker=markers[i], s=100, color=colors[i], label=labels[i])
        handles_df_list.append(handle)
        labels_df_list.append(labels[i])

    ax.axhline(y=100, linestyle=':', color='gray')

    x_min, x_max = ax.get_xlim()
    x_center = (x_min + x_max) / 2

    ax.text(
        x_center,
        101,
        '10-min fixed',
        ha='center', 
        va='bottom', 
        color='gray', 
        fontsize=10 
    )

    ax.set_xlabel('3rd Quartile App Cold Start (%)')
    ax.set_ylabel('Normalized Wasted Memory Time (%)')
    ax.set_xlim(right=100)
    ax.set_ylim(top=135)
    ax.legend(loc='lower right', title='CV')


    if save_file:
        if not file_name:
            raise ValueError("File name must be given if save_file is True")
        plt.savefig(file_name)

    plt.show()



def plot_pareto_front(df_list, normalize_value, markers, save_file=False, file_name="", hour_data=None, weighted_avg_data=None):
    x_vals = []
    y_vals = []
    labels = []

    for idx, result in enumerate(df_list):
        df = result['DF']

        x_val = np.percentile(df['ColdStartPercentage'], 75)
        y_val = df['WastedMemoryRatio'].mean()

        x_vals.append(x_val)
        y_vals.append(y_val)
        labels.append(result['Label'])

    y_vals_normalized = [(y / normalize_value) * 100 for y in y_vals]

    if hour_data is not None:
        x_vals_hour = []
        y_vals_hour = []
        labels_hour = []

        for idx, result in enumerate(hour_data):
            df = result['DF']
            x_val = np.percentile(df['ColdStartPercentage'], 75)
            y_val = df['WastedMemoryRatio'].mean()
            x_vals_hour.append(x_val)
            y_vals_hour.append(y_val)
            labels_hour.append(result['Label'])

        y_vals_normalized_hour = [(y / normalize_value) * 100 for y in y_vals_hour]

    if weighted_avg_data is not None:
        x_vals_weighted = []
        y_vals_weighted = []
        labels_weighted = []
        for idx, result in enumerate(weighted_avg_data):
            df = result['DF']
            x_val = np.percentile(df['ColdStartPercentage'], 75)
            y_val = df['WastedMemoryRatio'].mean()
            x_vals_weighted.append(x_val)
            y_vals_weighted.append(y_val)
            labels_weighted.append(result['Label'])
        y_vals_normalized_weighted = [(y / normalize_value) * 100 for y in y_vals_weighted]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    handles_df_list = []
    labels_df_list = []
    for i in range(len(x_vals)):
        handle = ax.scatter(x_vals[i], y_vals_normalized[i], marker=markers[i], s=100, color='red', label=labels[i])
        handles_df_list.append(handle)
        labels_df_list.append(labels[i])

    ax.plot(x_vals, y_vals_normalized, 'red', linestyle=':', linewidth=1, label='_nolegend_')

    if hour_data is not None:
        handles_hour_data = []
        labels_hour_data = []
        for i in range(len(x_vals_hour)):
            handle = ax.scatter(x_vals_hour[i], y_vals_normalized_hour[i], marker=markers[i], s=100, color='green', label=labels_hour[i])
            handles_hour_data.append(handle)
            labels_hour_data.append(labels_hour[i])

        ax.plot(x_vals_hour, y_vals_normalized_hour, 'green', linestyle=':', linewidth=1, label='_nolegend_')

    if weighted_avg_data is not None:
        handles_weighted_data = []
        labels_weighted_data = []
        for i in range(len(x_vals_weighted)):
            handle = ax.scatter(x_vals_weighted[i], y_vals_normalized_weighted[i], marker='X', s=100, color='blue', label=labels_weighted[i])
            handles_weighted_data.append(handle)
            labels_weighted_data.append(labels_weighted[i])

    ax.set_xlabel('3rd Quartile App Cold Start (%)')
    ax.set_ylabel('Normalized Wasted Memory Time (%)')
    ax.grid(True)

    legend1 = ax.legend(handles_df_list, labels_df_list, title='Fixed', loc='upper right')

    if hour_data is not None:
        legend2 = ax.legend(handles_hour_data, labels_hour_data, title='Hybrid', loc='lower left')
        ax.add_artist(legend1)

    if weighted_avg_data is not None:
        legend3 = ax.legend(handles_weighted_data, labels_weighted_data, loc='lower center')
        if hour_data is not None:
            ax.add_artist(legend2)
        ax.add_artist(legend1)

    if save_file:
        if not file_name:
            raise ValueError("File name must be given if save_file is True")
        plt.savefig(file_name)

    plt.show()


def pareto_front(df_hybrid, df_fixed):
    wmr_at_10 = df_fixed[df_fixed['KeepAlive'] == 10]['WastedMemoryRatio'].mean()

    summary = df_fixed.groupby('KeepAlive').agg({
        'ColdStartPercentage': lambda x: x.quantile(0.75),
        'WastedMemoryRatio': 'mean'
    }).reset_index()

    summary.rename(columns={
        'ColdStartPercentage': 'ColdStart75thPercentile',
        'WastedMemoryRatio': 'MeanWastedMemoryRatio'
    }, inplace=True)

    summary['NormalizedWastedMemoryRatio'] = (summary['MeanWastedMemoryRatio'] / wmr_at_10) * 100
    summary.sort_values('ColdStart75thPercentile', inplace=True)

    markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h', '*', '+', 'x']

    plt.figure(figsize=(10, 6))

    for i, (keepalive, group) in enumerate(summary.groupby('KeepAlive')):
        marker = markers[i % len(markers)]
        plt.scatter(
            group['ColdStart75thPercentile'],
            group['NormalizedWastedMemoryRatio'],
            marker=marker,
            label=f"KeepAlive = {keepalive}",
            s=100
        )

    plt.xlabel('75th Percentile of ColdStartPercentage')
    plt.ylabel('Normalized Wasted Memory Ratio (%)')
    plt.grid(True)
    plt.legend(title='Fixed')
    plt.show()


def plot_trigger_events(self):
    data = self.data 
    df = data.df_invocations_per_function

    minute_columns = [str(minute + 3) for minute in range(1, data.minute +1)]

    trigger_types = ['http', 'queue', 'event', 'timer', 'orchestration', 'storage', 'others']
    function_counts = []
    invocation_counts = []

    for trigger in trigger_types:
        trigger_df = df[df['Trigger'] == trigger] 
        function_count = trigger_df.shape[0]
        function_counts.append(function_count)

        total_invocations = trigger_df[minute_columns].sum(axis=1).sum()
        invocation_counts.append(total_invocations)

    df_grouped = pd.DataFrame({
        'Trigger': trigger_types,
        'FunctionCount': function_counts,
        'InvocationCount': invocation_counts 
    })

    df_grouped['FunctionRelative'] = df_grouped['FunctionCount'] / df_grouped['FunctionCount'].sum()
    df_grouped['InvocationRelative'] = df_grouped['InvocationCount'] / df_grouped['InvocationCount'].sum()

    df_grouped = df_grouped.sort_values(by='InvocationRelative', ascending=False)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1]})

    triggers = df_grouped['Trigger']
    y_pos = range(len(triggers))

    ax1.barh(y_pos, df_grouped['FunctionRelative'] * 100, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(triggers)
    ax1.set_xlabel('Percentage')
    ax1.set_ylabel('Triggers') 
    ax1.set_title('% Functions')
    ax1.grid(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.barh(y_pos, df_grouped['InvocationRelative'] * 100, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  
    ax2.set_xlabel('Percentage')
    ax2.set_title('% Invocations')
    ax2.grid(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    for i in y_pos:
        ax1.text(df_grouped['FunctionRelative'].iloc[i] * 100 + 0.5, i, f"{df_grouped['FunctionRelative'].iloc[i] * 100:.1f}%", va='center')
        ax2.text(df_grouped['InvocationRelative'].iloc[i] * 100 + 0.5, i, f"{df_grouped['InvocationRelative'].iloc[i] * 100:.1f}%", va='center')


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    plt.show()

