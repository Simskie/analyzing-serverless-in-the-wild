# Student project for course **CE7490: ADVANCED TOPICS IN DISTRIBUTED SYSTEMS**

## Preface

**Authors**:  
Jeremy Gerster  
Simon Menke

We thank Prof. Tang Xueyan for teaching the course and providing insightful lectures.

## Description

This project is an implementation of the paper "Serverless in the Wild: Characterizing and
Optimizing the Serverless Workload at a
Large Cloud Provider" [[1](https://www.usenix.org/system/files/atc20-shahrad.pdf)]. A detailed description about the implementation and our experiments can be found in the report under `report/report.pdf`.

## Instructions

Run `simulate_data.ipynb`to compute simulation results of input data. Note that this will take a while as simulations are quite compute-intensive.  
When the simulations are done, `analysis/policy/plot_results.ipynb` can be run to get relevant plots.  
In addition, `analysis/workload/exploratory_data_analysis.ipynb` contains code for plotting the analysis of the workload data.

### Requirements

Package requirements can be found in `requirements.txt`