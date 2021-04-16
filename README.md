# Online_Robust_Tensor_Recovery (OLRTR)

Streaming data preprocessing via online tensor recovery for large environmental sensor networks

Apr. 2021

## Overview
This repository contains the source code and tests developed for online robust tensor recovery for urban sensing network data preprocessing. This results are reported in "Streaming data preprocessing via online tensor recovery for large environmental sensor networks" by Y.Hu et. al.

## Structures
- `/code/` The source code folder.
  - `OLRTR.m` is the main algorithm. It solves online tensor robust complepetion under fiber-sparse corruption.
  - `solve_proj_21.m` and `update_L_col.m` are helper function for the main algorithm `OLRTR.m`
  - `/PROPACK/` Prerequisit packages, including code for efficient PCA.
  - `/tensor_toolbox-master/` is a tensor manipulation package
  - `test_numerical_simulate.m` is an experiment of OLRTR on numerically simulated tensor data
  - `test_NOAA.m` is the test code for recoverying the manually corrupted NOAA temperature data.
  - `test_AoT.m` is the test code for recoverying raw AoT data
  - `simulate_tensor.m` is the helper function for manually corrupting tensors.
- `/Data/` Contains dataset for testing. 
  -`/NOAA_12M.mat/` contains the 12 month original NOAA data in Chicago.
  -`/aot_12M.mat/` contains the 12 month raw AOT data of 52 sensors.
  -`/noaa_chi_12M.mat/` contains the NOAA record of the nearest noaa sensor to the AOT nodes at the same time stamps. 


## Usage
The code can be run in Matlab. `code/test_numerical_simulate.m`, `test_NOAA.m` and `test_AoT.m` are simulation test, NOAA data test and Chicago AOT case study, respectively.

## Contact
+ Author: Yue Hu, Institute for Software Integrated Systems, Vanderbilt University
+ Email: yue.hu (at) vanderbilt.edu
