#!/usr/bin/env python3
import os

# os.system("mv *.csv csvs/")
num_gpus = [4]
num_gpus = [8]

for gpu in num_gpus:
    os.system("./0_bench_MGG.py {0} > MGG_{0}GPU.log".format(gpu))
    os.system("./1_analysis_MGG.py MGG_{0}GPU.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU.log logs/".format(gpu))