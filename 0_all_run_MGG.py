#!/usr/bin/env python3
import os

# num_gpus = [2, 3, 4]
# num_gpus = [2, 3, 4, 5, 6, 7, 8]
num_gpus = [2]

os.system("mv *.csv csvs/")
for gpu in num_gpus:
    os.system("./0_bench.py {0}| tee MGG_{0}GPU.log".format(gpu))
    os.system("./1_analysis.py MGG_{0}GPU.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU.log logs/".format(gpu))