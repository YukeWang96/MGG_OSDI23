#!/usr/bin/env python3
import os
os.system("mv *.csv csvs/")

num_gpus = [8]

for gpu in num_gpus:
    os.system("./0_bench_MGG_WL.py {0} > MGG_{0}GPU_WL.log".format(gpu))
    os.system("./1_analysis_MGG.py MGG_{0}GPU_WL.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU_WL.log logs/".format(gpu))

for gpu in num_gpus:
    os.system("./0_bench_MGG_WO_WL.py {0} > MGG_{0}GPU_WO_WL.log".format(gpu))
    os.system("./1_analysis_MGG.py MGG_{0}GPU_WO_WL.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU_WO_WL.log logs/".format(gpu))