#!/usr/bin/env python3
import os
# os.system("mv *.csv csvs/")
os.system("mv *.log logs/")

num_gpus = [4]

for gpu in num_gpus:
    os.system("./0_bench_UVM.py {0}| tee UVM_{0}GPU.log".format(gpu))
    os.system("./1_analysis_UVM.py UVM_{0}GPU.log {0}".format(gpu))
    os.system("mv UVM_{0}GPU.log logs/".format(gpu))