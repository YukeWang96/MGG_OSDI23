#!/usr/bin/env python3
import os
# os.system("mv *.csv csvs/")
# os.system("mv *.log logs/")

num_gpus = [8]

for gpu in num_gpus:
    os.system("./0_bench_UVM.py {0}| tee UVA_{0}GPU.log".format(gpu))
    os.system("./1_analysis_UVA.py UVA_{0}GPU.log {0}".format(gpu))
    os.system("mv UVA_{0}GPU.log logs/".format(gpu))