#!/usr/bin/env python3
import os
# os.system("mv *.csv csvs/")
# os.system("mv *.log logs/")

num_gpus = [8]

for gpu in num_gpus:
    os.system("./bench_UVM.py {0}| tee UVM_{0}GPU.log".format(gpu))
    os.system("./analysis_UVM.py UVM_{0}GPU.log {0}".format(gpu))
    os.system("mv UVM_{0}GPU.log logs/".format(gpu))