#!/usr/bin/env python3
import os
# os.system("mv *.csv csvs/")
os.system("mv *.log logs/")
os.system("mv *.err logs/")

num_gpus = [8]

for gpu in num_gpus:
    os.system("./bench_UVM_GIN.py {0}| tee UVM_GIN_{0}GPU.log".format(gpu))
    os.system("./analysis_UVM.py UVM_GIN_{0}GPU.log {0}".format(gpu))
    os.system("mv UVM_GIN_{0}GPU.log logs/".format(gpu))