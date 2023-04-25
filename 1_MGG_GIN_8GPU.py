#!/usr/bin/env python3
import os

os.system("mv *.log logs/")
os.system("mv *.err logs/")
# os.system("mv *.csv csvs/")
num_gpus = [8]

for gpu in num_gpus:
    os.system("./bench_MGG_GIN.py {0}| tee MGG_GIN_{0}GPU.log 2>MGG_GIN_{0}GPU.err".format(gpu))
    os.system("./analysis_MGG.py MGG_GIN_{0}GPU.log {0}".format(gpu))
    os.system("mv MGG_GIN_{0}GPU.log logs/".format(gpu))