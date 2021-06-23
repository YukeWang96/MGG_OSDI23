#!/usr/bin/env python3
import os

num_gpus = [2, 3, 4, 5, 6, 7, 8]

os.system("mv *.csv csvs/")
for gpu in num_gpus:
    os.system("./4_bench_UVA.py {0}| tee UVA_{0}GPU.log".format(gpu))
    os.system("./1_analysis_UVA.py UVA_{0}GPU.log {0}".format(gpu))
    os.system("mv UVA_{0}GPU.log logs/".format(gpu))