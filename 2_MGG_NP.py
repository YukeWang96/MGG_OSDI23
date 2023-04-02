#!/usr/bin/env python3
import os
os.system("mv *.csv csvs/")

num_gpus = [4]

for gpu in num_gpus:
    os.system("./bench_MGG_NP.py {0} > MGG_NP.log".format(gpu))
    os.system("./analysis_MGG.py MGG_NP.log {0}".format(gpu))
    os.system("mv MGG_NP.log logs/".format(gpu))

for gpu in num_gpus:
    os.system("./bench_MGG_WO_NP.py {0} > MGG_WO_NP.log".format(gpu))
    os.system("./analysis_MGG.py MGG_WO_NP.log {0}".format(gpu))
    os.system("mv MGG_WO_NP.log logs/".format(gpu))

os.system("python3 combine_NP.py")