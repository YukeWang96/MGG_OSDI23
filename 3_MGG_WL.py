#!/usr/bin/env python3
import os
os.system("mv *.csv csvs/")

num_gpus = [4]

for gpu in num_gpus:
    os.system("./bench_MGG_WL.py {0} > MGG_WL.log".format(gpu))
    os.system("./analysis_MGG.py MGG_WL.log {0}".format(gpu))
    os.system("mv MGG_WL.log logs/".format(gpu))

for gpu in num_gpus:
    os.system("./bench_MGG_WO_WL.py {0} > MGG_WO_WL.log".format(gpu))
    os.system("./analysis_MGG.py MGG_WO_WL.log {0}".format(gpu))
    os.system("mv MGG_WO_WL.log logs/".format(gpu))

os.system("python combine_WL.py")