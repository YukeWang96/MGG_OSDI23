#!/usr/bin/env python3
import os
os.system("mv *.csv csvs/")

num_gpus = [8]

for gpu in num_gpus:
    os.system("./bench_MGG_API.py {0} {1} > MGG_{0}GPU_API_Thread.log".format(gpu, 0))
    os.system("./analysis_MGG.py MGG_{0}GPU_API_Thread.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU_API_Thread.log logs/".format(gpu))

# for gpu in num_gpus:
#     os.system("./bench_MGG_API.py {0} {1} > MGG_{0}GPU_API_Warp.log".format(gpu, 1))
#     os.system("./analysis_MGG.py MGG_{0}GPU_API_Warp.log {0}".format(gpu))
#     os.system("mv MGG_{0}GPU_API_Warp.log logs/".format(gpu))

for gpu in num_gpus:
    os.system("./bench_MGG_API.py {0} {1} > MGG_{0}GPU_API_Block.log".format(gpu, 2))
    os.system("./analysis_MGG.py MGG_{0}GPU_API_Block.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU_API_Block.log logs/".format(gpu))