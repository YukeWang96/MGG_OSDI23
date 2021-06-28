#!/usr/bin/env python3
import os

# num_gpus = [2, 3, 4]
# num_gpus = [2, 3, 4, 5, 6, 7, 8]
num_gpus = [2, 4, 8]

# os.system("mv *.csv csvs/")
# for gpu in num_gpus:
#     os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU.log".format(gpu))
#     os.system("./1_analysis.py MGG_{0}GPU.log {0}".format(gpu))
#     os.system("mv MGG_{0}GPU.log logs/".format(gpu))

# for dist in [1,2,4,8,16,32]:
#     os.system("./0_bench_MGG.py {0}| tee MGG_Dist_{0}.log".format(dist))
#     os.system("./1_analysis.py MGG_Dist_{0}.log {1}".format(dist, 4))
#     os.system("mv MGG_Dist_{0}.log logs/".format(dist))

# for wpb in [1,2,4,8,16,32]:
#     os.system("./0_bench_MGG.py {0}| tee MGG_wpb_{0}.log".format(wpb))
#     os.system("./1_analysis.py MGG_wpb_{0}.log {1}".format(wpb, 4))
#     os.system("mv MGG_wpb_{0}.log logs/".format(wpb))


for gpu in num_gpus:
    os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU_metrics.log".format(gpu))