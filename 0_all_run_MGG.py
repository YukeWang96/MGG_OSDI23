#!/usr/bin/env python3
import os

# num_gpus = [2, 3, 4, 5, 6, 7, 8]
# num_gpus = [2, 4, 8]
os.system("mv *.csv csvs/")
num_gpus = [2]

for gpu in num_gpus:
    os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU.log".format(gpu))
    # os.system("./1_analysis.py MGG_{0}GPU.log {0}".format(gpu))
    # os.system("mv MGG_{0}GPU.log logs/".format(gpu))

# for dist in [1,2,4,8,16,32]:
# for dist in [1,2,3,4]:
# for dist in [8,16,32,64]:
# for dist in [32]:
    # os.system("./0_bench_MGG.py {0}| tee MGG_Dist_{0}.log".format(dist))
    # os.system("./1_analysis.py MGG_Dist_{0}.log {1}".format(dist, 4))
    # os.system("mv MGG_Dist_{0}.log logs/".format(dist))

# for wpb in [1,2,4,8,16,32]:
# for wpb in [16]:
#     os.system("./0_bench_MGG.py {0}| tee MGG_wpb_{0}.log".format(wpb))
#     os.system("./1_analysis.py MGG_wpb_{0}.log {1}".format(wpb, 4))
#     os.system("mv MGG_wpb_{0}.log logs/".format(wpb))

# for ps in [1,2,4,8,16,32]:
# for ps in [64]:
# for ps in [16,32]:
    # os.system("./0_bench_MGG.py {0}| tee MGG_ps_{0}.log".format(ps))
    # os.system("./1_analysis.py MGG_ps_{0}.log {1}".format(ps, 4))
    # os.system("mv MGG_ps_{0}.log logs/".format(ps))

# for gpu in num_gpus:
#     os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU_metrics.log".format(gpu))

# for dist in [1,2,4,8,16]:
#     for ps in [1,2,4,8,16,32]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_dist_{0}_ps_{1}.log".format(dist, ps))
#         os.system("./1_analysis.py MGG_dist_{0}_ps_{1}.log {2}".format(dist, ps, 8))
#         os.system("mv MGG_dist_{0}_ps_{1}.log logs/".format(dist, ps))


# for wpb in [1,2,4,8,16]:
#     for dist in [1,2,4,8,16]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_wpb_{0}_dist_{1}.log".format(wpb, dist))
#         os.system("./1_analysis.py MGG_wpb_{0}_dist_{1}.log {2}".format(wpb, dist, 8))
#         os.system("mv MGG_wpb_{0}_dist_{1}.log logs/".format(wpb, dist))