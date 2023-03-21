#!/usr/bin/env python3
import os

os.system("mv *.csv csvs/")
# num_gpus = [2, 3, 4, 5, 6, 7, 8]
# num_gpus = [2, 4, 8]
# num_gpus = [2, 4]
# num_gpus = [8]
num_gpus = [4]

for gpu in num_gpus:
    os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU.log".format(gpu))
    os.system("./1_analysis.py MGG_{0}GPU.log {0}".format(gpu))
    os.system("mv MGG_{0}GPU.log logs/".format(gpu))

# =====================================================================
# num_gpus = 4
# for dist in [1,2,4,8,16]:
#     for ps in [1,2,4,8,16,32]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_dist_{0}_ps_{1}.log".format(dist, ps))
#         os.system("./1_analysis.py MGG_dist_{0}_ps_{1}.log {2}".format(dist, ps, num_gpus))
#         os.system("mv MGG_dist_{0}_ps_{1}.log logs/".format(dist, ps))
# os.system("./10_extract_search.py 6 > A100x8_dist_ps.csv")

# for dist in [1,2,4,8,16]:
#     for wpb in [1,2,4,8,16,32]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_dist_{0}_wpb_{1}.log".format(dist, wpb))
#         os.system("./1_analysis.py MGG_dist_{0}_wpb_{1}.log {2}".format(dist, wpb, num_gpus))
#         os.system("mv MGG_dist_{0}_wpb_{1}.log logs/".format(dist, wpb))
# os.system("./10_extract_search.py 6 > A100x8_dist_wpb.csv")


# =====================================================================   
#  ps -- wpb -- dist
# =====================================================================   
# num_gpus = 8
# for wpb in [1,2,4,8,16]:
#     for ps in [1,2,4,8,16,32]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_wpb_{0}_ps_{1}.log".format(wpb, ps))
#         os.system("./1_analysis.py MGG_wpb_{0}_ps_{1}.log {2}".format(wpb, ps, num_gpus))
#         os.system("mv MGG_wpb_{0}_ps_{1}.log logs/".format(wpb, ps))
# os.system("./10_extract_search.py 6 > A100x8_wpb_ps.csv")


# for wpb in [1,2,4,8,16]:
#     for dist in [1,2,4,8,16]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_wpb_{0}_dist_{1}.log".format(wpb, dist))
#         os.system("./1_analysis.py MGG_wpb_{0}_dist_{1}.log {2}".format(wpb, dist, num_gpus))
#         os.system("mv MGG_wpb_{0}_dist_{1}.log logs/".format(wpb, dist))
# os.system("./10_extract_search.py 5 > A100x8_wpb_dist.csv")
# =====================================================================   



# =====================================================================   
# dist -- wpb -- ps
# =====================================================================     
# num_gpus = 4

# for wpb in [1,2,4,8,16]:
#     for dist in [1,2,4,8,16]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_wpb_{0}_dist_{1}.log".format(wpb, dist))
#         os.system("./1_analysis.py MGG_wpb_{0}_dist_{1}.log {2}".format(wpb, dist, num_gpus))
#         os.system("mv MGG_wpb_{0}_dist_{1}.log logs/".format(wpb, dist))
# os.system("./10_extract_search.py 5 > A100x8_wpb_dist.csv")

# for wpb in [1,2,4,8,16]:
#     for ps in [1,2,4,8,16,32]:
#         os.system("./0_bench_MGG.py {0} {1}| tee MGG_wpb_{0}_ps_{1}.log".format(wpb, ps))
#         os.system("./1_analysis.py MGG_wpb_{0}_ps_{1}.log {2}".format(wpb, ps, num_gpus))
#         os.system("mv MGG_wpb_{0}_ps_{1}.log logs/".format(wpb, ps))
# os.system("./10_extract_search.py 6 > A100x8_wpb_ps.csv")
# =====================================================================   
# for dist in [1,2,4,8,16,32]:
# for dist in [1,2,3,4]:
# for dist in [8,16,32,64]:
# for dist in [32]:
    # os.system("./0_bench_MGG.py {0}| tee MGG_Dist_{0}.log".format(dist))
    # os.system("./1_analysis.py MGG_Dist_{0}.log {1}".format(dist, 4))
    # os.system("mv MGG_Dist_{0}.log logs/".format(dist))

# num_gpus = 4
# for wpb in [1,2,4,8,16,32]:
# for wpb in [16]:
    # os.system("./0_bench_MGG.py {0}| tee MGG_wpb_{0}.log".format(wpb))
    # os.system("./1_analysis.py MGG_wpb_{0}.log {1}".format(wpb, num_gpus))
    # os.system("mv MGG_wpb_{0}.log logs/".format(wpb))

# for ps in [1,2,4,8,16,32,64,128,256]:
# for ps in [64,128,256]:
# for ps in [16,32]:
    # os.system("./0_bench_MGG.py {0}| tee MGG_ps_{0}.log".format(ps))
    # os.system("./1_analysis.py MGG_ps_{0}.log {1}".format(ps, 4))
    # os.system("mv MGG_ps_{0}.log logs/".format(ps))

# for gpu in num_gpus:
#     os.system("./0_bench_MGG.py {0}| tee MGG_{0}GPU_metrics.log".format(gpu))