#!/usr/bin/env python3
import os
import sys

dim_li = [16,32,64,128,256,512,1024]
neighbors_li = [2,4,8,16,32,64,128,256,512]

# dim_li = [16]
# neighbors_li = [32]

for dim in dim_li:
    for nbs in neighbors_li:
        os.system("build/uvm_profile dataset/cora.mtx {} {}".format(dim, nbs))
# Usage: ./main graph.mtx num_GPUs dim nodeOfInterest
# nsys profile --stats='true' --gpu-metrics-device=0 --cuda-um-gpu-page-faults='true' --cuda-um-cpu-page-faults='true' --show-output='true' \
# build/uvm_profile dataset/cora.mtx 16 16