#!/usr/bin/env python3
import os
import sys

dim_li = [16,32,64,128,256,512,1024]
neighbors_li = [2,4,8,16,32,64,128,256,512]

# dim_li = [16]
# neighbors_li = [32]

for dim in dim_li:
    for nbs in neighbors_li:
        os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh \
                mpirun --allow-run-as-root -np 2 build/mgg_profile dataset/ppi.mtx 2 {} {}".format(dim, nbs))
# Usage: ./main graph.mtx num_GPUs dim nodeOfInterest