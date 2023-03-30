#!/usr/bin/env python3
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/openmpi-4.1.1/lib/'
os.environ["PATH"] += os.pathsep + 'local/openmpi-4.1.1/openmpi-4.1.1/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/cudnn-v8.2/lib64'

hidden = 64
# hidden = [int(sys.argv[1])]

# num_GPUs = 2
num_GPUs = int(sys.argv[1])

partSize = 16
# partSize = int(sys.argv[1])

warpPerblock = 4
# warpPerblock = int(sys.argv[1])

dataset = [
        ( 'Reddit'                      , 602      	, 41),
        ( 'enwiki-2013'	                , 100	        , 12),   
        ( 'it-2004'                     , 128           , 172),
        ( 'paper100M'                   , 128           , 172),
        ( 'ogbn-products'	        , 100	        , 47),   
        ( 'ogbn-proteins'	        , 128		, 112),
        ( 'com-Orkut'		        , 128		, 128),
]

GPU_avail = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "

command = GPU_avail + "build/unified_memory_updated "
# command = GPU_avail + "build/unified_memory "
# command = "build/MGG_gcn_2layer "
# command = "build/MGG_sgc_2layer "
# command = "build/MGG_agnn_2layer "

for data, d, c in dataset:
        beg_file = "dataset/bin/{}_beg_pos.bin".format(data)
        csr_file = "dataset/bin/{}_csr.bin".format(data)
        weight_file = "dataset/bin/{}_weight.bin".format(data)
        os.system(command + "{0} {1} {2} {3} {4} {5} {6} {7} {8}".\
                format(beg_file, csr_file, weight_file, 
                        num_GPUs, partSize, warpPerblock, 
                        hidden, hidden, hidden))