#!/usr/bin/env python3
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/openmpi-4.1.1/lib/'
os.environ["PATH"] += os.pathsep + 'local/openmpi-4.1.1/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/cudnn-v8.2/lib64'

# os.environ["NVSHMEM_SYMMETRIC_SIZE"] = '3690987520' # paper100M
# os.environ["NVSHMEM_SYMMETRIC_SIZE"] = '7381975040' # paper100M
os.environ["NVSHMEM_SYMMETRIC_SIZE"] = '14763950080' # paper100M

hidden = 16
partSize = 16
num_GPUs = 8

interleaved_dist = int(sys.argv[1])
warpPerblock = int(sys.argv[2])

dataset = [
        ( 'Reddit'                      , 602      	, 41),
        # ( 'enwiki-2013'	                , 100	        , 12),   
        # ( 'it-2004'                     , 128           , 172),
        # ( 'paper100M'                   , 128           , 172),
        # ( 'ogbn-products'	        , 100	        , 47),   
        # ( 'ogbn-proteins'	        , 128		, 112),
        # ( 'com-Orkut'		        , 128		, 128),
]

GPU_avail = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "

pre_condit = GPU_avail + 'OMPI_MCA_plm_rsh_agent=sh\
              mpirun --allow-run-as-root -np {} '.format(num_GPUs)

command = "build/MGG_np_div "

for data, d, c in dataset:
        beg_file = "dataset/bin/{}_beg_pos.bin".format(data)
        csr_file = "dataset/bin/{}_csr.bin".format(data)
        weight_file = "dataset/bin/{}_weight.bin".format(data)
        if data != 'enwiki-2013':
                os.system(pre_condit + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {0}".
                format(command, beg_file, csr_file, weight_file,  
                        num_GPUs, partSize, warpPerblock, 
                        hidden, interleaved_dist, hidden))
        else:
                os.system(pre_condit + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {0}".
                format(command, beg_file, csr_file, weight_file,  
                        num_GPUs, partSize, 16, 
                        hidden, interleaved_dist, hidden))