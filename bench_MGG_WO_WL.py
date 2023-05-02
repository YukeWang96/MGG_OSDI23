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
num_GPUs = int(sys.argv[1])
warpPerblock = 4 
partSize = 16
interleaved_dist = 1024

dataset = [
        ( 'Reddit'                      , 602      	, 41),
        ( 'enwiki-2013'	                , 100	        , 12),   
        ( 'ogbn-products'	        , 100	        , 47),   
]

# GPU_avail = "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
GPU_avail = "CUDA_VISIBLE_DEVICES=4,5,6,7 "

pre_condit = GPU_avail + 'OMPI_MCA_plm_rsh_agent=sh\
              mpirun --allow-run-as-root -np {} '.format(num_GPUs)

# command = "build/MGG "
# command = "build/MGG_basic "
# command = "build/MGG_np "
# command = "build/MGG_np_div "
command = "build/MGG_np_div_kernel "
# command = "build/MGG_np_pipeline "

for data, in_dim, out_dim in dataset:
        beg_file = "dataset/bin/{}_beg_pos.bin".format(data)
        csr_file = "dataset/bin/{}_csr.bin".format(data)
        weight_file = "dataset/bin/{}_weight.bin".format(data)
        if data != 'enwiki-2013':
                os.system(pre_condit + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}".
                format(command, beg_file, csr_file, weight_file,  
                        num_GPUs, partSize, warpPerblock, interleaved_dist,
                        in_dim, hidden, out_dim))
        else:
                os.system(pre_condit + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}".
                format(command, beg_file, csr_file, weight_file,  
                        num_GPUs, partSize, 16, interleaved_dist, 
                        in_dim, hidden, out_dim))