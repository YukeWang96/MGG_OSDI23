#!/usr/bin/env python3
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/openmpi-4.1.1/lib/'
os.environ["PATH"] += os.pathsep + 'local/openmpi-4.1.1/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/cudnn-v8.2/lib64'

hidden = 32
# hidden = [int(sys.argv[1])]

# num_GPUs = 4
num_GPUs = int(sys.argv[1])
###############################################
partSize = 4
warpPerblock = 16
interleaved_dist = 2
###############################################
# # interleaved_dist = 1
# interleaved_dist = int(sys.argv[1])

# # partSize = 16
# partSize = int(sys.argv[2])

# warpPerblock = 1
# # warpPerblock = int(sys.argv[1])
###############################################
# partSize = 8
# # partSize = int(sys.argv[2])

# # warpPerblock = 8
# warpPerblock = int(sys.argv[1])

# # interleaved_dist = 1
# interleaved_dist = int(sys.argv[2])
###############################################

dataset = [
        # ('citeseer'	        		, 3703	    , 6   ),  
        # ('cora' 	        		, 1433	    , 7   ),  
        # ('pubmed'	        		, 500	    , 3   ),      
        # ('ppi'	            		, 50	    , 121 ),   
        
        # ('PROTEINS'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeas'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('SW-620H'                   , 66       , 2) ,

        # ( 'amazon0505'               , 96	  , 22),
        # ( 'artist'                   , 100	  , 12),
        # ( 'com-amazon'               , 96	  , 22),
        # ( 'soc-BlogCatalog'	         , 128	  , 39),      
        # ( 'amazon0601'  	         , 96	  , 22), 

        ( 'Reddit'                      , 128      	, 41),
        ( 'enwiki-2013'	                , 100	        , 12),      
        ( 'ogbn-products'	        , 100	        , 47),
        ( 'ogbn-proteins'		, 128		, 112),
        # ( 'com-Orkut'		        , 128		, 128),

        # ( 'web-Google'				    , 128		, 128),
        # ( 'wiki-Talk'				    , 128		, 128),
]

pre_condit = 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh\
              mpirun --allow-run-as-root -np {} '.format(num_GPUs)

# command = "build/MGG "
# command = "build/MGG_basic "
# command = "build/MGG_np "
command = "build/MGG_np_div "
# command = "build/MGG_np_pipeline "

for data, d, c in dataset:
        beg_file = "dataset/bin/{}_beg_pos.bin".format(data)
        csr_file = "dataset/bin/{}_csr.bin".format(data)
        weight_file = "dataset/bin/{}_weight.bin".format(data)
        os.system(pre_condit + "{0} {1} {2} {3} {4} {5} {6} {7} {8} {0}".
        format(command, beg_file, csr_file, weight_file,  
                num_GPUs, partSize, warpPerblock, 
                hidden, interleaved_dist, hidden))
#     os.system(pre_condit + command + "{0}.mtx {1} {2} {3} {4} {5} {6}".\
#     format(data, num_GPUs, partSize, warpPerblock, hidden, interleaved_dist, hidden))
       