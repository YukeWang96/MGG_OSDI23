#!/usr/bin/env python3
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + '$PWD/local/openmpi-4.1.1/lib/'
os.environ["PATH"] += os.pathsep + '$PWD/local/openmpi-4.1.1/openmpi-4.1.1/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + '$PWD/local/cudnn-v8.2/lib64'

model = 'gcn'

# hidden = [16,32,64,128,256]
# hidden = [int(sys.argv[1])]
hidden = [256]

num_GPUs = 2
# num_GPUs = int(sys.argv[1])

partSize = 10
# partSize = 180
# partSize = int(sys.argv[1])

warpPerblock = 4
# warpPerblock = int(sys.argv[1])

interleaved_dist = 1
# interleaved_dist = int(sys.argv[1])

# single = False
# basic_MGG = False
# interleaved_MGG = False

# host2device_unified_mem = False
# device2device = False
test_neighbor_part = True

dataset = [
        # ('toy'	        , 3	    , 2   ),  
        # ('tc_gnn_verify'	, 16	, 2),
        # ('tc_gnn_verify_2x'	, 16	, 2),

        ('citeseer'	        		, 3703	    , 6   ),  
        ('cora' 	        		, 1433	    , 7   ),  
        ('pubmed'	        		, 500	    , 3   ),      
        ('ppi'	            		, 50	    , 121 ),   
        
        # ('PROTEINS_full'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeast'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('SW-620H'                   , 66       , 2) ,

        # ( 'amazon0505'               , 96	  , 22),
        # ( 'artist'                   , 100	  , 12),
        # ( 'com-amazon'               , 96	  , 22),
        # ( 'soc-BlogCatalog'	         , 128	  , 39),      
        # ( 'amazon0601'  	         , 96	  , 22), 

        # ( 'Reddit'                      , 602      	, 41),
        # ( 'enwiki-2013'	                , 100	    , 12),      
        # ( 'ogbn-products'	            , 100	    , 47),
        # ( 'ogbn-proteins'		        , 8		    , 112),
        # ( 'com-Orkut'				    , 128		, 128),
        # ( 'web-Google'				    , 128		, 128),
        # ( 'wiki-Talk'				    , 128		, 128),

       # ( 'amazon_also_bought'          , 96        , 22),
        # ( 'ogbn-arxiv'		            , 128	    , 40),
        # ('kmer_V1r', 128,128),
        # ('mawi_201512020330', 128,128),
        # ('YeastH'                    , 75       , 2) ,   
        # ( 'web-BerkStan'             , 100	  , 12),
        # ( 'wiki-topcats'             , 300	  , 12),
        # ( 'COLLAB'                   , 100      , 3) ,
        # ( 'wiki-topcats'             , 300	  , 12),
]


data_path = './dataset'
pre_condit = 'CUDA_VISIBLE_DEVICES=0,1,2,3 OMPI_MCA_plm_rsh_agent=sh \
              mpirun --allow-run-as-root -np {} '.format(num_GPUs)

# if single:
#     command = "./main_single data_path"
# if basic_MGG:
#     command = "./main_mgg {}".format(num_GPUs, data_path)
# if interleaved_MGG:	
#     command = "./main_mgg_interleave {}".format(num_GPUs, data_path)


# if host2device_unified_mem:
#     command = "./host2device_unified_mem {}".format(data_path)
# if device2device:
#     command = "./device2device {}".format(data_path)
if test_neighbor_part:
    command = "./test_neighbor_part {}".format(data_path)

assert command is not ""

for hid in hidden:
    for data, d, c in dataset:
        os.system(pre_condit + command + "{0}.mtx {1} {2} {3} {4} {5}".\
        format(data, num_GPUs, partSize, warpPerblock, hid, interleaved_dist))