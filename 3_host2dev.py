#!/usr/bin/env python3
import os
import sys

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/openmpi-4.1.1/lib/'
os.environ["PATH"] += os.pathsep + 'local/openmpi-4.1.1/openmpi-4.1.1/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'local/cudnn-v8.2/lib64'

hidden = 16
num_GPUs = 2

dataset = [
        # ('citeseer'	        		, 3703	    , 6   ),  
        # ('cora' 	        		, 1433	    , 7   ),  
        # ('pubmed'	        		, 500	    , 3   ),      
        # ('ppi'	            		, 50	    , 121 ),   
        
        # ('PROTEINS'             , 29       , 2) ,   
        # ('OVCAR-8H'                  , 66       , 2) , 
        # ('Yeast'                     , 74       , 2) ,
        # ('DD'                        , 89       , 2) ,
        # ('SW-620H'                   , 66       , 2) ,

        # ( 'amazon0505'               , 96	  , 22),
        # ( 'artist'                   , 100	  , 12),
        # ( 'com-amazon'               , 96	  , 22),
        # ( 'soc-BlogCatalog'	         , 128	  , 39),      
        # ( 'amazon0601'  	         , 96	  , 22), 

        ( 'Reddit'                      , 602      	, 41),
        ( 'enwiki-2013'	                , 100	    , 12),      
        # ( 'ogbn-products'	            , 100	    , 47),
        # ( 'ogbn-proteins'		        , 8		    , 112),
        # ( 'com-Orkut'				    , 128		, 128),
        # ( 'web-Google'				    , 128		, 128),
        # ( 'wiki-Talk'				    , 128		, 128),
]


data_path = 'dataset/'
prefix = 'CUDA_VISIBLE_DEVICES=0,1,2,3 OMPI_MCA_plm_rsh_agent=sh \
              mpirun -np {} '.format(num_GPUs)
command = "build/HOST2DEV {}".format(data_path)

for data, d, c in dataset:
    os.system(prefix + command + "{0}.mtx {1}".\
    format(data, num_GPUs))