#!/usr/bin/env python3
import os
import sys

dimension = 128
num_GPU = 2

dataset = [
        ( 'Reddit'        , 232965),
        ( 'enwiki-2013'	  , 4203323),
        ( 'ogbn-products' , 1511819904),
]

for data, nodes in dataset:
    os.system("build/NCCL {} {}".format(num_GPU, int(nodes/num_GPU)*dimension))