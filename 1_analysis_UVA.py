#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 3:
    raise ValueError("Usage: ./1_log2csv.py result.log num_gpus")

fp = open(sys.argv[1], "r")
num_GPUs = int(sys.argv[2])

dataset_li = []
time_li = []
for line in fp:
    if "Graph File:" in line:
        dataset = line.split("/")[-1].strip('.mtx').strip('\n')
        for i in range(num_GPUs):
            dataset_li.append(dataset)
    if "Time (ms):" in line:
        time = line.split("Time (ms):")[1].rstrip("\n")
        print(time)
        time_li.append(float(time))
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("Dataset,Time (ms)\n")
# print(time_li)

cnt = 0
for data, time in zip(dataset_li, time_li):
    if cnt % num_GPUs == 0:
        tmp_t = max(time_li[cnt:cnt+num_GPUs])
        fout.write("{},{}\n".format(data, tmp_t))
    cnt += 1
fout.close()