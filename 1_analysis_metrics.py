#!/usr/bin/env python3
import sys 

if len(sys.argv) < 3:
    raise ValueError("Usage: ./1_log2csv.py result.log num_gpus")

fp = open(sys.argv[1], "r")
num_GPUs = int(sys.argv[2])

metric = 'SM [%]'
# metric = 'Achieved Occupancy'

dataset_li = []
time_li = []
for line in fp:
    if "Graph File:" in line:
        dataset = line.split("/")[-1].strip('.mtx').strip('\n')
        dataset_li.append(dataset)
    if metric in line:
        time = line.strip(metric).strip('\n').strip('%').strip()
        print(time)
        time_li.append(float(time))
fp.close()

print(time_li)
# fout = open(sys.argv[1].strip(".log")+".csv", 'w')
# fout.write("Dataset,Time (ms)\n")
# print(time_li)

# cnt = 0
# for data in dataset_li:
#     if cnt % num_GPUs == 0:
#         tmp_t = time_li[int(cnt/num_GPUs)]
#         fout.write("{},{}\n".format(data, tmp_t))
#     cnt += 1
# fout.close()