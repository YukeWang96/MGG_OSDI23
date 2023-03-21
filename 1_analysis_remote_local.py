#!/usr/bin/env python3
import sys 
import statistics

if len(sys.argv) < 3:
    raise ValueError("Usage: ./1_log2csv.py result.log num_gpus")

fp = open(sys.argv[1], "r")
num_GPUs = int(sys.argv[2])

dataset_li = []
local_li = []
remote_li = []
for line in fp:
    if "Graph File:" in line:
        dataset = line.split("/")[-1].strip('.mtx').strip('\n')
        dataset_li.append(dataset)
    if "local" in line:
        line_spt = line.split(":")
        # print(line_spt)
        local_li.append(int(line_spt[1].split(",")[0]))
        remote_li.append(int(line_spt[-1].strip('\n')))
fp.close()

# print(local_li)
# print(remote_li)
local_li_new = [min(local_li[i:i+num_GPUs])*1.0/num_GPUs for i in range(0, len(local_li), num_GPUs)]
remote_li_new = [min(remote_li[i:i+num_GPUs])*1.0/num_GPUs for i in range(0, len(remote_li), num_GPUs)]

dataset_li_new = [dataset_li[i] for i in range(0, len(dataset_li), num_GPUs)]

# print(local_li_new)
# print(remote_li_new)

for dat, local, remote in zip(dataset_li_new, local_li_new, remote_li_new):
    print(dat,',', local,',',remote)
# print(time_li)
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