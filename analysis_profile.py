#!/usr/bin/env python3
import sys 

if len(sys.argv) < 3:
    raise ValueError("Usage: ./main.py result.log ldm")

fp = open(sys.argv[1], "r")
ldm = int(sys.argv[2])

time_li = []
for line in fp:
    if "MPI time (ms)" in line:
        time = line.strip("MPI time (ms)").strip('\n')
        print(time)
        time_li.append(float(time))
    if "Time (ms)" in line:
        time = line.strip("Time (ms):").strip('\n')
        print(time)
        time_li.append(float(time))
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
# fout.write("Dataset,Time (ms)\n")
# print(time_li)

cnt = 0
for time_val in time_li:
    if (cnt + 1) % ldm == 0:
        fout.write("{}\n".format(time_val))
    else:
        fout.write("{},".format(time_val))
    cnt += 1
fout.close()