#!/usr/bin/env python3
import os, sys

num_div = int(sys.argv[1])
files = os.listdir("./")
files.sort()
# print(files)

csvs = []
for name in files:
    if ".csv" in name and "MGG_dist" in name and "ps" in name:
        csvs.append(name)
# print(csvs)
csvs.sort(key=lambda x: (int(x.split("_")[2]), int(x.split("_")[4].strip(".csv"))))


out = []
for fname in csvs:
    fp = open(fname)
    for line in fp:
       out.append((fname.strip('.csv'), line.split(",")[1].strip('\n')))

print("dist\ps,1,2,4,8,16,32")
row_head = [1,2,4,8,16]
for idx, it in enumerate(out):
    if idx == 0:
        print(row_head.pop(0), end=",")

    print(it[1], end=",")

    if (idx + 1) % num_div == 0:
        print()
        if idx + 1 < len(out):
            # print(idx, len(out))
            # print(row_head)
            print(row_head.pop(0), end=",")