#!/usr/bin/env python3
import os, sys

num_div = int(sys.argv[1])
files = os.listdir("./")
files.sort()
# print(files)

csvs = []
for name in files:
    if ".csv" in name and "MGG" in name:
        csvs.append(name)
# print(csvs)
csvs.sort(key=lambda x: (int(x.split("_")[2]), int(x.split("_")[4].strip(".csv"))))


out = []
for fname in csvs:
    fp = open(fname)
    for line in fp:
       out.append((fname.strip('.csv'), line.split(",")[1].strip('\n')))

for idx, it in enumerate(out):
    print(it[1], end=",")
    if (idx + 1) % num_div == 0:
        print()