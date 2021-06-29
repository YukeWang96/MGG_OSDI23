#!/usr/bin/env python3
import os
import sys

if len(sys.argv) < 2:
    print("usage: ./exe ldim")

files = os.listdir('.')
fname_li = []
for fname in files:
    if not ".csv" in fname:
        continue
    fname_li.append(fname)

fname_li.sort(key=lambda x: (int(x.rstrip('.csv').split('_')[-3]), int(x.rstrip('.csv').split('_')[-1]) ) )
# print(fname_li)

output_dict = {}
for fname in fname_li:
    fp = open(fname, "r")
    for line in fp:
        time  = float(line.rstrip('\n').split(',')[1])
        break
    output_dict[fname] = time


ldim = int(sys.argv[1])
# ldim = 5
for i, fname in enumerate(fname_li):
    if i != 0 and i%ldim == 0:
        print()
    print(str(output_dict[fname])+",", end="")

print()