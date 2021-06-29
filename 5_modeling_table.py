#!/usr/bin/env python3
import os
import sys

files = os.listdir('.')
fname_li = []
for fname in files:
    if not ".csv" in fname:
        continue
    fname_li.append(fname)

fname_li.sort(lambda x: (int(x.rstrip('.csv').split('_')[-3]), int(x.rstrip('.csv').split('_')[-1]) ) )
print(fname_li)

output_dict = {}
for fname in fname_li:
    fp = open(fname, "r")
    for line in fp:
        time  = float(line.rstrip('\n').split(','))
        break
    output_dict[fname] = time


ldim = 6
for i, fname in enumerate(fname_li):
    if i != 0 and i%ldim == 0:
        print()
    else:
        print(output_dict[fname]+",", end="")

