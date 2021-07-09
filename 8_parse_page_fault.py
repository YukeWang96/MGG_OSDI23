import json
import os
import sys

files = os.listdir('.')

flist = []
for fname in files:
    if '.json' in fname:
        flist.append(fname)

flist.sort()

for fname in flist:

    f = open(fname)

    page_li = []
    for line in f:
        if 'CudaUvmGpuPageFaultEvent' in line:
            page_li.append(line)

    total_cnt = 0
    for line in page_li:
        test = json.loads(line)
        pagefaults = test['CudaUvmGpuPageFaultEvent']['numberOfPageFaults']
        total_cnt += int(pagefaults)

    print("{}, {}".format(fname, total_cnt))