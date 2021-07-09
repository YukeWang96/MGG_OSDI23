import json
import os
import sys

fname = sys.argv[1]
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

print("=> pagefault_cnt:", total_cnt)