#!/usr/bin/env python3
import os
import sys

graph = os.listdir(".")

for gr in graph:
    if ".mtx" not in gr: continue
    fp = open("{}".format(gr))
    fout = open("{}/{}".format("base_0", gr.strip(".mtx")), "w")
    print(gr)
    flag = False
    for line in fp:
        if "%" in line: continue
        elif flag == False: 
            flag = True 
            continue 
        tmp = line.rstrip("\n").split()
        src, trg = int(tmp[0]) - 1, int(tmp[1]) - 1
        fout.write("{} {}\n".format(src, trg))
    fp.close()
    fout.close()