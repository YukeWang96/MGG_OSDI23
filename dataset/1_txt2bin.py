#!/usr/bin/env python3
import os

graph = os.listdir("base_0/")

for gr in graph:
    os.system("./text_to_bin.bin base_0/{} 0 0 32".format(gr))
    # break