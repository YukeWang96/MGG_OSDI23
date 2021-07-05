# Usage: ./main graph.mtx num_GPUs dim nodeOfInterest
nsys profile --stats='true' --gpu-metrics-device=0 --cuda-um-gpu-page-faults='true' --cuda-um-cpu-page-faults='true' --show-output='true' \
build/uvm_profile dataset/cora.mtx 16 16
# build/uvm_profile dataset/Reddit.mtx 0 1024 2