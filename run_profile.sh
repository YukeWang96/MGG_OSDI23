# ./6_profile_uvm.py | tee profile_uvm.log
# ./1_analysis_profile.py profile_uvm.log 9

./7_profile_mgg.py | tee profile_mgg.log
./1_analysis_profile.py profile_mgg.log 9

# nsys profile --stats='true' --gpu-metrics-device=0 --cuda-um-gpu-page-faults='true' --cuda-um-cpu-page-faults='true' --show-output='true' \
# nsys profile --stats=true \
#             --force-overwrite=true	\
#             --cuda-um-gpu-page-faults=true \
#             --cuda-um-cpu-page-faults=true \
#             --show-output=true \
#             --export=json \
#             build/uvm_profile dataset/Reddit.mtx 16 16