./6_profile_uvm.py | tee profile_uvm.log
./1_analysis_profile.py profile_uvm.log 9

./7_profile_mgg.py | tee profile_mgg.log
./1_analysis_profile.py profile_mgg.log 9