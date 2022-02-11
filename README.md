# Multi-GPU GNNs

# Download libraries and datasets
+ Libraries [link](https://drive.google.com/file/d/1C1N6_v-dt_JChx6RemjLAkk7ulbWADu3/view?usp=sharing): `cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1`. You can also use the following command.
```
wget https://storage.googleapis.com/project_source_files/GLCC/local.tar.gz
tar -zxvf local.tar.gz
```
+ Datasets [link](https://drive.google.com/file/d/1MwxbZJxSXBJrfNWQkD2N655lfNcvenYS/view?usp=sharing). You can also use the following command
```
wget https://project-datasets.s3.us-west-2.amazonaws.com/mgg/dataset.tar.gz
tar -zxvf dataset.tar.gz
```

# Compile

+ Use the following commands
```
cd local/nvshmem_src_2.0.3-0/ 
tar -zxvf build_cu112.tar.gz  # where cu112 stands for CUDA 11.2
cd ../../ && mkdir build && cd build
cd Docker && launch.sh && cd ..
cmake ..
make
```

# Run experiment.
+ Run MGG, `./0_all_run_MGG.py`.
+ Run UVA baseline, `./0_all_run_UVA.py`.