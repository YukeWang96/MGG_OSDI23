# Multi-GPU GNNs

# Download libraries and datasets
+ Libraries [link](https://drive.google.com/file/d/1C1N6_v-dt_JChx6RemjLAkk7ulbWADu3/view?usp=sharing): `cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1`.
+ Datasets [link](https://drive.google.com/file/d/1MwxbZJxSXBJrfNWQkD2N655lfNcvenYS/view?usp=sharing).
+ `tar -zxvf local.tar.gz`
+ `tar -zxvf dataset.tar.gz`

# Compile
+ `mkdir build && cd build`
+ `cmake ..`

# Run experiment.
+ Change the path to corresponding library at `0_bench.py`.
```
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'path/to/openmpi/lib/'
os.environ["PATH"] += os.pathsep + 'path/to/openmpi/bin/'
os.environ["LD_LIBRARY_PATH"] += os.pathsep + 'path/to/cuDNN/lib64'
```
+ `0_all_run.py`, it will generate report in `*.csv`.