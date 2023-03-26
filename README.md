# Artifact for OSDI'23 paper "Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms"

# 1. Setup

## 1.1. Clone this project from Github.
```
git clone git@github.com:YukeWang96/MGG_new.git
```

## 1.2. Download libraries and datasets.
> + Download libraries, including `cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1` with the following commands. (around 1 minute)
```
wget https://storage.googleapis.com/project_source_files/GLCC/local.tar.gz
tar -zxvf local.tar.gz
```
+ Download datasets. (around 0.5 minute)
```
cd dataset
wget https://storage.googleapis.com/project_source_files/GLCC/graph_bin.tar.gz
tar -zxvf graph_bin.tar.gz
cd ..
```

## 1.3. Launch Docker 
```
cd local/nvshmem_src_2.0.3-0/ 
tar -zxvf build_cu112.tar.gz 
cd ../../
cd Docker 
launch.sh
```

## 1.4. Compile implementation.
```
mkdir build && cd build
./build.sh
```
# 2. Run initial test experiment.
+ Run MGG, `./0_all_run_MGG.py`.
+ Run UVA baseline, `./0_all_run_UVA.py`.


# 3. Reproduce the major results from paper.
+ 
