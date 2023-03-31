# Artifact for OSDI'23 paper 
> Yuke Wang, et al. *Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms.* OSDI'23.

# 1. Setup

## 1.1. Clone this project from Github.
```
git clone --recursive git@github.com:YukeWang96/MGG_new.git
```

## 1.2. Download libraries and datasets.
+ Download libraries (`cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1`).
```
wget https://storage.googleapis.com/mgg_data/local.tar.gz
tar -zxvf local.tar.gz
```
+ Download datasets. (around 3 minutes)
```
wget https://storage.googleapis.com/mgg_data/dataset.tar.gz
tar -zxvf dataset.tar.gz
```

## 1.3. Launch Docker 
```
tar -zxvf local/nvshmem_src_2.0.3-0/build_cu112.tar.gz 
cd Docker 
./launch.sh
```

## 1.4. Compile implementation.
```
mkdir build && cd build
./build.sh
```
# 2. Run initial test experiment.
+ Run MGG initial test, `./0_all_run_MGG.py`.
+ Run UVA initial test, `./0_all_run_UVA.py`.


# 3. Reproduce the major results from paper.

## Run MGG on GCN.
```
./0_test_MGG_4GPU.py
./0_test_MGG_8GPU.py
```


## Compare with DGL on 4xA100 and 8xA100 (Fig.7a and Fig.7b).
```
cd Docker
./launch.sh
wget https://storage.googleapis.com/mgg_data/graphdata.tar.gz
./0_run_pydirect.sh
```

## Compare with UVM on 4xA100 and 8xA100 (Fig.8a and Fig.8b).
```
./0_test_UVM_4GPU.py
./0_test_UVM_8GPU.py
```

## Compare with ROC on 8xA100 (Fig.9).
```
cd mgg-roc-internal
git submodule update --init --recursive
wget https://storage.googleapis.com/mgg_data/data.tar.gz && tar -zxvf data.tar.gz && rm -rf data.tar.gz
./docker/launch.sh
```

## Compare NP with w/o NP (Fig.10a).
```
python 3_MGG_NP.py
```

## Compare WL with w/o WL (Fig.10b).
```
python 4_MGG_WL.py
```

## Compare API (Fig.10c).


## Design Space Search (Fig.11a)
```
python 6_MGG_DSE.py
```