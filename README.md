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

## 3.1 Compare with DGL on 4xA100 and 8xA100 (Fig.7a and Fig.7b).
```
wget https://storage.googleapis.com/mgg_data/graphdata.tar.gz && tar -zxvf graphdata.tar.gz
cd Docker
./launch.sh
conda activate dgl
./0_run_dgl_gcn.sh
```
> Note that the results can be found at `1_dgl_gcn.csv`.


## 3.2 Compare with UVM on 4xA100 and 8xA100 (Fig.8a and Fig.8b).
```
./0_run_MGG_UVM_4G.sh
./0_run_MGG_UVM_8G.sh
```
> Note that the results can be found at `UVM_MGG_4GPU_study.csv` and `UVM_MGG_8GPU_study.csv`.


## 3.3 Compare with ROC on 8xA100 (Fig.9).
```
cd mgg-roc-internal
git submodule update --init --recursive
wget https://storage.googleapis.com/mgg_data/data.tar.gz && tar -zxvf data.tar.gz && rm -rf data.tar.gz
./docker/launch.sh
```
> Note that the results can be found at ` ` and compared with ``, respectively.

## 3.4 Compare NP with w/o NP (Fig.10a).
```
python 2_MGG_NP.py
```

> Note that the results can be found at `MGG_NP_study.csv`.

## 3.5 Compare WL with w/o WL (Fig.10b).
```
python 3_MGG_WL.py
```
> Note that the results can be found at `MGG_WL_study.csv`.

## 3.6 Compare API (Fig.10c).
```
python 4_MGG_API.py
```
> Note that the results can be found at `MGG_API_study.csv`.

## 3.7 Design Space Search (Fig.11a)
```
python 5_MGG_DSE.py
```
> Note that the results can be found at `Reddit_8xA100_dist_ps.csv` and `Reddit_8xA100_dist_wpb.csv`.


## Reference
* **NVIDIA OpenSHMEM Library (NVSHMEM) Documentation.** <br>
https://docs.nvidia.com/nvshmem/api/index.html

* **NVIDIA Unified Memory.** <br>
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

* **cuDNN Example for MNIST.** <br>
https://github.com/haanjack/mnist-cudnn

* [**Deep Graph Library**](https://github.com/dmlc/dgl) <br>
Wang, Minjie, et al. 
**Deep graph library: A graph-centric, highly-performant package for graph neural networks.**. *The International Conference on Learning Representations (ICLR), 2019.*

* [**ROC**](https://github.com/jiazhihao/ROC) <br>
Jia, Zhihao, et al. 
**Improving the accuracy, scalability, and performance of graph neural networks with roc.** *Proceedings of Machine Learning and Systems 2 (2020): 187-198*.

