# Artifact for OSDI'23 paper 
> Yuke Wang, et al. *Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms.* OSDI'23.

# 1. Setup

## 1.1. Clone this project from Github.
```
git clone --recursive git@github.com:YukeWang96/MGG-OSDI23-AE-internal.git
```

## 1.2. Download libraries and datasets.
+ Download libraries (`cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1`).
```
wget https://storage.googleapis.com/mgg_data/local.tar.gz
tar -zxvf local.tar.gz && rm local.tar.gz
tar -zxvf local/nvshmem_src_2.0.3-0/build_cu112.tar.gz
```
+ Download datasets. 
```
wget https://storage.googleapis.com/mgg_data/dataset.tar.gz && tar -zxvf dataset.tar.gz && rm dataset.tar.gz
```

## 1.3. Launch Docker 
```
cd Docker 
./launch.sh
```

## 1.4. Compile implementation.
```
mkdir build && cd build && cmake .. && cd ..
./build.sh
```
# 2. Run initial test experiment.
+ Run MGG initial test, `./0_all_run_MGG.py`.
+ Run UVA initial test, `./0_all_run_UVA.py`.


# 3. Reproduce the major results from paper.

## 3.1 Compare with DGL on 4xA100 and 8xA100 (Fig.7a and Fig.7b).
```
wget https://storage.googleapis.com/mgg_data/graphdata.tar.gz && tar -zxvf graphdata.tar.gz && rm graphdata.tar.gz
./launch_docker.sh
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
> Note that the results can be found at ` ` and compared with ``, respectively. Results of ROC is similar as

| Dataset       | Time (ms) |
|---------------|----------:|
| reddit        |    425.67 |
| enwiki-2013   |    619.33 |
| it-2004       |   5160.18 |
| paper100M     |   8179.35 |
| ogbn-products |    529.74 |
| ogbn-proteins |    423.82 |
| com-orkut     |    571.62 |


## 3.4 Compare NP with w/o NP (Fig.10a).
```
python 2_MGG_NP.py
```
> Note that the results can be found at `MGG_NP_study.csv`. Similar to following table.

| Dataset      | MGG_WO_NP | MGG_W_NP | Speedup (x) |
|--------------|----------:|---------:|------------:|
| Reddit       |    76.797 |   16.716 |       4.594 |
| enwiki-2013  |   290.169 |   88.249 |       3.288 |
| ogbn-product |    86.362 |   26.008 |       3.321 |

## 3.5 Compare WL with w/o WL (Fig.10b).
```
python 3_MGG_WL.py
```
> Note that the results can be found at `MGG_WL_study.csv`. Results are similar to

| Dataset      | MGG_WO_NP | MGG_W_NP | Speedup (x) |
|--------------|----------:|---------:|------------:|
| Reddit       |    75.035 |    18.92 |       3.966 |
| enwiki-2013  |   292.022 |  104.878 |       2.784 |
| ogbn-product |    86.632 |   29.941 |       2.893 |

## 3.6 Compare API (Fig.10c).
```
python 4_MGG_API.py
```
> Note that the results can be found at `MGG_API_study.csv`. Results are similar to 

| Norm.Time w.r.t. Thread | MGG_Thread | MGG_Warp | MGG_Block |
|-------------------------|------------|----------|-----------|
| Reddit                  | 1.0        | 0.299    | 0.295     |
| enwiki-2013             | 1.0        | 0.267    | 0.263     |
| ogbn-product            | 1.0        | 0.310    | 0.317     |


## 3.7 Design Space Search (Fig.11a)
```
python 5_MGG_DSE_4GPU.py
```
> Note that the results can be found at `Reddit_4xA100_dist_ps.csv` and `Reddit_4xA100_dist_wpb.csv`. Results similar to 

+ `Reddit_4xA100_dist_ps.csv`

| dist\ps |      1 |      2 |      4 |      8 |     16 |     32 |
|---------|-------:|-------:|-------:|-------:|-------:|-------:|
| 1       | 17.866 | 17.459 | 16.821 | 16.244 | 16.711 | 17.125 |
| 2       | 17.247 | 16.722 | 16.437 | 16.682 | 17.053 | 17.808 |
| 4       | 16.826 |  16.41 | 16.583 | 17.217 | 17.627 | 18.298 |
| 8       | 16.271 | 16.725 | 17.193 | 17.655 | 18.426 |  18.99 |
| 16      | 16.593 | 17.214 | 17.617 | 18.266 | 19.009 | 19.909 |

+ `Reddit_4xA100_dist_wpb.csv`

| dist\wpb |      1 |      2 |      4 |      8 |     16 |
|----------|-------:|-------:|-------:|-------:|-------:|
| 1        | 34.773 | 23.164 | 16.576 | 15.235 | 16.519 |
| 2        | 34.599 | 23.557 | 17.254 | 15.981 |  19.56 |
| 4        | 34.835 | 23.616 | 17.674 | 17.034 | 22.084 |
| 8        | 34.729 | 23.817 | 18.302 | 18.708 | 25.656 |
| 16       | 34.803 | 24.161 | 18.879 |  23.44 | 32.978 |



```
python 5_MGG_DSE_8GPU.py
```
> Note that the results can be found at `Reddit_8xA100_dist_ps.csv` and `Reddit_8xA100_dist_wpb.csv`. Results similar to



## Reference
* **NVIDIA OpenSHMEM Library (NVSHMEM) Documentation.** <br>
https://docs.nvidia.com/nvshmem/api/index.html

* **NVIDIA Unified Memory.** <br>
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

* **cuDNN Example for MNIST.** <br>
https://github.com/haanjack/mnist-cudnn

* [**Deep Graph Library**](https://github.com/dmlc/dgl) <br>
Wang, Minjie, et al. 
**Deep graph library: A graph-centric, highly-performant package for graph neural networks.**. *The International Conference on Learning Representations (ICLR'19).*

* [**ROC**](https://github.com/jiazhihao/ROC) <br>
Jia, Zhihao, et al. 
**Improving the accuracy, scalability, and performance of graph neural networks with roc.** *Proceedings of Machine Learning and Systems 2 (MLsys'20).*

* [**GNNAdvisor**](https://github.com/YukeWang96/OSDI21_AE) <br>
Wang, Yuke, et al. **GNNAdvisor: An adaptive and efficient runtime system for GNN acceleration on GPUs.** *15th USENIX symposium on operating systems design and implementation (OSDI'21)*.

* [**GE-SpMM**](https://github.com/hgyhungry/ge-spmm) <br>
Huang, Guyue, et al. **Ge-spmm: General-purpose sparse matrix-matrix multiplication on gpus for graph neural networks.** *International Conference for High Performance Computing, Networking, Storage and Analysis (SC'20)*.
