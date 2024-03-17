# Artifact for OSDI'23 paper 
> Yuke Wang, et al. *Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms.* OSDI'23.

[**[Paper]**](https://arxiv.org/pdf/2209.06800.pdf) 
[**[Bibtex]**](https://github.com/YukeWang96/personal_page/blob/7bf44569abca19ac93fc5f0ffddb2e9fed1f2c29/_publications/publication.bib#L143-L148)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7853910.svg)](https://doi.org/10.5281/zenodo.7853910)

# 1. Setup (Skip to Section-2 if evaluated on provided GCP)

## 1.1. Clone this project from Github.
```
git clone --recursive git@github.com:YukeWang96/MGG-OSDI23-AE.git
```

## 1.2. Download libraries and datasets.
+ Download libraries (`cudnn-v8.2, nvshmem_src_2.0.3-0, openmpi-4.1.1`).
```
wget https://proj-dat.s3.us-west-1.amazonaws.com/local.tar.gz
tar -zxvf local.tar.gz && rm local.tar.gz
tar -zxvf local/nvshmem_src_2.0.3-0/build_cu112.tar.gz
wget https://proj-dat.s3.us-west-1.amazonaws.com/dataset.tar.gz && tar -zxvf dataset.tar.gz && rm dataset.tar.gz
```

+ Setup baseline DGL
```
cd dgl_pydirect_internal
wget https://proj-dat.s3.us-west-1.amazonaws.com/graphdata.tar.gz && tar -zxvf graphdata.tar.gz && rm graphdata.tar.gz
cd ..
```

+ Setup baseline ROC

```
wget https://proj-dat.s3.us-west-1.amazonaws.com/roc-new.tar.gz && tar -zxvf roc-new.tar.gz && rm roc-new.tar.gz
```

## 1.3. Launch Docker for MGG.
```
cd docker 
./launch.sh
```

## 1.4. Compile implementation.
```
mkdir build && cd build && cmake .. && cd ..
./0_mgg_build.sh
```
# 2. Run initial test experiment.
+ Please try study experiments in below **Section-3.4** and **Section-3.5**


# 3. Reproduce the major results from paper.

## 3.1 Compare with UVM on 4xA100 and 8xA100 (Fig.8a and Fig.8b).
```
./0_run_MGG_UVM_4GPU_GCN.sh
./0_run_MGG_UVM_4GPU_GIN.sh
./0_run_MGG_UVM_8GPU_GCN.sh
./0_run_MGG_UVM_8GPU_GIN.sh
```
> Note that the results can be found at `Fig_8_UVM_MGG_4GPU_GCN.csv`, `Fig_8_UVM_MGG_4GPU_GIN.csv`, `Fig_8_UVM_MGG_8GPU_GCN.csv`, and `Fig_8_UVM_MGG_8GPU_GIN.csv`.


## 3.2 Compare with DGL on 8xA100 for GCN and GIN (Fig.7a and Fig.7b).
```
./launch_docker.sh
cd gcn/
./0_run_gcn.sh
cd ../gin/
./0_run_gin.sh
```

> Note that the results can be found at `1_dgl_gin.csv` and `1_dgl_gcn.csv` and our MGG reference is in `MGG_GCN_8GPU.csv` and `MGG_8GPU_GIN.csv`.


## 3.3 Compare with ROC on 8xA100 (Fig.9).
```
cd roc-new/docker
./launch.sh
./run_all.sh
```
> Note that the results can be found at `Fig_9_ROC_MGG_8GPU_GCN.csv` and `Fig_9_ROC_MGG_8GPU_GIN.csv`.

Results of ROC is similar as

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
> Note that the results can be found at `Reddit_8xA100_dist_ps.csv` and `Reddit_8xA100_dist_wpb.csv`.


# 4. Use MGG as a Tool or Library for your project.

Building a new design based on MGG with NVSHMEM is simple, there are only several steps:

## 4.1 Build the C++ design based on our existing examples 
+ Create a new `.cu` file under `src/`. An example is shown below.

https://github.com/YukeWang96/MGG_OSDI23/blob/9f2e7abc6ef433b6d0f6a4f7e88be162f948df75/src/mgg_np_div_kernel.cu#L78-L87

## 4.2 Build the CUDA kernel design based on our existing examples. 
+ Add a kernel design in `include/neighbor_utils.cuh`. An example is shown below.

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/include/neighbor_utils.cuh#L787-L802

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/include/neighbor_utils.cuh#L1351-L1366

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/include/neighbor_utils.cuh#L277C1-L292

## 4.3 Register the new design to CMake. 
+ Add a compilation entry in `CMakeLists.txt`).
+ Add a command `make filename.cu` in `0_mgg_build.cu`. 
+ An example is shown below. Note that please match the filename with your newly created `.cu` in step-1.

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/CMakeLists.txt#L60-L64

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/CMakeLists.txt#L218-L249

## 4.4 Launch the MGG docker and recompile, 
+ The compiled executable will be located under `build/`.
```
cd docker 
./launch.sh
cd build && cmake ..
cd .. && ./0_mgg_build.sh
```

## 4.5 Run the compiled executable.

https://github.com/YukeWang96/MGG_OSDI23/blob/73e1866f23d001491f0c69d5216dec680593de27/bench_MGG.py#L5-L51

## Reference
* **NVIDIA OpenSHMEM Library (NVSHMEM).** <br>
https://docs.nvidia.com/nvshmem/api/index.html

* **NVIDIA Unified Memory.** <br>
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

* **NVIDIA Unified Virtual Memory.** <br>
https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/

* **NVIDIA cuBLAS.** <br>
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Level-3/gemm

* **cuDNN Example for MNIST.** <br>
https://github.com/haanjack/mnist-cudnn

* **graph_project_start** <br>
Hang Liu. https://github.com/asherliu/graph_project_start.git

* [**Deep Graph Library**](https://github.com/dmlc/dgl) <br>
Wang, Minjie, et al. 
**Deep graph library: A graph-centric, highly-performant package for graph neural networks.**. *The International Conference on Learning Representations (ICLR'19).*

* [**ROC**](https://github.com/jiazhihao/ROC) <br>
Jia, Zhihao, et al. 
**Improving the accuracy, scalability, and performance of graph neural networks with roc.** *Proceedings of Machine Learning and Systems (MLsys'20).*

* [**GNNAdvisor**](https://github.com/YukeWang96/OSDI21_AE) <br>
Wang, Yuke, et al. **GNNAdvisor: An adaptive and efficient runtime system for GNN acceleration on GPUs.** *15th USENIX symposium on operating systems design and implementation (OSDI'21)*.

* [**GE-SpMM**](https://github.com/hgyhungry/ge-spmm) <br>
Huang, Guyue, et al. **Ge-spmm: General-purpose sparse matrix-matrix multiplication on gpus for graph neural networks.** *International Conference for High Performance Computing, Networking, Storage and Analysis (SC'20)*.

* [**Bit-Tensor-Core**](https://github.com/pnnl/TCBNN) <br>
Li, Ang, and Simon Su. **Accelerating binarized neural networks via bit-tensor-cores in turing gpus.** *IEEE Transactions on Parallel and Distributed Systems (TPDS'20)*.
