export LD_LIBRARY_PATH=/usr/local/lib:$PWD/local/openmpi-4.1.1/lib/:$LD_LIBRARY_PATH
export PATH=$PWD/local/openmpi-4.1.1/bin/:$PATH
export MPI_HOME=$PWD/local/openmpi-4.1.1/
export CUDA_HOME=/usr/local/cuda-11.0/
export CUDNN_HOME=/home/yuke/cudnn-v8.2

export NVSHMEM_HOME=$PWD/local/nvshmem_src_2.0.3-0/build
export NVCC_GENCODE=sm_70
export NVSHMEM_USE_GDRCOPY=0

/usr/local/cuda-11.0/bin/nvcc -rdc=true \
                            -ccbin g++ \
                            -arch=$NVCC_GENCODE \
                            -I$NVSHMEM_HOME/include \
                            -Iinclude \
                            -I$CUDNN_HOME/include/ \
                            -I$MPI_HOME/include \
                            src/test_neighbor_part.cu \
                            include/loss.cu \
                            include/layer.cu \
                            -o test_neighbor_part \
                            -L$NVSHMEM_HOME/lib \
                            -lnvshmem \
                            -lcuda\
                            -Xcompiler \
                            -pthread \
                            -L$MPI_HOME/lib \
                            -lmpi_cxx \
                            -lmpi \
                            -L/usr/local/cuda/lib \
                            -L$CUDNN_HOME/lib64 \
                            -lcublas \
                            -lcudnn \
                            -lgomp \
                            -lcurand