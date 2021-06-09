cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MGG LANGUAGES CXX CUDA)

set($ENV{PATH} "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1/bin/:$ENV{PATH}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.0/")
set(MPI_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1/")
set(CUDA_HOME "/usr/local/cuda-11.0/")
set(CUDNN_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/cudnn-v8.2")
set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/nvshmem_src_2.0.3-0/build")
set(NCCL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/nccl_2.9.8-1+cuda11.0_x86_64/")


set(SM_ARCH "sm_70")
set($ENV{NVCC_GENCODE} ${SM_ARCH})
set($ENV{NVSHMEM_USE_GDRCOPY} "0")

link_directories(${NVSHMEM_HOME}/lib)
link_directories(${CUDNN_HOME}/lib64)
link_directories(${MPI_HOME}/lib)
link_directories(/usr/local/lib)
link_directories(${NCCL_HOME}/lib)

add_executable(MGG 
    src/test_neighbor_part.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(NCCL 
    src/nccl_example.cu 
)

add_executable(HOST2DEV 
    src/host2device.cu 
)

target_include_directories(MGG 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
target_link_libraries(MGG
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)


set_target_properties(NCCL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(NCCL PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

target_include_directories(NCCL 
    PRIVATE 
    ${NCCL_HOME}/include 
    include
)

target_link_libraries(NCCL
    cuda
    nccl
)


target_include_directories(HOST2DEV 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(HOST2DEV PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(HOST2DEV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
target_link_libraries(HOST2DEV
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
)