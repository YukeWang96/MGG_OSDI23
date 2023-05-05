cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MGG LANGUAGES CXX CUDA)


#############################################################################
set($ENV{PATH} "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1/bin/:$ENV{PATH}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.0/")
set(MPI_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/openmpi-4.1.1/")
set(CUDA_HOME "/usr/local/cuda-11.0/")
set(CUDNN_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/cudnn-v8.2")
set(NVSHMEM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/nvshmem_src_2.0.3-0/build")
set(NCCL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/local/nccl_2.9.8-1+cuda11.0_x86_64/")

#############################################################################

set(SM_ARCH "sm_80")
set($ENV{NVCC_GENCODE} ${SM_ARCH})
set($ENV{NVSHMEM_USE_GDRCOPY} "0")


#############################################################################
link_directories(${NVSHMEM_HOME}/lib)
link_directories(${CUDNN_HOME}/lib64)
link_directories(${MPI_HOME}/lib)
link_directories(/usr/local/lib)
link_directories(${NCCL_HOME}/lib)




#############################################################################
add_executable(uvm_gcn_2layer_mem 
    src/uvm_gcn_2layer_mem.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(uvm_gin_5layer_mem 
    src/uvm_gin_5layer_mem.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_np_div_gin_mem 
    src/mgg_np_div_gin_mem.cu 
    include/loss.cu 
    include/layer.cu
)



add_executable(MGG_np_div_mem 
    src/mgg_np_div_mem.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div_kernel 
    src/mgg_np_div_kernel.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div_wo_np_kernel 
    src/mgg_np_div_wo_np_kernel.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_tag_2layer 
    src/mgg_tag_2layer.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_sgc_2layer 
    src/mgg_sgc_2layer.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_agnn_2layer 
    src/mgg_agnn_2layer.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_gcn_2layer 
    src/mgg_gcn_2layer.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG 
    src/test_neighbor_part.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_basic 
    src/mgg_basic.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np 
    src/mgg_np.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div 
    src/mgg_np_div.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div_th 
    src/mgg_np_div_th.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div_wrp 
    src/mgg_np_div_wrp.cu 
    include/loss.cu 
    include/layer.cu
)

add_executable(MGG_np_div_blk 
    src/mgg_np_div_blk.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_np_div_gin 
    src/mgg_np_div_gin.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(MGG_np_pipeline 
    src/mgg_np_pipeline.cu 
    include/loss.cu 
    include/layer.cu
)


add_executable(NCCL 
    src/nccl_example.cu 
)

add_executable(HOST2DEV 
    src/host2device.cu 
)

add_executable(unified_memory
src/unified_memory.cu
)

add_executable(unified_memory_updated
src/unified_memory_updated.cu
)

add_executable(uvm_gcn_2layer
src/uvm_gcn_2layer.cu
)

add_executable(uvm_gin_5layer
src/uvm_gin_5layer.cu
)

add_executable(uvm_profile
src/uvm_profile.cu
)

add_executable(mgg_profile 
    src/mgg_profile.cu 
    include/loss.cu 
    include/layer.cu
)

#############################################################################
target_include_directories(MGG_tag_2layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_tag_2layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_tag_2layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_tag_2layer PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_tag_2layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_np_div_kernel 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_kernel PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_kernel
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)


#############################################################################
target_include_directories(MGG_np_div_wo_np_kernel 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_wo_np_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_wo_np_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_wo_np_kernel PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_wo_np_kernel
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_np_div_mem 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_mem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_mem PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_mem
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)


#############################################################################
target_include_directories(MGG_np_div_gin_mem 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_gin_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_gin_mem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_gin_mem PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_gin_mem
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_sgc_2layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_sgc_2layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_sgc_2layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_sgc_2layer PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_sgc_2layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_agnn_2layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_agnn_2layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_agnn_2layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_agnn_2layer PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_agnn_2layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_gcn_2layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_gcn_2layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_gcn_2layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_gcn_2layer PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_gcn_2layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
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
set_target_properties(MGG PROPERTIES CUDA_ARCHITECTURES "80")

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
#############################################################################
target_include_directories(MGG_np 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(MGG_basic 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_basic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_basic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_basic PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_basic
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_np_pipeline 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_pipeline PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_pipeline
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(MGG_np_div 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(MGG_np_div_gin 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_gin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_gin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_gin PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_gin
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(MGG_np_div_blk 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_blk PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_blk PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_blk PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_blk
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)

#############################################################################
target_include_directories(MGG_np_div_wrp 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_wrp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_wrp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_wrp PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_wrp
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(MGG_np_div_th 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(MGG_np_div_th PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(MGG_np_div_th PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MGG_np_div_th PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(MGG_np_div_th
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
target_include_directories(mgg_profile 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(mgg_profile PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(mgg_profile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mgg_profile PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(mgg_profile
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)
#############################################################################
set_target_properties(NCCL PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(NCCL PROPERTIES CUDA_ARCHITECTURES "80")

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
set_target_properties(HOST2DEV PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(HOST2DEV
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
)

##############################################################################
target_include_directories(unified_memory 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(unified_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(unified_memory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unified_memory PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(unified_memory
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
)

##############################################################################
target_include_directories(unified_memory_updated 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(unified_memory_updated PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(unified_memory_updated PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unified_memory_updated PROPERTIES CUDA_ARCHITECTURES "80")
target_link_libraries(unified_memory_updated
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn
)

##############################################################################
target_include_directories(uvm_gcn_2layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(uvm_gcn_2layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(uvm_gcn_2layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(uvm_gcn_2layer PROPERTIES CUDA_ARCHITECTURES "80")
target_link_libraries(uvm_gcn_2layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn
)
##############################################################################
target_include_directories(uvm_gin_5layer 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(uvm_gin_5layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(uvm_gin_5layer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(uvm_gin_5layer PROPERTIES CUDA_ARCHITECTURES "80")
target_link_libraries(uvm_gin_5layer
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn
)
##############################################################################

target_include_directories(uvm_profile 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    include
)

target_compile_options(uvm_profile PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(uvm_profile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(uvm_profile PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(uvm_profile
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
)

##############################################################################
target_include_directories(uvm_gin_5layer_mem 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(uvm_gin_5layer_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(uvm_gin_5layer_mem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(uvm_gin_5layer_mem PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(uvm_gin_5layer_mem
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)


##############################################################################
target_include_directories(uvm_gcn_2layer_mem 
    PRIVATE 
    ${NVSHMEM_HOME}/include 
    ${CUDA_HOME}/include 
    ${MPI_HOME}/include
    ${CUDNN_HOME}/include
    include
)

target_compile_options(uvm_gcn_2layer_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler 
                        -pthread 
                        -rdc=true 
                        -ccbin g++ 
                        -arch ${SM_ARCH}
                       >)

set_target_properties(uvm_gcn_2layer_mem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(uvm_gcn_2layer_mem PROPERTIES CUDA_ARCHITECTURES "80")

target_link_libraries(uvm_gcn_2layer_mem
    nvshmem 
    cuda
    mpi_cxx 
    mpi 
    cublas 
    cudnn 
    gomp 
    curand
)