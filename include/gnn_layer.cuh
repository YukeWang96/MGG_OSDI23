#ifndef gnn_layer_cuh
#define gnn_layer_cuh

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include "cublas_utils.h"
#include <cudnn.h>

#include "neighbor_utils.cuh"
#include "gnn_kernel.cuh"

using namespace cooperative_groups;
using namespace std;

void softmax_forward(softmax_param* smx){
    cudnnSoftmaxForward(smx->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &(smx->alpha), smx->srcTensorDesc, smx->d_in,  &(smx->beta), smx->sftTensorDesc, smx->d_out);

    cudaDeviceSynchronize();
}

void dense_beg_forward(dense_param_beg* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_out, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));
}

void dense_hidden_forward(dense_param_hidden* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_out, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));
}

///////////////////////// new-memory reduction /////////////////////////

void softmax_new_forward(softmax_new_param* smx){
    cudnnSoftmaxForward(smx->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
            &(smx->alpha), smx->srcTensorDesc, smx->d_in,  &(smx->beta), smx->sftTensorDesc, smx->d_out);
    // cudaDeviceSynchronize();
}

void dense_beg_new_forward(dense_param_new_beg* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_out, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));
}

void dense_hidden_new_forward(dense_param_new_hidden* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_out, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));
}
////////////////////////////////////////////////////




void dense_beg_forward_uvm(dense_param_beg_uvm* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_in, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ dense_beg_forward_uvm %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}



void dense_hidden_forward_uvm(dense_param_hidden_uvm* dp)
{
    CUBLAS_CHECK(cublasSgemm(dp->cublasH, dp->transa, dp->transb, dp->m, dp->n, dp->k, 
        &(dp->alpha), dp->d_W, dp->ldw, dp->d_in, dp->ldx, &(dp->beta), dp->d_out, dp->ldout));
    
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ dense_hidden_forward_uvm %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

void sparse_beg_forward(sparse_param_beg*sp)
{
    #ifdef PROFILE
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<PROFILE; i++) {
        warmup<<<1,1>>>();
    }
	
    cudaEventRecord(start, 0);
    for (int i=0; i<PROFILE; i++)        
    #endif

    SpMM_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, sp->d_row_ptr, sp->d_col_ind, 
                                                            sp->numNodes, sp->dim, 
                                                            sp->partSize, sp->warpPerBlock); 

    #ifdef PROFILE
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SpMM_cuda_kernel -- Time (ms): %.3f\n", milliseconds/PROFILE);
    float gflop = 2*num_edges*1.0f/1e6*dim;
    printf("SpMM_cuda_kernel -- Time (ms): %.3f, GFLOPs: %.3f\n", milliseconds/PROFILE, gflop/(milliseconds/PROFILE));
    #endif

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ sparse_beg_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


void sparse_hidden_forward(sparse_param_hidden*sp)
{
    #ifdef PROFILE
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<PROFILE; i++) {
        warmup<<<1,1>>>();
    }
	
    cudaEventRecord(start, 0);
    for (int i=0; i<PROFILE; i++)        
    #endif

    SpMM_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, sp->d_row_ptr, sp->d_col_ind, 
                                                                sp->numNodes, sp->dim, 
                                                                sp->partSize, sp->warpPerBlock); 

    #ifdef PROFILE
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("SpMM_cuda_kernel -- Time (ms): %.3f\n", milliseconds/PROFILE);
    float gflop = 2*num_edges*1.0f/1e6*dim;
    printf("SpMM_cuda_kernel -- Time (ms): %.3f, GFLOPs: %.3f\n", milliseconds/PROFILE, gflop/(milliseconds/PROFILE));
    #endif

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ sparse_hidden_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

void AGNN_beg_forward(AGNN_param_beg*sp){

    AGNN_base_cuda_kernel<<<sp->grid, sp->block>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
                                                    sp->d_row_ptr, sp->d_col_ind, 
                                                    sp->numNodes, sp->dim, sp->warpPerBlock); 

    // AGNN_v2_cuda_kernel<<<sp->grid, sp->block>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
    //                                                 sp->d_row_ptr, sp->d_col_ind, 
    //                                                 sp->numNodes, sp->dim, sp->partSize, sp->warpPerBlock);

//   AGNN_updated_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
//                                                                         sp->d_row_ptr, sp->d_col_ind, 
//                                                                         sp->numNodes, sp->dim, sp->partSize, sp->warpPerBlock); 

    // SpMM_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, sp->d_row_ptr, sp->d_col_ind, 
    //                                                             sp->numNodes, sp->dim, 
    //                                                             sp->partSize, sp->warpPerBlock); 

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ AGNN_base_cuda_kernel: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

void AGNN_hidden_forward(AGNN_param_hidden*sp)
{                               
    AGNN_base_cuda_kernel<<<sp->grid, sp->block>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
                                                    sp->d_row_ptr, sp->d_col_ind, 
                                                    sp->numNodes, sp->dim, sp->warpPerBlock);

    // AGNN_v2_cuda_kernel<<<sp->grid, sp->block>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
    //                                                 sp->d_row_ptr, sp->d_col_ind, 
    //                                                 sp->numNodes, sp->dim, sp->partSize, sp->warpPerBlock);
    // AGNN_updated_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
    //                                                                         sp->d_row_ptr, sp->d_col_ind, 
    //                                                                         sp->numNodes, sp->dim, sp->partSize, sp->warpPerBlock);  

    // SpMM_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, sp->d_row_ptr, sp->d_col_ind, 
    //                                                         sp->numNodes, sp->dim, 
    //                                                         sp->partSize, sp->warpPerBlock); 
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ AGNN_hidden_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


void SGC_beg_forward(SGC_param_beg*sp)
{                                   
    // const int dev = 0;
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // int numBlocksPerSm;

    // cudaFuncSetAttribute(SGC_cuda_kernel_beg_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, sp->shared_memory);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SGC_cuda_kernel_beg_wrapper, sp->block, sp->shared_memory);

    // void* args[] = {&(sp->gpu)};      
    // printf("numBlocksPerSm: %d, SMs: %d,  Total: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocksPerSm*deviceProp.multiProcessorCount);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_beg_wrapper, numBlocksPerSm * deviceProp.multiProcessorCount, sp->block, args, sp->shared_memory);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_beg_wrapper, sp->grid, sp->block, args, sp->shared_memory);
    
    // for (int i = 0; i < sp->khop; i++){
        SGC_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, 
                                                                    sp->d_row_ptr, sp->d_col_ind, 
                                                                    sp->numNodes, sp->dim, 
                                                                    sp->partSize, sp->warpPerBlock); 
    // }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ SGC_beg_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


void SGC_hidden_forward(SGC_param_hidden* sp)
{                                
    // const int dev = 0;
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // int numBlocksPerSm;

    // cudaFuncSetAttribute(SGC_cuda_kernel_hidden_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, sp->shared_memory);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SGC_cuda_kernel_hidden_wrapper, sp->block, sp->shared_memory);

    // void* args[] = {&(sp->gpu)};  
    // printf("numBlocksPerSm: %d, SMs: %d,  Total: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocksPerSm*deviceProp.multiProcessorCount);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_hidden_wrapper, numBlocksPerSm * deviceProp.multiProcessorCount, sp->block, args, sp->shared_memory);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_hidden_wrapper, sp->grid, sp->block, args, sp->shared_memory);


//   for (int i = 0; i < sp->khop; i++){
    SGC_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, 
                                                                sp->d_row_ptr, sp->d_col_ind, 
                                                                sp->numNodes, sp->dim, 
                                                                sp->partSize, sp->warpPerBlock);     
//   }
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ SGC_hidden_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

#endif // gnn_layer_cuh