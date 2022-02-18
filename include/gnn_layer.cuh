#ifndef gnn_layer_cuh
#define gnn_layer_cuh

#include <cooperative_groups.h>

using namespace cooperative_groups;
using namespace std;

#include "neighbor_utils.cuh"
#include "gnn_kernel.cuh"

void AGNN_beg_forward(AGNN_param_beg*sp){

    AGNN_base_cuda_kernel<<<sp->grid, sp->block>>>(sp->d_out, sp->d_edge_att, sp->d_in, 
                                                    sp->d_row_ptr, sp->d_col_ind, 
                                                    sp->numNodes, sp->dim, sp->warpPerBlock); 
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
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ AGNN_hidden_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


// __global__ void SGC_cuda_kernel_wrapper(SGC_param_beg* sp_beg)
// {
//     grid_group grid = this_grid();
//     SGC_cuda_kernel(sp_beg->d_out, sp_beg->d_in, sp_beg->d_row_ptr, sp_beg->d_col_ind, 
//                     sp_beg->numNodes, sp_beg->dim, sp_beg->partSize, sp_beg->warpPerBlock);
//     grid.sync();
// }

void SGC_beg_forward(SGC_param_beg*sp)
{                                   
    const int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;

    cudaFuncSetAttribute(SGC_cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sp->shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SGC_cuda_kernel, sp->block, sp->shared_memory);

    void* args[] = {sp};
    printf("numBlocksPerSm: %d, SMs: %d,  Total: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocksPerSm*deviceProp.multiProcessorCount);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_wrapper, numBlocksPerSm*deviceProp.multiProcessorCount, sp->block, args, sp->shared_memory);
    // cudaLaunchCooperativeKernel((void*)SGC_cuda_kernel_wrapper, sp->grid, sp->block, args, sp->shared_memory);

    SGC_cuda_kernel<<<numBlocksPerSm*deviceProp.multiProcessorCount, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, sp->d_row_ptr, sp->d_col_ind, 
                                                                sp->numNodes, sp->dim, 
                                                                sp->partSize, sp->warpPerBlock); 

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ SGC_beg_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}


void SGC_hidden_forward(SGC_param_hidden*sp)
{                                
    // SAG_inPart_cuda_kernel<<<sp->grid, sp->block, sp->shared_memory>>>(sp->d_out, sp->d_in, 4->d_row_ptr, sp->d_col_ind, 
    //                                                         sp->numNodes, sp->dim, 
    //                                                         sp->partSize, sp->warpPerBlock); 
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error @ SGC_hidden_forward: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

#endif // gnn_layer_cuh