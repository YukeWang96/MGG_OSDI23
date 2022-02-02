
#ifndef KERNEL_H
#define KERNEL_H 

#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define remote_fetch
// #define enable_kernel_counter

__global__ 
void init_counter(int* counter){
   counter[0] = 0; 
}

__global__ 
void print_counter_local(int* counter){
    printf("local access: %d\n", counter[0]);
}

__global__ 
void print_counter_remote(int* counter){
    printf("remote access: %d\n", counter[0]);
}


__device__ inline 
void atomicAdd_float(float* address, float value)

{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);

};

__global__ 
void SAG_cuda_kernel_ref(float* d_output, const float* d_input, const int* d_row_ptr, const int* d_col_ind, const int lb_src, const int pe_num_nodes, const int dim){
    const int tid =  blockIdx.x * blockDim.x + threadIdxx.x;
    const int wid = tid/32;
    const int lanid = tid%32;
    
    if (wid < pe_num_nodes){
        const int src_nid = wid + lb_src;
        const int eidx_s =  d_row_ptr[src_nid];
        const int eidx_e = d_row_ptr[src_nid + 1];
        for (int eidx = eidx_s; eidx < eidx_e; eidx++){
            int nid = d_col_ind[eidx]; 
            for (int d = 0; d < dim; d++)
                d_output[src_nid * dim + d] = d_input[nid * dim + d];
        }
    }
}

__global__ 
void SAG(float *update_node, const float *sheme_node, const int *edgeList, const int *nodePtr, \
        const int numWarps, const int numNodes, const int numEdges, const int ebdDim, const int partSize,
        int *local_counter, int *remote_counter) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid/32;
    int laneId = tid%32;
    int warpSHEME_offset = (threadIdx.x/32)*ebdDim;
    
    extern __shared__ float tmp_local[];

    if (warpId < numWarps)
    {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        
        int g_nodeId = mype*partSize + warpId; // node ID for indexing global nodes.
        int local_addr_node = warpId * ebdDim;

        int e_start = nodePtr[g_nodeId];
        int e_end = nodePtr[g_nodeId+1];

        // printf("mype: %d, npes: %d, g_nodeId: %d\n", mype, npes, g_nodeId);
    
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            // printf("nIdx: %d\n", nIdx);
            int nid = edgeList[nIdx];
            // printf("nid: %d\n", nid);
            int remote_pe = nid / partSize;
            int remote_addr_neighbor = (nid % partSize) * ebdDim;

            if (remote_pe == mype){    // fetch from local.
                // printf("nid: %d, local_addr: %d\n", nid, local_addr);

                // #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    update_node[local_addr_node+d] += sheme_node[remote_addr_neighbor+d];
                }
                
                #ifdef enable_kernel_counter
                atomicAdd(local_counter, 1);
                #endif
                // if (warpId == 7 && laneId == 0){
                //     printf("[%d] ", warpId);
                //     for (int d  = 0; d < ebdDim; d++){
                //         printf("%.3f ", update_node[local_addr_node+d]);
                //         // printf("%.3f ", sheme_node[remote_addr_neighbor+d]);
                //     }
                //     printf("\n");
                // }
            }
            #ifdef remote_fetch
            else{   // fetch from remote for remote_pe != mype

                // if (remote_pe == 2){
                //     printf("mype: %d, remote_pe: %d\n", mype, remote_pe);
                //     printf("nid: %d, partSize: %d\n", nid, partSize);
                //     // exit(-1);
                // }
                #ifdef enable_kernel_counter
                atomicAdd(remote_counter, 1);
                #endif
                
                nvshmemx_float_get_warp(&tmp_local[warpSHEME_offset], &sheme_node[remote_addr_neighbor], ebdDim, remote_pe);
                // nvshmemx_float_get_nbi_warp(&tmp_local[warpSHEME_offset], &sheme_node[remote_addr_neighbor], ebdDim, remote_pe);

                // #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    update_node[local_addr_node+d] += tmp_local[warpSHEME_offset+d];
                }
            }
            #endif
        }
    }
}


__global__ 
void SAG_interleave(float *update_node, const float *sheme_node, const int *edgeList, const int *nodePtr, \
                     const int ebdDim, const int numWarps, const int nodesPerGPU, const int interleave_dist, const int num_GPUs) 
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid/32;
    int laneId = tid%32;
    int warpSHEME_offset = (threadIdx.x/32)*ebdDim;

    extern __shared__ float tmp_local[];

    if (warpId < numWarps)
    {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
                
        int g_nodeId = mype*nodesPerGPU + warpId / (2*interleave_dist);         // global node ID.
        int remote_flag =  (warpId % (2*interleave_dist)) / interleave_dist;    // if 0, local aggregation, else 1, remote aggregation.
        int local_addr_node = (g_nodeId % nodesPerGPU) * ebdDim;

        int e_start = nodePtr[g_nodeId];
        int e_end = nodePtr[g_nodeId+1];
        // printf("mype: %d, npes: %d, g_nodeId: %d\n", mype, npes, g_nodeId);
    
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            // printf("nIdx: %d\n", nIdx);
            int nid = edgeList[nIdx];
            // printf("nid: %d\n", nid);
            int remote_pe = nid / nodesPerGPU;
            int remote_addr_neighbor = (nid % nodesPerGPU) * ebdDim;

            if (remote_pe == mype && remote_flag == 0){
                // #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    tmp_local[warpSHEME_offset+d] += sheme_node[remote_addr_neighbor+d];
                }
            }

            if (remote_pe != mype && remote_flag == 1){
                nvshmemx_float_get_warp(&tmp_local[warpSHEME_offset], &sheme_node[remote_addr_neighbor], ebdDim, remote_pe);
                // #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    // update_node[local_addr_node+d] += tmp_local[warpSHEME_offset+d];
                    atomicAdd_float(&update_node[local_addr_node+d], tmp_local[warpSHEME_offset+d]);
                }
            }
        }

        if (remote_flag == 0)
        // #pragma unroll
        for (int d  = laneId; d < ebdDim; d += 32){
            // update_node[local_addr_node+d] += tmp_local[remote_addr_neighbor+d];
            atomicAdd_float(&update_node[local_addr_node+d], tmp_local[warpSHEME_offset+d]);
        }
    }
}


__global__ 
void SAG_pipeline(float *update_node, const float *sheme_node, const int *edgeList, const int *nodePtr, \
                    const int numBlocks, const int numNodes, const int numEdges, 
                    const int ebdDim, const int partSize) {

    int bid = blockIdx.x;
    int wid = threadIdx.x/32;
    int laneId = threadIdx.x%32;

    extern __shared__ float shmem[]; // 2*ebdDim*sizeof(float)
    float* tmp_local = shmem;
    float* tmp_acc = (float*)&shmem[ebdDim];

    if (bid < numBlocks)
    {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        
        int g_nodeId = bid; // node ID for indexing global nodes.
        int local_addr_node = (bid % partSize) * ebdDim;

        int e_start = nodePtr[g_nodeId];
        int e_end = nodePtr[g_nodeId+1];

        // printf("mype: %d, npes: %d, g_nodeId: %d\n", mype, npes, g_nodeId);
        if (wid == 0)
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            int nid = edgeList[nIdx];
            int remote_pe = nid / partSize;
            int remote_addr_neighbor = (nid % partSize) * ebdDim;

            if (remote_pe == mype){    // fetch from local.
                #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    update_node[local_addr_node+d] += sheme_node[remote_addr_neighbor+d];
                }
            }
        }

        if (wid == 1)
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            int nid = edgeList[nIdx];
            int remote_pe = nid / partSize;
            int remote_addr_neighbor = (nid % partSize) * ebdDim;

            if (remote_pe != mype){   
                // aggregate to shared memory first.
                nvshmemx_float_get_warp(&tmp_local[0], &sheme_node[remote_addr_neighbor], ebdDim, remote_pe);

                // aggregate to global memory.
                #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    tmp_acc[d] += tmp_local[d];
                }
            }
        }

        __syncthreads();

        // aggregate to local global memory.
        if (wid == 1)
        #pragma unroll
        for (int d  = laneId; d < ebdDim; d += 32){
            update_node[local_addr_node+d] += tmp_acc[d];
        }
    }
}


__global__ 
void SAG_pipeline_opt(float *update_node, const float *sheme_node, const int *edgeList, const int *nodePtr, \
                    const int numBlocks, const int numNodes, const int numEdges, 
                    const int ebdDim, const int partSize) {

    int bid = blockIdx.x;
    int wid = threadIdx.x/32;
    int laneId = threadIdx.x%32;

    extern __shared__ float shmem[]; // 2*ebdDim*sizeof(float)
    float* tmp_local = shmem;
    float* tmp_acc = (float*)&shmem[ebdDim];

    if (bid < numBlocks)
    {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        
        int g_nodeId = bid; // node ID for indexing global nodes.
        int local_addr_node = (bid % partSize) * ebdDim;

        int e_start = nodePtr[g_nodeId];
        int e_end = nodePtr[g_nodeId+1];

        // printf("mype: %d, npes: %d, g_nodeId: %d\n", mype, npes, g_nodeId);
        if (wid == 0)
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            int nid = edgeList[nIdx];
            int remote_pe = nid / partSize;
            int remote_addr_neighbor = (nid % partSize) * ebdDim;

            if (remote_pe == mype){    // fetch from local.
                #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    update_node[local_addr_node+d] += sheme_node[remote_addr_neighbor+d];
                }
            }
        }

        if (wid == 1)
        for (int nIdx = e_start; nIdx < e_end; nIdx++)
        {
            int nid = edgeList[nIdx];
            int remote_pe = nid / partSize;
            int remote_addr_neighbor = (nid % partSize) * ebdDim;

            if (remote_pe != mype){   
                // aggregate to shared memory first.
                nvshmemx_float_get_warp(&tmp_local[0], &sheme_node[remote_addr_neighbor], ebdDim, remote_pe);

                // aggregate to global memory.
                #pragma unroll
                for (int d  = laneId; d < ebdDim; d += 32){
                    tmp_acc[d] += tmp_local[d];
                    atomicAdd_float(&tmp_acc[d], tmp_local[d]);
                }
            }
        }
    }
}
#endif