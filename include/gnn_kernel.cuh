#ifndef gnn_kernel_cuh
#define gnn_kernel_cuh

#include <cooperative_groups.h>

using namespace cooperative_groups;


__global__ 
void SpMM_cuda_kernel(
    float*  output,
    const float* input,
    const int* row_pointers, 
    const int* column_index,
    const int num_nodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock
) 
{
    int srcId = blockIdx.x;                                     // each node allocate 
    int block_warpId = threadIdx.x / WARP_SIZE;                 // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                       // warp thread-id -- laneid

    extern __shared__ int part_meta[];                          // part information.
    int*  warp_nbs = (int*)&part_meta[warpPerBlock*dim];        // cache neighbor id (warpPerBlock*partsize)
    
    // grid_group grid = this_grid();

    // for (int k = 0; k < 2; k++){ // iterate through k-hop
    if (srcId < num_nodes){
    // for (int srcId = blockIdx.x; srcId < num_nodes; srcId += gridDim.x){

        const int neighborBeg = row_pointers[srcId];        // partitioning pointer start
        const int neighborEnd = row_pointers[srcId + 1];    // part pointer end

        #pragma unroll
        for (int d = laneid; d < dim; d += 32){
            part_meta[block_warpId*dim + d] = 0.0f;
        }

        __syncwarp();

        for (int nidx_b = neighborBeg; nidx_b < neighborEnd; nidx_b += partSize*warpPerBlock){

            const int w_start = nidx_b + partSize * block_warpId;
            const int w_end = w_start + partSize < neighborEnd?  w_start + partSize: neighborEnd;
            
            const int n_base = block_warpId * partSize;
            for(int nidx_w = w_start + laneid; nidx_w < w_end; nidx_w += WARP_SIZE){  
                warp_nbs[n_base + nidx_w - w_start] = column_index[nidx_w];
            }

            __syncwarp();

            for(int nidx = 0; nidx < w_end - w_start; nidx++){  
                int nid = warp_nbs[n_base + nidx];
                #pragma unroll
                for (int d = laneid; d < dim; d += 32){
                    part_meta[block_warpId*dim + d] += input[nid * dim + d];

                }
            }
        }
        for (int d = laneid; d < dim; d += 32){
            atomicAdd_F((float*)&output[srcId * dim + d], part_meta[block_warpId*dim + d]);
        }
    }    
    // grid.sync();
    // } // end K iteration
}



__global__ 
void AGNN_base_cuda_kernel(  //https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=dotgat#agnnconv
    float*  output,
    float* edge_feat,   // [1, E]
    const float* input,
    const int* row_pointers, 
    const int* column_index,
    const int num_nodes, 
    const int dim,
    const int warpPerBlock
)
{
    #define FULL_MASK 0xffffffff
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;         // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    if (warpId < num_nodes){
        const int nb_begin = row_pointers[warpId];       
        const int nb_end = row_pointers[warpId + 1];   
        
        float src_v2 = 0.0f;
        float cos_edge_sum = 0.0f;

        // compute the current node |v_src|
        for (int d = laneid; d < dim; d += 32){
            src_v2 += input[warpId*dim + d] * input[warpId * dim + d];
        }

        // warp_reduce for src_v2 -> lanid-0
        for (int offset = 16; offset > 0; offset /= 2)
            src_v2 += __shfl_down_sync(FULL_MASK, src_v2, offset);

        // compute the e_i,j = v_src . v_dst
        for (int nidx = nb_begin; nidx < nb_end; nidx++){
            int nid = column_index[nidx];

            float dot_sum = 0.0f;
            float dst_v2 = 0.0f;   
            float cos_edge = 0.0f;

            for (int d = laneid; d < dim; d += 32){
                float dst_remote = input[nid*dim + d];
                dot_sum += input[warpId*dim + d] * dst_remote; // add atomics
                dst_v2 += dst_remote * dst_remote;
            }

            // warp_reduce dot_prod -> lanid-0
            for (int offset = 16; offset > 0; offset /= 2)
                dot_sum += __shfl_down_sync(FULL_MASK, dot_sum, offset);

            // warp_reduce dst_v2 -> lanid-0
            for (int offset = 16; offset > 0; offset /= 2)
                dst_v2 += __shfl_down_sync(FULL_MASK, dst_v2, offset);

            // compute cosine function + softmax accumulation.
            // if (laneid == 0){   
                cos_edge = dot_sum / (__fsqrt_rn(dst_v2) * __fsqrt_rn(src_v2));
                edge_feat[nidx] = __expf(cos_edge);
                cos_edge_sum += edge_feat[nidx];
            // }
        }

        // broadcast edge_feat from lanid-0 to all threads in a warp.
        cos_edge_sum = __shfl_sync(FULL_MASK, cos_edge_sum, 0);

        // aggregation with attention. p_j = \sum a_(i,j) * p_j
        for (int nidx = nb_begin; nidx < nb_end; nidx++){
            int nid = column_index[nidx];
            float tmp = edge_feat[nidx];
            float att = tmp / cos_edge_sum;
            for (int d = laneid; d < dim; d += 32){
                // atomicAdd_F((float*)&output[warpId * dim + d], att*input[nid * dim + d]);
                output[warpId * dim + d] += att*input[nid * dim + d];
            }
        }
    }
}


__global__ 
void AGNN_v2_cuda_kernel(  //https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=dotgat#agnnconv
    float*  output,
    float* edge_feat,   // [1, E]
    const float* input,
    const int* row_pointers, 
    const int* column_index,
    const int num_nodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock
)
{
    #define FULL_MASK 0xffffffff
    // int tid =  blockIdx.x * blockDim.x + threadIdx.x;         // global thread-id
    // int warpId = tid / WARP_SIZE;                             // global warp-id
    int block_warpId =  threadIdx.x / WARP_SIZE;
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    int srcId = blockIdx.x;

    __shared__ float cos_edge_sum[1];
    __shared__ float src_v2[1];

    if (srcId < num_nodes){

        const int nb_begin = row_pointers[srcId];       
        const int nb_end = row_pointers[srcId + 1];   
        
        float src_tmp = 0.0f;
        cos_edge_sum[0] = 0.0f;
        // float cos_edge_sum = 0.0f;
        // float cos_edge_sum = 1.0f;

        if (block_warpId == 0){
            // compute the current node |v_src|
            for (int d = laneid; d < dim; d += 32){
                src_tmp += input[srcId * dim + d] * input[srcId * dim + d];
            }

            // warp_reduce for src_v2 -> lanid-0
            for (int offset = 16; offset > 0; offset /= 2)
                src_tmp += __shfl_down_sync(FULL_MASK, src_tmp, offset);

            if (laneid == 0)
                src_v2[0] = src_tmp;
        }

        __syncthreads();
        // compute the e_i,j = v_src . v_dst
        // for (int nidx = nb_begin; nidx < nb_end; nidx++){
         for (int nidx_b = nb_begin; nidx_b < nb_end; nidx_b += partSize*warpPerBlock){

            const int w_start = nidx_b + partSize * block_warpId;
            const int w_end = w_start + partSize < nb_end?  w_start + partSize: nb_end;

            for(int nidx = w_start; nidx < w_end; nidx++){  

                int nid = column_index[nidx];

                float dot_sum = 0.0f;
                float dst_v2 = 0.0f;   
                float cos_edge = 0.0f;

                for (int d = laneid; d < dim; d += 32){
                    dot_sum += input[srcId*dim + d] * input[nid*dim + d]; // add atomics
                    dst_v2 += input[nid*dim + d] * input[nid*dim + d];
                }

                // warp_reduce dot_prod -> lanid-0
                for (int offset = 16; offset > 0; offset /= 2)
                    dot_sum += __shfl_down_sync(FULL_MASK, dot_sum, offset);

                // warp_reduce dst_v2 -> lanid-0
                for (int offset = 16; offset > 0; offset /= 2)
                    dst_v2 += __shfl_down_sync(FULL_MASK, dst_v2, offset);

                // compute cosine function + softmax accumulation.
                if (laneid == 0){   
                    cos_edge = dot_sum / (sqrt(dst_v2) * sqrt(src_v2[0]));
                    edge_feat[nidx] = expf(cos_edge);
                    // cos_edge_sum += edge_feat[nidx];
                    atomicAdd_F((float*)&cos_edge_sum[0], edge_feat[nidx]);
                }
            }
        }

        // broadcast edge_feat from lanid-0 to all threads in a warp.
        // cos_edge_sum = __shfl_sync(FULL_MASK, cos_edge_sum, 0);

        // aggregation with attention. p_j = \sum a_(i,j) * p_j
        for (int nidx = nb_begin; nidx < nb_end; nidx++){
            int nid = column_index[nidx];
            float tmp = edge_feat[nidx];
            float att = tmp / cos_edge_sum[0];
            for (int d = laneid; d < dim; d += 32){
                // atomicAdd_F((float*)&output[srcId * dim + d], att*input[nid * dim + d]);
                output[srcId * dim + d] += att*input[nid * dim + d];
            }
        }
    }
}


__global__ 
void AGNN_updated_cuda_kernel(  //https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=dotgat#agnnconv
    float*  output,
    float* edge_feat,   // [1, E]
    const float* input,
    const int* row_pointers, 
    const int* column_index,
    const int num_nodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock
)
{
    int srcId = blockIdx.x;                                     // each node allocate 
    int block_warpId = threadIdx.x / WARP_SIZE;                 // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                       // warp thread-id -- laneid

    extern __shared__ int part_meta[];                          // part information.
    int* warp_nbs = (int*)&part_meta[warpPerBlock*dim];        // cache neighbor id (warpPerBlock*partsize)

    if (srcId < num_nodes){

        const int neighborBeg = row_pointers[srcId];        // partitioning pointer start
        const int neighborEnd = row_pointers[srcId + 1];    // part pointer end

        #pragma unroll
        for (int d = laneid; d < dim; d += 32){
            part_meta[block_warpId*dim + d] = 0.0f;
        }

        __syncwarp();

        for (int nidx_b = neighborBeg; nidx_b < neighborEnd; nidx_b += partSize*warpPerBlock){

            const int w_start = nidx_b + partSize * block_warpId;
            const int w_end = w_start + partSize < neighborEnd?  w_start + partSize: neighborEnd;
            
            const int n_base = block_warpId * partSize;
            for(int nidx_w = w_start + laneid; nidx_w < w_end; nidx_w += WARP_SIZE){  
                warp_nbs[n_base + nidx_w - w_start] = column_index[nidx_w];
            }

            __syncwarp();

            for(int nidx = 0; nidx < w_end - w_start; nidx++){  
                int nid = warp_nbs[n_base + nidx];
                #pragma unroll
                for (int d = laneid; d < dim; d += 32){
                    part_meta[block_warpId*dim + d] += input[nid * dim + d];

                }
            }
        }
        for (int d = laneid; d < dim; d += 32){
            atomicAdd_F((float*)&output[srcId * dim + d], part_meta[block_warpId*dim + d]);
        }
    }    
}


__global__ 
void SGC_cuda_kernel(
    float*  output,
    const float* input,
    const int* row_pointers, 
    const int* column_index,
    const int num_nodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock
) 
{
    int srcId = blockIdx.x;                                   // each node allocate 
    int block_warpId = threadIdx.x / WARP_SIZE;                 // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                       // warp thread-id -- laneid

    extern __shared__ int part_meta[];                          // part information.
    int*  warp_nbs = (int*)&part_meta[warpPerBlock*dim];        // cache neighbor id (warpPerBlock*partsize)

    // grid_group grid = this_grid();
    for (int k = 0; k < 2; k++){ // iterate through k-hop
    if (srcId < num_nodes){
    // for (int srcId = blockIdx.x; srcId < num_nodes; srcId += gridDim.x){
        const int neighborBeg = row_pointers[srcId];        // partitioning pointer start
        const int neighborEnd = row_pointers[srcId + 1];    // part pointer end

        #pragma unroll
        for (int d = laneid; d < dim; d += 32){
            part_meta[block_warpId*dim + d] = 0.0f;
        }

        __syncwarp();

        for (int nidx_b = neighborBeg; nidx_b < neighborEnd; nidx_b += partSize*warpPerBlock){

            const int w_start = nidx_b + partSize * block_warpId;
            const int w_end = w_start + partSize < neighborEnd?  w_start + partSize: neighborEnd;
            
            const int n_base = block_warpId * partSize;
            for(int nidx_w = w_start + laneid; nidx_w < w_end; nidx_w += WARP_SIZE){  
                warp_nbs[n_base + nidx_w - w_start] = column_index[nidx_w];
            }

            __syncwarp();

            for(int nidx = 0; nidx < w_end - w_start; nidx++){  
                int nid = warp_nbs[n_base + nidx];
                #pragma unroll
                for (int d = laneid; d < dim; d += 32){
                    part_meta[block_warpId*dim + d] += input[nid * dim + d];

                }
            }
        }
        for (int d = laneid; d < dim; d += 32){
            atomicAdd_F((float*)&output[srcId * dim + d], part_meta[block_warpId*dim + d]);
        }
    }
    //  grid.sync();
    // __threadfence_system();
    __threadfence();
    } // end K iteration
}


// __global__ void SGC_cuda_kernel_beg_wrapper(SGC_param_beg* sp_beg)
// {
//     SGC_cuda_kernel(sp_beg->d_out, sp_beg->d_in, sp_beg->d_row_ptr, sp_beg->d_col_ind, 
//                     sp_beg->numNodes, sp_beg->dim, sp_beg->partSize, sp_beg->warpPerBlock);
// }

// __global__ void SGC_cuda_kernel_hidden_wrapper(SGC_param_hidden* sp_beg)
// {
//     SGC_cuda_kernel(sp_beg->d_out, sp_beg->d_in, sp_beg->d_row_ptr, sp_beg->d_col_ind, 
//                     sp_beg->numNodes, sp_beg->dim, sp_beg->partSize, sp_beg->warpPerBlock);
// }


#endif // gnn_kernel_cuh