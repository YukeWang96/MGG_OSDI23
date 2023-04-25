
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cublas_v2.h>

#include "graph.h"
#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "layer.h"

#include "cublas_utils.h"
#include "layer_new.cuh"
#include "gnn_layer.cuh"
#include "nccl.h"


// #define validate 1 // the number (< num_GPUs) indicates the validation on which PE.
// using nidType = size_t;
// using nidType = long;
using nidType = int;

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }
    
    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    int num_GPUs = atoi(argv[4]);           // 2
    int partSize = atoi(argv[5]);           // 32
    int warpPerBlock = atoi(argv[6]);       // 4
    int interleaved_dist = atoi(argv[7]);   // 2
    int dim = atoi(argv[8]);                // 16
    int hiddenSize = atoi(argv[9]);
    int outdim = atoi(argv[10]);

    int max_dim = max({hiddenSize, outdim});
    int max_dim_1 = max({dim, outdim});

    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    cout << "Complete loading graphs !!" << endl;
    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();    


    // std::cout << "max node: " << *std::max_element(std::begin(global_col_ind), std::end(global_col_ind)) << '\n';
    
    double t1, t2; 
    // print_array<int>("global_row_ptr", global_row_ptr, global_row_ptr.size());
    // print_array<int>("global_col_ind", global_col_ind, global_col_ind.size());
    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // Set the workload on each device.
    nidType nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    // printf("numNodes: %d, nodesPerPE: %d\n", numNodes, nodesPerPE);
    nidType lb = nodesPerPE * mype_node;
    nidType ub = (lb + nodesPerPE) < numNodes? (lb + nodesPerPE) : numNodes;
    nidType local_edges = global_row_ptr[ub] - global_row_ptr[lb];
    nidType edge_beg = global_row_ptr[lb];

    std::clock_t c_start_proc = std::clock();    
    // Divide the CSR into the local and remote for each GPU.
    auto split_output = split_CSR<nidType>(global_row_ptr, global_col_ind, lb, ub);
    std::clock_t c_end_proc = std::clock();
    float preproc_time_elapsed_ms = 1000.0 * (c_end_proc - c_start_proc) / CLOCKS_PER_SEC;
    if (mype_node == 0)
    printf("Preproc (ms): %.3f\n", preproc_time_elapsed_ms);

    // printf("lb: %d, ub: %d\n", lb, ub);
    auto local_ptr_vec = split_output[0];       // with the base start from lb.
    auto remote_ptr_vec = split_output[1];      // with the base start from ub.
    auto local_col_idx_vec = split_output[2];
    auto remote_col_idx_vec = split_output[3];
    // printf("PE[%d]. local: %d, remote: %d\n", mype_node, local_col_idx_vec.size(), remote_col_idx_vec.size());

    // Allocate memory on each device.
    float *h_input;
    float *d_buff_1, *d_buff_2, *d_buff_nvshmem;

    // d_input = (float *) nvshmem_malloc (nodesPerPE * dim * sizeof(float));  // NVSHMEM global memory for input embedding.
    /*
    gpuErrchk(cudaMalloc((void**)&d_input, nodesPerPE * dim * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out, nodesPerPE * hiddenSize * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out_1, nodesPerPE * outdim * sizeof(float))); 
    */
    

    // buffers for switching between input and output for each layer.
    printf("d_buff_1: %.3f GB\n", (nodesPerPE * 1.0f * max_dim_1 * sizeof(float))/1e9);
    printf("nodesPerPE: %d, dim: %d\n", nodesPerPE, dim);
    gpuErrchk(cudaMalloc((void**)&d_buff_1, static_cast<size_t>(nodesPerPE) * max_dim_1 * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&d_buff_2, static_cast<size_t>(nodesPerPE) * hiddenSize * sizeof(float))); 
   
    h_input = (float *) malloc (static_cast<size_t>(nodesPerPE) * max_dim_1 * sizeof(float));                  // CPU host memory (input)
    // h_output = (float *) malloc (nodesPerPE * hiddenSize * sizeof(float));         //  CPU host memory (output)
    // hsp_output_1 = (float *) malloc (nodesPerPE * outdim * sizeof(float));         //  CPU host memory (output)
    std::fill_n(h_input, static_cast<size_t>(nodesPerPE)*dim, 1.0f);                                 // filled with all ones for input embeddings.
    // std::fill_n(h_output, nodesPerPE*hiddenSize, 0.0f);                         // filled with all zeros for output embeddings.
    
     /*
    gpuErrchk(cudaMemcpy(d_input, h_input, nodesPerPE * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out, h_output, nodesPerPE * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out_1, hsp_output_1, nodesPerPE * outdim * sizeof(float), cudaMemcpyHostToDevice));
    */

    gpuErrchk(cudaMemcpy(d_buff_1, h_input, static_cast<size_t>(nodesPerPE) * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_buff_2, 0, static_cast<size_t>(nodesPerPE) * hiddenSize * sizeof(float)));
    d_buff_nvshmem = (float *) nvshmem_malloc (static_cast<size_t>(nodesPerPE) * hiddenSize * sizeof(float)); 

    // Initialize the parameters. 
    // d_buf_1 - (dense-1) -> d_buff_nvshmem -> (sp-1) 
    // -> d_buff_2 -> (dense-2) -> d_buff_nvshmem - (sp-2) -> d_buff_2 
    // -> smx-2 -> d_buff_2
    dense_param_new_beg* dp1 = new dense_param_new_beg("d-1", d_buff_1, d_buff_nvshmem, nodesPerPE, dim, hiddenSize);
    dense_param_new_hidden* dp2 = new dense_param_new_hidden("d-2", d_buff_2, d_buff_nvshmem, nodesPerPE, hiddenSize, outdim);
    softmax_new_param* smx2 = new softmax_new_param("smx-2", d_buff_1, d_buff_1, nodesPerPE, outdim);
 
    #ifdef validate
    float *h_input_ref, *h_output_ref,  *d_input_ref, *d_output_ref;
    if (mype_node == validate)
    {
        h_input_ref = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_output_ref = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_input_ref, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_output_ref, numNodes * dim, 0.0f); // filled with all zeros.
        gpuErrchk(cudaMalloc((void**)&d_input_ref, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        gpuErrchk(cudaMalloc((void**)&d_output_ref, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    }
    #endif

    nidType *d_row_ptr_l, *d_col_ind_l,  *d_row_ptr_r, *d_col_ind_r;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_l, local_ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_l, local_col_idx_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_r, remote_ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_r, remote_col_idx_vec.size()*sizeof(nidType))); 

    gpuErrchk(cudaMemcpy(d_row_ptr_l, &local_ptr_vec[0], local_ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_l, &local_col_idx_vec[0], local_col_idx_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr_r, &remote_ptr_vec[0], remote_ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_r, &remote_col_idx_vec[0], remote_col_idx_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));

    #ifdef validate
    int* d_row_ptr_ref, *d_col_ind_ref;
    if (mype_node == validate)
    {
        gpuErrchk(cudaMalloc((void**)&d_row_ptr_ref, global_row_ptr.size()*sizeof(int))); 
        gpuErrchk(cudaMalloc((void**)&d_col_ind_ref, global_col_ind.size()*sizeof(int))); 
        gpuErrchk(cudaMemcpy(d_row_ptr_ref, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_col_ind_ref, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_input_ref, h_input_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_output_ref, h_output_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        //
        // Compute the result [lb, ub] based on the whole graph CSR.
        //
        SAG_host_ref(d_output_ref, d_input_ref, 
                    d_row_ptr_ref, d_col_ind_ref, 
                    lb, ub, dim);

        gpuErrchk(cudaMemcpy(h_output_ref, d_output_ref, numNodes * dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD); 

    //
    // Compute on each GPU device.
    //
    // for (int i = 0; i < 10; i++)
    // {
    //     mgg_SAG_np_div(dsp_out, d_input, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
    //                     lb, ub, dim, nodesPerPE, mype_node, partSize, warpPerBlock, interleaved_dist);
    //     MPI_Barrier(MPI_COMM_WORLD); 
    // }
    
    int num_profiles = 100;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {
        dense_beg_new_forward(dp1);
        mgg_SAG_np_div(d_buff_2, dp1->d_out, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
                        lb, ub, hiddenSize, nodesPerPE, mype_node, partSize, warpPerBlock, interleaved_dist);
        // MPI_Barrier(MPI_COMM_WORLD); 
        // MPI_Barrier(MPI_COMM_WORLD); 
        // nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_W_new, dp2->d_W, dp2->dim1*dp2->dim2);

        dense_hidden_new_forward(dp2);
        mgg_SAG_np_div(d_buff_1, dp2->d_out, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
                        lb, ub, hiddenSize, nodesPerPE, mype_node, partSize, warpPerBlock, interleaved_dist);
        // MPI_Barrier(MPI_COMM_WORLD); 
        // MPI_Barrier(MPI_COMM_WORLD); 
        // nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_W_new, dp2->d_W, dp2->dim1*dp2->dim2);
        softmax_new_forward(smx2);
    }

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3/num_profiles); 
    
    // gpuErrchk(cudaMemcpy(h_output, dsp_out, nodesPerPE*dim*sizeof(float), cudaMemcpyDeviceToHost));

    #ifdef validate
    if (mype_node == validate){
        for (int nid = 0; nid < 10; nid++){
            printf("out [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output[nid * dim + d]);
            }
            printf("\n");
        }
        printf("==============================\n");
        for (int nid = 0; nid < 10; nid++){
            printf("ref [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output_ref[lb * dim + nid * dim + d]);
            }
            printf("\n");
        }
        bool val_status = check_equal(h_output_ref, h_output, (ub - lb) * dim, dim, lb * dim);
        printf("Validation on PE-{%d}, status: ", validate);
        if (val_status) printf("True\n"); else printf("False\n");
    }
    #endif

    // release memory.
    // cudaFree(dsp_out);
    cudaFree(d_row_ptr_l);
    cudaFree(d_col_ind_l);
    cudaFree(d_row_ptr_r);
    cudaFree(d_col_ind_r);
    // cudaFree(d_input);
    // cudaDeviceReset();
    nvshmem_finalize();

    // free(h_input);
    // free(h_output);

    MPI_Finalize();

    #ifdef validate
    if (mype_node == validate){
        cudaFree(d_output_ref);
        free(h_output_ref);
    }
    #endif


    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}