
#include <iostream>
#include <stdio.h>
#include <ctime>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cublas_v2.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "layer.h"

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();
    
    int num_GPUs = atoi(argv[2]);           // 2
    int partSize = atoi(argv[3]);           // 32
    int warpPerBlock = atoi(argv[4]);       // 4
    int dim = atoi(argv[5]);                // 16
    int interleaved_dist = atoi(argv[6]);   // 2
    int hiddenSize = atoi(argv[7]);

    double t1, t2; 
    // print_array<int>("asym.row_ptr", asym.row_ptr, asym.row_ptr.size());
    // print_array<int>("asym.col_ind", asym.col_ind, asym.col_ind.size());

    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Get the current GPU ID.
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // range of node.
    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    int lb = nodesPerPE * mype_node;
    int ub = (lb+nodesPerPE) < numNodes? (lb+nodesPerPE):numNodes;


    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    t1 = MPI_Wtime(); 

    auto split_output = split_CSR<int>(asym.row_ptr, asym.col_ind, lb, ub);

    auto local_ptr = split_output[0]; // with the base start from lb.
    auto remote_ptr = split_output[1]; // with the base start from ub.
    auto local_col_idx = split_output[2];
    auto remote_col_idx = split_output[3];

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1.0f, beta = 0.0f;

    float *init_input, *d_output, *d_input, *d_weight_1;

    gpuErrchk(cudaMalloc((void**)&d_weight_1, dim*hiddenSize*sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&init_input, (ub-lb)*dim*sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&d_output, (ub-lb)*hiddenSize*sizeof(float))); 

    d_input = (float *) nvshmem_malloc ((ub-lb)*hiddenSize*sizeof(float)); // NVSHMEM global memory
    d_input = (float *) nvshmem_malloc ((ub-lb)*hiddenSize*sizeof(float)); // NVSHMEM global memory

    int *d_row_ptr_local, *d_col_ind_local;
    int *d_row_ptr_remote, *d_col_ind_remote;

    gpuErrchk(cudaMalloc((void**)&d_row_ptr_local, local_ptr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_local, local_col_idx.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_remote, remote_ptr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_remote, remote_col_idx.size()*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_row_ptr_local, &local_ptr[0], local_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_local, &local_col_idx[0], local_col_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr_remote, &remote_ptr[0], remote_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_remote, &remote_col_idx[0], remote_col_idx.size()*sizeof(int), cudaMemcpyHostToDevice));


    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "PreProcess time (s) %.3f\n", (t2 - t1)); 
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    
    t1 = MPI_Wtime(); 

    mgg_SAG_basic(d_output, d_input,  d_row_ptr, d_col_ind,
                    lb, ub, dim, nodePerPE);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("PE-%d, Total (ms): %.3f, Dense (ms): %.3f\n", mype_node, time_elapsed_ms, dense_time_elapsed_ms);

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3); 


    // release nvshmem objects and finalize context.
    cudaFree(d_output);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);

    nvshmem_free(d_input);
    nvshmem_finalize();
    MPI_Finalize();
    // printf("--*-- PEID: %d, End NVSHMEM --*--\n", mype_node);
    if (mype_node == 0) printf("===================================\n");

    return 0;
}