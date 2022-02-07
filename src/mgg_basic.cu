
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

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
    std::cout << "max node: " << *std::max_element(std::begin(asym.col_ind), std::end(asym.col_ind)) << '\n';

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

    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // Set the workload on each device.
    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    printf("numNodes: %d, nodesPerPE: %d\n", numNodes, nodesPerPE);
    int lb = nodesPerPE * mype_node;
    int ub = (lb + nodesPerPE) < numNodes? (lb + nodesPerPE) : numNodes;
    int local_nodes = ub - lb;
    int local_edges = asym.row_ptr[ub] - asym.row_ptr[lb];
    int edge_beg = asym.row_ptr[lb];

    // Allocate memory on each device.
    float *d_input, *d_output;
    gpuErrchk(cudaMalloc((void**)&d_output, nodesPerPE * dim * sizeof(float))); 
    d_input = (float *) nvshmem_malloc (nodesPerPE * dim * sizeof(float)); // NVSHMEM global memory
    int *d_row_ptr, *d_col_ind;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, (local_nodes + 1)*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, local_edges*sizeof(int))); 
    gpuErrchk(cudaMemcpy(d_row_ptr, &asym.row_ptr[lb], (local_nodes + 1)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[edge_beg], local_edges*sizeof(int), cudaMemcpyHostToDevice));
    
    // if (mype_node == 0)
    // {
    //     for (int i = 0; i < local_edges; i++){
    //         printf("asym.col_ind[%d]: %d\n", edge_beg+i, asym.col_ind[edge_beg + i]);
    //     }
    //     print_dev_column_index<<<1,1>>>(d_col_ind, local_edges);    
    // }

    MPI_Barrier(MPI_COMM_WORLD); 


    // Compute on each device.
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    mgg_SAG_basic(d_output, d_input, d_row_ptr, d_col_ind,
                    lb, ub, dim, nodesPerPE);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3); 

    // release memory.
    cudaFree(d_output);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);

    nvshmem_free(d_input);
    nvshmem_finalize();
    MPI_Finalize();
    
    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}