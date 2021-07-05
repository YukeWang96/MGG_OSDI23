
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
	
    if (argc < 5){
        printf("Usage: ./main graph.mtx num_GPUs dim nodeOfInterest\n");
        return -1;
    }

    // cout << "\n\n=====================\n";
    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();
    
    int num_GPUs = atoi(argv[2]);        
    int dim = atoi(argv[3]);                 
    int nodeOfInterest = atoi(argv[4]);

    int warpPerBlock = 1;      
    float *d_output, *d_input;
    int *d_col_ind, *d_row_ptr;

    double t1, t2; 

    // create NVSHMEM common world.
    cudaStream_t stream;
    int rank, nranks;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    // <- end initialization.

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    int lb = nodesPerPE * mype_node;
    int ub = (lb+nodesPerPE) < numNodes? (lb+nodesPerPE):numNodes;

    d_input = (float *) nvshmem_malloc ((ub-lb)*dim*sizeof(float)); // NVSHMEM allocation (ub - lb) nodes on current GPU.
    gpuErrchk(cudaMalloc((void**)&d_output, (ub-lb)*dim*sizeof(float)));  // private global memory (ub-lb) nodes on current GPU.
    
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, numNodes*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, numEdges*sizeof(int))); 
    
    gpuErrchk(cudaMemcpy(d_row_ptr, &asym.row_ptr[0], numNodes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[0], numEdges*sizeof(int), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    t1 = MPI_Wtime(); 
    mgg_profile<<<1, 32*warpPerBlock>>>(d_output, d_input, 
                                        d_row_ptr, d_col_ind, 
                                        nodeOfInterest, dim, nodesPerPE);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    t2 = MPI_Wtime(); 

    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3); 

    // release nvshmem objects and finalize context.
    cudaFree(d_output);
    nvshmem_free(d_input);
    nvshmem_finalize();
    MPI_Finalize();

    return 0;
}