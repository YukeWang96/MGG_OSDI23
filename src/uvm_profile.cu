
#include <iostream>
#include <stdio.h>
#include <omp.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"

using namespace std;


int main(int argc, char* argv[]){
	
    if (argc < 4){
        printf("Usage: ./main graph.mtx dim nodeOfInterest\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();

    int warpPerBlock = 1;
    int dim = atoi(argv[2]);

    // float* input = (float*)malloc(numNodes*dim*sizeof(float));
    float *d_output, *d_input;
    int *d_col_ind, *d_row_ptr;

    int mype_node = 0;
    cudaSetDevice(mype_node);

    // Load the corresponding tiles.
    const int lb_src = 0; // node of interest
    const int ub_src = lb_src + 1;  // the node next to the node of interest.
    const int e_lb = 0;
    const int e_ub = e_lb + atoi(argv[3]);
    printf("node [%d]: %d neighbors\n", lb_src, asym.row_ptr[ub_src] - asym.row_ptr[lb_src]);
    
    gpuErrchk(cudaMallocManaged((void**)&d_input, numNodes*dim*sizeof(float)));    // UVM allocation
    // gpuErrchk(cudaMalloc((void**)&d_output, (ub_src-lb_src)*dim*sizeof(float)));   
    gpuErrchk(cudaMalloc((void**)&d_output, (ub_src-lb_src)*dim*sizeof(float)));   

    gpuErrchk(cudaMalloc((void**)&d_row_ptr, numNodes*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, numEdges*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_row_ptr, &asym.row_ptr[0], numNodes*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[0], numEdges*sizeof(int), cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    uvm_profile<<<1, 32*warpPerBlock>>>(d_output, d_input, 
                                        d_row_ptr, d_col_ind, 
                                        e_lb, e_ub, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("[%d] CUDA error at uvm_profile: %s\n", mype_node, cudaGetErrorString(error));
        exit(-1);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);
    printf("===================================\n");

    cudaFree(d_input);    
    cudaFree(d_output);
    cudaFree(d_col_ind);

    return 0;
}