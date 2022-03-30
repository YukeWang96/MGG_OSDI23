
#include <iostream>
#include <stdio.h>
#include <omp.h>

#include "graph.h"
#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"

// #define validate //--> for results validation
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 5){
        printf("Usage: ./main beg_file.bin csr_file.bin weight_file.bin num_GPUs partSize warpPerBlock dim\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    
    graph<long, long, int, int, int, int>* ginst = new graph<long, long, int, int, int, int>(beg_file, csr_file, weight_file);
    std::vector<int> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<int> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    int numNodes = global_row_ptr.size() - 1;
    int numEdges = global_col_ind.size();    

    int num_GPUs = atoi(argv[4]);
    int partSize = atoi(argv[5]);
    int warpPerBlock = atoi(argv[6]);
    int dim = atoi(argv[7]);

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float** h_input = new float*[num_GPUs];
    float** h_output = new float*[num_GPUs];
    float **d_output, **d_input;
    // std::fill(h_ref, h_ref+nodesPerPE*dim, 0.0);          // sets every value in the array to 0.0

    // memset(input, 0, numNodes*dim*sizeof(float));
    int **d_row_ptr = new int*[num_GPUs]; 
    int **d_col_ind = new int*[num_GPUs]; 
    // int *d_row_ptr, *d_col_ind;
    // float *d_ref;


#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);

    // h_input[mype_node] = (float*)malloc(nodesPerPE*dim*sizeof(float));
    // h_output[mype_node] = (float*)malloc(nodesPerPE*dim*sizeof(float));
    // float* h_ref = (float*)malloc(nodesPerPE*dim*sizeof(float));

    // std::fill(h_input[mype_node], h_input[mype_node]+nodesPerPE*dim, 1.0);      // sets every value in the array to 1.0
    // std::fill(h_output[mype_node], h_output[mype_node]+nodesPerPE*dim, 0.0);    // sets every value in the array to 0.0

    printf("mype_node: %d, nodesPerPE: %d\n", mype_node, nodesPerPE);

    gpuErrchk(cudaMalloc((void**)&d_input,  num_GPUs*sizeof(float*))); 
    gpuErrchk(cudaMalloc((void**)&d_output,  num_GPUs*sizeof(float*))); 
    // UVM data: output, input, row_ptr, col_ind 
    gpuErrchk(cudaMallocManaged((void**)&h_output[mype_node],  nodesPerPE*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&h_input[mype_node],   nodesPerPE*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_row_ptr[mype_node], (numNodes+1)*sizeof(int)));
    gpuErrchk(cudaMallocManaged((void**)&d_col_ind[mype_node], numEdges*sizeof(int))); 

    // cudaMemset(d_output[mype_node],            0,         nodesPerPE*dim*sizeof(float));
    // gpuErrchk(cudaMemcpy(d_input[mype_node],   h_input[mype_node],   nodesPerPE*dim*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr[mype_node], &global_row_ptr[0],   (numNodes+1)*sizeof(int),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind[mype_node], &global_col_ind[0],   numEdges*sizeof(int),         cudaMemcpyHostToDevice));


}


// #ifdef validate
// cudaSetDevice(0);
// SAG_host_single_ref(d_ref, d_input, d_row_ptr, d_col_ind, numNodes, dim);
// gpuErrchk(cudaMemcpy(h_ref,     d_ref,       numNodes*dim*sizeof(float),   cudaMemcpyDeviceToHost));
// #endif

// One GPU per threads
#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);
    gpuErrchk(cudaMemcpy(d_input, h_input, num_GPUs*sizeof(float*), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_output, h_output, num_GPUs*sizeof(float*), cudaMemcpyHostToDevice));

    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    SAG_host_UVM_updated(h_output[mype_node], d_input, 
                        d_row_ptr[mype_node], d_col_ind[mype_node], 
                        lb_src, ub_src, dim, num_GPUs, 
                        mype_node, nodesPerPE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);
}
return;

#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    gpuErrchk(cudaMemcpy(h_output[mype_node],  d_output[mype_node], nodesPerPE*dim*sizeof(float), cudaMemcpyDeviceToHost));

    // #ifdef validate
    // bool status = compare_array(h_ref, h_output, numNodes*dim);
    // printf(status ? "validate: True\n" : "validate: False\n");
    // #endif

    // cudaFree(d_ref);    
    // cudaFree(d_input[mype_node]);    
    // cudaFree(d_output[mype_node]);
    // cudaFree(d_col_ind[mype_node]);
    // cudaFree(d_row_ptr[mype_node]);
    // cudaFree(d_input);
}


    // free(d_output);
    // free(d_col_ind);
    // free(d_row_ptr);

    // free(h_ref);
    // free(h_output);
    // free(h_input);

    return 0;
}