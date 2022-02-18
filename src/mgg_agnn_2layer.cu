
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>
#include <cublas_v2.h>

#include "graph.h"
#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "cublas_utils.h"
#include "layer_new.cuh"
#include "gnn_kernel.cuh"

// using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim\n");
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

    int num_GPUs = atoi(argv[4]);           // 2
    int partSize = atoi(argv[5]);           // 32
    int warpPerBlock = atoi(argv[6]);       // 4
    int dim = atoi(argv[7]);                // 16
    int dim1 = 128;               
    int dim2 = 128;

    int lb = 0;
    int ub = numNodes;
    
    int* d_row_ptr, *d_col_ind;
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, global_row_ptr.size()*sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(int))); 
    CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
    // 
    // define AGNN model
    // 
    AGNN_param_beg* sp1 = new AGNN_param_beg("s-1", d_row_ptr, d_col_ind, numNodes, numEdges, dim);
    dense_param_hidden* dp1 = new dense_param_hidden("d-1", sp1->d_out, numNodes, dim, dim1);
    AGNN_param_hidden* sp2 = new AGNN_param_hidden("s-2",  dp1->d_out, d_row_ptr, d_col_ind, numNodes, numEdges, dim1);
    dense_param_hidden* dp2 = new dense_param_hidden("d-2", sp2->d_out, numNodes, dim1, dim2);

    //
    // xecute model.
    //
    std::clock_t c_start = std::clock();    
    // AGNN layer-1
    AGNN_beg_forward(sp1);
    // dense layer-1
    dense_hidden_forward(dp1);
    // AGNN layer-2
    AGNN_hidden_forward(sp2);
    // dense layer-2
    dense_hidden_forward(dp2);
    // model end
    std::clock_t c_end = std::clock();

    float time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    printf("Time (ms): %.3f\n", time_elapsed_ms);
    printf("===================================\n");

    return 0;
}