
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
#include "gnn_layer.cuh"

// using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph_beg.bin graph_csr.bin graph_weight.bin num_GPUs partSize warpPerblock dim\n");
        return -1;
    }
    
    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];

    graph<long, long, int, int, int, int>* ginst = new graph<long, long, int, int, int, int>(beg_file, csr_file, weight_file);
    std::vector<int> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<int> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);
    // cout << "Complete loading graphs !!" << endl;
    int numNodes = global_row_ptr.size() - 1;
    int numEdges = global_col_ind.size();    

    int num_GPUs = atoi(argv[4]);           // 2
    int partSize = atoi(argv[5]);           // 32
    int warpPerBlock = atoi(argv[6]);       // 4
    int dim = atoi(argv[7]);                // input
    int dim1 = atoi(argv[8]);               // hidden
    int dim2 = atoi(argv[9]);               // output

    int num_profiles = 200;
    std::vector<float> time_li;

    int* d_row_ptr, *d_col_ind;
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, global_row_ptr.size()*sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(int))); 
    CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
    // 
    // define model
    // 
    dense_param_beg* dp1 = new dense_param_beg("d-1", numNodes, dim, dim1);
    sparse_param_hidden* sp1 = new sparse_param_hidden("s-1", dp1->d_out, d_row_ptr, d_col_ind, numNodes, dim1, partSize, warpPerBlock);

    dense_param_hidden* dp2 = new dense_param_hidden("d-2", sp1->d_out, numNodes, dim1, dim2);
    sparse_param_hidden* sp2 = new sparse_param_hidden("s-2", dp2->d_out, d_row_ptr, d_col_ind, numNodes, dim2, partSize, warpPerBlock);
    
     for (int i = 0; i < 10; i++)
    {
        dense_beg_forward(dp1);
        sparse_hidden_forward(sp1);
        dense_hidden_forward(dp2);
        sparse_hidden_forward(sp2);
    }

    //
    // Execute Model
    //
    std::clock_t c_start = std::clock();    
    for (int i = 0; i < num_profiles; i++)
    {
        dense_beg_forward(dp1);
        sparse_hidden_forward(sp1);
        dense_hidden_forward(dp2);
        sparse_hidden_forward(sp2);
    }
    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("Time (ms): %.3f\n", time_elapsed_ms);
    //
    // Profile model by layers.
    //
    /*
    std::clock_t c_start = std::clock();    

    // dense layer-1
    dense_beg_forward(dp1);

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    time_li.push_back(time_elapsed_ms);

    c_start = std::clock();    
    // sparse layer-1
    sparse_hidden_forward(sp1);

    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    time_li.push_back(time_elapsed_ms);

    c_start = std::clock();    
    // dense layer-2
    dense_hidden_forward(dp2);

    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    time_li.push_back(time_elapsed_ms);

    c_start = std::clock();    
    // sparse layer-2
    sparse_hidden_forward(sp2);

    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    time_li.push_back(time_elapsed_ms);

    for (int i = 0; i < time_li.size(); i++){
        printf("layer-%d (ms): %.3f\n", time_li[i]);
    }
    */

    printf("===================================\n");

    return 0;
}