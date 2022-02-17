
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
#include "cublas_utils.h"
// #include "csr_formatter.h"
// #include "layer.h"

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
    // cout << "Complete loading graphs !!" << endl;
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
    
    float* d_row_ptr, *d_col_ind;


    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, global_row_ptr.size()*sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(int))); 
    CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));



    sparse_param_beg* sp1 = new sparse_param_beg(d_row_ptr, d_col_ind, numNodes, dim);
    dense_param_hidden* dp1 = new dense_param_hidden(sp1->d_out, numNodes, dim, dim1);
    sparse_param_hidden* sp2 = new sparse_param_hidden(dp1->d_out, d_row_ptr, d_col_ind, numNodes, dim1);
    dense_param_hidden* dp1 = new dense_param_hidden(sp2->d_out, numNodes, dim1, dim2);

    //
    // xecute model.
    //
    std::clock_t c_start = std::clock();    
    // sparse layer-1
    SAG_host_ref(sp1->d_out, sp1->d_in, d_row_ptr, d_col_ind, lb, ub, dim1, global_col_ind.size());
    // dense layer-1
    CUBLAS_CHECK(cublasSgemm(dp1->cublasH, dp1->transa, dp1->transb, dp1->m, dp1->n, dp1->k, &(dp1->malpha), dp1->d_W, dp1->ldw, dp1->d_out, dp1->ldx, &(dp1->beta), dp1->d_out, dp1->ldout));
    // sparse layer-2
    SAG_host_ref(sp2->d_out, sp2->d_in, d_row_ptr, d_col_ind, lb, ub, dim1, global_col_ind.size());
    // dense layer-2
    CUBLAS_CHECK(cublasSgemm(dp2->cublasH, dp2->transa, dp2->transb, dp2->m, dp2->n, dp2->k, &(dp2->malpha), dp2->d_W, dp2->ldw, dp2->d_out, dp2->ldx, &(dp2->beta), dp2->d_out, dp2->ldout));

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("Total (ms): %.3f\n", time_elapsed_ms);


    // CUDA_CHECK(cudaMemcpy(h_output_ref, d_output_ref, numNodes * dim * sizeof(float), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream));  
    //
    // sparse-1 param
    // 
    // float *h_input_ref, *h_output_ref,  *d_input_ref, *d_output_ref;
    // h_input_ref = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
    // h_output_ref = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
    // std::fill_n(h_input_ref, numNodes * dim, 1.0f); // filled with all zeros.
    // std::fill_n(h_output_ref, numNodes * dim, 0.0f); // filled with all zeros.
    // CUDA_CHECK(cudaMalloc((void**)&d_input_ref, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
    // CUDA_CHECK(cudaMalloc((void**)&d_output_ref, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)

    // int* d_row_ptr, *d_col_ind;
    // CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, global_row_ptr.size()*sizeof(int))); 
    // CUDA_CHECK(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(int))); 
    // CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_input_ref, h_input_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_output_ref, h_output_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    
    //
    // dense-1 layer param
    //
    // const float alpha = 1.0f;
    // const float beta = 0.0;

    // cublasOperation_t transa = CUBLAS_OP_N;
    // cublasOperation_t transb = CUBLAS_OP_N;
    // cublasHandle_t cublasH = NULL;
    // cudaStream_t stream = NULL;

    // CUBLAS_CHECK(cublasCreate(&cublasH));
    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    
    // float* h_W1, *d_W1, *d_out1, *d_out1_sag;
    // h_W1 = (float *) malloc (dim * dim1 * sizeof(float));      // CPU host memory (input_ref)
    // std::fill_n(h_W1, dim * dim1, 1.0f); // filled with all zeros.
    // CUDA_CHECK(cudaMalloc((void**)&d_W1, dim * dim1 * sizeof(float))); // GPU device memory (input_ref)
    // CUDA_CHECK(cudaMemcpy(d_W1, h_W1, dim * dim1 * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc((void**)&d_out1, numNodes * dim1 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc((void**)&d_out1_sag, numNodes * dim1 * sizeof(float)));

    // const int m1 = dim1, n1 = numNodes, k1 = dim; // (XW) --> W_T x X_T for column-major store.
    // const int ldx1 = dim, ldw1 = dim1, ldout1 = dim1;
    //
    // dense-2 layer param
    //    
    // float* h_W2, *d_W2, *d_out2;
    // h_W2 = (float *) malloc (dim1 * dim2 * sizeof(float));      // CPU host memory (input_ref)
    // std::fill_n(h_W1, dim1 * dim2, 1.0f); // filled with all zeros.
    // CUDA_CHECK(cudaMalloc((void**)&d_W2,  dim1 * dim2 * sizeof(float))); // GPU device memory (input_ref)
    // CUDA_CHECK(cudaMemcpy(d_W2, h_W2,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc((void**)&d_out2, numNodes * dim2 * sizeof(float)));

    // const int m2 = dim2, n2 = numNodes, k2 = dim1; // (XW) --> W_T X X_T for column-major store.
    // const int ldx2 = dim1, ldw2 = dim2, ldout2 = dim2;

    // for (int nid = 0; nid < 10; nid++){
    //     printf("out [%d] ", nid);
    //     for (int d = 0; d < dim; d++){
    //         printf("%.3f,", h_output_ref[nid * dim + d]);
    //     }
    //     printf("\n");
    // }

    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess){
    //     printf("CUDA error @ SAG_cuda_kernel_ref: %s\n", cudaGetErrorString(error));
    //     exit(-1);
    // }


    cudaFree(d_output_ref);
    free(h_output_ref);
    printf("===================================\n");

    return 0;
}