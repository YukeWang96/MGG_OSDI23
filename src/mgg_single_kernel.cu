
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

    float *h_input_ref, *h_output_ref,  *d_input_ref, *d_output_ref;
    h_input_ref = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
    h_output_ref = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
    std::fill_n(h_input_ref, numNodes * dim, 1.0f); // filled with all zeros.
    std::fill_n(h_output_ref, numNodes * dim, 0.0f); // filled with all zeros.
    gpuErrchk(cudaMalloc((void**)&d_input_ref, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
    gpuErrchk(cudaMalloc((void**)&d_output_ref, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)

    int* d_row_ptr_ref, *d_col_ind_ref;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_ref, global_row_ptr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_ref, global_col_ind.size()*sizeof(int))); 
    gpuErrchk(cudaMemcpy(d_row_ptr_ref, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_ref, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input_ref, h_input_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_output_ref, h_output_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    //
    // dense-1 layer param
    //
    const float alpha = 1.0f;
    const float beta = 0.0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    
    float* h_W1, *d_W1, *d_out1, *d_out1_sag;
    h_W1 = (float *) malloc (dim * dim1 * sizeof(float));      // CPU host memory (input_ref)
    std::fill_n(h_W1, dim * dim1, 1.0f); // filled with all zeros.
    gpuErrchk(cudaMalloc((void**)&d_W1, dim * dim1 * sizeof(float))); // GPU device memory (input_ref)
    gpuErrchk(cudaMemcpy(d_W1, h_W1, dim * dim1 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&d_out1, numNodes * dim1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_out1_sag, numNodes * dim1 * sizeof(float)));

    const int m1 = dim1, n1 = numNodes, k1 = dim; // (XW) --> W_T x X_T for column-major store.
    const int ldx1 = dim, ldw1 = dim1, ldout1 = dim1;
    //
    // dense-2 layer param
    //    
    float* h_W2, *d_W2, *d_out2;
    h_W2 = (float *) malloc (dim1 * dim2 * sizeof(float));      // CPU host memory (input_ref)
    std::fill_n(h_W1, dim1 * dim2, 1.0f); // filled with all zeros.
    gpuErrchk(cudaMalloc((void**)&d_W2,  dim1 * dim2 * sizeof(float))); // GPU device memory (input_ref)
    gpuErrchk(cudaMemcpy(d_W2, h_W2,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**)&d_out2, numNodes * dim2 * sizeof(float)));

    const int m2 = dim2, n2 = numNodes, k2 = dim1; // (XW) --> W_T X X_T for column-major store.
    const int ldx2 = dim1, ldw2 = dim2, ldout2 = dim2;

    //
    // xecute model.
    //
    std::clock_t c_start = std::clock();    
    // sparse layer-1
    SAG_host_ref(d_output_ref, d_input_ref, 
                d_row_ptr_ref, d_col_ind_ref, 
                lb, ub, dim, global_col_ind.size());
    // gpuErrchk(cudaMemcpy(h_output_ref, d_output_ref, numNodes * dim * sizeof(float), cudaMemcpyDeviceToHost));
    // dense layer-1
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m1, n1, k1, &alpha, d_W1, ldw1, d_output_ref, ldx1, &beta, d_out1, ldout1));
    // CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream));    
    // sparse layer-2
    SAG_host_ref(d_out1_sag, d_out1, 
        d_row_ptr_ref, d_col_ind_ref, 
        lb, ub, dim1, global_col_ind.size());
    // dense layer-2
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m2, n2, k2, &alpha, d_W2, ldw2, d_out1_sag, ldx2, &beta, d_out2, ldout2));
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // for (int nid = 0; nid < 10; nid++){
    //     printf("out [%d] ", nid);
    //     for (int d = 0; d < dim; d++){
    //         printf("%.3f,", h_output_ref[nid * dim + d]);
    //     }
    //     printf("\n");
    // }
    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("Total (ms): %.3f\n", time_elapsed_ms);

    cudaFree(d_output_ref);
    free(h_output_ref);
    printf("===================================\n");

    return 0;
}