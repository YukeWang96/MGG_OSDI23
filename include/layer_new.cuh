#ifndef layer_new_cuh
#define layer_new_cuh

#include <cublas_v2.h>
#include "cublas_utils.h"
#include "utils.cuh"

class sparse_param_beg{

public:
    sparse_param_beg(int *d_row_ptr_in, int *d_col_ind_in, int numNodes_in, int dim_in){

        numNodes = numNodes_in;
        dim = dim_in;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;
        // allocate memory space.
        _mem_alloc();
    }    

    void _mem_alloc(){

        h_in = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_out = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_in, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_out, numNodes * dim, 0.0f); // filled with all zeros.
        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    
        // CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, h_out, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

public:
    int numNodes, dim;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;
};


class sparse_param_hidden{


public:
    sparse_param_hidden(float* d_in_input, int *d_row_ptr_in, int *d_col_ind_in,  int numNodes_in, int dim_in){

        numNodes = numNodes_in;
        dim = dim_in;
        d_in = d_in_input;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;

        // allocate memory space.
        _mem_alloc();
    }    

    void _mem_alloc(){
        // h_in = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_out = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        // std::fill_n(h_in, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_out, numNodes * dim, 0.0f); // filled with all zeros.
        // CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    
        // CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, global_row_ptr.size()*sizeof(int))); 
        // CUDA_CHECK(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(int))); 
        // CUDA_CHECK(cudaMemcpy(d_row_ptr, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, h_out, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

public:
    int numNodes, dim;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;
};


class dense_param_beg{

public:
    dense_param_beg(int numNodes_in, int dim1_in, int dim2_in){
        numNodes = numNodes_in;
        dim1 = dim1_in;
        dim2 = dim2_in;

        m = dim2, n = numNodes, k = dim1; // (XW) --> W_T x X_T for column-major store.
        ldx = dim1, ldw = dim2, ldout = dim2;

        alpha = 1.0f;
        beta = 0.0;

        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_N;
        cublasH = NULL;
        
        CUBLAS_CHECK(cublasCreate(&cublasH));
        // allocate memory space.
        _mem_alloc();
    }    


    void _mem_alloc(){
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));              // CPU host memory (input_ref)
        std::fill_n(h_W, dim1 * dim2, 1.0f);                                // filled with all zeros.

        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim1 * sizeof(float))); // GPU device memory (output_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMemcpy(d_W, h_W,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_out, *d_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;
};


class dense_param_hidden{

public:
    dense_param_hidden(float *d_in_input, int numNodes_in, int dim1_in, int dim2_in){
        numNodes = numNodes_in;
        dim1 = dim1_in;
        dim2 = dim2_in;
        d_in = d_in_input;

        m = dim2, n = numNodes, k = dim1; // (XW) --> W_T x X_T for column-major store.
        ldx = dim1, ldw = dim2, ldout = dim2;

        alpha = 1.0f;
        beta = 0.0;

        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_N;
        cublasH = NULL;
        
        CUBLAS_CHECK(cublasCreate(&cublasH));
        // allocate memory space.
        _mem_alloc();
    }    


    void _mem_alloc(){
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));              // CPU host memory (input_ref)
        std::fill_n(h_W, dim1 * dim2, 1.0f);                                // filled with all zeros.

        CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMemcpy(d_W, h_W,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_out, *d_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;
};
#endif // layer_new_cuh