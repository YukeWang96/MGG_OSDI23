#ifndef layer_new_cuh
#define layer_new_cuh

#include <cublas_v2.h>
#include <cudnn.h>
#include "cublas_utils.h"
#include "utils.cuh"
#include <nvshmem.h>
#include <nvshmemx.h>

int n_partition = 1;

class softmax_param{

public:
    softmax_param(const char* name_in, float *d_in_input, int numNodes_in, int dim_in){

        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_in = d_in_input;

        alpha = 1.0f;
        beta = 0.0;
        cudnnCreate(&cudnnHandle);
        
        // allocate memory space.
        _mem_alloc();

        // softmaxForward(n, c, h, w, dstData, &srcData);
        cudnnCreateTensorDescriptor(&srcTensorDesc);
        cudnnCreateTensorDescriptor(&sftTensorDesc);
        cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                numNodes, dim, 1, 1);
        cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                numNodes, dim, 1, 1);
    }    


    void _mem_alloc(){                           
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim * sizeof(float)));
    }

public:
    int numNodes, dim;
    float *d_out, *d_in;
    float alpha, beta;

    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;

    char name[8];
};



class softmax_new_param{

public:
    softmax_new_param(const char* name_in, float *d_in_input, float *d_out_ext, 
                    int numNodes_in, int dim_in){

        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_in = d_in_input;
        d_out = d_out_ext;

        alpha = 1.0f;
        beta = 0.0;
        cudnnCreate(&cudnnHandle);
        
        // allocate memory space.
        _mem_alloc();

        // softmaxForward(n, c, h, w, dstData, &srcData);
        cudnnCreateTensorDescriptor(&srcTensorDesc);
        cudnnCreateTensorDescriptor(&sftTensorDesc);
        cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                numNodes, dim, 1, 1);
        cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                numNodes, dim, 1, 1);
    }    


    void _mem_alloc(){                           
        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));
        // CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim * sizeof(float)));
    }

public:
    int numNodes, dim;
    float *d_out, *d_in;
    float alpha, beta;

    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;

    char name[8];
};



class sparse_param_beg{

public:
    sparse_param_beg(const char* name_in, int *d_row_ptr_in, int *d_col_ind_in, 
                    int numNodes_in, int dim_in, int partSize_in, int warpPerBlock_in){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;

        warpPerBlock = warpPerBlock_in;
        partSize = partSize_in;
        
        _mem_alloc();
        _kernel_param();
    }    

    void _mem_alloc(){

        h_in = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_out = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_in, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_out, numNodes * dim, 0.0f); // filled with all zeros.
        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    
        CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, h_out, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    void _kernel_param(){
        
        // warpPerBlock = 8;
        // partSize = 16;
       WARP_SIZE = 32;
       block = warpPerBlock * WARP_SIZE;
       grid = numNodes/n_partition;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, dim;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;
    char name[8];
};


class sparse_param_hidden{

public:
    sparse_param_hidden(const char* name_in, float* d_in_input, int *d_row_ptr_in, int *d_col_ind_in,  
                        int numNodes_in, int dim_in, int partSize_in, int warpPerBlock_in){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        
        d_in = d_in_input;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;

        warpPerBlock = warpPerBlock_in;
        partSize = partSize_in;

        _mem_alloc();
        _kernel_param();

    }    

    void _mem_alloc(){
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim * sizeof(float)));
    }

    void _kernel_param(){
        
        // warpPerBlock = 8;
        // partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = numNodes/n_partition;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, dim;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;

    char name[8];
};


class dense_param_beg{

public:
    dense_param_beg(const char* name_in, float* d_in_ext, int numNodes_in, int dim1_in, int dim2_in){
        
        strncpy(name, name_in, 8);


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
        
        d_in = d_in_ext;

        CUBLAS_CHECK(cublasCreate(&cublasH));
        // allocate memory space.
        _mem_alloc();
    }    


    void _mem_alloc(){
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));     
        h_in = (float *) malloc (numNodes * dim1 * sizeof(float));                   
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               
        // std::fill_n(h_in, numNodes * dim1, 1.0f);                               

        // CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim1 * sizeof(float))); 
        // CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float)));

        // d_in = (float *) nvshmem_malloc (numNodes * dim1 * sizeof(float)); 
        d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  
        // d_W_new = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float)); 
        d_out = (float *) nvshmem_malloc (numNodes * dim2 * sizeof(float)); 

        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_W, h_W, dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim1 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in, *h_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};


class dense_param_new_beg{

public:
    dense_param_new_beg(const char* name_in, float* d_in_ext, float* d_out_ext, 
                    int numNodes_in, int dim1_in, int dim2_in){
        
        strncpy(name, name_in, 8);


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
        
        d_in = d_in_ext;
        d_out = d_out_ext;

        CUBLAS_CHECK(cublasCreate(&cublasH));
        // allocate memory space.
        _mem_alloc();
    }    


    void _mem_alloc(){
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));     
        h_in = (float *) malloc (numNodes * dim1 * sizeof(float));                   
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               
        // std::fill_n(h_in, numNodes * dim1, 1.0f);                               

        // CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim1 * sizeof(float))); 
        // CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float)));

        // d_in = (float *) nvshmem_malloc (numNodes * dim1 * sizeof(float)); 
        d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  
        // d_W_new = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float)); 
        // d_out = (float *) nvshmem_malloc (numNodes * dim2 * sizeof(float)); 

        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_W, h_W, dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim1 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in, *h_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};


class dense_param_new_hidden{

public:
    dense_param_new_hidden(const char* name_in, float *d_in_input, float *d_out_ext, 
                        int numNodes_in, int dim1_in, int dim2_in){

        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim1 = dim1_in;
        dim2 = dim2_in;
        d_in = d_in_input;
        d_out = d_out_ext;

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
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));             
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               

        // CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float))); 
        // d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.
        // d_W_new = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.

        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
        // d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.
        CUDA_CHECK(cudaMalloc((void**)&d_W, dim1 * dim2 * sizeof(float)));
        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float))); 
        // d_out = (float *) nvshmem_malloc (numNodes * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.

        CUDA_CHECK(cudaMemcpy(d_W, h_W,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};


class dense_param_hidden{

public:
    dense_param_hidden(const char* name_in, float *d_in_input, int numNodes_in, int dim1_in, int dim2_in){

        strncpy(name, name_in, 8);

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
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));             
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               

        // CUDA_CHECK(cudaMalloc((void**)&d_W,  dim1 * dim2 * sizeof(float))); 
        // d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.
        // d_W_new = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.

        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float)));
        // d_W = (float *) nvshmem_malloc (dim1 * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.
        CUDA_CHECK(cudaMalloc((void**)&d_W, dim1 * dim2 * sizeof(float)));
        // CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim2 * sizeof(float))); 
        d_out = (float *) nvshmem_malloc (numNodes * dim2 * sizeof(float));  // NVSHMEM global memory for input embedding.

        CUDA_CHECK(cudaMemcpy(d_W, h_W,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};



class dense_param_beg_uvm{

public:
    dense_param_beg_uvm(const char* name_in, float* d_in_ext, int gpu_id, float** d_out_arr,
                        int numNodes_in, int dim1_in, int dim2_in){
        
        strncpy(name, name_in, 8);

        gpuid = gpu_id;
        _d_out_arr = d_out_arr;

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
        
        d_in = d_in_ext;

        CUBLAS_CHECK(cublasCreate(&cublasH));
        _mem_alloc();
    }    


    void _mem_alloc(){
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));     
        h_in = (float *) malloc (numNodes * dim1 * sizeof(float));                   
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               
 
        CUDA_CHECK(cudaMallocManaged((void**)&_d_out_arr[gpuid], numNodes * dim2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_W, dim1 * dim2 * sizeof(float)));
        d_out = _d_out_arr[gpuid];

        CUDA_CHECK(cudaMemcpy(d_W, h_W, dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(_d_out_arr[gpuid], 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout;
    int numNodes, dim1, dim2;
    int gpuid;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in, *h_in;
    float** _d_out_arr;
    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};

class dense_param_hidden_uvm{

public:
    dense_param_hidden_uvm(const char* name_in, float *d_in_input, int gpu_id, float** d_out_arr, int numNodes_in, int dim1_in, int dim2_in){

        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim1 = dim1_in;
        dim2 = dim2_in;
        d_in = d_in_input;
                            
        gpuid = gpu_id;
        _d_out_arr = d_out_arr;

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
        h_W = (float *) malloc (dim1 * dim2 * sizeof(float));             
        std::fill_n(h_W, dim1 * dim2, 1.0f);                               

        CUDA_CHECK(cudaMalloc((void**)&d_W, dim1 * dim2 * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged((void**)&_d_out_arr[gpuid], numNodes * dim2 * sizeof(float)));
        d_out = _d_out_arr[gpuid];
        
        CUDA_CHECK(cudaMemcpy(d_W, h_W,  dim1 * dim2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim2 * sizeof(float)));
    }

public:
    int m, n, k, ldx, ldw, ldout, gpuid;
    int numNodes, dim1, dim2;
    float* h_W, *d_W, *d_W_new, *d_out, *d_in;
    float** _d_out_arr;

    float alpha, beta;
    cublasOperation_t transa, transb;
    cublasHandle_t cublasH;

    char name[8];
};


class AGNN_param_beg{
    // https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=dotgat#agnnconv
public:
    AGNN_param_beg(const char* name_in, int *d_row_ptr_in, int *d_col_ind_in, 
                    int numNodes_in, int numEdges_in, int dim_in,  int warpPerBlock_in, int partSize_in){
        strncpy(name, name_in, 8);

        numEdges = numEdges_in;
        numNodes = numNodes_in;

        dim = dim_in;
        
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;

        warpPerBlock = warpPerBlock_in;
        partSize = partSize_in;

        _mem_alloc();
        _kernel_param();
    }    

    void _mem_alloc(){

        h_in = (float *) malloc (numNodes * dim * sizeof(float));   
        std::fill_n(h_in, numNodes * dim, 1.0f); 
        
        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); 
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_edge_att, numEdges * sizeof(float))); 

        CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0,  numNodes * dim  * sizeof(float)));    
        CUDA_CHECK(cudaMemset(d_edge_att, 0, numEdges * sizeof(float))); 
    }

    void _kernel_param(){
        
        // warpPerBlock = 2;
        // partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = (numNodes * WARP_SIZE + block - 1) / block;
        // grid  = numNodes;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    float beta;
    int numNodes, numEdges, dim;
    float *h_in, *h_out, *d_in, *d_out, *d_edge_att;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;
    char name[8];
};



class AGNN_param_hidden{
    // https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=dotgat#agnnconv
public:
    AGNN_param_hidden(const char* name_in, float* d_in_input, int *d_row_ptr_in, int *d_col_ind_in,  
                        int numNodes_in, int numEdges_in, int dim_in, int warpPerBlock_in, int partSize_in){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        numEdges = numEdges_in;

        dim = dim_in;
        d_in = d_in_input;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;

        warpPerBlock = warpPerBlock_in;
        partSize = partSize_in;

        _mem_alloc();
        _kernel_param();
    }    

    void _mem_alloc(){
        CUDA_CHECK(cudaMalloc((void**)&d_edge_att, numEdges * sizeof(float))); 
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));

        CUDA_CHECK(cudaMemset(d_edge_att, 0, numEdges * sizeof(float))); 
        CUDA_CHECK(cudaMemset(d_out, 0,  numNodes * dim  * sizeof(float)));
    }

    void _kernel_param(){
        
        // warpPerBlock = 2;
        // partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = (numNodes * WARP_SIZE + block - 1) / block;
    //    grid = numNodes;
    //    shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, numEdges, dim;
    float *h_in, *h_out, *d_in, *d_out, *d_edge_att;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;

    char name[8];
};



class SGC_param_beg{

public:
    SGC_param_beg(const char* name_in, int *d_row_ptr_in, int *d_col_ind_in, int numNodes_in, int dim_in, int k=2){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;
        khop = k;

        _mem_alloc();
        _kernel_param();
        _gpu_ready();
    }    

    void _gpu_ready()
    {
        CUDA_CHECK(cudaMalloc((void**)&gpu, sizeof(SGC_param_beg)));
        CUDA_CHECK(cudaMemcpy(this->gpu, this, sizeof(SGC_param_beg), cudaMemcpyHostToDevice));
        // return this->gpu;
    }

    void _mem_alloc(){

        h_in = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_out = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_in, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_out, numNodes * dim, 0.0f); // filled with all zeros.
        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    
        CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, h_out, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    void _kernel_param(){
        
        // warpPerBlock = 4;
        // partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = numNodes;
    //    grid = 6 * 108;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, dim, khop;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;


    SGC_param_beg* gpu;
    char name[8];
};


class SGC_param_hidden{

public:
    SGC_param_hidden(const char* name_in, float* d_in_input, int *d_row_ptr_in, int *d_col_ind_in,  int numNodes_in, int dim_in, int k=2){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_in = d_in_input;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;
        khop = k;

        _mem_alloc();
        _kernel_param();
        _kernel_launch();

        _gpu_ready();
    }    

    void _gpu_ready()
    {
        CUDA_CHECK(cudaMalloc((void**)&gpu, sizeof(SGC_param_hidden)));
        CUDA_CHECK(cudaMemcpy(this->gpu, this, sizeof(SGC_param_hidden), cudaMemcpyHostToDevice));
        // return this->gpu;
    }

    void _mem_alloc(){
        h_out = (float *) malloc (numNodes * dim * sizeof(float));  
        std::fill_n(h_out, numNodes * dim, 0.0f); 

        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim * sizeof(float)));
    }

    void _kernel_param(){
        
        warpPerBlock = 4;
        partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = numNodes;
        // grid = 6 * 108;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

    void _kernel_launch(){
        const int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        multiProcessorCount = deviceProp.multiProcessorCount;
        // cudaFuncSetAttribute(SGC_cuda_kernel_hidden_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
        // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SGC_cuda_kernel_hidden_wrapper, block, shared_memory);
    }

public:
    int numNodes, dim, khop;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;
    int numBlocksPerSm, multiProcessorCount;

    SGC_param_hidden* gpu;

    char name[8];
};

class TAG_param_beg{

public:
    TAG_param_beg(const char* name_in, int *d_row_ptr_in, int *d_col_ind_in, int numNodes_in, int dim_in, int k=2){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;
        khop = k;

        _mem_alloc();
        _kernel_param();
    }    

    void _mem_alloc(){

        h_in = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_out = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_in, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_out, numNodes * dim, 0.0f); // filled with all zeros.
        CUDA_CHECK(cudaMalloc((void**)&d_in, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    
        CUDA_CHECK(cudaMemcpy(d_in, h_in, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out, h_out, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

    void _kernel_param(){
        
        warpPerBlock = 4;
        partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = numNodes;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, dim, khop;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;
    char name[8];
};


class TAG_param_hidden{

public:
    TAG_param_hidden(const char* name_in, float* d_in_input, int *d_row_ptr_in, int *d_col_ind_in,  int numNodes_in, int dim_in, int k=2){
        strncpy(name, name_in, 8);

        numNodes = numNodes_in;
        dim = dim_in;
        d_in = d_in_input;
        d_row_ptr = d_row_ptr_in;
        d_col_ind = d_col_ind_in;
        khop = k;

        _mem_alloc();
        _kernel_param();

    }    

    void _mem_alloc(){
        h_out = (float *) malloc (numNodes * dim * sizeof(float));  
        std::fill_n(h_out, numNodes * dim, 0.0f); 

        CUDA_CHECK(cudaMalloc((void**)&d_out, numNodes * dim * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out, 0, numNodes * dim * sizeof(float)));
    }

    void _kernel_param(){
        
        warpPerBlock = 4;
        partSize = 16;
        WARP_SIZE = 32;

       block = warpPerBlock * WARP_SIZE;
       grid = numNodes;
       shared_memory = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(int);
    }

public:
    int numNodes, dim, khop;
    float *h_in, *h_out, *d_in, *d_out;
    int* d_row_ptr, *d_col_ind;

    int warpPerBlock, partSize, WARP_SIZE;
    int block, grid, shared_memory;

    char name[8];
};


#endif // layer_new_cuh