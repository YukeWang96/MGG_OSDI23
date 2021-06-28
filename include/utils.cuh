
#ifndef UTIL
#define UTIL 

#include <cuda.h>
#include <vector>
#include <iostream>
#include <cublas_v2.h>

using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define min_val(a,b) ((a)<(b)?(a):(b))
#define max_val(a,b) ((a)>(b)?(a):(b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define cuBLASErrchk(ans) { cuBLASAssert((ans), __FILE__, __LINE__); }
inline void cuBLASAssert(cublasStatus_t status, const char *file, int line, bool abort=true)
{
      if(status !=CUBLAS_STATUS_SUCCESS)
      {
          fprintf(stderr, "Error in cublasGemmEx()\n");
          std::string error_msg;
          switch(status)
          {
              case CUBLAS_STATUS_SUCCESS:          error_msg.assign("CUBLAS_STATUS_SUCCESS");
              case CUBLAS_STATUS_NOT_INITIALIZED:  error_msg.assign( "CUBLAS_STATUS_NOT_INITIALIZED");
              case CUBLAS_STATUS_NOT_SUPPORTED:    error_msg.assign( "CUBLAS_STATUS_NOT_SUPPORTED");
              case CUBLAS_STATUS_INVALID_VALUE:    error_msg.assign( "CUBLAS_STATUS_INVALID_VALUE"); 
              case CUBLAS_STATUS_ARCH_MISMATCH:    error_msg.assign( "CUBLAS_STATUS_ARCH_MISMATCH"); 
              case CUBLAS_STATUS_MAPPING_ERROR:    error_msg.assign( "CUBLAS_STATUS_MAPPING_ERROR");
              case CUBLAS_STATUS_EXECUTION_FAILED: error_msg.assign( "CUBLAS_STATUS_EXECUTION_FAILED"); 
              case CUBLAS_STATUS_INTERNAL_ERROR:   error_msg.assign( "CUBLAS_STATUS_INTERNAL_ERROR"); 
              default:                             error_msg.assign( "unknown error");
          }
          std::cout << error_msg << std::endl;
         if (abort) exit(status);
      }
}

__global__ 
void init_float_array(float* arr, float val, int len){
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   if (tid < len){
      arr[tid] = val;
   }
}


__global__ 
void print_float_array(float* arr, int my_pe, int len=10){
   printf("[%d] ", my_pe);
   for (int i = 0; i < len; i++){
      printf("%.3f ", arr[i]);
   }
   printf("\n");
}
// reference implementation for the most basic scatter and gather kernel.
void SAG_ref(float *update, float *input, \
               int *node_ptr, int *column_index, \
               int numNode, int numEdges, int ebdDim)
{

   for (int nid = 0; nid < numNode; nid++){
      int eidx_start = node_ptr[nid];
      int eidx_end = node_ptr[nid + 1];
      for (int eidx = eidx_start; eidx < eidx_end; eidx++){
         int eid = column_index[eidx];
         for (int d = 0; d < ebdDim; d++){
            update[nid*ebdDim + d] += input[eid*ebdDim + d];
         }
      }
   }
}

// fill all origin node embdding with all ones
void init_origin_node(float *origin_node, int ebdDim, int numNodes, float init_val){
   for (int i = 0; i < numNodes; i++){
      for (int d = 0; d < ebdDim; d++){
         origin_node[i*ebdDim + d] = init_val;
      }
   }
}

// print all origin node embdding with all ones
void print_update_node(float *update_node, int ebdDim, int start, int end){
   for (int i = start; i < end; i++){
      printf("[%d]", i);
      for (int d = 0; d < ebdDim; d++){
         printf("%.2f ", update_node[i*ebdDim + d]);
      }
      printf("\n");
   }
}

template <typename T>
void print_array(string name, std::vector<T>& array, int len=10){
    cout << name << ":\t";
    for (int i = 0; i < len; i++){
        if (typeid(T) == typeid(int))
            printf("%d ", array[i]);
        else
            printf("%.3f ", array[i]);
    }
    cout << endl;
}

#endif