#include <stdio.h>
#include <ctime>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  if (argc < 3){
    printf("./exe ndevices size\n");
    exit(-1);
  }
  //managing 4 devices
  int nDev = atoi(argv[1]);
  int size = atoi(argv[2]);
  int *devs = new int[nDev];
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);
  // ncclComm_t comms[4];

  for (int i = 0; i < nDev; i++)
      devs[i] = i;

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);
  
  std::clock_t dense_start = std::clock();

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    ncclSend(sendbuff[i], size, ncclFloat, (i+1)%nDev, comms[i], s[i]);
    ncclRecv(recvbuff[i], size, ncclFloat, (i-1+nDev)%nDev, comms[i], s[i]);
    // NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], 
    //                         size, ncclFloat, ncclSum, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());


  std::clock_t dense_end = std::clock();
  float dense_time_elapsed_ms = 1000.0 * (dense_end - dense_start) / CLOCKS_PER_SEC;
  printf("CPU-Wall (ms): %.3f\n", dense_time_elapsed_ms);

  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // printf("kernel time (ms): %.3f\n", milliseconds);

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}