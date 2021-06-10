
#include <iostream>
#include <stdio.h>
#include <ctime>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "layer.h"

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 7){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist\n");
        return -1;
    }

    // cout << "\n\n=====================\n";
    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;
    float dense_time_elapsed_ms = 0.0f;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();
    int dimWorker = 32;
    
    int num_GPUs = atoi(argv[2]);           // 2
    int partSize = atoi(argv[3]);           // 32
    int warpPerBlock = atoi(argv[4]);       // 4
    int dim = atoi(argv[5]);                // 16
    int interleaved_dist = atoi(argv[6]);   // 2

    int hiddenSize = dim;
    int outputClass = 128; // 10 by default.

    // print_array<int>("asym.row_ptr", asym.row_ptr, asym.row_ptr.size());
    // print_array<int>("asym.col_ind", asym.col_ind, asym.col_ind.size());

    //
    // create NVSHMEM common world.
    //
    cudaStream_t stream;

    int rank, nranks;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    int lb = nodesPerPE * mype_node;
    int ub = (lb+nodesPerPE) < numNodes? (lb+nodesPerPE):numNodes;

    auto split_output = split_CSR<int>(asym.row_ptr, asym.col_ind, lb, ub);
    // printf("lb: %d, ub: %d\n", lb, ub);

    auto local_ptr = split_output[0]; // with the base start from lb.
    auto remote_ptr = split_output[1]; // with the base start from ub.
    auto local_col_idx = split_output[2];
    auto remote_col_idx = split_output[3];

    // print_array<int>("local_ptr", local_ptr, local_ptr.size());
    // print_array<int>("remote_ptr", remote_ptr, remote_ptr.size());
    // print_array<int>("local_col_idx", local_col_idx, local_col_idx.size());
    // print_array<int>("remote_col_idx", remote_col_idx, remote_col_idx.size());

    // return 0;
    // len: ub - lb, base: lb.
    auto local_info = build_part<int>("PE-" + std::to_string(mype_node) + "-local", local_ptr, ub - lb, partSize);
    auto remote_info = build_part<int>("PE-" + std::to_string(mype_node) + "-remote", remote_ptr, ub - lb, partSize);
    // return 0;
    
    auto local_partPtr = local_info[0];
    auto local_part2Node = local_info[1];

    auto remote_partPtr = remote_info[0];
    auto remote_part2Node = remote_info[1];

    // if (mype_node == 0)
    // for (int i = 0; i < remote_part2Node.size(); i++){
        // printf("remote_part: %d -- %d\n", i, remote_part2Node[i]);
    // }

    float *d_output, *d_input;
    gpuErrchk(cudaMalloc((void**)&d_output, (ub-lb)*dim*sizeof(float))); 
    // gpuErrchk(cudaMalloc((void**)&d_input, (ub-lb)*dim*sizeof(float))); 
    d_input = (float *) nvshmem_malloc ((ub-lb)*dim*sizeof(float)); // shared node embedding part on each GPU

    int *d_row_ptr_local, *d_col_ind_local, *d_part_ptr_local, *d_part2Node_local;
    int *d_row_ptr_remote, *d_col_ind_remote, *d_part_ptr_remote, *d_part2Node_remote;

    gpuErrchk(cudaMalloc((void**)&d_row_ptr_local, local_ptr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_local, local_col_idx.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part_ptr_local, local_partPtr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part2Node_local, local_part2Node.size()*sizeof(int))); 

    gpuErrchk(cudaMalloc((void**)&d_row_ptr_remote, remote_ptr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_remote, remote_col_idx.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part_ptr_remote, remote_partPtr.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part2Node_remote, remote_part2Node.size()*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_row_ptr_local, &local_ptr[0], local_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_local, &local_col_idx[0], local_col_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part_ptr_local, &local_partPtr[0], local_partPtr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part2Node_local, &local_part2Node[0], local_part2Node.size()*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_row_ptr_remote, &remote_ptr[0], remote_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_remote, &remote_col_idx[0], remote_col_idx.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part_ptr_remote, &remote_partPtr[0], remote_partPtr.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part2Node_remote, &remote_part2Node[0], remote_part2Node.size()*sizeof(int), cudaMemcpyHostToDevice));

    // for sync gradient.
    float* sendbuff, *recvbuff;
    gpuErrchk(cudaMalloc((void**)&sendbuff, nodesPerPE*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&recvbuff, nodesPerPE*sizeof(float)));

    int block = 1024;
    int grid = (nodesPerPE + block - 1) / block;
    init_float_array<<<grid, block>>>(sendbuff, 1.0f, nodesPerPE);
    init_float_array<<<grid, block>>>(recvbuff, 0.0f, nodesPerPE);
    
    // define the input.
    Blob<float> *output = new Blob<float>(nodesPerPE, hiddenSize);
    Blob<float> *target = new Blob<float>(nodesPerPE);
    Blob<float> *gradient = target;
    float learning_rate = 0.02f;

    // build the dense layers.
    Layer* dense1 = new Dense("linear1", hiddenSize);
    Layer* act1 = new Activation("relu1", CUDNN_ACTIVATION_RELU);
    Layer* dense2 = new Dense("linear2", outputClass);
    Layer* softmax = new Softmax("softmax");

    // set the GPU cuBLAS/cuDNN context.
    CudaContext *cuda_ = new CudaContext();
    dense1->set_cuda_context(cuda_);
    dense2->set_cuda_context(cuda_);
    act1->set_cuda_context(cuda_);
    softmax->set_cuda_context(cuda_);

    //
    // 
    //  For the 1-layer end-to-end training.
    // 
    std::clock_t c_start = std::clock();
    
    double t1, t2; 
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    t1 = MPI_Wtime(); 

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    // Forward computation.
    SAG_host_fused_interleaved<int, float, int>(d_output, d_input,
                                                // local access param.
                                                d_row_ptr_local, d_col_ind_local,
                                                d_part_ptr_local, d_part2Node_local,
                                                local_partPtr.size()-1, 
                                                // remote access param.
                                                d_row_ptr_remote, d_col_ind_remote,
                                                d_part_ptr_remote, d_part2Node_remote,
                                                remote_partPtr.size()-1,
                                                // other param.
                                                local_ptr.size(),
                                                // nodesPerPE,
                                                lb, partSize, dim, 
                                                dimWorker, warpPerBlock, 
                                                interleaved_dist);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // SAG_host_fused<int, float, int>(d_output, d_input,
    //                                 // local access param.
    //                                 d_row_ptr_local, d_col_ind_local,
    //                                 d_part_ptr_local, d_part2Node_local,
    //                                 local_partPtr.size()-1, 
    //                                 // remote access param.
    //                                 d_row_ptr_remote, d_col_ind_remote,
    //                                 d_part_ptr_remote, d_part2Node_remote,
    //                                 remote_partPtr.size()-1,
    //                                 // other param.
    //                                 local_ptr.size(),
    //                                 // nodesPerPE,
    //                                 lb, partSize, dim, 
    //                                 dimWorker, warpPerBlock);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Time (ms): %.2f\n", milliseconds);

    // #define dense_layer 
    #ifdef dense_layer
    std::clock_t dense_start = std::clock();

    output->to(cuda);
    gradient->to(cuda);

    // forward for output computation.
    // if (mype_node == 0) printf("--*-- Forward...--*--\n");
    output = dense1->forward(output);
    output = act1->forward(output);
    output = dense2->forward(output);
    // output = softmax->forward(output);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // output->print("Output");

    // backward for graiden
    // if (mype_node == 0) printf("--*-- Backward... --*--\n");
    // Reduce all graident across fiferent device.
    nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, recvbuff, sendbuff, nodesPerPE);
    
    // #define BP
    #ifdef BP
    gradient = softmax->backward(gradient);
    gradient = dense2->backward(gradient);
    gradient = act1->backward(gradient);
    gradient = dense1->backward(gradient);

    // weight update.
    // if (mype_node == 0) printf("--*-- Weight Update --*--\n");
    dense1->update_weights_biases(learning_rate);
    act1->update_weights_biases(learning_rate);
    dense2->update_weights_biases(learning_rate);
    // softmax->update_weights_biases(learning_rate);
    #endif

    std::clock_t dense_end = std::clock();
    dense_time_elapsed_ms = 1000.0 * (dense_end - dense_start) / CLOCKS_PER_SEC;

    // backward graident for Sparse operation.
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // Gradient backward.
    SAG_host_fused_interleaved<int, float, int>(d_output, d_input,
                                                // local access param.
                                                d_row_ptr_local, d_col_ind_local,
                                                d_part_ptr_local, d_part2Node_local,
                                                local_partPtr.size()-1, 
                                                // remote access param.
                                                d_row_ptr_remote, d_col_ind_remote,
                                                d_part_ptr_remote, d_part2Node_remote,
                                                remote_partPtr.size()-1,
                                                // other param.
                                                local_ptr.size(),
                                                // nodesPerPE,
                                                lb, partSize, dim, 
                                                dimWorker, warpPerBlock, 
                                                interleaved_dist);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("[Backward] Sparse Time (ms): %.2f\n", milliseconds);

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("PE-%d, Total (ms): %.3f, Dense (ms): %.3f\n", mype_node, time_elapsed_ms, dense_time_elapsed_ms);
    // std::cout << "CPU time %: " << time_elapsed_ms << " ms\n";

    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */

    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3); 

    // if (mype_node == 0) printf("END forward and backward\n");

    // release nvshmem objects and finalize context.
    cudaFree(d_output);
    cudaFree(d_row_ptr_local);
    cudaFree(d_col_ind_local);
    cudaFree(d_part_ptr_local);
    cudaFree(d_part2Node_local);

    cudaFree(d_row_ptr_remote);
    cudaFree(d_col_ind_remote);
    cudaFree(d_part_ptr_remote);
    cudaFree(d_part2Node_remote);

    nvshmem_free(d_input);
    nvshmem_finalize();
    MPI_Finalize();
    // printf("--*-- PEID: %d, End NVSHMEM --*--\n", mype_node);
    if (mype_node == 0) printf("===================================\n");

    return 0;
}