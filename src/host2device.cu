
#include <iostream>
#include <stdio.h>
#include <ctime>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"

using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 3){
        printf("Usage: ./main mtx_graph_file_path num_GPUS\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();
    // printf("numNodes: %d, numEdges: %d\n", numNodes, numEdges);

    int num_GPUs = atoi(argv[2]);
    int partSize = 3;
    int dim = atoi(argv[3]);
    int dimWorker = 32;
    int warpPerBlock = 2;
    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float* input = (float*)malloc(numNodes*dim*sizeof(float));

    float *d_output, *d_input;
    int *d_col_ind, *d_part_ptr, *d_part2Node;

    // create NVSHMEM common world.
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

    // printf("Before build the part\n");
    // Build the partitions for the entire graph.
    auto global_part_info = build_part<int>("whole graph", asym.row_ptr, numNodes, partSize);
    // printf("After build the part\n");

    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);
    const int edge_lb = asym.row_ptr[lb_src];
    const int edge_ub = asym.row_ptr[ub_src];
    const int num_edges_range = edge_ub - edge_lb;

    auto partPtr = global_part_info[0];
    auto part2Node = global_part_info[1];
    auto node2Part = global_part_info[2];

    const int part_lb = node2Part[lb_src];
    const int part_ub = node2Part[ub_src];

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    const int local_nodes = ub_src - lb_src;
    const int num_parts_range = part_ub - part_lb;

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    gpuErrchk(cudaMalloc((void**)&d_output, (ub_src-lb_src)*dim*sizeof(float))); 

    gpuErrchk(cudaMalloc((void**)&d_col_ind, num_edges_range*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part_ptr, (num_parts_range+1)*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part2Node, num_parts_range*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[edge_lb], num_edges_range*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part_ptr, &partPtr[part_lb], num_parts_range*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part2Node, &part2Node[part_lb], num_parts_range*sizeof(int), cudaMemcpyHostToDevice));
    std::clock_t dense_start, dense_end, total = 0; 

//
// iterate through all partitions
//
for (int part_iter = 0; part_iter < num_GPUs; part_iter++){

    // Iterative different range of partition based on [lb, ub).
    int lb = nodesPerPE * part_iter;
    int ub = min_val(lb+nodesPerPE, numNodes);

    // Load the corresponding tiles.
    // printf("[%d], ub: %d, lb: %d\n", part_iter, ub, lb);
    gpuErrchk(cudaMalloc((void**)&d_input, (ub-lb)*dim*sizeof(float))); 
    gpuErrchk(cudaMemcpy(d_input, input, (ub - lb)*dim*sizeof(float), cudaMemcpyHostToDevice));

    dense_start = std::clock();
    // Aggregation on the local tile.
    SAG_host_range_only<int, float, int>(d_output, 
                                        d_input,
                                        d_col_ind,
                                        d_part_ptr, 
                                        d_part2Node,
                                        num_parts_range, 
                                        lb, 
                                        ub, 
                                        edge_lb,
                                        local_nodes,
                                        // other param.
                                        partSize, 
                                        dim, 
                                        dimWorker, 
                                        warpPerBlock);
    
    cudaDeviceSynchronize();
    dense_end = std::clock();
    total += dense_end - dense_start;

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at SAG_host_range_only: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaFree(d_input);
} // END FOR (int part_iter = 0; i < part_iter; i++)

    float dense_time_elapsed_ms = 1000.0 * total / CLOCKS_PER_SEC;
    printf("PE-%d, CPU-Wall (ms): %.3f\n", mype_node, dense_time_elapsed_ms);

    // release nvshmem objects and finalize context.
    cudaFree(d_output);
    cudaFree(d_col_ind);
    cudaFree(d_part_ptr);
    cudaFree(d_part2Node);

    nvshmem_finalize(); \
    printf("--*-- PEID: %d, End NVSHMEM --*--\n", mype_node); 
    MPI_Finalize();

    return 0;
}