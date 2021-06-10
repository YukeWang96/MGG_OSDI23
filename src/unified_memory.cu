
#include <iostream>
#include <stdio.h>
#include <omp.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"

using namespace std;


int main(int argc, char* argv[]){
	
    if (argc < 5){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerBlock\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    int numEdges = asym.col_ind.size();
    int dim = 16;
    int dimWorker = 32;

    int num_GPUs = atoi(argv[2]);
    int partSize = atoi(argv[3]);
    int warpPerBlock = atoi(argv[4]);
    
    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float* input = (float*)malloc(numNodes*dim*sizeof(float));
    // memset(input, 0, numNodes*dim*sizeof(float));

    float *d_output, *d_input;
    int *d_col_ind, *d_part_ptr, *d_part2Node;

    // Build the partitions.
    auto global_part_info = build_part<int>("Overall Graph", asym.row_ptr, numNodes, partSize);
    auto partPtr = global_part_info[0];
    auto part2Node = global_part_info[1];
    auto node2Part = global_part_info[2];
    gpuErrchk(cudaMallocManaged((void**)&d_input, numNodes*dim*sizeof(float))); 


// individual GPUs
#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{

    cudaSetDevice(mype_node);

    // Load the corresponding tiles.
    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);

    const int edge_lb = asym.row_ptr[lb_src];
    const int edge_ub = asym.row_ptr[ub_src];
    const int num_edges_range = edge_ub - edge_lb;

    const int part_lb = node2Part[lb_src];
    const int part_ub = node2Part[ub_src];

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    const int num_parts_range = part_ub - part_lb;

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    gpuErrchk(cudaMalloc((void**)&d_output, (ub_src-lb_src)*dim*sizeof(float))); 

    gpuErrchk(cudaMalloc((void**)&d_col_ind, num_edges_range*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part_ptr, (num_parts_range+1)*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_part2Node, num_parts_range*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[edge_lb], num_edges_range*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part_ptr, &partPtr[part_lb], (num_parts_range + 1)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_part2Node, &part2Node[part_lb], num_parts_range*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Aggregation on the local tile.
    SAG_host_unified<int, float, int>(d_output, 
                                        d_input,
                                        d_col_ind,
                                        d_part_ptr, 
                                        d_part2Node,
                                        edge_lb, 
                                        num_parts_range, 
                                        nodesPerPE,
                                        // other param.
                                        partSize, 
                                        dim, 
                                        dimWorker, 
                                        warpPerBlock);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("[%d] CUDA error at SAG_host_unified: %s\n", mype_node, cudaGetErrorString(error));
        exit(-1);
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);
    printf("===================================\n");
}

    cudaFree(d_input);    
    cudaFree(d_output);
    cudaFree(d_col_ind);
    cudaFree(d_part_ptr);
    cudaFree(d_part2Node);

    return 0;
}