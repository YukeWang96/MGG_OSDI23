
#include <iostream>
#include <stdio.h>
#include <omp.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"

// #define validate //--> for results validation
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
    // int dimWorker = 32;

    int num_GPUs = atoi(argv[2]);
    int partSize = atoi(argv[3]);
    int warpPerBlock = atoi(argv[4]);
    int dim = atoi(argv[5]);

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float* h_input = (float*)malloc(numNodes*dim*sizeof(float));
    float* h_output = (float*)malloc(numNodes*dim*sizeof(float));
    
    std::fill(h_input, h_input+numNodes*dim, 1.0); // sets every value in the array to 1.0
    std::fill(h_input, h_input+numNodes*dim, 0.0); // sets every value in the array to 0.0

    // memset(input, 0, numNodes*dim*sizeof(float));
    float *d_output, *d_input;
    int *d_row_ptr, *d_col_ind;
    
    #ifdef validate
    float *d_ref;
    #endif
    
    // int *d_col_ind, *d_part_ptr, *d_part2Node;
    // Build the partitions.
    // auto global_part_info = build_part<int>("Overall Graph", asym.row_ptr, numNodes, partSize);
    // auto partPtr = global_part_info[0];
    // auto part2Node = global_part_info[1];
    // auto node2Part = global_part_info[2];

    // UVM data: output, input, row_ptr, col_ind 
    gpuErrchk(cudaMallocManaged((void**)&d_output,  numNodes*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_input,   numNodes*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_row_ptr, (numNodes+1)*sizeof(int)));
    gpuErrchk(cudaMallocManaged((void**)&d_col_ind, numEdges*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_input,   h_output,           numNodes*dim*sizeof(float),   cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input,   h_input,            numNodes*dim*sizeof(float),   cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr, &asym.row_ptr[0],   (numNodes+1)*sizeof(int),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[0],   numEdges*sizeof(int),         cudaMemcpyHostToDevice));

// individual GPUs
#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{

    cudaSetDevice(mype_node);

    // Load the corresponding tiles.
    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);
    // const int row_ptr_range = ub_src - lb_src + 1;

    // const int edge_lb = asym.row_ptr[lb_src];
    // const int edge_ub = asym.row_ptr[ub_src];
    // const int num_edges_range = edge_ub - edge_lb;

    // const int part_lb = node2Part[lb_src];
    // const int part_ub = node2Part[ub_src];

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    // const int num_parts_range = part_ub - part_lb;

    // printf("part_ub: %d, part_lb: %d\n", part_ub, part_lb);
    // gpuErrchk(cudaMalloc((void**)&d_output, (ub_src-lb_src)*dim*sizeof(float))); 
    // #ifdef validate
    // gpuErrchk(cudaMalloc((void**)&d_ref, (ub_src-lb_src)*dim*sizeof(float))); 
    // #endif

    // gpuErrchk(cudaMalloc((void**)&d_col_ind, num_edges_range*sizeof(int))); 
    // gpuErrchk(cudaMalloc((void**)&d_part_ptr, (num_parts_range+1)*sizeof(int))); 
    // gpuErrchk(cudaMalloc((void**)&d_part2Node, num_parts_range*sizeof(int))); 


    // gpuErrchk(cudaMemcpy(d_row_ptr, &asym.row_ptr[lb_src], row_ptr_range*sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_col_ind, &asym.col_ind[edge_lb], num_edges_range*sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_part_ptr, &partPtr[part_lb], (num_parts_range + 1)*sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_part2Node, &part2Node[part_lb], num_parts_range*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // #ifdef validate
    SAG_host_ref(d_output, d_input, d_row_ptr, d_col_ind, lb_src, ub_src, dim);
    // #endif

    // // Aggregation on the local tile.
    // SAG_host_unified<int, float, int>(d_output, 
    //                                     d_input,
    //                                     d_col_ind,
    //                                     d_part_ptr, 
    //                                     d_part2Node,
    //                                     edge_lb, 
    //                                     num_parts_range, 
    //                                     nodesPerPE,
    //                                     // other param.
    //                                     partSize, 
    //                                     dim, 
    //                                     dimWorker, 
    //                                     warpPerBlock);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);
    printf("===================================\n");
}

    cudaFree(d_input);    
    cudaFree(d_output);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);
    // cudaFree(d_part_ptr);
    // cudaFree(d_part2Node);

    return 0;
}