
#include <iostream>
#include <stdio.h>
#include <omp.h>

#include "graph.h"
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
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    
    graph<long, long, int, int, int, int>* ginst = new graph<long, long, int, int, int, int>(beg_file, csr_file, weight_file);
    std::vector<int> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<int> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    int numNodes = global_row_ptr.size() - 1;
    int numEdges = global_col_ind.size();    

    int num_GPUs = atoi(argv[4]);
    int partSize = atoi(argv[5]);
    int warpPerBlock = atoi(argv[6]);
    int dim = atoi(argv[7]);

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float* h_input = (float*)malloc(numNodes*dim*sizeof(float));
    float* h_output = (float*)malloc(numNodes*dim*sizeof(float));
    float* h_ref = (float*)malloc(numNodes*dim*sizeof(float));

    std::fill(h_input, h_input+numNodes*dim, 1.0);      // sets every value in the array to 1.0
    std::fill(h_output, h_output+numNodes*dim, 0.0);    // sets every value in the array to 0.0
    std::fill(h_ref, h_ref+numNodes*dim, 0.0);          // sets every value in the array to 0.0

    // memset(input, 0, numNodes*dim*sizeof(float));
    float *d_output, *d_input;
    int *d_row_ptr, *d_col_ind;
    float *d_ref;
    
    // int *d_col_ind, *d_part_ptr, *d_part2Node;
    // Build the partitions.
    // auto global_part_info = build_part<int>("Overall Graph", global_row_ptr, numNodes, partSize);
    // auto partPtr = global_part_info[0];
    // auto part2Node = global_part_info[1];
    // auto node2Part = global_part_info[2];

    // UVM data: output, input, row_ptr, col_ind 
    gpuErrchk(cudaMallocManaged((void**)&d_ref,     numNodes*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_output,  numNodes*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_input,   numNodes*dim*sizeof(float))); 
    gpuErrchk(cudaMallocManaged((void**)&d_row_ptr, (numNodes+1)*sizeof(int)));
    gpuErrchk(cudaMallocManaged((void**)&d_col_ind, numEdges*sizeof(int))); 


    cudaMemset(d_ref,       0, numNodes*dim*sizeof(float));
    cudaMemset(d_output,    0, numNodes*dim*sizeof(float));
    gpuErrchk(cudaMemcpy(d_input,   h_input,            numNodes*dim*sizeof(float),   cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr, &global_row_ptr[0],   (numNodes+1)*sizeof(int),     cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[0],   numEdges*sizeof(int),         cudaMemcpyHostToDevice));

    #ifdef validate
    cudaSetDevice(0);
    SAG_host_single_ref(d_ref, d_input, d_row_ptr, d_col_ind, numNodes, dim);
    gpuErrchk(cudaMemcpy(h_ref,     d_ref,       numNodes*dim*sizeof(float),   cudaMemcpyDeviceToHost));
    #endif

    // One GPU per threads
#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{

    cudaSetDevice(mype_node);

    // Load the corresponding tiles.
    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);
    // const int row_ptr_range = ub_src - lb_src + 1;

    // const int edge_lb = global_row_ptr[lb_src];
    // const int edge_ub = global_row_ptr[ub_src];
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

    // gpuErrchk(cudaMemcpy(d_row_ptr, &global_row_ptr[lb_src], row_ptr_range*sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[edge_lb], num_edges_range*sizeof(int), cudaMemcpyHostToDevice));
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
}

    gpuErrchk(cudaMemcpy(h_output,  d_output,    numNodes*dim*sizeof(float),   cudaMemcpyDeviceToHost));

    #ifdef validate
    bool status = compare_array(h_ref, h_output, numNodes*dim);
    printf(status ? "validate: True\n" : "validate: False\n");
    #endif
    printf("===================================\n");

    cudaFree(d_ref);    
    cudaFree(d_input);    
    cudaFree(d_output);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);


    free(h_ref);
    free(h_output);
    free(h_input);
    // delete asym;
    // cudaFree(d_part_ptr);
    // cudaFree(d_part2Node);

    return 0;
}