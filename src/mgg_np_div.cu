
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cublas_v2.h>

#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "layer.h"

// #define validate 1 // the number (< num_GPUs) indicates the validation on which PE.

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
	CSR asym = assemble_csr_matrix_new(argv[1]);
    cout << "Complete loading graphs !!" << endl;

    int numNodes = asym.row_ptr.size() - 1;
    // std::cout << "max node: " << *std::max_element(std::begin(asym.col_ind), std::end(asym.col_ind)) << '\n';
    int numEdges = asym.col_ind.size();    
    int num_GPUs = atoi(argv[2]);           // 2
    int partSize = atoi(argv[3]);           // 32
    int warpPerBlock = atoi(argv[4]);       // 4
    int dim = atoi(argv[5]);                // 16
    int interleaved_dist = atoi(argv[6]);   // 2
    int hiddenSize = atoi(argv[7]);

    double t1, t2; 
    // print_array<int>("asym.row_ptr", asym.row_ptr, asym.row_ptr.size());
    // print_array<int>("asym.col_ind", asym.col_ind, asym.col_ind.size());
    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // Set the workload on each device.
    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    // printf("numNodes: %d, nodesPerPE: %d\n", numNodes, nodesPerPE);
    int lb = nodesPerPE * mype_node;
    int ub = (lb + nodesPerPE) < numNodes? (lb + nodesPerPE) : numNodes;
    int local_edges = asym.row_ptr[ub] - asym.row_ptr[lb];
    int edge_beg = asym.row_ptr[lb];

    // Divide the CSR into the local and remote for each GPU.
    auto split_output = split_CSR<int>(asym.row_ptr, asym.col_ind, lb, ub);
    // printf("lb: %d, ub: %d\n", lb, ub);
    auto local_ptr_vec = split_output[0];       // with the base start from lb.
    auto remote_ptr_vec = split_output[1];      // with the base start from ub.
    auto local_col_idx_vec = split_output[2];
    auto remote_col_idx_vec = split_output[3];

    printf("local_col_idx_vec = %d, remote_col_idx_vec = %d, local_edges = %d\n",
            local_col_idx_vec.size(), remote_col_idx_vec.size(), local_edges);
    // exit(0);

    // Allocate memory on each device.
    float *d_input, *d_output, *h_input, *h_output;
    gpuErrchk(cudaMalloc((void**)&d_output, nodesPerPE * dim * sizeof(float))); 
    d_input = (float *) nvshmem_malloc (nodesPerPE * dim * sizeof(float));  // NVSHMEM global memory for input embedding.
    h_input = (float *) malloc (nodesPerPE * dim * sizeof(float));          // CPU host memory (input)
    h_output = (float *) malloc (nodesPerPE * dim * sizeof(float));         //  CPU host memory (output)
    std::fill_n(h_input, nodesPerPE*dim, 1.0f); // filled with all ones for input embeddings.
    std::fill_n(h_output, nodesPerPE*dim, 0.0f); // filled with all zeros for output embeddings.
    gpuErrchk(cudaMemcpy(d_input, h_input, nodesPerPE * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_output, h_output, nodesPerPE * dim * sizeof(float), cudaMemcpyHostToDevice));

    #ifdef validate
    float *h_input_ref, *h_output_ref,  *d_input_ref, *d_output_ref;
    if (mype_node == validate)
    {
        h_input_ref = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_output_ref = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_input_ref, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_output_ref, numNodes * dim, 0.0f); // filled with all zeros.
        gpuErrchk(cudaMalloc((void**)&d_input_ref, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        gpuErrchk(cudaMalloc((void**)&d_output_ref, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    }
    #endif

    int *d_row_ptr_l, *d_col_ind_l,  *d_row_ptr_r, *d_col_ind_r;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_l, local_ptr_vec.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_l, local_col_idx_vec.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_row_ptr_r, remote_ptr_vec.size()*sizeof(int))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind_r, remote_col_idx_vec.size()*sizeof(int))); 

    gpuErrchk(cudaMemcpy(d_row_ptr_l, &local_ptr_vec[0], local_ptr_vec.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_l, &local_col_idx_vec[0], local_col_idx_vec.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_ptr_r, &remote_ptr_vec[0], remote_ptr_vec.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind_r, &remote_col_idx_vec[0], remote_col_idx_vec.size()*sizeof(int), cudaMemcpyHostToDevice));

    #ifdef validate
    int* d_row_ptr_ref, *d_col_ind_ref;
    if (mype_node == validate)
    {
        gpuErrchk(cudaMalloc((void**)&d_row_ptr_ref, asym.row_ptr.size()*sizeof(int))); 
        gpuErrchk(cudaMalloc((void**)&d_col_ind_ref, asym.col_ind.size()*sizeof(int))); 
        gpuErrchk(cudaMemcpy(d_row_ptr_ref, &asym.row_ptr[0], asym.row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_col_ind_ref, &asym.col_ind[0], asym.col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_input_ref, h_input_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_output_ref, h_output_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        //
        // Compute the result [lb, ub] based on the whole graph CSR.
        //
        SAG_host_ref(d_output_ref, d_input_ref, 
                    d_row_ptr_ref, d_col_ind_ref, 
                    lb, ub, dim);

        gpuErrchk(cudaMemcpy(h_output_ref, d_output_ref, numNodes * dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD); 

    //
    // Compute on each GPU device.
    //
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    mgg_SAG_np_div(d_output, d_input, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
                    lb, ub, dim, nodesPerPE);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3); 
    
    gpuErrchk(cudaMemcpy(h_output, d_output, nodesPerPE*dim*sizeof(float), cudaMemcpyDeviceToHost));

    #ifdef validate
    if (mype_node == validate){
        for (int nid = 0; nid < 10; nid++){
            printf("out [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output[nid * dim + d]);
            }
            printf("\n");
        }
        printf("==============================\n");
        for (int nid = 0; nid < 10; nid++){
            printf("ref [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output_ref[lb * dim + nid * dim + d]);
            }
            printf("\n");
        }
        bool val_status = check_equal(h_output_ref, h_output, (ub - lb) * dim, dim, lb * dim);
        printf("Validation on PE-{%d}, status: ", validate);
        if (val_status) printf("True\n"); else printf("False\n");
    }
    #endif

    // release memory.
    cudaFree(d_output);
    cudaFree(d_row_ptr_l);
    cudaFree(d_col_ind_l);
    cudaFree(d_row_ptr_r);
    cudaFree(d_col_ind_r);
    nvshmem_free(d_input);
    nvshmem_finalize();

    free(h_input);
    free(h_output);

    #ifdef validate
    cudaFree(d_output_ref);
    free(h_output_ref);
    #endif

    MPI_Finalize();
    
    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}