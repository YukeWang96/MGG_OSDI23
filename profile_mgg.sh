# Usage: ./main graph.mtx num_GPUs dim nodeOfInterest
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMPI_MCA_plm_rsh_agent=sh \
        mpirun --allow-run-as-root -np 2 build/mgg_profile dataset/ppi.mtx 2 16 1000