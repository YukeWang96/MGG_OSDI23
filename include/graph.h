#ifndef __GRAPH_H__
#define __GRAPH_H__
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "wtime.h"
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
inline off_t fsize(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

template<
typename file_vert_t, typename file_index_t, typename file_weight_t,
typename new_vert_t, typename new_index_t, typename new_weight_t>
class graph
{
	public:
		new_index_t *beg_pos;
		new_vert_t *csr;
//		new_weight_t *weight;
        new_vert_t* weight;
		new_index_t vert_count;
		new_index_t edge_count;

	public:
		graph(){};
		~graph(){};
		graph(const char *beg_file, 
				const char *csr_file,
				const char *weight_file);
};
#include "graph.hpp"
#endif
