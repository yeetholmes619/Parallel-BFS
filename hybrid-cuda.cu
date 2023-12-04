#include "cuda_runtime.h"
#include<iomanip>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<chrono>
#include <math.h>
#include<limits.h>
#define NUM_NODES 50'000'005
#define NUM_EDGES 300'000'005

using namespace std;
typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

Node node[NUM_NODES];
int edges[NUM_EDGES];
bool curr_frontier[NUM_NODES] = { false }, next_frontier[NUM_NODES] = { false };
bool visited[NUM_NODES] = { false };
int source = 1;
int num_nodes, num_edges;
int degree[NUM_NODES];
const int alpha = 14;
const int beta = 24;

//pointers
int *num_nodes_ptr;
Node* Va;
int* Ea;
bool* Cf;
bool* Nf;
bool* Xa;
bool* done;
unsigned int *nf_ptr;
unsigned int *mf_ptr;
unsigned int *m_unvisited_ptr;
int* degree_ptr;
int* num_edges_ptr;
int n_visited = 0;

//Xa -> Visited
//Ea -> CSR edges
//Va -> CSR start and lengths
//Fa -> Frontier


__device__ bool valid_idx(int idx,int *num_nodes_ptr){
    return idx < *num_nodes_ptr;
}

__device__ void next_true(int idx,bool* Nf,unsigned int* nf_ptr,unsigned int* mf_ptr,int* degree_ptr){
	Nf[idx] = true;
	atomicInc(nf_ptr,INT_MAX);
	atomicAdd(mf_ptr,degree_ptr[idx]);

}

__device__ void current_false(int idx,unsigned int* nf_ptr,unsigned int* mf_ptr,int* degree_ptr,bool* Cf){
	Cf[idx] = false;
	atomicDec(nf_ptr,INT_MAX);
	atomicSub(mf_ptr,degree_ptr[idx]);
}

__device__ void visit(int idx,unsigned int*m_unvisited_ptr,int* degree_ptr,bool* Xa){
	Xa[idx] = true;
	atomicSub(m_unvisited_ptr,degree_ptr[idx]);
}
	
__global__ void TOPDOWN_BFS_KERNEL(Node *Va, int *Ea, bool *Cf, bool *Nf, bool *Xa, bool *done,unsigned int*nf_ptr,unsigned int*mf_ptr,unsigned int* m_unvisited_ptr,int*num_nodes_ptr,int* degree_ptr)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;


	if ( valid_idx(id,num_nodes_ptr) && Cf[id] == true )
	{
		//printf("%d ", id); //This printf gives the order of vertices in BFS	
		visit(id,m_unvisited_ptr,degree_ptr,Xa);
		current_false(id,nf_ptr,mf_ptr,degree_ptr,Cf);
		int start = Va[id].start;
		int end = start + Va[id].length;
		for (int i = start; i < end; i++) 
		{
			int nid = Ea[i];

			if (Cf[nid] == false and Xa[nid] == false)
			{
				next_true(nid,Nf,nf_ptr,mf_ptr,degree_ptr);
				*done = false;
			}

		}

	}



}

__global__ void BOTTOMUP_BFS_KERNEL(Node *Va, int *Ea, bool *Cf, bool *Nf, bool *Xa, bool *done,unsigned int*nf_ptr,unsigned int*mf_ptr,unsigned int*
		 m_unvisited_ptr,int*num_nodes_ptr,int* degree_ptr)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;


	if ( valid_idx(id,num_nodes_ptr) && Xa[id] == false)
	{
		if(Cf[id] == true){
			//printf("%d ", id); //This printf gives the order of vertices in BFS	
			visit(id,m_unvisited_ptr,degree_ptr,Xa);
		}else{
			int start = Va[id].start;
			int end = start + Va[id].length;
			for (int i = start; i < end; i++) 
			{
				int nid = Ea[i];

				if (Cf[nid] == true)
				{
					next_true(id,Nf,nf_ptr,mf_ptr,degree_ptr);
					*done = false;
					break;
				}

			}
		}
	}

	__syncthreads();
    if(valid_idx(id,num_nodes_ptr))
	    current_false(id,nf_ptr,mf_ptr,degree_ptr,Cf);

}




int num_blks;
int threads;
int state; //0 -> TOPDOWN, 1 -> BOTTOMUP
//TODO : account for increase or decrease in nodes and edges in frontier
__global__ void swapPointers(bool** a, bool** b){
	bool* temp = *a;
	*a = *b;
	*b = temp;
}

int n_top_down = 0;
int n_bottom_up = 0;

void BFS_BOILERPLATE()
{  


    	auto start_pp = std::chrono::high_resolution_clock::now();
	int nf = 1;
	int mf = degree[source];
	int m_unvisited = num_edges;
	bool done_val = true;
	curr_frontier[source] = true;

	cudaMalloc((void**)&num_nodes_ptr, sizeof(int));
	cudaMemcpy(num_nodes_ptr, &num_nodes, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&nf_ptr, sizeof(unsigned int));
	cudaMemcpy(nf_ptr,&nf,sizeof(unsigned int),cudaMemcpyHostToDevice);

        cudaMalloc((void**)&mf_ptr, sizeof(unsigned int));
	cudaMemcpy(mf_ptr,&mf,sizeof(unsigned int),cudaMemcpyHostToDevice);

        cudaMalloc((void**)&m_unvisited_ptr, sizeof(unsigned int));
	cudaMemcpy(m_unvisited_ptr,&num_edges,sizeof(unsigned int),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&num_edges_ptr,sizeof(unsigned int));
	cudaMemcpy(num_edges_ptr,&num_edges,sizeof(unsigned int),cudaMemcpyHostToDevice);

        cudaMalloc((void**)&degree_ptr, sizeof(int)*num_nodes);
        cudaMemcpy(degree_ptr, degree, sizeof(int)*num_nodes, cudaMemcpyHostToDevice);


        cudaMalloc((void**)&Va, sizeof(Node)*num_nodes);
        cudaMemcpy(Va, node, sizeof(Node)*num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&Ea, sizeof(int)*num_edges);
        cudaMemcpy(Ea, edges, sizeof(int)*num_edges, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&Cf, sizeof(bool)*num_nodes);
        cudaMemcpy(Cf, curr_frontier, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice);


        cudaMalloc((void**)&Nf, sizeof(bool)*num_nodes);
        cudaMemcpy(Nf, next_frontier, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&Xa, sizeof(bool)*num_nodes);
        cudaMemcpy(Xa, visited, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&done, sizeof(bool));


    	auto end_pp = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double, std::nano> duration_pp = end_pp - start_pp;
    	auto start_bfs = std::chrono::high_resolution_clock::now();
	do{
		done_val =true;
		cudaMemcpy(done,&done_val,sizeof(bool),cudaMemcpyHostToDevice);
		if(state == 0 and mf > m_unvisited/alpha){
			//printf("Going from top-down to bottom-up\n");
			state = 1;
		}
		else if(state == 1 and nf < num_nodes/beta){
			//printf("Going from bottom-up to top-down\n");
			state = 0;
		}

		if(state == 0){
			n_top_down++;	
			TOPDOWN_BFS_KERNEL<<<num_blks, threads>>>(Va, Ea,  Cf, Nf,  Xa,  done,nf_ptr,mf_ptr,m_unvisited_ptr,num_nodes_ptr, degree_ptr);
		}
		else{
			n_bottom_up++;	
			BOTTOMUP_BFS_KERNEL<<<num_blks, threads>>>(Va, Ea,  Cf, Nf,  Xa,  done,nf_ptr,mf_ptr,m_unvisited_ptr,num_nodes_ptr, degree_ptr);
		}

		cudaMemcpy(&done_val, done, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&mf, mf_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&nf, nf_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&m_unvisited, m_unvisited_ptr, sizeof(int), cudaMemcpyDeviceToHost);
		
		//swapping current and next frontiers..
		bool *tmp = Cf;
		Cf = Nf;
		Nf = tmp;
	}while(!done_val);
    	auto end_bfs = std::chrono::high_resolution_clock::now();
    	std::chrono::duration<double, std::nano> duration_bfs = end_bfs - start_bfs;
	cout<<fixed<<setprecision(12);
	cout<<num_nodes<<" "<<num_edges<<" "<<duration_pp.count()<<" "<<duration_bfs.count()<<"\n";

        //cudaMemcpy(visited, Xa, sizeof(bool)*num_nodes, cudaMemcpyDeviceToHost);
	//for(int i =0 ; i < num_nodes;i++) n_visited += visited[i];

	//TODO: write free for all pointers
        cudaFree(Va);
        cudaFree(Ea);
        cudaFree(Cf);
        cudaFree(Nf);
        cudaFree(Xa);
        cudaFree(done);
	cudaFree(nf_ptr);
	cudaFree(mf_ptr);
	cudaFree(m_unvisited_ptr);
	cudaFree(num_edges_ptr);
	cudaFree(degree_ptr);

}

// The BFS frontier corresponds to all the nodes being processed at the current level.


int main()
{
    cin>>num_nodes>>num_edges;
    //num_blks,threads
    threads = min((1<<10), num_nodes);
    num_blks = (num_nodes + threads-1)/threads;

    for(int i = 0; i < num_nodes; i++){
            cin>>node[i].start>>node[i].length;
            degree[i] = node[i].length;
    }

    for(int i = 0; i < num_edges; i++){
            cin>>edges[i];
    }



	BFS_BOILERPLATE();



}

