#include<omp.h>
//#include<random>
#include<cstdlib>
#include<cmath>
#include<limits.h>
#include<assert.h>
#include<iostream>
#include<chrono>
#include<string.h>
#include<stdio.h>
#include<ctime>

using namespace std;

#define currTime std::chrono::high_resolution_clock::now()
#define tellTime(l, r) std::chrono::duration_cast<std::chrono::milliseconds>(r - l).count()
double acc_time=0.0;


int number_bucalls=0;
int number_tdcalls=0;
const int num_threads = 18;
const int num_nodes = 10'000'001;
const int num_edges = 60'000'001;
const int alpha = 14;
const int beta = 24;
const int total_samples = 10;
int csr[num_edges];
int csr_in[num_edges];
int out_degree[num_nodes];
int in_degree[num_nodes];
int pref_sum[num_nodes];
int pref_sum_in[num_nodes];
int filled_upto_in[num_nodes];
int filled_upto[num_nodes];
int parent[num_nodes];
int local_count[num_nodes];
int edge_list[num_edges][2];
int edge_in[num_edges][2];
int num_frontier_edges = 0;
int num_frontier_nodes = 0;
int num_visited_nodes = 0;
int num_visited_edges = 0;
int num_unvisited_edges = 0;
int num_unvisited_nodes = 0;
int del_mf = 0;
int del_nf = 0;
int n, m;


//std::random_device rd;
//std::mt19937 gen(rd());

class frontier{
    int new_arr[num_nodes];
    public:
        int arr_size;
        int arr[num_nodes];
        bool is_arr;
        frontier(){
            arr_size = 0;
            is_arr = true;

        }
       void push(int data){
            //cout<<data<<" ";
            if(is_arr){
                arr[arr_size] = data;
            }else{
                arr[data] = 1;
            }
            arr_size++;
        }

        void conv(){
            if(is_arr){
                memset(new_arr, 0, sizeof(new_arr));
                for (int i=0; i<arr_size; i++){
                    new_arr[arr[i]] = 1;
                }
                copy(new_arr, new_arr + num_nodes, arr);
                // for (int i=0; i<num_nodes; i++){
                //     arr[i] = new_arr[i];
                // }
            }else{
                int j=0;
                for(int i=0; i<num_nodes; i++){
                    if (arr[i]){
                        new_arr[j++]=i;
                    }
                }
                copy(new_arr, new_arr + arr_size, arr);
                // for (int i=0; i<; i++){
                //     arr[i] = new_arr[i];
               // }

            }
            is_arr = !is_arr;
        }
        // void conv_bitmap_to_arr(){
        //     int new_arr[arr_size];
        //     for (int i=0; i<num_nodes; i++){

        //     }


        // }
};

void clear_parent(){
        for (int i=0; i<num_nodes; i++){
                parent[i] = -1;
        }
}

void preprocessing(){
    cin>>n>>m;
    for(int i=0; i<n; i++){
            out_degree[i] = 0;
            in_degree[i] = 0;
            parent[i] = -1;
            filled_upto[i] = 0;
            filled_upto_in[i] = 0;
    }
    auto addEdge = [&](int i, int u, int v){
            edge_list[i][0] = u; 
            edge_list[i][1] = v;
            edge_in[i][0] = edge_list[i][1];
            edge_in[i][1] = edge_list[i][0];

            out_degree[edge_list[i][0]]++;
            in_degree[edge_list[i][1]]++;
    };
    for(int i=0 ; i<m; i++){
            int u, v;
            cin>>u>>v;
	    u--;v--;
            addEdge(2*i,u,v);
            addEdge(2*i+1,v,u);
    }
    m *= 2;
    pref_sum[0] = out_degree[0];
    for(int i=1; i<n; i++){
            pref_sum[i] = pref_sum[i-1] + out_degree[i];
    }
    for(int i=0; i<m; i++){
            int u = edge_list[i][0];
            int v = edge_list[i][1];
            int index = pref_sum[u] - out_degree[u] + filled_upto[u];
//            cout<<u<<" "<<v<<" "<<pref_sum[u]<<" "<<out_degree[u]<<" "<<filled_upto[u]<<"\n";
            csr[index] = v;
            filled_upto[u]++;
    }

    pref_sum_in[0] = in_degree[0];
    for(int i=1; i<n; i++){
            pref_sum_in[i] = pref_sum_in[i-1] + in_degree[i];
    }
    for(int i=0; i<m; i++){
            int u = edge_in[i][0];
            int v = edge_in[i][1];
            int index = pref_sum_in[u] - in_degree[u] + filled_upto_in[u];
            csr_in[index] = v;
            filled_upto_in[u]++;
    }
    num_unvisited_nodes = n;
    num_unvisited_edges = m;
}


void top_down(frontier *curr_frontier, frontier *next_frontier){
    number_tdcalls++;
    //cout<<"top down : "<<curr_frontier->arr_size<<" "<<curr_frontier->arr[0]<<"\n";
    // cout<<"new iteration :"<<"\n";
    for (int i=0; i<curr_frontier->arr_size; i++){
        int v = curr_frontier->arr[i];
        // cout<<"parent node :"<<v<<"\n";
        for(int j=pref_sum[v]-out_degree[v]; j<pref_sum[v]; j++){
            // cout<<pref_sum[v]-out_degree[v]<<" "<<pref_sum[v]<<"\n";
            int u = csr[j];
            if (parent[u] == -1){
                parent[u] = v;
                next_frontier->push(u);
            }       
        }
        // cout<<"\n";
    }
}


void bottom_up(frontier *curr_frontier, frontier *next_frontier){
    number_bucalls++;
    curr_frontier->conv();
    next_frontier->conv();
    omp_set_num_threads(num_threads);

    int start, end;
    int tid, block;
    double beg, last;
    beg=omp_get_wtime();
#pragma omp parallel private(start, end, tid, block)
    {
	tid=omp_get_thread_num();
	block=ceil((1.0*n)/num_threads);
	start=tid*block;
	end=min(start+block,n);
	//printf("thread-id : %d , start : %d , end : %d , block : %d\n",tid,start,end,block);
	for(int i=start; i<end; i++){
		local_count[i] = 0;
		if (parent[i] == -1){
		    for(int j=pref_sum_in[i]-in_degree[i]; j<pref_sum_in[i]; j++){
			if (curr_frontier->arr[csr[j]] == 1){
				  parent[i] = csr[j];
//				  next_frontier->push(i);
				  next_frontier->arr[i] = 1;
				  local_count[i]++;
				  break;
			}
		    }
		}
	}
    }
    last=omp_get_wtime();
    //cout<<"Time taken inside parallel BFS "<<(last-beg)<<endl;
    acc_time+=(last-beg);
    //printf("temp : %d, n : %d\n",temp,n);
    //assert(temp == n);
    int count =0 ;
    for(int i = 0; i < n; i++){
           count += local_count[i];
    }
    next_frontier->arr_size = count;
    //printf("At the end of bottom_up before conversion: curr_frontier size : %d , next_frontier size : %d \n" , curr_frontier->arr_size, next_frontier->arr_size);
    curr_frontier->conv();
    next_frontier->conv();
    //printf("At the end of bottom_up after conversion: curr_frontier size : %d , next_frontier size : %d \n" , curr_frontier->arr_size, next_frontier->arr_size);
}


//void (*state)(frontier*, frontier*) = top_down;
//void (*TB)(frontier*, frontier*) = top_down;
//void (*BT)(frontier*, frontier*) = bottom_up;
string state = "top_down";
void hybrid(frontier *curr_frontier, frontier *next_frontier){
        int C_TB = num_unvisited_edges/alpha;
        int C_BT = n/beta;
        if(state == "top_down" and num_frontier_edges > C_TB and del_mf > 0){
		//printf("changing from top down to bottom up\n");
		state = "bottom_up";
        } 
	else if (state == "bottom_up" and num_frontier_nodes < C_BT and del_nf < 0){
		//printf("changing from bottom up to top down\n");
		state = "top_down";
        }

	if(state == "top_down"){
		top_down(curr_frontier, next_frontier);
	}else{
		bottom_up(curr_frontier,next_frontier);
	}
}
int cnt_vis = 0;
void bfs_boilerplate(void (*bfs_type)(frontier*, frontier*)){
    //std::uniform_int_distribution<int> distribution(0, n-1);
    //for (int trial =0; trial<total_samples; trial++){
	 
	//clear_parent();
	//int i = distribution(gen);
    for(int i = 0; i < n; i++){
        if(parent[i] != -1)
                continue;
            frontier *curr_frontier = new frontier();
            frontier *next_frontier = new frontier();
            curr_frontier->push(i);
            parent[i]=i;
            num_frontier_nodes = 1;
            num_frontier_edges = out_degree[i];
            num_unvisited_nodes--;
            num_unvisited_edges -= in_degree[i];
             //cout<<"boiler:"<<i<<"\n";

            while (curr_frontier->arr_size > 0){
		cnt_vis += curr_frontier->arr_size;
		if(curr_frontier->arr_size !=  1){
			//cout<<"curr_frontier size : "<<curr_frontier->arr_size<<" "<<state<<"\n";
		//	cout<<curr_frontier->arr_size<<endl;			
		//	printf("%d",curr_frontier->arr_size);
			//cout<<"";
			
		}
		//	printf("%d",curr_frontier->arr_size);
                bfs_type(curr_frontier, next_frontier);
		int old_nf = num_frontier_nodes;
		int old_mf = num_frontier_edges;
		for(int i = 0; i < curr_frontier->arr_size; i++){
			num_frontier_nodes--;
			num_frontier_edges -= out_degree[curr_frontier->arr[i]];

		}
		delete curr_frontier;
                curr_frontier = next_frontier;
		for(int i = 0; i < curr_frontier->arr_size; i++){
			num_frontier_nodes++;
			num_frontier_edges += out_degree[curr_frontier->arr[i]];
			num_unvisited_edges -= in_degree[curr_frontier->arr[i]];

		}
		del_nf = num_frontier_nodes - old_nf;
		del_mf = num_frontier_edges - old_mf;
                next_frontier = new frontier();
            }
	    delete curr_frontier;
	    delete next_frontier;
    }
}



int main(){
    //freopen("sample.txt", "r", stdin);
    preprocessing();
    clock_t start, end;

    clear_parent();
    //start=clock();
    //for(int i = 0; i < total_samples; i++)
    //        bfs_boilerplate(top_down);
    //end=clock();
    ////cout<<"TIME FOR TOP DOWN PLATE "<<(float)(end-start)/CLOCKS_PER_SEC*1000<<endl;
    ////cout<<"\n";
    //clear_parent();
    //auto start_BT = currTime;
    ////for(int i = 0; i < total_samples; i++)
    //       // bfs_boilerplate(bottom_up);
    //auto end_BT = currTime;
    //clear_parent();
    ////cout<<"\n";
    auto start_hybrid = currTime;
	    bfs_boilerplate(hybrid);
    auto end_hybrid=currTime;
    //cout<<"TIME FOR BOILER PLATE "<<(float)(end-start)/CLOCKS_PER_SEC*1000<<endl;
    //cout<<n << " "<< m<<" "<<1.0*tellTime(start_TB, end_TB)/total_samples<<" "<<1.0*tellTime(start_BT, end_BT)/total_samples<<" "<<1.0*tellTime(start_hybrid, end_hybrid)/total_samples<<"\n";
    cout<<n<<" "<<m<<"  "<<tellTime(start_hybrid, end_hybrid)<<"\n";
    //cout<<"number of bottom-up bfs calls :"<<1.0*number_bucalls<<"\n";
    //cout<<"number of top-down calls :"<<1.0*number_tdcalls<<"\n";
    return 0;
}


