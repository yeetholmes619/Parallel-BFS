#include<vector>
#include<chrono>
#include<iostream>
#include<queue>
#define currTime std::chrono::high_resolution_clock::now()
#define tellTime(l, r) std::chrono::duration_cast<std::chrono::milliseconds>(r - l).count()

const int maxnodes = 300'000'000;
const int maxedges = 600'000'000;

std::vector <int> out_degree;
std::vector <int> prefix_sum;
std::vector <int> csr;
std::vector <int> visited;
std::queue <int> bfs_queue;

int main(){
	//this version of bfs takes csr as input..
        auto start = currTime;
        int nodes , edges;
	int start_idx_not_required;
        std::cin >> nodes >> edges;
	visited.assign(nodes,0);
	csr.assign(edges,0);
	prefix_sum.assign(nodes,0);
	out_degree.assign(nodes,0);
	for (int i = 0; i < nodes; i++){
		std::cin >> start_idx_not_required >> out_degree[i];
	}
        for(int i = 0 ; i < edges; i++){
                std::cin>>csr[i];
        }
        for(int i = 0; i < nodes; i++){
                visited[i] = false;
        }
        prefix_sum[0] = out_degree[0];
        for(int i = 1; i < nodes; i++){
                prefix_sum[i] = prefix_sum[i - 1] + out_degree[i];
        }

        auto bfs = [&](int x)->void{
                bfs_queue.push(x);
                visited[x] = true;
                while(!bfs_queue.empty()){
                        int node = bfs_queue.front();
                        bfs_queue.pop();
                        for(int i = prefix_sum[node] - out_degree[node]; i < prefix_sum[node]; i++){
                                if(!visited[csr[i]]){
                                        bfs_queue.push(csr[i]);
                                        visited[csr[i]] = true;
                                }
                        }
                }
        };

        auto start_bfs = currTime;
	bfs(0);
        auto end = currTime;
        std::cout<<nodes<<" "<<edges<<" "<<tellTime(start_bfs, end)<<"\n";
        return 0;
}
