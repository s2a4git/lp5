// Design and implement Parallel Breadth First Search and Depth First Search based on existingalgorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS.

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph
{
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    // PARALLEL BFS
    void parallelBFS(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS Traversal: ";

        while (!q.empty())
        {
            int size = q.size();

#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                int node;

#pragma omp critical
                {
                    if (!q.empty())
                    {
                        node = q.front();
                        q.pop();
                        cout << node << " ";
                    }
                }

                for (int neighbor : adj[node])
                {
                    if (!visited[neighbor])
                    {
#pragma omp critical
                        {
                            if (!visited[neighbor])
                            {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }

    // PARALLEL DFS
    void parallelDFSUtil(int node, vector<bool> &visited)
    {
        visited[node] = true;

#pragma omp critical
        cout << node << " ";

        for (int neighbor : adj[node])
        {
            if (!visited[neighbor])
            {
#pragma omp task shared(visited)
                parallelDFSUtil(neighbor, visited);
            }
        }
    }

    void parallelDFS(int start)
    {
        vector<bool> visited(V, false);
        cout << "Parallel DFS Traversal: ";

#pragma omp parallel
        {
#pragma omp single
            {
                parallelDFSUtil(start, visited);
            }
        }
        cout << endl;
    }
};

int main()
{
    int V = 6;
    Graph g(V);

    // Example Undirected Graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}

// COMMAND: 
// g++ -fopenmp -O3 bfs_dfs_traversal.cpp -o graph_traversal
// ./graph_traversal