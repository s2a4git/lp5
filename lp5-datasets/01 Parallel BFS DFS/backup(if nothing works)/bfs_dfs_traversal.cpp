// Parallel BFS & DFS (Dev-C++ Compatible FIXED)

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
        adj[v].push_back(u);
    }

    // ✅ FIXED PARALLEL BFS
    void parallelBFS(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS Traversal: ";

        while (!q.empty())
        {
            int node = -1;

#pragma omp critical
            {
                if (!q.empty())
                {
                    node = q.front();
                    q.pop();
                    cout << node << " ";
                }
            }

            if (node == -1)
                continue; // ✅ moved outside critical

#pragma omp parallel for
            for (int i = 0; i < adj[node].size(); i++)
            {
                int neighbor = adj[node][i];

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
        cout << endl;
    }

    // ✅ FIXED PARALLEL DFS
    void parallelDFSUtil(int node, vector<bool> &visited)
    {
        bool shouldVisit = false;

#pragma omp critical
        {
            if (!visited[node])
            {
                visited[node] = true;
                cout << node << " ";
                shouldVisit = true;
            }
        }

        if (!shouldVisit)
            return; // ✅ moved outside critical

        for (int i = 0; i < adj[node].size(); i++)
        {
            int neighbor = adj[node][i];

#pragma omp task shared(visited)
            parallelDFSUtil(neighbor, visited);
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
    Graph g(6);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    g.parallelBFS(0);
    g.parallelDFS(0);

    return 0;
}