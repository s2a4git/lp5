#include <iostream>
#include <cuda.h>
using namespace std;

#define INF 99999
#define N 4   // change size if needed

// GPU Kernel
__global__ void floydWarshall(int *dist, int k, int n) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n && j < n) {
        int via_k = dist[i*n + k] + dist[k*n + j];
        if (via_k < dist[i*n + j]) {
            dist[i*n + j] = via_k;
        }
    }
}

// Main
int main() {
    int h_dist[N][N] = {
        {0,   5,  INF, 10},
        {INF, 0,   3, INF},
        {INF, INF, 0,   1},
        {INF, INF, INF, 0}
    };

    int *d_dist;
    size_t size = N * N * sizeof(int);

    cudaMalloc(&d_dist, size);
    cudaMemcpy(d_dist, h_dist, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);


    // DP Loop (k is sequential)

    for (int k = 0; k < N; k++) {
        floydWarshall<<<blocks, threads>>>(d_dist, k, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_dist, d_dist, size, cudaMemcpyDeviceToHost);


    // Print result

    cout << "Shortest distance matrix:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (h_dist[i][j] == INF)
                cout << "INF ";
            else
                cout << h_dist[i][j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_dist);
    return 0;
}

// COMMAND:
// nvcc mini_non_serial_dp.cu -o mini_non_serial_dp
// ./mini_non_serial_dp

// CODE RUNS ON COLLAB
// !nvcc mini_non_serial_dp.cu -o mini_non_serial_dp && ./mini_non_serial_dp