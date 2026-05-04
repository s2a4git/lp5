#include <iostream>
#include <cuda.h>
using namespace std;

// GPU Kernel: Compute cost
__global__ void computeCost(int *sizes, int *plans, int *costs, int n_plans) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n_plans) {
        int a = plans[idx * 3 + 0];
        int b = plans[idx * 3 + 1];
        int c = plans[idx * 3 + 2];

        // Simple cost model
        int cost1 = sizes[a] * sizes[b];
        int cost2 = cost1 * sizes[c];

        costs[idx] = cost1 + cost2;
    }
}

// Main
int main() {
    // Table sizes
    int h_sizes[3] = {1000, 500, 2000}; // A, B, C

    // Plans (permutations of joins)
    int h_plans[3][3] = {
        {0, 1, 2}, // (A B) C
        {0, 2, 1}, // (A C) B
        {1, 2, 0}  // (B C) A
    };

    int h_costs[3];

    int *d_sizes, *d_plans, *d_costs;

    cudaMalloc(&d_sizes, 3 * sizeof(int));
    cudaMalloc(&d_plans, 9 * sizeof(int));
    cudaMalloc(&d_costs, 3 * sizeof(int));

    cudaMemcpy(d_sizes, h_sizes, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plans, h_plans, 9 * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel
    computeCost<<<1, 3>>>(d_sizes, d_plans, d_costs, 3);

    cudaMemcpy(h_costs, d_costs, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Find best plan
    int min_cost = h_costs[0];
    int best_plan = 0;

    for (int i = 1; i < 3; i++) {
        if (h_costs[i] < min_cost) {
            min_cost = h_costs[i];
            best_plan = i;
        }
    }

    cout << "Costs of plans:\n";
    for (int i = 0; i < 3; i++) {
        cout << "Plan " << i << ": " << h_costs[i] << endl;
    }

    cout << "\nBest Plan: " << best_plan
         << " with cost " << min_cost << endl;

    cudaFree(d_sizes);
    cudaFree(d_plans);
    cudaFree(d_costs);

    return 0;
}

// COMMAND:
// nvcc -o mini_database_query mini_database_query.cu
// ./mini_database_query

// CODE RUNS ON COLLAB
// !nvcc -o mini_database_query mini_database_query.cu && ./mini_database_query