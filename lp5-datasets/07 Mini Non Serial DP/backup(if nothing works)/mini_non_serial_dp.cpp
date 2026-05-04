#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

using namespace std;

int matrixChainOrder(vector<int>& p) {
    int n = p.size() - 1;

    // DP table
    vector<vector<int>> dp(n, vector<int>(n, 0));

    // l = chain length
    for (int l = 2; l <= n; l++) {

        // Parallelize this loop (wavefront)
        #pragma omp parallel for
        for (int i = 0; i < n - l + 1; i++) {
            int j = i + l - 1;
            int min_cost = INT_MAX;

            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k+1][j] + p[i] * p[k+1] * p[j+1];
                if (cost < min_cost)
                    min_cost = cost;
            }

            dp[i][j] = min_cost;
        }
    }

    return dp[0][n-1];
}

int main() {
    vector<int> p = {10, 20, 30, 40, 30};

    double start = omp_get_wtime();

    int result = matrixChainOrder(p);

    double end = omp_get_wtime();

    cout << "Minimum Multiplication Cost: " << result << endl;
    cout << "Time: " << (end - start) << " seconds" << endl;

    return 0;
}