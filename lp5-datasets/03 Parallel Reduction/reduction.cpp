// Implement Min, Max, Sum and Average operations using Parallel Reduction.

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <limits>

using namespace std;

int main()
{

    int n = 1000000; // Size of array
    vector<int> arr(n);

    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 1000; // Random numbers 0–999
    }

    long long sum = 0;
    int min_val = numeric_limits<int>::max();
    int max_val = numeric_limits<int>::min();

    double start = omp_get_wtime();

#pragma omp parallel for reduction(+ : sum) reduction(min : min_val) reduction(max : max_val)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];

        if (arr[i] < min_val)
            min_val = arr[i];

        if (arr[i] > max_val)
            max_val = arr[i];
    }

    double end = omp_get_wtime();

    double average = (double)sum / n;

    cout << "Parallel Reduction Results:\n";
    cout << "Sum = " << sum << endl;
    cout << "Minimum = " << min_val << endl;
    cout << "Maximum = " << max_val << endl;
    cout << "Average = " << average << endl;
    cout << "Execution Time = " << end - start << " seconds\n";

    return 0;
}