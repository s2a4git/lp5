// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP.Useexisting algorithms and measure the performance of sequential and parallel algorithms.

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// SEQUENTIAL BUBBLE SORT 
void sequentialBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// PARALLEL BUBBLE SORT (ODD-EVEN METHOD) 
void parallelBubbleSort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n; i++)
    {

// Even phase
#pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }

// Odd phase
#pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// MERGE FUNCTION 
void merge(vector<int> &arr, int left, int mid, int right)
{
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];

    for (int i = left, k = 0; i <= right; i++, k++)
        arr[i] = temp[k];
}

// SEQUENTIAL MERGE SORT 
void sequentialMergeSort(vector<int> &arr, int left, int right)
{
    if (left >= right)
        return;

    int mid = (left + right) / 2;
    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// PARALLEL MERGE SORT 
void parallelMergeSort(vector<int> &arr, int left, int right)
{
    if (left >= right)
        return;

    int mid = (left + right) / 2;

#pragma omp task shared(arr)
    parallelMergeSort(arr, left, mid);

#pragma omp task shared(arr)
    parallelMergeSort(arr, mid + 1, right);

#pragma omp taskwait
    merge(arr, left, mid, right);
}

// MAIN FUNCTION 
int main()
{
    int n = 10000; // Change size for testing
    vector<int> arr(n);

    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 10000;

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds\n";

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << " seconds\n";

    // Parallel Merge Sort
    start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        parallelMergeSort(arr4, 0, n - 1);
    }
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";

    return 0;
}

// COMMAND:
// g++ -fopenmp -O3 bubble_merge_sort.cpp -o bubble_merge_sort
// ./bubble_merge_sort