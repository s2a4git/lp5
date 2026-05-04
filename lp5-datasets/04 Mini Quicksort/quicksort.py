# Mini Project: Evaluate performance enhancement of parallel Quicksort Algorithm using MPI

from mpi4py import MPI
import numpy as np
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# SERIAL QUICKSORT
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# MAIN
def main():

    if rank == 0:
        print(f"\nRunning with {size} MPI Processes")

        # Generate dataset ONLY in root
        data = np.array([random.randint(0, 10000) for _ in range(1000000)])

        # Serial timing
        start = time.time()
        serial_sorted = quicksort(data.tolist())
        serial_time = time.time() - start

    else:
        data = None
        serial_time = None

    # Broadcast serial time
    serial_time = comm.bcast(serial_time, root=0)

    # PARALLEL PART
    chunk_size = 1000000 // size

    # Allocate local buffer
    local_data = np.zeros(chunk_size, dtype=int)

    # Scatter equal chunks
    comm.Scatter(data, local_data, root=0)

    # Local sort
    start = time.time()
    local_sorted = quicksort(local_data.tolist())
    parallel_local_time = time.time() - start

    # Gather sorted chunks
    gathered = comm.gather(local_sorted, root=0)

    if rank == 0:
        # Merge all sorted chunks
        final_sorted = []
        for chunk in gathered:
            final_sorted.extend(chunk)

        # Final merge sort for correctness
        final_sorted = sorted(final_sorted)

        parallel_time = parallel_local_time

        print(f"Sequential Time: {serial_time:.4f} seconds")
        print(f"Parallel Time: {parallel_time:.4f} seconds")
        print(f"Speedup: {serial_time/parallel_time:.2f}")


if __name__ == "__main__":
    main()
