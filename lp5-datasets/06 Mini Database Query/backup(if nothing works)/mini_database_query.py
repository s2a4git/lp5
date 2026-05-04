import pandas as pd
import numpy as np
import time
import os
from multiprocessing import Pool, cpu_count


# STEP 1: CREATE SAMPLE DATA (if not exists)
def create_data():
    if not os.path.exists("./users.csv"):
        print("Creating sample dataset...")

        users = pd.DataFrame(
            {
                "id": np.arange(1, 100001),
                "age": np.random.randint(18, 60, 100000),
                "name": ["User_" + str(i) for i in range(100000)],
            }
        )

        orders = pd.DataFrame(
            {
                "order_id": np.arange(1, 200001),
                "user_id": np.random.randint(1, 100001, 200000),
                "amount": np.random.randint(100, 10000, 200000),
            }
        )

        users.to_csv("./users.csv", index=False)
        orders.to_csv("./orders.csv", index=False)

        print("Dataset created!")


# STEP 2: LOAD DATA
def load_data():
    users = pd.read_csv("./users.csv")
    orders = pd.read_csv("./orders.csv")
    return users, orders


# STEP 3: SEQUENTIAL QUERY
def sequential_query(users, orders):
    filtered = users[users["age"] > 30]
    result = filtered.merge(orders, left_on="id", right_on="user_id")
    return result


# STEP 4: PARALLEL FILTER
def filter_chunk(chunk):
    return chunk[chunk["age"] > 30]


def parallel_filter(users):
    chunks = np.array_split(users, cpu_count())

    with Pool(cpu_count()) as p:
        results = p.map(filter_chunk, chunks)

    return pd.concat(results)


# STEP 5: PARALLEL QUERY
def parallel_query(users, orders):
    filtered_users = parallel_filter(users)
    result = filtered_users.merge(orders, left_on="id", right_on="user_id")
    return result


# STEP 6: OPTIMIZED QUERY (Push-down)
def optimized_query(users, orders):
    # Apply filter first (optimization)
    users_filtered = users[users["age"] > 30]

    # Then join
    result = users_filtered.merge(orders, left_on="id", right_on="user_id")
    return result


# STEP 7: BAD QUERY (No Optimization)
def bad_query(users, orders):
    # Join first (inefficient)
    result = users.merge(orders, left_on="id", right_on="user_id")
    result = result[result["age"] > 30]
    return result


# STEP 8: PERFORMANCE TEST
def run_tests(users, orders):
    print("\nRunning performance comparison...\n")

    start = time.time()
    res1 = bad_query(users, orders)
    print("Bad Query Time:", time.time() - start)

    start = time.time()
    res2 = sequential_query(users, orders)
    print("Sequential Optimized Time:", time.time() - start)

    start = time.time()
    res3 = parallel_query(users, orders)
    print("Parallel Time:", time.time() - start)

    # Check correctness
    print("\nResults match:", res2.shape == res3.shape)


# MAIN
if __name__ == "__main__":
    create_data()
    users, orders = load_data()

    print("Users:", users.shape)
    print("Orders:", orders.shape)

    run_tests(users, orders)
