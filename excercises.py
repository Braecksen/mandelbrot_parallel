from multiprocessing import Pool
import os
import time
import statistics
import random

def estimate_pi_chunk(num_samples):

    hits = 0

    for i in range(num_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def estimate_pi_parallel(num_samples, num_processes):
    
    samples_per_worker = num_samples // num_processes

    tasks = [samples_per_worker] * num_processes

    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)

    total_hits = sum(results)
    total_samples = samples_per_worker * num_processes

    return 4.0 * total_hits / total_samples



def benchmark_parallel(num_samples):
    
    max_procs = os.cpu_count()

    for p in range(1, max_procs + 1):
        t0 = time.perf_counter()
        pi_est = estimate_pi_parallel(num_samples, p)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        speedup =  1.7396 / elapsed

        print(
            f"Procs: {p:2d} | π ≈ {pi_est:.6f} | "
            f"time = {elapsed:.4f} s | speedup = {speedup:.2f}x"
        )


if __name__ == "__main__":
    NUM_SAMPLES = 10000000
    benchmark_parallel(NUM_SAMPLES)