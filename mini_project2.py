import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time
import statistics
import os
from multiprocessing import Pool


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for j in range(max_iter):
        if z_real*z_real + z_imag*z_imag > 4.0:
            return j

        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2.0 * z_real * z_imag + c_imag

        z_real = new_real
        z_imag = new_imag

    return max_iter


@njit
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):

    height = row_end - row_start
    out = np.empty((height, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(height):
        c_imag = y_min + (r + row_start) * dy
        c_real = x_min

        for col in range(N):
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)
            c_real += dx  

    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter):
    return mandelbrot_chunk(
        row_start=0,
        row_end=N,
        N=N,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        max_iter=max_iter
    )



def benchmark(func, *args, n_runs=3):
    """Time func, return median runtime and result."""
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)
    print(
        f"Median: {median_t:.4f}s "
        f"(min={min(times):.4f}, max={max(times):.4f})"
    )

    return median_t, result



def worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(pool, chunks):
    parts = pool.map(worker, chunks)
    return np.vstack(parts)

def build_chunks(N, x_min, x_max, y_min, y_max, max_iter, n_workers):

    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)

        chunks.append((
            row, row_end, N,
            x_min, x_max, y_min, y_max, max_iter
        ))

        row = row_end

    return chunks

def benchmark_parallel(pool, chunks, n_runs=3):

    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()

        result = mandelbrot_parallel(pool, chunks)

        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)

    print(f"Median: {median_t:.4f}s")

    return median_t, result

if __name__ == "__main__":
    
    N = 1024
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    max_iter = 100
    
    mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

     # serial 
    t_serial, _ = benchmark(mandelbrot_serial, N, x_min, x_max, y_min, y_max, max_iter)

    print(f"\nSerial time: {t_serial:.4f}s\n")
    
    
    for p in range(1, os.cpu_count() + 1):

        chunks = build_chunks(
            N, x_min, x_max, y_min, y_max, max_iter, p
        )

        with Pool(p) as pool:

            # warm-up (NOT timed)
            pool.map(worker, chunks)

            # benchmark
            t_parallel, result = benchmark_parallel(pool, chunks)

        speedup = t_serial / t_parallel
        

        (f"{p:2d} workers: {t_parallel:.3f}s, speedup={speedup:.2f}x, eff={speedup/p*100:.0f}%")
        

        