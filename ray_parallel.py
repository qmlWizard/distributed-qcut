import ray
import time
import numpy as np

ray.init()


@ray.remote
def process(data):
    return data * data


# --------------------------------------------------
# DATA
# --------------------------------------------------

data = np.random.randint(
    2,
    10000,
    size=1000000
)

print("Data Shape:", data.shape)


# --------------------------------------------------
# RAY
# --------------------------------------------------

# 8 independent chunks
chunks = np.array_split(data, 8)

start_ray = time.perf_counter()

futures = [
    process.remote(chunk)
    for chunk in chunks
]

ray_results = np.concatenate(
    ray.get(futures)
)

end_ray = time.perf_counter()


# --------------------------------------------------
# SERIAL
# --------------------------------------------------

start_serial = time.perf_counter()

serial_results = np.asarray([
    x * x
    for x in data
])

end_serial = time.perf_counter()


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

assert np.array_equal(
    ray_results,
    serial_results
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

ray_time = end_ray - start_ray
serial_time = end_serial - start_serial

print("=" * 40)
print("SUMMARY")
print("=" * 40)

print(f"Ray Time:    {ray_time:.6f} seconds")
print(f"Serial Time: {serial_time:.6f} seconds")

print(
    f"Speedup:     {serial_time / ray_time:.3f}x"
)

ray.shutdown()