import numpy as np

import warp as wp

# CUDA snippet for atomicCAS function
atomic_cas_snippet = """
return atomicCAS(&address[slot], compare, val);
"""


# Register the atomicCAS function
@wp.func_native(atomic_cas_snippet)
def atomicCAS(address: wp.array(dtype=int), slot: int, compare: int, val: int) -> int:
    ...


# Hash function to convert vec4i to int32
@wp.func
def hash_func(key: wp.vec4i, capacity: int) -> int:
    # Simple hash function for demonstration
    hash_val = key[0] ^ key[1] ^ key[2] ^ key[3]
    # hash_val = key[0] + key[1] + key[2] + key[3]
    return hash_val % capacity


@wp.func
def vec_equal(a: wp.vec4i, b: wp.vec4i) -> bool:
    for i in range(4):
        if a[i] != b[i]:
            return False
    return True


# Warp kernel for inserting into the hashmap
@wp.kernel
def insert_kernel(
    table_keys: wp.array(dtype=int),
    table_vals: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    table_capacity: int,
):
    idx = wp.tid()
    value = idx + 1
    slot = hash_func(vec_keys[idx], table_capacity)
    while True:
        prev = atomicCAS(table_keys, slot, -1, slot)
        # Insertion successful.
        if prev == -1:
            table_vals[slot] = value
            return
        slot = (slot + 1) % table_capacity


# Warp kernel for searching in the hashmap
@wp.kernel
def search_kernel(
    table_keys: wp.array(dtype=int),
    table_vals: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    search_keys: wp.array(dtype=wp.vec4i),
    search_results: wp.array(dtype=int),
    table_capacity: int,
):
    idx = wp.tid()
    key = search_keys[idx]
    slot = hash_func(key, table_capacity)
    while True:
        current_key = table_keys[slot]
        if current_key != -1:
            vec_val = table_vals[slot] - 1
            if vec_equal(vec_keys[vec_val], key):
                search_results[idx] = vec_val
                return
        else:
            search_results[idx] = -1
            return
        slot = (slot + 1) % table_capacity


def test():
    # Define device
    device = "cuda"

    # Define hashmap size and create arrays
    capacity = 32
    table_keys = wp.array(np.full(capacity, -1, dtype=int), dtype=int, device=device)
    table_vals = wp.zeros(capacity, dtype=int, device=device)

    # Create example keys
    N = 16
    table_vec_keys = wp.array(
        np.random.randint(0, 100, size=(N, 4), dtype=int), dtype=wp.vec4i, device=device
    )

    # Launch the kernel
    wp.launch(
        kernel=insert_kernel,
        dim=N,
        inputs=[table_keys, table_vals, table_vec_keys, capacity],
        device=device,
    )

    # Output results
    print("Keys:", table_vec_keys.numpy())
    print("Table Keys:", table_keys.numpy())
    print("Table Values:", table_vals.numpy())

    search_keys = table_vec_keys
    results = wp.empty(len(search_keys), dtype=int, device=device)

    # Launch the search kernel
    wp.launch(
        kernel=search_kernel,
        dim=len(search_keys),
        inputs=[table_keys, table_vals, table_vec_keys, search_keys, results, capacity],
        device=device,
    )

    # Output search results
    print("Search Keys:", search_keys.numpy())
    print("Search Results:", results.numpy())


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    __loader__ = None

    wp.init()

    test()
