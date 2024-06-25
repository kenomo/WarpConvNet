import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

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
def hash_fnv1a(key: wp.vec4i, capacity: int) -> int:
    # Simple hash function for demonstration
    hash_val = int(2166136261)
    for i in range(4):
        hash_val ^= key[i]
        hash_val *= 16777619
    # Make sure the hash value is positive
    return ((hash_val % capacity) + capacity) % capacity


@wp.func
def murmur_hash3(key: wp.vec4i, capacity: int) -> int:
    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xE6546B64

    hash_val = 0

    for i in range(4):
        k = key[i]
        k = k * c1
        k = (k << r1) | (k >> (32 - r1))
        k = k * c2

        hash_val = hash_val ^ k
        hash_val = (hash_val << r2) | (hash_val >> (32 - r2))
        hash_val = hash_val * m + n

    return (hash_val % capacity + capacity) % capacity


@wp.func
def city_hash(key: wp.vec4i, capacity: int) -> int:
    hash_val = 0
    for i in range(4):
        hash_val += key[i] * 0x9E3779B9
        hash_val ^= hash_val >> 16
        hash_val *= 0x85EBCA6B
        hash_val ^= hash_val >> 13
        hash_val *= 0xC2B2AE35
        hash_val ^= hash_val >> 16

    return (hash_val % capacity + capacity) % capacity


@wp.func
def xx_hash(key: wp.vec4i, capacity: int) -> int:
    prime1 = 0x9E3779B1
    prime2 = 0x85EBCA77
    prime3 = 0xC2B2AE3D
    prime4 = 0x27D4EB2F
    prime5 = 0x165667B1

    hash_val = 0

    for i in range(4):
        hash_val += key[i] * prime3
        hash_val = (hash_val << 13) | (hash_val >> (32 - 13))
        hash_val *= prime1

    hash_val += 4 * prime5
    hash_val ^= hash_val >> 15
    hash_val *= prime2
    hash_val ^= hash_val >> 13
    hash_val *= prime4
    hash_val ^= hash_val >> 16

    return (hash_val % capacity + capacity) % capacity


@wp.func
def djb_hash(key: wp.vec4i, capacity: int) -> int:
    hash_val = 5381
    for i in range(4):
        hash_val = ((hash_val << 5) + hash_val) + key[i]

    return (hash_val % capacity + capacity) % capacity


@wp.func
def vec_equal(a: wp.vec4i, b: wp.vec4i) -> bool:
    for i in range(4):
        if a[i] != b[i]:
            return False
    return True


# Warp kernel for inserting into the hashmap
@wp.kernel
def insert_kernel(
    table_kvs: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    table_capacity: int,
):
    idx = wp.tid()
    slot = hash_fnv1a(vec_keys[idx], table_capacity)
    initial_slot = slot
    while True:
        prev = atomicCAS(table_kvs, 2 * slot, -1, slot)
        # Insertion successful.
        if prev == -1:
            table_kvs[2 * slot + 1] = idx
            return
        slot = (slot + 1) % table_capacity

        # If we circle back to the initial slot, the table is full
        if slot == initial_slot:
            return  # This indicates that the table is full and we couldn't insert the unique item


@wp.kernel
def insert_unique_kernel(
    table_kvs: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    table_capacity: int,
):
    idx = wp.tid()
    vec_key = vec_keys[idx]
    slot = hash_fnv1a(vec_key, table_capacity)
    initial_slot = slot

    while True:
        prev = atomicCAS(table_kvs, 2 * slot, -1, slot)
        if prev == -1:  # insertion success
            table_kvs[2 * slot + 1] = idx
            return
        else:  # collision exists
            current_idx = table_kvs[2 * slot + 1]
            if vec_equal(vec_keys[current_idx], vec_key):
                return  # Item already exists in the table
        slot = (slot + 1) % table_capacity

        # If we circle back to the initial slot, the table is full
        if slot == initial_slot:
            return  # This indicates that the table is full and we couldn't insert the unique item


# Warp kernel for searching in the hashmap
@wp.kernel
def search_kernel(
    table_kvs: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    search_keys: wp.array(dtype=wp.vec4i),
    search_results: wp.array(dtype=int),
    table_capacity: int,
):
    idx = wp.tid()
    key = search_keys[idx]
    slot = hash_fnv1a(key, table_capacity)
    while True:
        current_key = table_kvs[2 * slot + 0]
        if current_key == -1:
            search_results[idx] = -1
            return
        else:
            vec_val = table_kvs[2 * slot + 1]
            if vec_equal(vec_keys[vec_val], key):
                search_results[idx] = vec_val
                return
        slot = (slot + 1) % table_capacity


# Warp kernel for preparing key-value pairs
@wp.kernel
def prepare_key_value_pairs(table_kv: wp.array(dtype=int)):
    tid = wp.tid()
    table_kv[2 * tid + 0] = -1
    table_kv[2 * tid + 1] = -1


class VectorHashTable:
    _table_kv: wp.array(dtype=wp.vec2i) = None
    _vector_keys: wp.array(dtype=wp.vec4i) = None
    capacity: int = 0

    def __init__(self, capacity: int):
        self.capacity = capacity

    # setter and getter for table_kv
    @property
    def table_kv(self):
        assert self._table_kv is not None, "Table key-value pairs are not initialized"
        return self._table_kv

    @table_kv.setter
    def table_kv(self, value):
        self._table_kv = value

    @property
    def table_keys(self):
        return self.table_kv[::2]

    @property
    def table_values(self):
        return self.table_kv[1::2]

    def insert(self, vec_keys: wp.array(dtype=wp.vec4i)):
        assert (
            len(vec_keys) <= self.capacity / 2
        ), f"Number of keys {len(vec_keys)} exceeds capacity {self.capacity}"
        assert vec_keys.dtype == wp.vec4i
        assert self._table_kv is None and self._vector_keys is None
        self.table_kv = wp.empty(2 * self.capacity, dtype=int, device=vec_keys.device)
        wp.launch(
            kernel=prepare_key_value_pairs,
            dim=self.capacity,
            inputs=[self.table_kv],
        )
        self._vector_keys = vec_keys
        wp.launch(
            kernel=insert_kernel,
            dim=len(vec_keys),
            inputs=[self.table_kv, vec_keys, self.capacity],
        )

    def search(self, search_keys: wp.array(dtype=wp.vec4i)) -> wp.array(dtype=int):
        results = wp.empty(len(search_keys), dtype=int)
        wp.launch(
            kernel=search_kernel,
            dim=len(search_keys),
            inputs=[self.table_kv, self._vector_keys, search_keys, results, self.capacity],
        )
        return results

    def unique_index(self) -> Int[Tensor, "N"]:  # noqa: F821
        # table_values = wp.to_torch(self.table_values)
        indices = self.search(self._vector_keys)
        return torch.unique(wp.to_torch(indices))

    @property
    def device(self):
        return self.table_kv.device

    def to_dict(self):
        table_kv_np = self.table_kv.numpy().reshape(-1, 2)
        vec_keys_np = self._vector_keys.numpy()
        return {
            "table_kvs": table_kv_np,
            "vec_keys": vec_keys_np,
        }

    @classmethod
    def from_dict(cls, data: dict):
        capacity = len(data["table_kvs"]) // 2
        device = data["table_kvs"].device
        obj = cls(capacity)
        obj._table_kv = wp.array(data["table_kvs"], dtype=wp.vec2i, device=device)
        obj._vector_keys = wp.array(data["vec_keys"], dtype=wp.vec4i, device=device)
        return obj


def test():
    # Define device
    device = "cuda"

    # Define hashmap size and create arrays
    capacity = 128
    table_kvs = wp.empty(2 * capacity, dtype=int, device=device)

    wp.launch(
        kernel=prepare_key_value_pairs,
        dim=capacity,
        inputs=[table_kvs],
    )

    # Create example keys
    N = 48
    table_vec_keys = wp.array(
        np.random.randint(0, 100, size=(N, 4), dtype=int), dtype=wp.vec4i, device=device
    )

    # Launch the kernel
    wp.launch(
        kernel=insert_kernel,
        dim=N,
        inputs=[table_kvs, table_vec_keys, capacity],
        device=device,
    )

    # Output results
    print("Keys:", table_vec_keys.numpy())
    table_kvs_np = table_kvs.numpy().reshape(-1, 2)
    print("Table Keys:", table_kvs_np[:, 0])
    print("Table Values:", table_kvs_np[:, 1])

    search_keys = table_vec_keys
    results = wp.empty(len(search_keys), dtype=int, device=device)

    # Launch the search kernel
    wp.launch(
        kernel=search_kernel,
        dim=len(search_keys),
        inputs=[table_kvs, table_vec_keys, search_keys, results, capacity],
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
