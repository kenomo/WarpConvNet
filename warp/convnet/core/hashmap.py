import enum

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor

import warp as wp


class HashMethod(enum.Enum):
    FNV1A = 0
    CITY = 1
    MURMUR = 2


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
def murmur_32_scramble(k: int) -> int:
    k *= 0xCC9E2D51
    k = (k << 15) | (k >> 17)
    k *= 0x1B873593
    return k


@wp.func
def hash_murmur(key: wp.vec4i, capacity: int) -> int:
    h = 0x9747B28C

    # Process each of the 4 integers in the vec4i key
    for i in range(4):
        k = key[i]
        h ^= murmur_32_scramble(k)
        h = (h << 13) | (h >> 19)
        h = h * 5 + 0xE6546B64

    # Finalize
    h ^= 16  # Length of the key in bytes (4 ints * 4 bytes each)
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16

    # Ensure the hash value is positive
    return ((h % capacity) + capacity) % capacity


@wp.func
def hash_city(key: wp.vec4i, capacity: int) -> int:
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
def hash_selection(hash_method: int, key: wp.vec4i, capacity: int) -> int:
    if hash_method == 0:
        return hash_fnv1a(key, capacity)
    elif hash_method == 1:
        return hash_city(key, capacity)
    elif hash_method == 2:
        return hash_murmur(key, capacity)
    else:
        return hash_fnv1a(key, capacity)


@wp.func
def vec_equal(a: wp.vec4i, b: wp.vec4i) -> bool:
    for i in range(4):
        if a[i] != b[i]:
            return False
    return True


@wp.struct
class HashStruct:
    table_kvs: wp.array(dtype=int)
    vector_keys: wp.array(dtype=wp.vec4i)
    capacity: int
    hash_method: int

    def insert(self, vec_keys: wp.array(dtype=wp.vec4i)):
        assert self.capacity > 0
        assert self.hash_method in [0, 1, 2]
        assert (
            len(vec_keys) <= self.capacity / 2
        ), f"Number of keys {len(vec_keys)} exceeds capacity {self.capacity}"

        device = vec_keys.device
        self.table_kvs = wp.empty(2 * self.capacity, dtype=int, device=device)
        wp.launch(
            kernel=prepare_key_value_pairs,
            dim=self.capacity,
            inputs=[self.table_kvs],
            device=device,
        )

        self.vector_keys = vec_keys
        wp.launch(
            kernel=insert_kernel,
            dim=len(vec_keys),
            inputs=[self.table_kvs, vec_keys, self.capacity, self.hash_method],
            device=device,
        )

    def search(self, search_keys: wp.array(dtype=wp.vec4i)) -> wp.array(dtype=int):
        device = search_keys.device
        results = wp.empty(len(search_keys), dtype=int, device=device)
        wp.launch(
            kernel=search_kernel,
            dim=len(search_keys),
            inputs=[
                self,
                search_keys,
                results,
            ],
            device=device,
        )
        return results

    def to_dict(self):
        table_kv_np = self.table_kvs.numpy().reshape(-1, 2)
        vec_keys_np = self.vector_keys.numpy()
        return {
            "table_kvs": table_kv_np,
            "vec_keys": vec_keys_np,
        }

    def from_dict(self, data: dict, device: str):
        self.table_kv = wp.array(data["table_kvs"], dtype=wp.vec2i, device=device)
        self.vector_keys = wp.array(data["vec_keys"], dtype=wp.vec4i, device=device)
        return self


# Warp kernel for inserting into the hashmap
@wp.kernel
def insert_kernel(
    table_kvs: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    table_capacity: int,
    hash_method: int,
):
    idx = wp.tid()
    slot = hash_selection(hash_method, vec_keys[idx], table_capacity)
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


@wp.func
def search_func(
    table_kvs: wp.array(dtype=int),
    vec_keys: wp.array(dtype=wp.vec4i),
    query_key: wp.vec4i,
    table_capacity: int,
    hash_method: int,
) -> int:
    slot = hash_selection(hash_method, query_key, table_capacity)
    initial_slot = slot
    while True:
        current_key = table_kvs[2 * slot + 0]
        if current_key == -1:
            return -1
        else:
            vec_val = table_kvs[2 * slot + 1]
            if vec_equal(vec_keys[vec_val], query_key):
                return vec_val
        slot = (slot + 1) % table_capacity
        if slot == initial_slot:
            return -1


# Warp kernel for searching in the hashmap
@wp.kernel
def search_kernel(
    hash_struct: HashStruct,
    search_keys: wp.array(dtype=wp.vec4i),
    search_results: wp.array(dtype=int),
):
    idx = wp.tid()
    key = search_keys[idx]
    result = search_func(
        hash_struct.table_kvs,
        hash_struct.vector_keys,
        key,
        hash_struct.capacity,
        hash_struct.hash_method,
    )
    search_results[idx] = result


# Warp kernel for preparing key-value pairs
@wp.kernel
def prepare_key_value_pairs(table_kv: wp.array(dtype=int)):
    tid = wp.tid()
    table_kv[2 * tid + 0] = -1
    table_kv[2 * tid + 1] = -1


class VectorHashTable:
    _hash_struct: HashStruct = None

    def __init__(self, capacity: int, hash_method: HashMethod = HashMethod.CITY):
        assert isinstance(hash_method, HashMethod)
        self._hash_struct = HashStruct()
        self._hash_struct.capacity = capacity
        self._hash_struct.hash_method = hash_method.value

    @property
    def capacity(self):
        return self._hash_struct.capacity

    @property
    def hash_method(self) -> HashMethod:
        return HashMethod(self._hash_struct.hash_method)

    @property
    def device(self):
        return self._hash_struct.table_kvs.device

    def insert(self, vec_keys: wp.array(dtype=wp.vec4i)):
        self._hash_struct.insert(vec_keys)

    @classmethod
    def from_keys(cls, vec_keys: wp.array(dtype=wp.vec4i)):
        if isinstance(vec_keys, torch.Tensor):
            vec_keys = wp.from_torch(vec_keys, dtype=wp.vec4i)
        obj = cls(2 * len(vec_keys))
        obj.insert(vec_keys)
        return obj

    def search(self, search_keys: wp.array(dtype=wp.vec4i)) -> wp.array(dtype=int):
        return self._hash_struct.search(search_keys)

    def unique_index(self) -> Int[Tensor, "N"]:  # noqa: F821
        # table_values = wp.to_torch(self.table_values)
        indices = self.search(self._hash_struct.vector_keys)
        return torch.unique(wp.to_torch(indices))

    def to_dict(self):
        return self._hash_struct.to_dict()

    def hashmap_struct(self) -> HashStruct:
        return self._hash_struct


def test():
    # Define device
    device = "cuda:0"

    # Define hashmap size and create arrays
    capacity = 128
    table_kvs = wp.empty(2 * capacity, dtype=int, device=device)

    wp.launch(
        kernel=prepare_key_value_pairs,
        dim=capacity,
        inputs=[table_kvs],
        device=device,
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
        inputs=[table_kvs, table_vec_keys, capacity, 0],
        device=device,
    )

    # Output results
    print("Keys:", table_vec_keys.numpy())
    table_kvs_np = table_kvs.numpy().reshape(-1, 2)
    print("Table Keys:", table_kvs_np[:, 0])
    print("Table Values:", table_kvs_np[:, 1])

    search_keys = table_vec_keys

    # Test struct
    hash_struct = HashStruct()
    hash_struct.capacity = capacity
    hash_struct.hash_method = 0
    hash_struct.insert(table_vec_keys)

    search_results = hash_struct.search(table_vec_keys)
    print("Search Keys:", search_keys.numpy())
    print("Search Results:", search_results.numpy())


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    __loader__ = None

    wp.init()
    test()
