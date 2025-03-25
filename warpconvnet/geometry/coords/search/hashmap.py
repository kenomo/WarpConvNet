# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Union

import numpy as np
import torch
import warp as wp
from jaxtyping import Int
from torch import Tensor


class HashMethod(enum.Enum):
    """Hash method enumeration for the vector hash table.

    Attributes:
        FNV1A: FNV-1a hash algorithm
        CITY: CityHash algorithm
        MURMUR: MurmurHash algorithm
    """

    FNV1A = 0
    CITY = 1
    MURMUR = 2


# CUDA snippet for atomicCAS function
atomic_cas_snippet = """
return atomicCAS(&address[slot], compare, val);
"""


# Register the atomicCAS function
@wp.func_native(atomic_cas_snippet)
def atomicCAS(address: wp.array2d(dtype=int), slot: int, compare: int, val: int) -> int: ...


# Hash function to convert an array of int32 to int32
@wp.func
def _hash_fnv1a_impl(hash_val: int, key: int) -> int:
    """Internal implementation of FNV-1a hash update step"""
    hash_val ^= key
    hash_val *= 16777619
    return hash_val


@wp.func
def hash_fnv1a_array(key: wp.array(dtype=int), capacity: int) -> int:
    """Hash function for array input"""
    hash_val = int(2166136261)
    for i in range(key.shape[0]):
        hash_val = _hash_fnv1a_impl(hash_val, key[i])
    return ((hash_val % capacity) + capacity) % capacity


@wp.func
def hash_fnv1a_vec4i(coord: wp.vec4i, capacity: int) -> int:
    """Hash function for vec4i input"""
    hash_val = int(2166136261)
    for i in range(4):
        hash_val = _hash_fnv1a_impl(hash_val, coord[i])
    return ((hash_val % capacity) + capacity) % capacity


@wp.func
def murmur_32_scramble(k: int) -> int:
    k *= 0xCC9E2D51
    k = (k << 15) | (k >> 17)
    k *= 0x1B873593
    return k


@wp.func
def _hash_murmur_impl(h: int, k: int) -> int:
    """Internal implementation of Murmur hash update step"""
    h ^= murmur_32_scramble(k)
    h = (h << 13) | (h >> 19)
    h = h * 5 + 0xE6546B64
    return h


@wp.func
def _hash_murmur_finalize(h: int, length: int) -> int:
    """Finalize step for Murmur hash"""
    h ^= length  # Length of the key in bytes
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


@wp.func
def hash_murmur_array(key: wp.array(dtype=int), capacity: int) -> int:
    """Murmur hash function for array input"""
    h = int(0x9747B28C)

    # Process each of the integers in the key
    for i in range(key.shape[0]):
        h = _hash_murmur_impl(h, key[i])

    # Finalize
    h = _hash_murmur_finalize(h, 16)  # 16 = 4 ints * 4 bytes each
    return ((h % capacity) + capacity) % capacity


@wp.func
def _hash_city_impl(hash_val: int, key: int) -> int:
    """Internal implementation of City hash update step"""
    hash_val += key * 0x9E3779B9
    hash_val ^= hash_val >> 16
    hash_val *= 0x85EBCA6B
    hash_val ^= hash_val >> 13
    hash_val *= 0xC2B2AE35
    hash_val ^= hash_val >> 16
    return hash_val


@wp.func
def hash_city_array(key: wp.array(dtype=int), capacity: int) -> int:
    """City hash function for array input"""
    hash_val = int(0)
    for i in range(key.shape[0]):
        hash_val = _hash_city_impl(hash_val, key[i])
    return (hash_val % capacity + capacity) % capacity


@wp.func
def hash_city_vec4i(coord: wp.vec4i, capacity: int) -> int:
    """City hash function for vec4i input"""
    hash_val = int(0)
    for i in range(4):
        hash_val = _hash_city_impl(hash_val, coord[i])
    return (hash_val % capacity + capacity) % capacity


@wp.func
def hash_murmur_vec4i(coord: wp.vec4i, capacity: int) -> int:
    """Murmur hash function for vec4i input"""
    h = int(0x9747B28C)
    for i in range(4):
        h = _hash_murmur_impl(h, coord[i])

    # Finalize
    h = _hash_murmur_finalize(h, 16)  # 16 = 4 ints * 4 bytes each
    return ((h % capacity) + capacity) % capacity


@wp.func
def hash_selection(hash_method: int, key: wp.array(dtype=int), capacity: int) -> int:  # noqa: F811
    """Hash selection for array input"""
    if hash_method == 0:
        return hash_fnv1a_array(key, capacity)
    elif hash_method == 1:
        return hash_city_array(key, capacity)
    elif hash_method == 2:
        return hash_murmur_array(key, capacity)
    else:
        return hash_fnv1a_array(key, capacity)


@wp.func
def hash_selection(hash_method: int, coord: wp.vec4i, capacity: int) -> int:  # noqa: F811
    """Hash selection for vec4i input"""
    if hash_method == 0:
        return hash_fnv1a_vec4i(coord, capacity)
    elif hash_method == 1:
        return hash_city_vec4i(coord, capacity)
    elif hash_method == 2:
        return hash_murmur_vec4i(coord, capacity)
    else:
        return hash_fnv1a_vec4i(coord, capacity)


@wp.func
def vec_equal(a: wp.array(dtype=int), b: wp.array(dtype=int)) -> bool:  # noqa: F811
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True


@wp.func
def vec_equal(vec: wp.array(dtype=int), coord: wp.vec4i) -> bool:  # noqa: F811
    """Compare a vector with a vec4i coordinate"""
    return vec[0] == coord[0] and vec[1] == coord[1] and vec[2] == coord[2] and vec[3] == coord[3]


@wp.struct
class HashStruct:
    """Internal structure for hash table implementation.

    Attributes:
        table_kvs: 2D array storing key-value pairs
        vector_keys: 2D array storing the original vector keys
        capacity: Maximum capacity of the hash table
        hash_method: Integer representing the hash method used
    """

    table_kvs: wp.array2d(dtype=int)
    vector_keys: wp.array2d(dtype=int)
    capacity: int
    hash_method: int

    def insert(self, vec_keys: wp.array2d(dtype=int)):
        """Insert vector keys into the hash table.

        Args:
            vec_keys: 2D array of vector keys to insert

        Raises:
            AssertionError: If capacity is invalid or number of keys exceeds capacity/2
        """
        assert self.capacity > 0
        assert self.hash_method in [0, 1, 2]
        assert (
            len(vec_keys) <= self.capacity / 2
        ), f"Number of keys {len(vec_keys)} exceeds capacity {self.capacity}"

        device = vec_keys.device
        self.table_kvs = wp.zeros((self.capacity, 2), dtype=int, device=device)
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

    def search(self, search_keys: wp.array2d(dtype=int)) -> wp.array(dtype=int):
        """Search for keys in the hash table.

        Args:
            search_keys: 2D array of keys to search for

        Returns:
            wp.array: Array of indices where keys were found (-1 if not found)
        """
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
        table_kv_np = self.table_kvs.numpy()
        vec_keys_np = self.vector_keys.numpy()
        return {
            "table_kvs": table_kv_np,
            "vec_keys": vec_keys_np,
            "hash_method": self.hash_method,
            "capacity": self.capacity,
        }

    def from_dict(self, data: dict, device: str):
        self.table_kvs = wp.array(data["table_kvs"], dtype=int, device=device)
        self.vector_keys = wp.array(data["vec_keys"], dtype=int, device=device)
        self.capacity = data["capacity"]
        self.hash_method = data["hash_method"]
        return self


# Warp kernel for inserting into the hashmap
@wp.kernel
def insert_kernel(
    table_kvs: wp.array2d(dtype=int),
    vec_keys: wp.array2d(dtype=int),
    table_capacity: int,
    hash_method: int,
):
    idx = wp.tid()
    slot = hash_selection(hash_method, vec_keys[idx], table_capacity)
    initial_slot = slot
    while True:
        prev = atomicCAS(table_kvs, 2 * slot + 0, -1, slot)
        # Insertion successful.
        if prev == -1:
            table_kvs[slot, 1] = idx
            return
        slot = (slot + 1) % table_capacity

        # If we circle back to the initial slot, the table is full
        if slot == initial_slot:
            return  # This indicates that the table is full and we couldn't insert the unique item


@wp.func
def search_func(  # noqa: F811
    table_kvs: wp.array2d(dtype=int),
    vec_keys: wp.array2d(dtype=int),
    query_key: wp.array(dtype=int),
    table_capacity: int,
    hash_method: int,
) -> int:
    slot = hash_selection(hash_method, query_key, table_capacity)
    initial_slot = slot
    while True:
        current_key = table_kvs[slot, 0]
        if current_key == -1:
            return -1
        else:
            vec_val = table_kvs[slot, 1]
            if vec_equal(vec_keys[vec_val], query_key):
                return vec_val
        slot = (slot + 1) % table_capacity
        if slot == initial_slot:
            return -1


@wp.func
def search_func(  # noqa: F811
    table_kvs: wp.array2d(dtype=int),
    vec_keys: wp.array2d(dtype=int),
    coord: wp.vec4i,
    table_capacity: int,
    hash_method: int,
) -> int:
    slot = hash_selection(hash_method, coord, table_capacity)
    initial_slot = slot
    while True:
        current_key = table_kvs[slot, 0]
        if current_key == -1:
            return -1
        else:
            vec_val = table_kvs[slot, 1]
            if vec_equal(vec_keys[vec_val], coord):
                return vec_val
        slot = (slot + 1) % table_capacity
        if slot == initial_slot:
            return -1


# Warp kernel for searching in the hashmap
@wp.kernel
def search_kernel(
    hash_struct: HashStruct,
    search_keys: wp.array2d(dtype=int),
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
def prepare_key_value_pairs(table_kv: wp.array2d(dtype=int)):
    tid = wp.tid()
    table_kv[tid, 0] = -1
    table_kv[tid, 1] = -1


class VectorHashTable:
    """A hash table implementation for efficient vector key storage and lookup.

    This class provides a hash table optimized for vector keys, supporting multiple
    hash methods and efficient search operations.

    Attributes:
        capacity: Maximum number of entries the hash table can store
        hash_method: Hash method used for key hashing (FNV1A, CITY, or MURMUR)
        device: Device where the hash table is stored
    """

    _hash_struct: HashStruct = None

    def __init__(self, capacity: int, hash_method: HashMethod = HashMethod.CITY):
        """Initialize the vector hash table.

        Args:
            capacity: Maximum number of entries the table can store
            hash_method: Hash method to use (default: CITY)
        """
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

    def insert(self, vec_keys: wp.array2d):
        self._hash_struct.insert(vec_keys)

    @classmethod
    def from_keys(cls, vec_keys: Union[wp.array2d, Tensor]):
        """Create a hash table from a set of vector keys.

        Args:
            vec_keys: Vector keys to initialize the table with (warp array or torch tensor)

        Returns:
            VectorHashTable: New hash table instance containing the keys
        """
        if isinstance(vec_keys, torch.Tensor):
            vec_keys = wp.from_torch(vec_keys)
        obj = cls(2 * len(vec_keys))
        obj.insert(vec_keys)
        return obj

    def search(self, search_keys: Union[wp.array2d, Tensor]) -> wp.array:
        """Search for keys in the hash table.

        Args:
            search_keys: Keys to search for (warp array or torch tensor)

        Returns:
            wp.array: Array of indices where keys were found (-1 if not found)
        """
        if isinstance(search_keys, torch.Tensor):
            search_keys = wp.from_torch(search_keys)
        return self._hash_struct.search(search_keys)

    def unique_index(self) -> Int[Tensor, "N"]:  # noqa: F821
        """Get sorted unique indices from the hash table.

        Returns:
            Tensor: Sorted tensor of unique indices
        """
        indices = self.search(self._hash_struct.vector_keys)
        return torch.unique(wp.to_torch(indices))

    def to_dict(self):
        return self._hash_struct.to_dict()

    def hashmap_struct(self) -> HashStruct:
        return self._hash_struct
