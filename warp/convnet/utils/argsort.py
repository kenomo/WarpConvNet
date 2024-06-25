import warp as wp


@wp.kernel
def prepare_key_value_pairs(
    data: wp.array(dtype=int), keys: wp.array(dtype=int), values: wp.array(dtype=int)
):
    tid = wp.tid()
    keys[tid] = data[tid]
    values[tid] = tid


def argsort(data: wp.array(dtype=int), device: str) -> wp.array(dtype=int):
    N = len(data)
    keys, values = wp.empty(N, dtype=int, device=device), wp.empty(N, dtype=int, device=device)

    # Prepare key-value pairs
    wp.launch(kernel=prepare_key_value_pairs, dim=N, inputs=[data, keys, values], device=device)
    wp.utils.radix_sort_pairs(keys, values)
    return values
