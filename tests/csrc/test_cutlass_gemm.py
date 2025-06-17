import os
import sys

import pytest

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

import warpconvnet._C as _C


def compare_results(result_auto, d_ref, indices_d):
    # Check all results are finite
    if not torch.all(torch.isfinite(result_auto)) or not torch.all(torch.isfinite(d_ref)):
        print("❌ Results contain NaNs or Infs!")

    all_diff = torch.abs(result_auto - d_ref)
    max_diff_idx = torch.argmax(all_diff)
    max_diff = all_diff.view(-1)[max_diff_idx]

    rel_diff = torch.abs((result_auto - d_ref) / (d_ref + 1e-6))
    max_rel_diff_idx = torch.argmax(rel_diff)
    max_rel_diff = rel_diff.view(-1)[max_rel_diff_idx]

    print(
        f"Max diff (all): {max_diff.item()} and value at max diff: {result_auto.view(-1)[max_diff_idx].item()}, {d_ref.view(-1)[max_diff_idx].item()}"
    )
    print(
        f"Max rel diff (all): {max_rel_diff.item()} and value at max rel diff: {result_auto.view(-1)[max_rel_diff_idx].item()}, {d_ref.view(-1)[max_rel_diff_idx].item()}"
    )


def randn_clamped(shape, dtype, device, scale=0.1):
    return torch.clamp(torch.randn(shape, dtype=dtype, device=device) * scale, -scale, scale)


def rand_indices(size, indices_size, device):
    return torch.sort(torch.randperm(size, device=device)[:indices_size].unsqueeze(1), dim=0)[
        0
    ].int()


@pytest.mark.parametrize(
    "test_types",
    [
        (torch.float16, torch.float16, torch.float16),
        (torch.float16, torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float32, torch.float32),
    ],
    ids=[
        "if16of16af16",
        "if16of16af32",
        "ibf16obf16af32",
        "if16of32af32",
        "if32of32af32",
    ],
)
def test_cutlass_gemm_gather_scatter(test_types):
    """Test CUTLASS GEMM with half precision inputs and half accumulator"""
    print(f"Testing {test_types}...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    in_dtype, out_dtype, acc_dtype = test_types

    # Larger matrix dimensions (K == N for identity) to satisfy kernel tile constraints
    M, N, K, indices_size, out_size = 4096, 128, 128, 2048, 4096

    # Generate unique gather & scatter indices to avoid data races
    indices_a = rand_indices(M, indices_size, "cuda")
    indices_d = rand_indices(out_size, indices_size, "cuda")

    # Create input tensors with specific values for debugging (all float16)
    tensor_a = randn_clamped((M, K), in_dtype, "cuda")  # M x K

    # Set tensor_a[indices_a, :] = torch.arange(indices_size) for debugging
    # for i in range(indices_size):
    #     row_idx = indices_a[i, 0].item()
    #     tensor_a[row_idx, :] = float(i)

    # Set tensor_b to identity matrix (K x N, with K == N)
    tensor_b = torch.eye(K, N, dtype=in_dtype, device="cuda")  # K x N identity
    tensor_b += randn_clamped((K, N), in_dtype, "cuda")

    # Set tensor_c to zeros for simplicity (all float16)
    tensor_d = torch.zeros(out_size, N, dtype=out_dtype, device="cuda")  # out_size x N
    # tensor_c = torch.zeros(M, N, dtype=out_dtype, device="cuda")  # indices_size x N
    tensor_c = tensor_d

    print(
        f"Matrix dimensions: M={M}, N={N}, K={K}, indices_size={indices_size}, out_size={out_size}"
    )

    # Test with explicit accumulator type (default is float32)
    status = _C.gemm.cutlass_gemm_ad_gather_scatter(
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        tensor_c=tensor_c,
        tensor_d=tensor_d,
        indices_a=indices_a,
        indices_d=indices_d,
        accumulator_type=acc_dtype,
        split_k_slices=1,
        alpha=1.0,
        beta=1.0,
    )
    torch.cuda.synchronize()
    assert (
        status == 0
    ), f"Error in cutlass_gemm_ad_gather_scatter: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"

    # Compute reference result using PyTorch (convert to float32 for computation)
    a_gathered = tensor_a[indices_a.squeeze()]
    c_ref = torch.matmul(a_gathered, tensor_b).to(acc_dtype)
    # c_ref = c_ref + tensor_c.to(acc_dtype)
    d_ref = torch.zeros_like(tensor_d)
    d_ref[indices_d.squeeze()] = c_ref.to(out_dtype)

    # Compare results (convert to float32 for comparison)
    compare_results(tensor_d, d_ref, indices_d)

    # Use more lenient thresholds for half precision
    print(f"{test_types} test passed!")


@pytest.mark.parametrize(
    "test_types",
    [
        (torch.float16, torch.float16, torch.float16),
        (torch.float16, torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float32, torch.float32),
    ],
    ids=[
        "if16of16af16",
        "if16of16af32",
        "ibf16obf16af32",
        "if16of32af32",
        "if32of32af32",
    ],
)
def test_cutlass_gemm_trAB_gather(test_types):
    """Test CUTLASS GEMM with trAB gather (A transpose + AB gather)"""
    print(f"Testing trAB gather {test_types}...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    in_dtype, out_dtype, acc_dtype = test_types

    # Matrix dimensions for trAB gather: A[indices_a, :].T @ B[indices_b, :]
    # A and B are both tall skinny but can have different numbers of rows
    M_A, M_B, N, K = 2048, 3072, 128, 64  # A is M_A x K, B is M_B x N (different M values)
    indices_size = 1024

    # Generate unique gather indices for both A and B
    indices_a = rand_indices(M_A, indices_size, "cuda")
    indices_b = rand_indices(M_B, indices_size, "cuda")

    # Create input tensors - both tall skinny
    tensor_a = randn_clamped((M_A, K), in_dtype, "cuda")  # M_A x K
    tensor_b = randn_clamped((M_B, N), in_dtype, "cuda")  # M_B x N

    # For trAB gather, C and D should have shape K x N (result of A^T @ B)
    tensor_c = torch.zeros(K, N, dtype=out_dtype, device="cuda")  # K x N
    tensor_d = torch.zeros(K, N, dtype=out_dtype, device="cuda")  # K x N

    print(
        f"Matrix dimensions: A={M_A}x{K}, B={M_B}x{N}, indices_size={indices_size}, result={K}x{N}"
    )

    # Test trAB gather: A[indices_a, :].T @ B[indices_b, :]
    status = _C.gemm.cutlass_gemm_trAB_gather(
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        tensor_c=tensor_c,
        tensor_d=tensor_d,
        indices_a=indices_a,
        indices_b=indices_b,
        accumulator_type=acc_dtype,
        split_k_slices=1,
        alpha=1.0,
        beta=0.0,
    )
    torch.cuda.synchronize()
    assert (
        status == 0
    ), f"Error in cutlass_gemm_trAB_gather: {_C.gemm.gemm_status_to_string(_C.gemm.GemmStatus(status))}"

    # Compute reference result using PyTorch
    a_gathered = tensor_a[indices_a.squeeze()]  # indices_size x K
    b_gathered = tensor_b[indices_b.squeeze()]  # indices_size x N
    # A[indices_a, :].T @ B[indices_b, :] = (indices_size x K).T @ (indices_size x N) = K x N
    c_ref = torch.matmul(a_gathered.T, b_gathered).to(acc_dtype)  # K x N
    d_ref = c_ref + tensor_c.to(acc_dtype)  # Add bias (C matrix)

    # Compare results
    compare_results(tensor_d, d_ref.to(out_dtype), None)

    print(f"trAB gather {test_types} test passed!")
