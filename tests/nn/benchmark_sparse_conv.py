import pytest
import torch
import warp as wp
from typing import Tuple
from contextlib import nullcontext

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import (
    SpatiallySparseConv,
    SPATIALLY_SPARSE_CONV_ALGO_MODE,
    STRIDED_CONV_MODE,
)


def pytest_benchmark_generate_json(config, benchmarks, output_json):
    """Customize benchmark JSON output."""
    # Filter out unwanted stats
    for benchmark in benchmarks:
        benchmark.stats = {"min": benchmark.stats["min"]}


def pytest_addoption(parser):
    """Add custom options to pytest."""
    parser.addoption(
        "--benchmark-only-min",
        action="store_true",
        default=True,
        help="Only show minimum time in benchmark results",
    )


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Update benchmark JSON format."""
    for benchmark in output_json["benchmarks"]:
        benchmark["stats"] = {"min": benchmark["stats"]["min"]}


@pytest.fixture
def setup_voxel_data():
    """Setup fixed coordinate and feature data for benchmarking."""
    wp.init()
    torch.manual_seed(0)
    device = "cuda:0"

    # Fixed configuration for data generation
    B, min_N, max_N = 3, 100000, 1000000
    base_channels = 32  # Use maximum channel size for feature generation

    # Generate fixed coordinates and features
    Ns = torch.randint(min_N, max_N, (B,))
    voxel_size = 0.01
    # default dtype is float32
    coords = [(torch.rand((N, 3), device=device) / voxel_size).int() for N in Ns]
    features = [torch.randn((N, base_channels), device=device) for N in Ns]

    return coords, features


def _gen_dtype_params(C_in=32, C_out=64, algo=SPATIALLY_SPARSE_CONV_ALGO_MODE.EXPLICIT_GEMM):
    """Generate dtype test parameters."""
    base_config = (C_in, C_out)
    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "none": None,
    }

    params = []
    # Format: (in_dtype, compute_dtype) pairs
    configs = [
        ("fp32", "none"),
        ("fp32", "fp32"),
        ("fp16", "fp32"),
        ("fp16", "fp16"),
        ("fp16", "bf16"),
        ("fp16", "none"),
        ("bf16", "fp16"),
        ("bf16", "bf16"),
        ("bf16", "none"),
        ("none", "none"),
    ]

    for dtype_key, compute_key in configs:
        params.append(
            pytest.param(
                (*base_config, dtypes[dtype_key], dtypes[compute_key], algo),
                id=f"{dtype_key}_{compute_key}",
            )
        )

    return params


@pytest.fixture(params=_gen_dtype_params())
def setup_sparse_conv(
    request, setup_voxel_data
) -> Tuple[SpatiallySparseConv, Voxels, torch.dtype]:
    """Setup sparse convolution with different configurations."""
    C_in, C_out, dtype, compute_dtype, algo_mode = request.param
    device = "cuda:0"

    coords, base_features = setup_voxel_data

    # Slice features if needed to match in_channels
    features = [f[:, :C_in] for f in base_features]
    input_voxels = Voxels(coords, features, device=device).unique()

    # Create conv layer with various configurations
    conv = SpatiallySparseConv(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        stride=(2, 2, 2),
        conv_algo=algo_mode,
        stride_mode=STRIDED_CONV_MODE.REDUCE_AND_STRIDE,
        out_code_backend="morton",
        compute_dtype=compute_dtype,
    ).to(device)

    return conv, input_voxels, dtype


@pytest.mark.benchmark(group="sparse_conv_forward")
def test_forward_sparse_conv(setup_sparse_conv, benchmark):
    """Benchmark forward pass of sparse convolution."""
    conv, input_voxels, test_dtype = setup_sparse_conv
    if test_dtype is not None:
        input_voxels = input_voxels.to(dtype=test_dtype)

    use_amp = test_dtype in [torch.float16, torch.bfloat16]

    def run_forward():
        with torch.no_grad():
            # Use autocast for FP16, nullcontext for FP32
            ctx = torch.cuda.amp.autocast(dtype=test_dtype) if use_amp else nullcontext()
            with ctx:
                return conv(input_voxels)

    result = benchmark.pedantic(
        run_forward,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="sparse_conv_backward")
def test_forward_backward_sparse_conv(setup_sparse_conv, benchmark):
    """Benchmark forward + backward pass of sparse convolution."""
    conv, input_voxels, test_dtype = setup_sparse_conv
    if test_dtype is not None:
        input_voxels = input_voxels.to(dtype=test_dtype)

    # Create GradScaler for FP16
    use_amp = test_dtype in [torch.float16, torch.bfloat16]
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    optimizer = torch.optim.Adam(conv.parameters())

    def run_forward_backward():
        # Use autocast for FP16, nullcontext for FP32
        ctx = torch.cuda.amp.autocast(dtype=test_dtype) if use_amp else nullcontext()
        with ctx:
            output = conv(input_voxels)
            loss = output.features.mean()

        if scaler is not None:
            # FP16 training path
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 training path
            loss.backward()
            optimizer.step()

        return output

    # Reset gradients before each benchmark iteration
    def run_with_reset():
        optimizer.zero_grad(set_to_none=True)
        return run_forward_backward()

    result = benchmark.pedantic(
        run_with_reset,
        iterations=10,
        rounds=3,
        warmup_rounds=1,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-columns=min,median"])
