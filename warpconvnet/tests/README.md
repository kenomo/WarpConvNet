# All Tests

## Running single GPU tests

```bash
python -m unittest warpconvnet.tests.test_all_single_gpu
```

## Running FSDP

Requires multi GPU

```bash
torchrun --nproc_per_node=2 warp/convnet/tests/test_fsdp.py
```