# GPU Setup Guide

## Current Status

Your system has:
- **NVIDIA GeForce RTX 4070 Ti SUPER** (16GB VRAM) - Primary GPU for training
- **AMD Radeon 780M Graphics** - Integrated GPU (not used for PyTorch)
- **CUDA Driver Version**: 12.8 (supports CUDA 12.x)

## PyTorch CUDA Installation

The training code is already configured to automatically use GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Installation Steps

1. **Install PyTorch with CUDA support** (choose one):

   **For CUDA 12.1:**
   ```bash
   poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   **For CUDA 11.8:**
   ```bash
   poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Verify GPU detection:**
   ```bash
   poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```

### Troubleshooting DLL Errors

If you encounter `OSError: [WinError 127]` when importing torch:

1. **Check Python Version:**
   - PyTorch requires stable Python versions (not beta/alpha)
   - Current: Python 3.11.0b5 (beta) - may cause compatibility issues
   - **Recommended:** Upgrade to Python 3.11.9 or 3.12.x stable release
   - After upgrading, recreate the virtual environment:
     ```bash
     poetry env remove python
     poetry install
     poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```

2. **Install Visual C++ Redistributables:**
   - Download and install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Install both x64 and x86 versions

3. **Install CUDA Toolkit (if DLL errors persist):**
   - Download [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
   - This provides additional CUDA runtime libraries that PyTorch may need

4. **Verify NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   Should show your RTX 4070 Ti SUPER with driver version

5. **Alternative: Try older PyTorch version (if Python upgrade not possible):**
   ```bash
   poetry run pip uninstall -y torch torchvision
   poetry run pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   ```

## Using GPU in Training

Once PyTorch detects CUDA, training will automatically use GPU. You'll see:
```
Using device: cuda
```

The model, input data, and targets will be automatically moved to GPU during training.

## Monitoring GPU Usage

During training, monitor GPU usage with:
```bash
nvidia-smi -l 1
```

This shows:
- GPU utilization percentage
- Memory usage
- Temperature
- Power consumption

## Performance Tips

1. **Batch size**: Increase `batch_size` in `configs/training_config.yaml` to fully utilize GPU memory (16GB available)
2. **Mixed precision**: Consider enabling automatic mixed precision (AMP) for faster training
3. **Data loading**: Use `num_workers > 0` in DataLoader for parallel data loading (if not already set)
