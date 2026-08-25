# PyTorch GPU Setup for RTX 4060 on Windows

## Problem
PyTorch 2.13+ doesn't support Python 3.14. The default `.venv` uses Python 3.14.7 which only has CPU wheels.

## Solution: Use Python 3.11

### 1. Install Python 3.11
```powershell
winget install Python.Python.3.11
```

### 2. Create Virtual Environment
```powershell
py -3.11 -m venv .venv
```

### 3. Install PyTorch with CUDA 12.1
```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify GPU Works
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
GPU: True
Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

### 5. Use in Scripts
```powershell
.\.venv\Scripts\python.exe your_script.py
```

### Activate Environment (Optional)
```powershell
$env:VIRTUAL_ENV = "D:\forex-main\.venv"
$env:PATH = "D:\forex-main\.venv\Scripts;" + $env:PATH
```

## Why This Works
- Python 3.11 has full PyTorch CUDA wheel support
- CUDA 12.1 is compatible with RTX 4060 (driver 560.76+)
- PyTorch 2.5.1+cu121 is the latest stable with CUDA 12.1

## Troubleshooting
- If `py -3.11` not found: Install Python 3.11 from python.org or winget
- If CUDA errors: Ensure NVIDIA driver 560+ is installed
- If DLL errors: Add CUDA bin to PATH: `$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"`