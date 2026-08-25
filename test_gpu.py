import torch

print(f'GPU Available: {torch.cuda.is_available()}')
print(f'Version: {torch.__version__}')
print(f'CUDA Version: {torch.version.cuda}')

if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device Count: {torch.cuda.device_count()}')
else:
    print('Device: CPU')