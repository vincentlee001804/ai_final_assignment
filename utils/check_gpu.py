"""
Script to verify PyTorch is using NVIDIA GPU and show GPU information
"""
import torch

print("=" * 60)
print("PyTorch GPU Information")
print("=" * 60)

print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    print(f"\nNumber of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test actual GPU computation
    print("\n" + "=" * 60)
    print("Testing GPU Computation")
    print("=" * 60)
    
    # Create tensors on GPU
    x = torch.randn(5000, 5000, device='cuda')
    y = torch.randn(5000, 5000, device='cuda')
    
    # Perform computation
    z = torch.matmul(x, y)
    
    print(f"\nTensor device: {z.device}")
    print(f"GPU used: {torch.cuda.get_device_name(z.device)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Verify it's actually on GPU
    if z.device.type == 'cuda':
        print("\n[SUCCESS] PyTorch is using NVIDIA GPU for computations!")
    else:
        print("\n[WARNING] Tensors are not on GPU!")
else:
    print("\n[ERROR] CUDA is not available!")

