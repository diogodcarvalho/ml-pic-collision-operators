import torch
import sys


def verify_installation():
    print(f"--- Environment Check for {sys.prefix} ---")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    # Check for CUDA (GPU) support
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "Warning: Torch is running on CPU. Check your --index-url during install."
        )


if __name__ == "__main__":
    verify_installation()
