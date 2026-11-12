"""Lab 00: check that Python, PyTorch and the GPU are ready.

Run from learning/labs with the .venv active:
    python 00-setup/check_env.py
"""

import platform
import shutil
import sys
import time


def row(label, value):
    print(f"{label:<13} {value}")


def is_wsl():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def matmul_tflops(torch, device, n=4096, repeats=10):
    """Time an n x n matrix multiply and return TFLOP/s."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b  # warm up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    flops = 2 * n**3 * repeats  # one multiply-add per term
    return flops / seconds / 1e12


def main():
    problems = []
    row("Python", platform.python_version())
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is needed.")

    row("Running in", "WSL2" if is_wsl() else platform.system())
    row("nvidia-smi", "found" if shutil.which("nvidia-smi") else "missing")

    try:
        import torch
    except ImportError:
        row("PyTorch", "not installed")
        print("\nActivate .venv and install 00-setup/requirements.txt.")
        return 1

    row("PyTorch", torch.__version__.split("+")[0])
    cuda = torch.cuda.is_available()
    row("CUDA ready", "yes" if cuda else "no")

    if cuda:
        props = torch.cuda.get_device_properties(0)
        row("GPU", props.name)
        row("GPU memory", f"{props.total_memory / 1e9:.1f} GB")
        row("bfloat16", "yes" if torch.cuda.is_bf16_supported() else "no")
    else:
        problems.append(
            "PyTorch cannot see a GPU. Update the Windows NVIDIA driver, "
            "run `wsl --update`, and check nvidia-smi works in WSL."
        )

    cpu = matmul_tflops(torch, "cpu", n=2048, repeats=3)
    row("CPU matmul", f"{cpu:.1f} TFLOP/s")
    if cuda:
        gpu = matmul_tflops(torch, "cuda")
        row("GPU matmul", f"{gpu:.1f} TFLOP/s")
        row("GPU speedup", f"about {gpu / cpu:.0f}x")

    print()
    if problems:
        for p in problems:
            print("PROBLEM:", p)
        print("Labs 01 and 04 still work on CPU, more slowly.")
        return 1
    print("All good. On to Lab 01.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
