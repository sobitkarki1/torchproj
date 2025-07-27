import torch
import platform
import os
import psutil
import shutil

def bytes_to_gb(x): return round(x / (1024 ** 3), 2)

# CPU Info
print("\n[CPU]")
print("Processor:", platform.processor())
print("Cores:", psutil.cpu_count(logical=False), "Logical:", psutil.cpu_count(logical=True))

# RAM Info
print("\n[RAM]")
mem = psutil.virtual_memory()
print("Total:", bytes_to_gb(mem.total), "GB")

# Disk Info
print("\n[Disk]")
total, used, free = shutil.disk_usage("/")
print("Total:", bytes_to_gb(total), "GB | Free:", bytes_to_gb(free), "GB")

# GPU Info
print("\n[GPU]")
if torch.cuda.is_available():
    print(f"CUDA is available. GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Total: {bytes_to_gb(torch.cuda.get_device_properties(i).total_memory)} GB")
else:
    print("No CUDA GPU detected.")

# Python + Torch
print("\n[Software]")
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
