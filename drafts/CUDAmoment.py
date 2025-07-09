import torch

print("=== PyTorch CUDA проверка ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Device capabilities: {torch.cuda.get_device_capability(0)}")
print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

import cv2

print("\n=== OpenCV CUDA проверка ===")
print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f"CUDA devices (OpenCV): {cv2.cuda.printCudaDeviceInfo(0)}")

import torch, torchvision
print(torch.cuda.is_available())  # Должно быть True
print(torchvision.ops.nms.__module__)  # Покажет, откуда импортируется NMS
    #conda install -c conda-forge opencv=*=*cuda128*  # Или cuda118, cuda128
    

print('сосать')
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0)}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")

print(f"OpenCV version: {cv2.__version__}")
print(f"OpenCV CUDA enabled: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
