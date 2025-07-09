import torch

# Проверь, доступен ли CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU
else:
    device = torch.device("cpu")   # CPU

# Создай тензор на GPU
tensor = torch.randn(10, 10).to(device)

# Создай модель на GPU
#model = YourModel().to(device)
print(tensor.device)  # Должно вывести "cuda:0" (или что-то подобное)
    
   
