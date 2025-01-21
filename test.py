import torch

# 定义 WSL 中的文件路径
file_path = '/mnt/d/AI/EFM/OAReactDiff-EFM-now - 12_08/mistake_batch_val/debug_batch_4.pt'

# 尝试加载文件
try:
    data = torch.load(file_path)
    # 判断数据类型
    if isinstance(data, dict):
        print("Loaded data is a dictionary.")
        print("Keys:", data.keys())
    elif isinstance(data, list):
        print("Loaded data is a list.")
        print("Length:", len(data))
    else:
        print(f"Loaded data is of type: {type(data)}")

except Exception as e:
    print("Error while loading the .pt file:", e)
