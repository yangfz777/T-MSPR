import pickle
import numpy as np
import matplotlib.pyplot as plt

# 指定文件路径
file_path = "/opt/data/private/yfz/PADE-main/data/output_features_epoch_120.pkl"

try:
    # 打开并加载 pickle 文件
    with open(file_path, "rb") as f:
        epoch_features = pickle.load(f)

    # 打印数据类型
    print(f"数据类型: {type(epoch_features)}")
except FileNotFoundError:
    print(f"文件未找到：{file_path}")
except pickle.UnpicklingError:
    print(f"文件格式错误，无法反序列化：{file_path}")
except Exception as e:
    print(f"读取文件时发生错误：{e}")