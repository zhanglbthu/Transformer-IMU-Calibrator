import numpy as np
import time
from torch.utils.data import Dataset
import torch

def random_index(data_len:int, sampling_rate=1.0, seed:int=None) -> list:
    """
    随机采样索引的函数

    参数:
    sample_size (int): 样本数量
    sampling_rate (float): 采样率，范围在(0, 1]
    seed (int or None): 随机数生成的种子，如果为None则不设置种子

    返回:
    list: 随机采样的索引列表
    """
    if not (0 < sampling_rate <= 1):
        raise ValueError("采样率必须在(0, 1]范围内")

    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))

    # 计算需要采样的样本数量
    num_samples_to_select = int(data_len * sampling_rate)

    # 生成样本索引的随机排列
    all_indices = np.arange(data_len)
    np.random.shuffle(all_indices)

    # 从随机排列的索引中选择需要的数量
    selected_indices = all_indices[:num_samples_to_select]

    return selected_indices

def data_shuffle(*args, seed=None):
    """
    Shuffle data at dim 0.
    Args:
        *args: Data with same length. Can be [torch.Tensor] or [ndarray].
        seed: Random seed, if None, using current time.

    Returns:
        Shuffled data. When multiple data was given as input, it will be tuple.

    """
    data = list(args)
    rand_index = random_index(data_len=len(data[0]), seed=seed)
    for i, d in enumerate(data):
        data[i] = data[i][rand_index]
    data = tuple(data)
    return data


class BaseDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.data_len = len(x)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        return self.x[i], self.y[i]
    @staticmethod
    def load_data(path:str) -> dict :
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files

        Returns:
            Dict of datas.
        """
        pass



