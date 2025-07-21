from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
# from pymatgen.core.structure import Structure # 不再需要，因为我们不再实时解析CIF
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    用于将数据集划分为训练、验证和测试集的实用函数。

    !!! 在使用此函数之前，数据集需要被打乱 !!!

    参数
    ----------
    dataset: torch.utils.data.Dataset
        要划分的完整数据集。
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
        是否返回测试数据集加载器。如果为False，最后的test_size数据将被隐藏。
    num_workers: int
    pin_memory: bool

    返回
    -------
    train_loader: torch.utils.data.DataLoader
        随机采样训练数据的数据加载器。
    val_loader: torch.utils.data.DataLoader
        随机采样验证数据的数据加载器。
    (test_loader): torch.utils.data.DataLoader
        随机采样测试数据的数据加载器，如果return_test=True则返回。
    """
    total_size = len(dataset)
    if kwargs.get('train_size') is None: # 使用 .get() 避免 KeyError
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    
    indices = list(range(total_size))
    
    if kwargs.get('train_size'):
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    
    if kwargs.get('test_size'):
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
        
    if kwargs.get('val_size'):
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    # 确保在切分后测试集不为空
    if test_size == 0 and return_test:
        warnings.warn("Test size is zero. Test loader will be empty.")
    
    # 调整验证集和训练集的索引以避免重叠和原始代码中的bug
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:train_size + valid_size])
    
    if return_test:
        test_sampler = SubsetRandomSampler(indices[train_size + valid_size:])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                   sampler=test_sampler,
                                   num_workers=num_workers,
                                   collate_fn=collate_fn, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    整理数据列表并返回一个用于预测晶体属性的批次。

    参数
    ----------
    dataset_list: list of tuples for each data point.
        (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        cif_id: str or int

    返回
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        来自原子类型的原子特征
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        每个原子的M个邻居的键特征
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        每个原子的M个邻居的索引
    crystal_atom_idx: list of torch.LongTensor of length N0
        从晶体索引到原子索引的映射
    target: torch.Tensor shape (N, 1)
        用于预测的目标值
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
        torch.stack(batch_target, dim=0), \
        batch_cif_ids


class GaussianDistance(object):
    """
    通过高斯基函数扩展距离。
    单位: 埃 (angstrom)
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        参数
        ----------
        dmin: float
            最小原子间距离
        dmax: float
            最大原子间距离
        step: float
            高斯滤波器的步长
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        对numpy距离数组应用高斯距离滤波器。
        参数
        ----------
        distance: np.array shape n-d array
            任何形状的距离矩阵
        返回
        -------
        expanded_distance: shape (n+1)-d array
            扩展后的距离矩阵，最后一个维度长度为len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    用于初始化原子向量表示的基类。
    !!! 每个数据集使用一个AtomInitializer !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    使用JSON文件初始化原子特征向量，该文件是一个python字典，
    将元素序数映射到表示元素特征向量的列表。
    参数
    ----------
    elem_embedding_file: str
        .json文件的路径
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


# --- 核心修改部分 ---

class CIFData(Dataset):
    """
    修改后的CIFData数据集。
    它现在直接从预处理好的 .pt 文件中加载数据，速度极快。
    """
    def __init__(self, root_dir, random_seed=123):
        """
        参数:
        ----------
        root_dir: str
            原始数据集的根目录路径。
            这个目录需要包含 'id_prop.csv' 和一个名为 'processed' 的子目录。
        random_seed: int
            用于打乱数据集的随机种子。
        """
        self.root_dir = root_dir
        self.processed_dir = os.path.join(self.root_dir, 'processed')
        
        # 检查必需的文件和目录
        assert os.path.exists(self.root_dir), f"根目录 '{self.root_dir}' 不存在！"
        assert os.path.exists(self.processed_dir), (
            f"预处理数据目录 '{self.processed_dir}' 不存在！\n"
            f"请先运行 preprocess_data.py 脚本。"
        )
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), f"'id_prop.csv' 在根目录中未找到！"
        
        # 读取 id 和 target，这决定了数据集的大小和顺序
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        
        # 打乱数据集
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        """
        这个方法现在变得极其简单和快速。
        它不再解析CIF文件，而是直接从磁盘加载一个预处理好的.pt文件。
        """
        cif_id, _ = self.id_prop_data[idx]
        
        # 直接加载预处理好的张量
        # torch.load返回在预处理脚本中保存的元组：
        # ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
        data = torch.load(os.path.join(self.processed_dir, cif_id + '.pt'))
        
        return data

