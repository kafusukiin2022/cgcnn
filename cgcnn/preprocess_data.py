import os
import csv
import json
import warnings
import argparse
from tqdm import tqdm

import numpy as np
import torch
from pymatgen.core.structure import Structure

# --- 从 data.py 复制过来的辅助类 ---
# 我们将这些类复制过来，让这个预处理脚本可以独立运行，不依赖其他文件。

class GaussianDistance(object):
    """
    通过高斯基函数扩展距离。
    单位: 埃 (angstrom)
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        对距离数组应用高斯滤波器。
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomCustomJSONInitializer(object):
    """
    使用JSON文件初始化原子特征向量。
    JSON文件是一个字典，将元素序数映射到其特征向量列表。
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        self.atom_types = set(elem_embedding.keys())
        self._embedding = {}
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

# --- 核心预处理函数 ---

def preprocess_data(root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2):
    """
    遍历原始数据集目录，将每个CIF文件转换为PyTorch张量并保存。

    参数:
    ----------
    root_dir: str
        原始数据集的根目录路径。
        (应包含 id_prop.csv, atom_init.json, 和所有 .cif 文件)
    max_num_nbr: int
        构建晶体图时的最大邻居数。
    radius: float
        搜索邻居的截止半径。
    dmin: float
        构建GaussianDistance的最小距离。
    step: float
        构建GaussianDistance的步长。
    """
    # 1. 检查路径和文件是否存在
    assert os.path.exists(root_dir), f"错误: 根目录 '{root_dir}' 不存在！"
    id_prop_file = os.path.join(root_dir, 'id_prop.csv')
    assert os.path.exists(id_prop_file), f"错误: '{id_prop_file}' 不存在！"
    atom_init_file = os.path.join(root_dir, 'atom_init.json')
    assert os.path.exists(atom_init_file), f"错误: '{atom_init_file}' 不存在！"

    # 2. 创建用于存放预处理后数据的目录
    processed_dir = os.path.join(root_dir, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"创建目录 '{processed_dir}' 用于存放预处理数据。")

    # 3. 初始化原子特征和距离扩展工具
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    # 4. 读取 id_prop.csv 文件
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    # 5. 遍历所有CIF文件并进行处理
    print("开始预处理CIF文件...")
    for cif_id, target in tqdm(id_prop_data):
        cif_path = os.path.join(root_dir, cif_id + '.cif')
        if not os.path.exists(cif_path):
            warnings.warn(f"警告: 文件 '{cif_path}' 未找到，跳过。")
            continue

        # --- 这部分逻辑与你原始的 __getitem__ 完全相同 ---
        crystal = Structure.from_file(cif_path)
        
        # 获取原子特征
        atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])

        # 获取邻居信息
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                warnings.warn(f'ID {cif_id} 未找到足够的邻居来构建图。'
                              '如果此警告频繁出现，请考虑增大切割半径(radius)。')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)

        # 转换为 PyTorch 张量
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target_tensor = torch.Tensor([float(target)])

        # 6. 将处理好的数据保存为 .pt 文件
        save_path = os.path.join(processed_dir, f'{cif_id}.pt')
        torch.save(((atom_fea, nbr_fea, nbr_fea_idx), target_tensor, cif_id), save_path)

    print("="*50)
    print(f"🎉 预处理完成！所有数据已保存到 '{processed_dir}' 目录下。")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIF文件预处理脚本')
    parser.add_argument('root_dir', type=str,
                        help='包含cif文件、id_prop.csv和atom_init.json的根目录路径。')
    parser.add_argument('--max-num-nbr', type=int, default=12,
                        help='构建晶体图时的最大邻居数。')
    parser.add_argument('--radius', type=float, default=8,
                        help='搜索邻居的截止半径。')
    parser.add_argument('--dmin', type=float, default=0,
                        help='构建GaussianDistance的最小距离。')
    parser.add_argument('--step', type=float, default=0.2,
                        help='构建GaussianDistance的步长。')

    args = parser.parse_args()

    preprocess_data(
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step
    )

