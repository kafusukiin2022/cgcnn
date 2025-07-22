import os
import csv
import json
import warnings
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
# 将处理单个CIF文件的逻辑封装成一个函数
def process_single_cif(args):
    cif_id, target, root_dir, max_num_nbr, radius, dmin, step, atom_init_file = args
    
    # 重新初始化这些对象，因为它们不能在进程间直接共享
    # 或者可以将它们作为参数传递，但对于当前结构，重新初始化更简单
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    cif_path = os.path.join(root_dir, cif_id + '.cif')
    processed_dir = os.path.join(root_dir, 'processed')

    if not os.path.exists(cif_path):
        warnings.warn(f"警告: 文件 '{cif_path}' 未找到，跳过。")
        return None # 返回 None 表示处理失败或跳过

    try:
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
                # 原始警告在并行环境中可能不好处理，这里简化
                # warnings.warn(f'ID {cif_id} 未找到足够的邻居来构建图。')
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

        # 保存为 .pt 文件
        save_path = os.path.join(processed_dir, f'{cif_id}.pt')
        torch.save(((atom_fea, nbr_fea, nbr_fea_idx), target_tensor, cif_id), save_path)
        return cif_id # 返回成功处理的ID
    except Exception as e:
        warnings.warn(f"处理文件 '{cif_path}' 时发生错误: {e}")
        return None


def preprocess_data_parallel(root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2):
    """
    遍历原始数据集目录，将每个CIF文件转换为PyTorch张量并保存。
    此版本使用多进程并行处理。

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

    # 3. 读取 id_prop.csv 文件
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    # 4. 准备并行任务的参数列表
    tasks = []
    for cif_id, target in id_prop_data:
        tasks.append((cif_id, target, root_dir, max_num_nbr, radius, dmin, step, atom_init_file))

    # 5. 使用多进程池进行处理
    print(f"开始使用 {cpu_count()} 个进程并行预处理CIF文件...")
    with Pool(cpu_count()) as pool: # 使用所有可用的CPU核心
        # 将 tqdm 包装在 pool.imap 的结果上，而不是 pool.map
        # 这样 tqdm 就会迭代地消费结果，并显示一个统一的进度条
        results = list(tqdm(pool.imap_unordered(process_single_cif, tasks), total=len(tasks),
                            desc="处理CIF文件"))
    
    successful_count = sum(1 for r in results if r is not None)
    print("="*50)
    print(f"🎉 预处理完成！成功处理 {successful_count} 个文件，所有数据已保存到 '{processed_dir}' 目录下。")
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

    # 调用并行版本
    preprocess_data_parallel(
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step
    )
