import os
import csv
import json
import warnings
import argparse
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

import numpy as np
import torch
from pymatgen.core.structure import Structure

# --- 从 data.py 复制过来的辅助类 ---
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
    # Unpack args to make it more readable inside the function
    cif_id, target, root_dir, max_num_nbr, radius, dmin, step, atom_init_file = args
    
    # Re-initialize these objects for each process as they are not shareable across processes directly.
    # This is a common pattern when using multiprocessing with objects that can't be pickled.
    # Note: If `AtomCustomJSONInitializer` and `GaussianDistance` were truly immutable and small,
    # they could be passed to `initializer` and `initargs` in `Pool` if you were managing the Pool directly.
    # For `process_map`, re-initialization per process is the typical workaround for non-picklable objects.
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    cif_path = os.path.join(root_dir, cif_id + '.cif')
    processed_dir = os.path.join(root_dir, 'processed')

    if not os.path.exists(cif_path):
        warnings.warn(f"Warning: File '{cif_path}' not found, skipping.")
        return None # Return None to indicate failure or skip

    try:
        crystal = Structure.from_file(cif_path)
        
        # Get atom features
        atom_fea = np.vstack([ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])

        # Get neighbor information
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                # warnings.warn(f'ID {cif_id} did not find enough neighbors to build graph.')
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

        # Convert to PyTorch tensors
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target_tensor = torch.Tensor([float(target)])

        # Save as .pt file
        save_path = os.path.join(processed_dir, f'{cif_id}.pt')
        torch.save(((atom_fea, nbr_fea, nbr_fea_idx), target_tensor, cif_id), save_path)
        return cif_id # Return the ID of the successfully processed file
    except Exception as e:
        warnings.warn(f"Error processing file '{cif_path}': {e}")
        return None


def preprocess_data_parallel(root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2):
    """
    Traverses the raw dataset directory, converts each CIF file to PyTorch tensors, and saves them.
    This version uses multiprocessing for parallel processing with a single tqdm progress bar.

    Parameters:
    ----------
    root_dir: str
        Path to the root directory of the raw dataset.
        (Should contain id_prop.csv, atom_init.json, and all .cif files)
    max_num_nbr: int
        Maximum number of neighbors when building the crystal graph.
    radius: float
        Cutoff radius for searching neighbors.
    dmin: float
        Minimum distance for constructing GaussianDistance.
    step: float
        Step size for constructing GaussianDistance.
    """
    # 1. Check if paths and files exist
    assert os.path.exists(root_dir), f"Error: Root directory '{root_dir}' does not exist!"
    id_prop_file = os.path.join(root_dir, 'id_prop.csv')
    assert os.path.exists(id_prop_file), f"Error: '{id_prop_file}' does not exist!"
    atom_init_file = os.path.join(root_dir, 'atom_init.json')
    assert os.path.exists(atom_init_file), f"Error: '{atom_init_file}' does not exist!"

    # 2. Create directory for processed data
    processed_dir = os.path.join(root_dir, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Creating directory '{processed_dir}' for processed data.")

    # 3. Read id_prop.csv file
    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    # 4. Prepare parameters list for parallel tasks
    tasks = []
    for cif_id, target in id_prop_data:
        tasks.append((cif_id, target, root_dir, max_num_nbr, radius, dmin, step, atom_init_file))

    # Calculate miniters for approximately 4% updates
    total_tasks = len(tasks)
    # Ensure miniters is at least 1 to avoid division by zero or overly frequent updates for small datasets
    min_iterations_for_update = max(1, int(total_tasks * 0.04)) 

    # 5. Use process_map for multiprocessing with a single, controlled progress bar
    print(f"Starting parallel preprocessing of CIF files using {cpu_count()} processes...")
    
    results = process_map(
        process_single_cif,
        tasks,
        max_workers=cpu_count(), # Use all available CPU cores
        chunksize=1,             # Smaller chunksize can sometimes provide smoother updates for fast tasks
        desc="Processing CIF files",
        mininterval=1.0,         # Minimum progress display update interval (in seconds).
                                 # Set a reasonable interval to prevent excessive updates.
        miniters=min_iterations_for_update # Update after this many iterations have passed.
    )
    
    successful_count = sum(1 for r in results if r is not None)
    print("="*50)
    print(f"🎉 Preprocessing completed! Successfully processed {successful_count} files. All data saved to '{processed_dir}'.")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIF file preprocessing script')
    parser.add_argument('root_dir', type=str,
                        help='Path to the root directory containing cif files, id_prop.csv, and atom_init.json.')
    parser.add_argument('--max-num-nbr', type=int, default=12,
                        help='Maximum number of neighbors when building the crystal graph.')
    parser.add_argument('--radius', type=float, default=8,
                        help='Cutoff radius for searching neighbors.')
    parser.add_argument('--dmin', type=float, default=0,
                        help='Minimum distance for constructing GaussianDistance.')
    parser.add_argument('--step', type=float, default=0.2,
                        help='Step size for constructing GaussianDistance.')

    args = parser.parse_args()

    # Call the parallel version
    preprocess_data_parallel(
        root_dir=args.root_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step
    )
