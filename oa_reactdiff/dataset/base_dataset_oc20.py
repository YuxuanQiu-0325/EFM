import lmdb
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import io
from oa_reactdiff.get_edge_index import generate_graph

from torch_geometric.data import DataLoader, Batch
import lmdb
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import io
from oa_reactdiff.get_edge_index import generate_graph
from torch_geometric.data import DataLoader, Batch
from torch_geometric.data import Data
import torch_geometric

ATOM_MAPPING = {
    1: 0,    # Hydrogen
    6: 1,    # Carbon
    7: 2,    # Nitrogen
    8: 3,    # Oxygen
    11: 4,   # Sodium
    13: 5,   # Aluminum
    14: 6,   # Silicon
    15: 7,   # Phosphorus
    16: 8,   # Sulfur
    17: 9,   # Chlorine
    19: 10,  # Potassium
    20: 11,  # Calcium
    21: 12,  # Scandium
    22: 13,  # Titanium
    23: 14,  # Vanadium
    24: 15,  # Chromium
    25: 16,  # Manganese
    26: 17,  # Iron
    27: 18,  # Cobalt
    28: 19,  # Nickel
    29: 20,  # Copper
    30: 21,  # Zinc
    31: 22,  # Gallium
    32: 23,  # Germanium
    33: 24,  # Arsenic
    34: 25,  # Selenium
    37: 26,  # Rubidium
    38: 27,  # Strontium
    39: 28,  # Yttrium
    40: 29,  # Zirconium
    41: 30,  # Niobium
    42: 31,  # Molybdenum
    43: 32,  # Technetium
    44: 33,  # Ruthenium
    45: 34,  # Rhodium
    46: 35,  # Palladium
    47: 36,  # Silver
    48: 37,  # Cadmium
    49: 38,  # Indium
    50: 39,  # Tin
    51: 40,  # Antimony
    52: 41,  # Tellurium
    55: 42,  # Cesium
    72: 43,  # Hafnium
    73: 44,  # Tantalum
    74: 45,  # Tungsten
    75: 46,  # Rhenium
    76: 47,  # Osmium
    77: 48,  # Iridium
    78: 49,  # Platinum
    79: 50,  # Gold
    80: 51,  # Mercury
    81: 52, ############# ood中的
    82: 53,  # Lead
    83: 54  # Bismuth
}


n_element = len(ATOM_MAPPING)

FRAG_MAPPING = {
    "adslab": "adslab",  # 单独的adslab不涉及reactant和product之间的关系
}

class BaseDataset(Dataset):
    def __init__(
        self,
        lmdb_path,
        center=True,
        zero_charge=False,
        device="cpu",
        remove_h=False,
        swapping_react_prod=False,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.center = center
        self.zero_charge = zero_charge
        self.remove_h = remove_h
        self.swapping_react_prod = swapping_react_prod

        # 打开 LMDB 数据库
        #self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        # 检查路径是否是一个目录，并读取其中的所有 .lmdb 文件
        if os.path.isdir(lmdb_path):
            self.env = []
            for file in os.listdir(lmdb_path):
                if file.endswith(".lmdb"):
                    env = lmdb.open(
                        os.path.join(lmdb_path, file),
                        subdir=False,
                        map_size=10 * 1024 * 1024 * 1024,  # 设置为 10GB，根据实际需求调整
                        readonly=True,
                        lock=False,
                        readahead=True,
                        meminit=False,
                        max_readers=1,
                    )
                    self.env.append(env)
        else:
            # 如果只是单个 lmdb 文件，直接打开
            self.env = lmdb.open(
                str(lmdb_path),
                subdir=False,
                map_size=10 * 1024 * 1024 * 1024,  # 设置为 10GB，根据实际需求调整
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1,
            )


        self.raw_dataset = self._load_data_from_lmdb()

    def _load_data_from_lmdb(self):
        raw_dataset = []
        for env in self.env:  # 遍历每个环境对象
            with env.begin() as txn:
                cursor = txn.cursor()
                for _, value in cursor:
                    # 解码 LMDB 中的值
                    sample = self._decode_lmdb_value(value)
                    # 将解码后的样本添加到 raw_dataset 列表
                    raw_dataset.append(sample)
        return raw_dataset

    def _decode_lmdb_value(self, value):
        # 使用 BytesIO 将 value 包装成类文件对象，以便 np.load 读取
        return np.load(io.BytesIO(value), allow_pickle=True)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        # 直接返回 raw_dataset 中的样本
        return self.raw_dataset[idx]



    @staticmethod
    def collate_fn(batch, swapping_react_prod=False):
        # 深拷贝 batch 数据，避免直接修改原始数据
        data_duplicated = copy.deepcopy(batch)

        # 初始化批处理结果的数据结构
        sizes, positions, one_hot_encodings, charges, masks = [], [], [], [], []
        conditions, edge_indices_original, edge_indices_pbc, fixed_atoms, tags_list = [], [], [], [], []
        filtered_batch = [data for data in batch if isinstance(data, torch_geometric.data.Data)]

        if not filtered_batch:
            print("All samples in batch are invalid. Returning empty batch.")
            return None, None, None, None, None, None

        for idx, data in enumerate(filtered_batch):
            # 提取每个样本的原子数量
            sizes.append(data.natoms)

            # 提取位置数据
            positions.append(data.pos)

            # 映射原子序数为 one-hot 编码索引
            mapped_indices = [ATOM_MAPPING.get(int(at.item()), -1) for at in data.atomic_numbers]
            mapped_indices_tensor = torch.tensor(mapped_indices, dtype=torch.long)

            # 生成 one-hot 编码
            one_hot = F.one_hot(mapped_indices_tensor, num_classes=n_element)
            one_hot_encodings.append(one_hot)

            # 使用原子序数作为电荷
            # charges.append(data.atomic_numbers.view(-1, 1))

            # **将所有电荷设置为 0**
            charge_zero = torch.zeros((data.natoms, 1), dtype=torch.float)
            charges.append(charge_zero)

            # 生成每个原子的样本索引 mask
            mask = torch.full((data.natoms,), idx, dtype=torch.long)
            masks.append(mask)

            # 提取能量数据 y 作为 condition
            #conditions.append(data.y)

            # 提取能量数据 y 作为 condition，如果不存在则默认值为 0
            if hasattr(data, 'y') and data.y is not None:
                conditions.append(data.y)
            else:
                print(f"Sample {idx} is missing 'y' attribute or its value is None. Setting default value to 0.")
                conditions.append(torch.tensor([0.0], dtype=torch.float))  # 默认值为 0.0


            # 提取 edge_index_original
            edge_indices_original.append(data.edge_index)

            # 提取 fixed 原子信息
            fixed_atoms.append(data.fixed)

            # 提取 tags 信息
            if hasattr(data, 'tags'):
                tags_list.append(data.tags)
            else:
                print(f"Sample {idx} is missing 'tags' attribute. Filling with default values.")
                tags_list.append(torch.zeros((data.natoms,), dtype=torch.long))  # 默认值

        # 如果所有数据都被跳过，返回空结果
        if not sizes:
            print("All samples were skipped.")
            return None, None, None, None, None, None

        # 组合并形成5维字典，并放入单个元素的列表中
        processed_batch = [{
            'size': torch.tensor(sizes),
            'pos': torch.cat(positions),
            'one_hot': torch.cat(one_hot_encodings),
            # 'charge': torch.cat(charges),
            'mask': torch.cat(masks)
        }]

        # 将 conditions 转换为一个 tensor
        condition_tensor = torch.tensor(conditions)

        # 将 edge_indices 转换为一个 tensor
        edge_index_original = torch.cat(edge_indices_original, dim=1)  # 假设 edge_index 是 [2, num_edges] 形状

        # 将 fixed 原子信息合并为一个 tensor
        fixed_tensor = torch.cat(fixed_atoms)

        # 将 tags 合并为一个 tensor
        tags_tensor = torch.cat(tags_list)

        # 假设 `batch` 是一个 Data 列表
        if isinstance(batch, list):  # 确保 batch 是列表形式
            filtered_batch = Batch.from_data_list(filtered_batch)  # 合并为 DataBatch

        edge_index_pbc, *_ = generate_graph(filtered_batch)

        # 返回包含 list、condition tensor 和 edge_index tensor 的 tupleddddd
        return processed_batch, condition_tensor, edge_index_pbc, edge_index_original, tags_tensor

