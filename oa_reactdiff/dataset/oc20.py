import torch
import lmdb
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torch.nn.functional as F
from oa_reactdiff.dataset.base_dataset_oc20 import BaseDataset, ATOM_MAPPING


n_element = len(list(ATOM_MAPPING.keys())) #这个要在base_dataset中定义

FRAG_MAPPING = {
    "adslab": "adslab",  # 只有一个
}

#拼接操作在base_dataset_oc20中，初始化这个父类的时候会执行
class ProcessedOC20(BaseDataset):
    def __init__(
        self,
        lmdb_path,
        center=True,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        swapping_react_prod=False,
        **kwargs,
    ):
        super().__init__(lmdb_path, center, zero_charge, device, remove_h, swapping_react_prod, **kwargs)
        # 调用 base_dataset 中的 process_oc20 方法进行处理
        #self.process_oc20()

