import torch
from torch.utils.data import DataLoader
import lmdb
import pickle
from pathlib import Path

class ProcessedAdsorption:
    def __init__(self, lmdb_dir):
        self.lmdb_dir = Path(lmdb_dir)
        print(f"Trying to open LMDB at: {str(self.lmdb_dir)}")
        self.env = lmdb.open(
            str(self.lmdb_dir),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )

        with self.env.begin() as txn:
            length_entry = txn.get("length".encode("ascii"))
            if length_entry is not None:
                self.num_samples = pickle.loads(length_entry)
            else:
                self.num_samples = self.env.stat()["entries"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            if datapoint_pickled is None:
                raise IndexError(f"Index {idx} is out of bounds.")
            data = pickle.loads(datapoint_pickled)
        return data

def find_unique_atomic_numbers(dataset):
    unique_atomic_numbers = set()
    for i in range(len(dataset)):
        sample = dataset[i]
        atomic_numbers = sample['atomic_numbers']
        unique_atomic_numbers.update(atomic_numbers.tolist())
    return sorted(list(unique_atomic_numbers))

if __name__ == "__main__":
    dataset_dir = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 12_31/oa_reactdiff/data/oc20/val_ood/data.0001.lmdb"
    dataset = ProcessedAdsorption(lmdb_dir=dataset_dir)

    # 查找数据集中所有独特的原子序数
    unique_atomic_numbers = find_unique_atomic_numbers(dataset)
    print(f"OC20-Dense 数据集中的原子种类: {unique_atomic_numbers}")
    print(f"总共有 {len(unique_atomic_numbers)} 种原子")
