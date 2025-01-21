import lmdb
import os
import pickle
from pathlib import Path

def split_lmdb(source_lmdb_path, target_dir):
    """
    将一个 LMDB 文件中的每个样本拆分为单独的 LMDB 文件。

    Args:
        source_lmdb_path (str): 源 LMDB 文件路径。
        target_dir (str): 拆分后的 LMDB 文件保存目录。
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 打开源 LMDB 文件
    source_env = lmdb.open(
        source_lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )

    sample_count = 0  # 样本计数

    with source_env.begin() as source_txn:
        for key, value in source_txn.cursor():
            # 跳过特殊的长度键（"length"）
            if key.decode("ascii") == "length":
                continue

            # 为每个样本创建一个单独的 LMDB 文件
            sample_lmdb_path = os.path.join(target_dir, f"sample_{sample_count}.lmdb")
            sample_env = lmdb.open(
                sample_lmdb_path,
                subdir=False,
                readonly=False,
                lock=True,
                readahead=False,
                meminit=False,
                map_size=10 ** 9  # 设置每个样本的文件大小上限
            )

            with sample_env.begin(write=True) as sample_txn:
                sample_txn.put(key, value)

            sample_env.close()
            print(f"Sample {sample_count} saved to {sample_lmdb_path}")
            sample_count += 1

    source_env.close()
    print(f"Splitting completed! Total samples: {sample_count}")

# 使用示例
source_lmdb_file = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/data/oc20/val_data/data.0003.lmdb"  # 源 LMDB 文件路径
target_directory = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/data/oc20/val_data/split_samples"  # 拆分后的 LMDB 文件保存目录

split_lmdb(source_lmdb_file, target_directory)
