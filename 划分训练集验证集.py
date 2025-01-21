import lmdb
import os
import pickle
import random
from pathlib import Path


def merge_and_split_lmdb(source_dir, target_lmdb_path, train_lmdb_path, val_lmdb_path, train_ratio=0.8):
    # 创建目标 LMDB 环境用于合并
    target_env = lmdb.open(
        target_lmdb_path,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=10 ** 11  # 设置目标文件大小上限
    )

    sample_count = 0  # 记录样本总数
    samples = []  # 保存所有样本键值对

    # 合并 .lmdb 文件
    with target_env.begin(write=True) as target_txn:
        for source_lmdb in Path(source_dir).glob("*.lmdb"):
            print(f"Merging {source_lmdb}...")
            source_env = lmdb.open(
                str(source_lmdb),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

            with source_env.begin() as source_txn:
                for key, value in source_txn.cursor():
                    new_key = f"{sample_count}".encode("ascii")
                    target_txn.put(new_key, value)
                    samples.append((new_key, value))
                    sample_count += 1

            source_env.close()

        # 保存总样本数到合并后的 LMDB
        target_txn.put("length".encode("ascii"), pickle.dumps(sample_count))

    target_env.close()
    print(f"Merging completed! Total samples: {sample_count}")

    # 打乱样本顺序
    random.shuffle(samples)

    # 按比例划分为训练集和验证集
    total_samples = len(samples)
    train_size = int(total_samples * train_ratio)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    # 保存训练集和验证集
    save_lmdb(train_lmdb_path, train_samples, "train")
    save_lmdb(val_lmdb_path, val_samples, "validation")


def save_lmdb(lmdb_path, samples, dataset_type):
    # 创建 LMDB 环境并保存数据
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=10 ** 11  # 设置目标文件大小上限
    )

    with env.begin(write=True) as txn:
        for idx, (key, value) in enumerate(samples):
            new_key = f"{idx}".encode("ascii")
            txn.put(new_key, value)
        # 保存样本总数
        txn.put("length".encode("ascii"), pickle.dumps(len(samples)))

    env.close()
    print(f"{dataset_type.capitalize()} dataset saved to {lmdb_path}.")


# 使用示例
#source_directory = "/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now/oa_reactdiff/data/oc20"  # 包含多个 .lmdb 文件的目录路径
source_directory = "/mnt/d/EFM/adsorbdiff_valID_lmdb/val_nonrelaxed_update" # 包含多个 .lmdb 文件的目录路径
merged_lmdb_file = "/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now - 12_08/oa_reactdiff/data/oc20/merge.lmdb"  # 合并后的 LMDB 文件路径
train_lmdb_file = "/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now - 12_08/oa_reactdiff/data/oc20/train_data/train.lmdb"  # 训练集 LMDB 文件路径
val_lmdb_file = "/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now - 12_08/oa_reactdiff/data/oc20/val_data/val.lmdb"  # 验证集 LMDB 文件路径

merge_and_split_lmdb(source_directory, merged_lmdb_file, train_lmdb_file, val_lmdb_file, train_ratio=0.8)
