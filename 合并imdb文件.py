import lmdb
import os
import pickle
from pathlib import Path


def merge_lmdb(source_dir, target_lmdb_path):
    """
    合并目标目录下的所有 .lmdb 文件到一个新的 LMDB 文件中。

    Args:
        source_dir (str): 包含多个 .lmdb 文件的目录路径。
        target_lmdb_path (str): 合并后的 LMDB 文件保存路径。
    """
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
                    # 重新生成唯一的 key
                    new_key = f"{sample_count}".encode("ascii")
                    target_txn.put(new_key, value)
                    sample_count += 1

            source_env.close()

        # 保存总样本数到合并后的 LMDB
        target_txn.put("length".encode("ascii"), pickle.dumps(sample_count))

    target_env.close()
    print(f"Merging completed! Total samples: {sample_count}")


# 使用示例
source_directory = "/mnt/d/AI/EFM/adsorbdiff_valood50_R1I0.1"  # 包含多个 .lmdb 文件的目录路径
merged_lmdb_file = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 12_31/oa_reactdiff/data/oc20/val_data/merge.lmdb"  # 合并后的 LMDB 文件路径

merge_lmdb(source_directory, merged_lmdb_file)
