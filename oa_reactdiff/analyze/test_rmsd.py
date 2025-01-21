import os
import json
import numpy as np
import math
import wandb  # 引入 wandb
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import (
    KabschMatcher,
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
)
from tqdm import tqdm  # 用于显示进度条


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)  # HungarianOrderMatcher  GeneticOrderMatcher
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
        mol1,
        mol2,
        ignore_chirality=False,
        threshold=0.5,
        same_order=False,
):
    # 核心逻辑：处理手性问题
    rmsd = rmsd_core(mol1, mol2, threshold, same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]  # 对 z 坐标取反，生成镜像分子
        mol2_reflect = Molecule(
            species=mol2.species,  # 保持原子种类不变
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold, same_order)
        rmsd = min(rmsd, rmsd_reflect)  # 返回 RMSD 的较小值
    return rmsd
if __name__ == "__main__":
    # 初始化 wandb
    wandb.init(project="rmsd-analysis", name="json-rmsd-logs", config={"threshold": 0.5, "ignore_chirality": True})

    # 指定 JSON 文件的目录
    directory = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol_1000/json"
    # 获取所有 JSON 文件路径
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")])

    # 用于保存所有文件 RMSD 结果的列表
    all_results = []

    for file_idx, file in enumerate(files):
        print(f"Processing file {file_idx + 1}/{len(files)}: {file}")

        # 加载 JSON 文件
        with open(file, "r") as f:
            data = json.load(f)

        # 用于存储当前文件中所有样本的 RMSD
        rmsds = []

        # 遍历当前文件的所有样本
        for sample_idx, sample in enumerate(tqdm(data, desc=f"Processing samples in File {file_idx + 1}")):
            print(f"  Processing sample {sample_idx + 1}/{len(data)}...")

            # 从 JSON 数据构造 mol1 和 mol2
            mol1 = Molecule.from_dict(sample["mol1"])
            mol2 = Molecule.from_dict(sample["mol2"])

            # 计算 RMSD
            try:
                rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=True, threshold=0.5, same_order=False)
                print(f"    RMSD: {rmsd}")
                rmsds.append(rmsd)
            except Exception as e:
                print(f"    Error calculating RMSD for sample {sample_idx + 1}: {e}")
                rmsds.append(float('nan'))

        # 计算文件的 RMSD 统计结果
        mean_rmsd = np.nanmean(rmsds)
        median_rmsd = np.nanmedian(rmsds)

        # 保存当前文件的 RMSD 结果
        all_results.append({
            "file_index": file_idx + 1,
            "file_name": os.path.basename(file),
            "rmsd_values": rmsds,
            "mean_rmsd": mean_rmsd,
            "median_rmsd": median_rmsd
        })

        # 打印统计结果
        print(f"File {file_idx + 1} - Mean RMSD: {mean_rmsd:.4f}")
        print(f"File {file_idx + 1} - Median RMSD: {median_rmsd:.4f}")

        # 记录到 wandb
        wandb.log({
            "file_index": file_idx + 1,
            "file_name": os.path.basename(file),
            "mean_rmsd": mean_rmsd,
            "median_rmsd": median_rmsd
        })

    # 保存所有 RMSD 结果为整合的 JSON 文件
    json_path = os.path.join(directory, "all_rmsd_results.json")
    with open(json_path, "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"All RMSD results saved to {json_path}")

    wandb.finish()
