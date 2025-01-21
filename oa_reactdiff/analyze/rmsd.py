from typing import List
import numpy as np

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.io.xyz import XYZ

from torch import Tensor
import json
import os
from datetime import datetime
from oa_reactdiff.dataset.base_dataset_oc20 import ATOM_MAPPING
import torch
from ase import Atoms
from ase.io import Trajectory


'''
def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol

'''


def xh2pmg(xh):
    # 获取 one-hot 编码区域 (xh 的第 3 至第 57 列)
    one_hot = xh[:, 3:57]

    # 获取 one-hot 编码对应的最大索引（即原子类型）
    atom_indices = torch.argmax(one_hot, dim=1).cpu().numpy()

    # 构建反向映射：从索引到元素符号
    reverse_mapping = {v: k for k, v in ATOM_MAPPING.items()}

    # 将索引转换为元素符号
    species = [reverse_mapping[idx] for idx in atom_indices]

    # 构建 pymatgen 的 Molecule 对象
    mol = Molecule(
        species=species,  # 元素符号列表
        coords=xh[:, :3].cpu().numpy(),  # 三维坐标
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:    #False
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  #表示两个分子中相同原子种类的所有可能排列组合数 # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)  #这个pairs用来储存分子和对应的rmsd
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
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)

    print(f"Calculating rmsd_core...")

    rmsd = rmsd_core(mol1, mol2, threshold)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd



def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    id_fixed: Tensor,  # 修改为 Tensor 而非 List
    idx: int = 0,
    threshold=0.5,
    save_dir=r"/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol_1000/json"  #存mol的文件夹
):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 初始化返回值
    rmsds = []
    out_samples_use = out_samples[idx]  # 模型预测的分子坐标+分子种类
    xh_use = xh[idx]  # 真实的分子坐标+分子种类
    nodes = fragments_nodes[idx].long().cpu().numpy()  # 获取每个样本的原子数量
    start_ind, end_ind = 0, 0

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 保存 .traj 文件路径
    traj_filename = os.path.join("/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol_1000/traj", f"samples_{current_time}.traj")
    traj = Trajectory(traj_filename, 'w')  # 创建一个共享的 Trajectory 文件

    # 保存同一时间的所有样本
    all_samples = []
    mae_list = []

    for jj, natoms in enumerate(nodes):
        print(f"Processing sample {jj + 1}/{len(nodes)} with {natoms} atoms...")
        end_ind += natoms

        # 对整个 id_fixed 张量进行切片，获取当前样本的基板信息
        id_fixed_sample = id_fixed[start_ind:end_ind]
        movable_mask = id_fixed_sample == 2  # 筛选吸附体原子（id_fixed == 0）

        # 生成 mol1 和 mol2，仅包含吸附体原子
        x = out_samples_use[start_ind:end_ind][movable_mask] #这个值有的原子很大，也就是预测出的adsordtant中的某几个原子坐标很大
        print(x.cpu().numpy())  # 转为 NumPy 格式并打印

        ############################# 计算 MAE################################
        #pred_coords = out_samples_use[start_ind:end_ind][movable_mask]
        #true_coords = xh_use[start_ind:end_ind][movable_mask]
        #error = torch.abs(pred_coords - true_coords)
        #mae_value = torch.mean(error).item()  # 平均绝对误差
        #print(f"MAE for sample {jj + 1}: {mae_value}")
        #mae_list.append(mae_value)

        ############################# 保存traj文件 ################################
        pred_coords_traj = out_samples_use[start_ind:end_ind]
        true_coords_traj = xh_use[start_ind:end_ind]

        one_hot_pred= out_samples_use[start_ind:end_ind][:, 3:57] #保存全部的原子
        atom_indices_pred = torch.argmax(one_hot_pred, dim=1).cpu().numpy()
        reverse_mapping_pred = {v: k for k, v in ATOM_MAPPING.items()}
        species_pred = [reverse_mapping_pred[idx] for idx in atom_indices_pred]

        one_hot_true = xh_use[start_ind:end_ind][:, 3:57] #保存全部的原子
        atom_indices_true = torch.argmax(one_hot_true, dim=1).cpu().numpy()
        reverse_mapping_true = {v: k for k, v in ATOM_MAPPING.items()}
        species_true = [reverse_mapping_true[idx] for idx in atom_indices_true]

        # 创建 ASE Atoms 对象保存到 Trajectory
        atoms_pred = Atoms(symbols=species_pred, positions=pred_coords_traj[:,0:3].cpu().numpy())
        atoms_true = Atoms(symbols=species_true, positions=true_coords_traj[:,0:3].cpu().numpy())
        print(f"Atoms for predicted coordinates: {atoms_pred}")
        print(f"Atoms for true coordinates: {atoms_true}")

        # 将分子对象写入到共享的 .traj 文件
        traj.write(atoms_pred)  # 写入预测的分子
        traj.write(atoms_true)  # 写入真实的分子
###############################################################################################


        mol1 = xh2pmg(out_samples_use[start_ind:end_ind][movable_mask]) #out_samples_use中坐标很小，mol1坐标很大
        mol2 = xh2pmg(xh_use[start_ind:end_ind][movable_mask])

        # 转换为字典或 JSON 可序列化的格式
        mol1_dict = mol1.as_dict()  # 假设 xh2pmg 返回的对象有 `as_dict()` 方法
        mol2_dict = mol2.as_dict()

        # 将 mol1 和 mol2 数据保存到列表中
        all_samples.append({
            "sample_id": jj + 1,
            "mol1": mol1_dict,
            "mol2": mol2_dict,
        })

        # 计算 RMSD
        #rmsd = rmsd_core(mol1, mol2, threshold)
        #rmsds.append(rmsd)

        # 更新起始索引
        start_ind = end_ind

    print(mae_list)

    # 关闭 Trajectory 文件
    traj.close()

    print(f"Saved all samples to {traj_filename}")

    # 保存 MAE 到本地文件
    #mae_filename = os.path.join("/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/mol/mae", f"mae_{current_time}.json")
    #with open(mae_filename, 'w') as f:
     #   json.dump(mae_list, f, indent=4)
    #print(f"Saved MAE list to {mae_filename}")


    # 保存到一个文件
    filename = os.path.join(save_dir, f"samples_{current_time}.json")
    with open(filename, 'w') as f:
        json.dump(all_samples, f, indent=4)

    print(f"Saved all samples to {filename}")
    # 返回 NaN 作为 RMSD 值
    rmsds = [float('NaN')] * len(nodes)
    return rmsds




'''

def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    tags: Tensor,  # 修改为 Tensor 而非 List
    idx: int = 0,
    threshold=0.5,
    save_dir=r"/mnt/d/AI/EFM/OAReactDiff-EFM-now - 12_31/mol"
):
    import os
    import json
    from datetime import datetime

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 初始化返回值
    rmsds = []
    out_samples_use = out_samples[idx]  # 模型预测的分子坐标+分子种类
    xh_use = xh[idx]  # 真实的分子坐标+分子种类
    nodes = fragments_nodes[idx].long().cpu().numpy()  # 获取每个样本的原子数量
    start_ind, end_ind = 0, 0

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存同一时间的所有样本
    all_samples = []

    for jj, natoms in enumerate(nodes):
        print(f"Processing sample {jj + 1}/{len(nodes)} with {natoms} atoms...")
        end_ind += natoms

        # 对整个 id_fixed 张量进行切片，获取当前样本的基板信息
        id_fixed_sample = tags[start_ind:end_ind]
        movable_mask = id_fixed_sample == 2  # 筛选吸附体原子（id_fixed == 2）

        # 提取吸附体和整体分子
        adsorbate_coords = out_samples_use[start_ind:end_ind][movable_mask]
        complete_coords = out_samples_use[start_ind:end_ind]

        # 保存完整分子
        mol_complete = xh2pmg(complete_coords)  # 假设 xh2pmg 可以处理完整分子
        complete_filename = os.path.join(save_dir, f"sample_{jj + 1}_surface_{current_time}.json")
        with open(complete_filename, 'w') as f:
            json.dump(mol_complete.as_dict(), f, indent=4)
        print(f"Saved complete molecule to {complete_filename}")

        # 保存吸附体
        mol_adsorbate = xh2pmg(adsorbate_coords)
        adsorbate_filename = os.path.join(save_dir, f"sample_{jj + 1}_adsorbent_{current_time}.json")
        with open(adsorbate_filename, 'w') as f:
            json.dump(mol_adsorbate.as_dict(), f, indent=4)
        print(f"Saved adsorbate molecule to {adsorbate_filename}")

        # 将数据保存到 all_samples
        all_samples.append({
            "sample_id": jj + 1,
            "mol_complete": mol_complete.as_dict(),
            "mol_adsorbate": mol_adsorbate.as_dict(),
        })

        # 更新起始索引
        start_ind = end_ind

    # 保存所有样本的汇总文件
    summary_filename = os.path.join(save_dir, f"samples_summary_{current_time}.json")
    with open(summary_filename, 'w') as f:
        json.dump(all_samples, f, indent=4)
    print(f"Saved summary of all samples to {summary_filename}")

    # 返回 NaN 作为 RMSD 值
    rmsds = [float('NaN')] * len(nodes)
    return rmsds

'''