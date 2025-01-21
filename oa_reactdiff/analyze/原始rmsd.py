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


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
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
    idx: int = 0,  #### 1 -> 0
    threshold=0.5,
):
    rmsds = []
    out_samples_use = out_samples[idx]  #模型预测的分子坐标+分子种类
    xh_use = xh[idx] #真实的分子坐标+分子种类
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        print(f"Processing sample {jj + 1}/{len(nodes)} with {natoms} atoms...")
        end_ind += natoms
        print(f"Processing mol1...")
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        print(f"Processing mol2...")
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            print(f"Calculating RMSD...")
            rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=True, threshold=threshold)
        except:
            rmsd = 1.0
        rmsd = min(rmsd, 1.0)
        print(f"Sample {jj + 1}: RMSD = {rmsd}")  # 打印每一个样本的 RMSD
        rmsds.append(rmsd)
    return rmsds
