import copy
import numpy as np
import torch
from oa_reactdiff.dataset.base_dataset import BaseDataset, ATOM_MAPPING


'''
从transition1x数据集中预处理-data：dict5.
处理之后的这个data是送进model用的
'''


n_element = len(list(ATOM_MAPPING.keys()))
FRAG_MAPPING = {
    "reactant": "product",
    "transition_state": "transition_state",
    "product": "reactant",
}


def reflect_z(x):
    x = np.array(x)
    x[:, -1] = -x[:, -1]
    return x


class ProcessedTS1x(BaseDataset):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        single_frag_only=True,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=False,
        only_ts=False,
        confidence_model=False,
        position_key="positions",
        ediff=None,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )
        if confidence_model:
            use_by_ind = False
        if remove_h:
            print("remove_h is ignored because it is not reasonble for TS.")
        if single_frag_only:   #统计dataset中single_fragment=1的索引（只选择数据集中的单片结构，不要分裂的分子片段）
            single_frag_inds = np.where(
                np.array(self.raw_dataset["single_fragment"]) == 1
            )[0]
        else:
            single_frag_inds = np.array(range(len(self.raw_dataset["single_fragment"])))
        if use_by_ind:
            use_inds = self.raw_dataset["use_ind"] #use_ind是什么？？？
        else:
            use_inds = range(len(self.raw_dataset["single_fragment"]))
        single_frag_inds = list(set(single_frag_inds).intersection(set(use_inds)))

        data_duplicated = copy.deepcopy(self.raw_dataset)
        for k, mapped_k in FRAG_MAPPING.items():
            for v, val in data_duplicated[k].items():
                self.raw_dataset[k][v] = [val[ii] for ii in single_frag_inds]
                if swapping_react_prod:
                    mapped_val = data_duplicated[mapped_k][v]
                    self.raw_dataset[k][v] += [
                        mapped_val[ii] for ii in single_frag_inds
                    ]
        if reflection:
            for k, mapped_k in FRAG_MAPPING.items():
                for v, val in self.raw_dataset[k].items():
                    if v in ["wB97x_6-31G(d).forces", position_key]:
                        self.raw_dataset[k][v] += [reflect_z(_val) for _val in val]
                    else:
                        self.raw_dataset[k][v] += val

        self.reactant = self.raw_dataset["reactant"]
        self.transition_state = self.raw_dataset["transition_state"]
        self.product = self.raw_dataset["product"]

        self.n_fragments = pad_fragments + 3 #????????
        self.device = torch.device(device)
        n_samples = len(self.reactant["charges"])  #分子个数
        self.n_samples = len(self.reactant["charges"])

        self.data = {}
        repeat = 2 if swapping_react_prod else 1  #???????
        if confidence_model:
            self.data["target"] = torch.tensor(
                self.raw_dataset["target"] * repeat
            ).unsqueeze(1)
            self.data["rmsd"] = torch.tensor(
                self.raw_dataset["rmsd"] * repeat
            ).unsqueeze(1)
        if ediff is not None:
            self.data["ediff"] = torch.tensor(
                self.raw_dataset[ediff]["ediff"] * repeat
            ).unsqueeze(1)
        if not only_ts:   #不是只处理TS，是同时处理R P TS (论文中的无条件情景)
            if not append_frag:
                self.process_molecules(
                    "reactant", n_samples, idx=0, position_key=position_key
                )
                self.process_molecules("transition_state", n_samples, idx=1)
                self.process_molecules(
                    "product", n_samples, idx=2, position_key=position_key
                )
            else:
                self.process_molecules(
                    "reactant",
                    n_samples,
                    idx=0,
                    append_charge=0,
                    position_key=position_key,
                )
                self.process_molecules(
                    "transition_state", n_samples, idx=1, append_charge=1
                )
                self.process_molecules(
                    "product",
                    n_samples,
                    idx=2,
                    append_charge=0,
                    position_key=position_key,
                )

            for idx in range(pad_fragments):
                self.patch_dummy_molecules(idx + 3)
        else:
            if not append_frag:
                self.process_molecules("transition_state", n_samples, idx=0)
            else:
                self.process_molecules(
                    "transition_state", n_samples, idx=0, append_charge=1
                )
            # for idx in range(2):
            #     self.patch_dummy_molecules(idx + 1)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]
