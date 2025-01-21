from typing import List, Optional, Tuple
from uuid import uuid4
import os
import shutil
import torch

from pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet


model_type = "leftnet"
version = "0"
project = "OC20ReactDiff_1_08"
# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn  8->
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",

)
leftnet_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn  8->57-1 （3+54）
    pos_require_grad=False,
    cutoff=12.0, ############应该改成12，再确认一下adsorbdiff中的config
    num_layers=6,
    hidden_channels=196, #196->256
    num_radial=96,
    in_hidden_channels=8,  #8->57-1
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    #lr=2.5e-4,
    lr=1e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 200    #datadir_train="/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now/oa_reactdiff/data/oc20",

T_mult = 2
training_config = dict(
    #datadir="/mnt/c/Users/Administrator/Desektop/EFM/OAReactDiff-EFM-now/oa_reactdiff/data/oc20",

    datadir_train="/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/data/oc20/train_data",
    datadir_val="/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/data/oc20/val_data",
    datadir_test="/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/data/oc20/test_data",
    #datadir_train="/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now/oa_reactdiff/data/oc20",
    #datadir_val="/mnt/c/Users/Administrator/Desktop/EFM/OAReactDiff-EFM-now/oa_reactdiff/data/oc20",

    remove_h=False,
    bz=16,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=False,
    append_frag=False,
    use_by_ind=False,
    reflection=False,
    single_frag_only=False,
    only_ts=True,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),  # step
)
training_data_frac = 1.0

node_nfs: List[int] = [58]  # Adjusted for the combined entity structure 3+55
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["adslab"]  # Single entity for adsorbate and substrate
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "adslab"
enforce_same_encoding = None
scales = [1.0]
fixed_idx: Optional[List] = None
eval_epochs = 1 #######################################################

# ----Normalizer---
norm_values: Tuple = (1.0, 1.0, 1.0)
norm_biases: Tuple = (0.0, 0.0, 0.0)

# ---Schedule---
noise_schedule: str = "cosine"
timesteps: int = 1000   #这是加噪的时间步（5000），去噪过程在inpaint中指定为1000
precision: float = 1e-5

norms = "_".join([str(x) for x in norm_values])
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

seed_everything(42, workers=True)
ddpm = DDPMModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    norm_values,
    norm_biases,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=None,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False)
    except:
        pass
ckpt_path ='/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/checkpoint_1000'
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
    every_n_epochs=1,
    save_top_k=-1,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))

if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
shutil.copy(f"../model/{model_type}.py", f"{ckpt_path}/{model_type}.py")

print("config: ", config)

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = None

'''
pretrained_model_path = "/mnt/d/AI/EFM/OAReactDiff-EFM-now - 12_08/oa_reactdiff/checkpoint/ddpm-epoch=018-val-totloss=1534.46.ckpt"
if os.path.exists(pretrained_model_path):
    print(f"Loading pretrained model from {pretrained_model_path}")
    ddpm.load_state_dict(torch.load(pretrained_model_path, map_location="cuda:0"))
else:
    print("No pretrained model found. Training from scratch.")
'''


trainer = Trainer(
    max_epochs=1000,
    accelerator="gpu",
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    #limit_train_batches=600, #200
    limit_val_batches=20,  #20
    resume_from_checkpoint="/mnt/d/AI/EFM/OAReactDiff-EFM-now - 1_08/oa_reactdiff/checkpoint_1000/ddpm-epoch=099-val-totloss=-39.85.ckpt", #########
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)

trainer.save_checkpoint("pretrained-oc20-diff.ckpt")
