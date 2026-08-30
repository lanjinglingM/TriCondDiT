
import os
import sys
import json
import argparse
import subprocess
import time

import torch
import pytorch_lightning as pl

from data.dataset import create_dataloader_from_config
from models.factory import create_model_from_config
from models.utils import load_ckpt_state_dict
from training.factory import create_training_wrapper_from_config


DATASET_CONFIGS = {
    "ptbxl": {
        "config": "configs/ptbxl_diffusion.json",
        "sample_size": 1000,
        "audio_channels": 12,
        "max_iters": 100000,
        "name": "ptbxl_diffusion",
        "needs_fold": False,
    },
    "qtdb": {
        "config": "configs/qtdb_diffusion.json",
        "sample_size": 500,
        "audio_channels": 1,
        "max_iters": 100000,
        "name": "qtdb_diffusion",
        "needs_fold": False,
    },
    "ludb": {
        "config": "configs/ludb_diffusion.json",
        "sample_size": 500,
        "audio_channels": 12,
        "max_iters": 100000,
        "name": "ludb_diffusion_fold_{fold}",
        "needs_fold": True,
    },
}

DEFAULT_5FOLD_GPUS = [0, 1, 2, 3, 4]


def update_config_for_fold(config, fold_idx):
    target_fold = f"fold_{fold_idx}"
    print(f"Configuring data paths for {target_fold}...")
    for key in ["data_path", "label_path", "patient_info_path", "mask_path"]:
        if key in config and isinstance(config[key], str):
            config[key] = config[key].replace("fold_0", target_fold)
            print(f"   - {key}: {config[key]}")
    return config


def train_single(dataset, fold, gpu_id, ckpt_path=None, max_iters=None):
    cfg = DATASET_CONFIGS[dataset]
    max_iters = int(max_iters) if max_iters is not None else cfg["max_iters"]

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(42, workers=True)

    with open(cfg["config"]) as f:
        model_config = json.load(f)

    if cfg["needs_fold"]:
        if fold is None:
            raise ValueError(f"--fold is required for dataset '{dataset}'")
        model_config = update_config_for_fold(model_config, fold)

    experiment_name = cfg["name"].format(fold=fold) if cfg["needs_fold"] else cfg["name"]

    print(f"[{experiment_name}] Preparing dataloader (length: {cfg['sample_size']})...")
    train_dl = create_dataloader_from_config(
        model_config,
        batch_size=32,
        num_workers=8,
        sample_rate=model_config.get("sample_rate", 500),
        sample_size=cfg["sample_size"],
        audio_channels=model_config.get("audio_channels", cfg["audio_channels"]),
    )

    print(f"[{experiment_name}] Initializing signal-space DiT model...")
    model = create_model_from_config(model_config)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    checkpoint_dir = os.path.join("outputs", experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_filename = f"ludb-fold{fold}-diff-{{step}}" if cfg["needs_fold"] else "ecg-diff-{step}"
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=5000,
        dirpath=checkpoint_dir,
        save_top_k=-1,
        filename=ckpt_filename
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        callbacks=[ckpt_callback],
        log_every_n_steps=10,
        max_steps=max_iters,
        max_epochs=-1,
        default_root_dir=os.path.join("outputs", experiment_name),
        gradient_clip_val=1.0,
        num_sanity_val_steps=0
    )

    channels = model_config.get("audio_channels", cfg["audio_channels"])
    print(f"[{experiment_name}] Starting {dataset} signal-space diffusion training ({max_iters} iterations)")
    if cfg["needs_fold"] and channels != 12:
        print(f"Warning: audio_channels ({channels}) != 12, please confirm.")

    trainer.fit(training_wrapper, train_dl, ckpt_path=ckpt_path)


def train_5fold(dataset, gpu_list, max_iters=None):
    if not DATASET_CONFIGS[dataset]["needs_fold"]:
        raise ValueError(f"5-fold training is only supported for datasets that use folds (got '{dataset}')")

    processes = []
    print(f"Starting 5-fold cross-validation on {len(gpu_list)} GPUs...")

    for fold_idx in range(5):
        gpu_id = gpu_list[fold_idx]
        log_file = open(f"train_fold_{fold_idx}.log", "w")

        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--fold", str(fold_idx),
            "--gpu_id", str(gpu_id)
        ]

        if max_iters is not None:
            cmd += ["--max_iters", str(max_iters)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"   -> Starting fold {fold_idx} on GPU {gpu_id} (log: train_fold_{fold_idx}.log)")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        processes.append((p, log_file))

        time.sleep(2)

    print("\nAll processes started and running in the background...")
    print("Use 'tail -f train_fold_0.log' to follow a specific fold.")

    for p, log_f in processes:
        p.wait()
        log_f.close()

    print("All 5 folds finished training!")


def main():
    parser = argparse.ArgumentParser(description="Train TriCondDiT")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to train on")
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold index (0-4) for fold-based datasets, or 'all' to run all folds in parallel")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (controls CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for the 5-fold parallel run (e.g. '0,1,2,3,4')")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Total training iterations (optimizer steps); defaults to the dataset config")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.fold == "all":
        gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else DEFAULT_5FOLD_GPUS
        if len(gpu_list) < 5:
            raise ValueError("Need at least 5 GPU IDs for the 5-fold parallel run (use --gpus)")
        train_5fold(args.dataset, gpu_list, max_iters=args.max_iters)
    else:
        fold = int(args.fold) if args.fold is not None else None
        train_single(args.dataset, fold, args.gpu_id, ckpt_path=args.ckpt, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
