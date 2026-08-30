
import os
import sys
import glob
import json
import argparse
import subprocess
import time

import torch
import numpy as np
from tqdm import tqdm

from models.factory import create_model_from_config
from models.utils import load_ckpt_state_dict
from inference.generation import generate_diffusion_cond


DATASET_CONFIGS = {
    "ptbxl": {
        "config": "configs/ptbxl_diffusion.json",
        "audio_channels": 12,
        "generation_length": 1000,
        "target_length": 1000,
        "cfg_scale": 1.0,
        "batch_size": 128,
        "needs_fold": False,
    },
    "qtdb": {
        "config": "configs/qtdb_diffusion.json",
        "audio_channels": 1,
        "generation_length": 500,
        "target_length": 500,
        "cfg_scale": 1.0,
        "batch_size": 512,
        "needs_fold": False,
    },
    "ludb": {
        "config": "configs/ludb_diffusion.json",
        "audio_channels": 12,
        "generation_length": 500,
        "target_length": 500,
        "cfg_scale": 1.0,
        "batch_size": 64,
        "needs_fold": True,
    },
}

DEFAULT_5FOLD_GPUS = [0, 1, 2, 3, 4]


def find_best_checkpoint(save_dir, fold_idx):
    ckpt_dir = os.path.join(save_dir, f"ludb_diffusion_fold_{fold_idx}", "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in directory: {ckpt_dir}")
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"[fold {fold_idx}] Using latest checkpoint: {os.path.basename(latest_ckpt)}")
    return latest_ckpt


def load_model(cfg, dataset, fold):
    with open(cfg["config"]) as f:
        model_config = json.load(f)

    print(f"Initializing model architecture (dataset: {dataset})...")
    model = create_model_from_config(model_config)

    if cfg["needs_fold"]:
        checkpoint_path = find_best_checkpoint("outputs", fold)
    else:
        checkpoint_path = cfg["checkpoint"]

    print(f"Loading checkpoint: {checkpoint_path}")
    raw_state_dict = load_ckpt_state_dict(checkpoint_path)

    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("diffusion_ema.ema_model."):
            cleaned_state_dict[k.replace("diffusion_ema.ema_model.", "")] = v
        elif k.startswith("diffusion."):
            new_key = k.replace("diffusion.", "")
            if new_key not in cleaned_state_dict:
                cleaned_state_dict[new_key] = v
        else:
            if k not in cleaned_state_dict:
                cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("Checkpoint loaded")
    return model, device


def load_conditioning_arrays(cfg, dataset, fold, mode):
    if dataset == "ptbxl":
        base_path = "data/ptbxl"
        labels = np.load(os.path.join(base_path, f"{mode}_label.npy"), mmap_mode='r')
        p_info = np.load(os.path.join(base_path, f"{mode}_p_infor.npy"), mmap_mode='r')
        return {"labels": labels, "p_info": p_info}
    if dataset == "qtdb":
        base_path = "data/qtdb"
        masks = np.load(os.path.join(base_path, f"{mode}_mask.npy"), mmap_mode='r')
        return {"masks": masks}
    if dataset == "ludb":
        base_path = f"data/ludb/fold_{fold}"
        labels = np.load(os.path.join(base_path, f"{mode}_labels.npy"), mmap_mode='r')
        p_info = np.load(os.path.join(base_path, f"{mode}_pinfor.npy"), mmap_mode='r')
        masks = np.load(os.path.join(base_path, f"{mode}_masks.npy"), mmap_mode='r')
        return {"labels": labels, "p_info": p_info, "masks": masks}
    raise ValueError(f"Unknown dataset: {dataset}")


def build_condition(dataset, cfg, arrays, idx):
    if dataset == "ptbxl":
        info_min = torch.tensor([0.0, 0.0, 0.0, 0.0])
        info_max = torch.tensor([100.0, 1.0, 210.0, 250.0])
        label = torch.from_numpy(arrays["labels"][idx].copy()).float()
        pinfo = torch.from_numpy(arrays["p_info"][idx].copy()).float()
        pinfo = (pinfo - info_min) / (info_max - info_min + 1e-8)
        return {"disease_labels": label, "patient_info": pinfo}

    if dataset == "qtdb":
        mask = torch.from_numpy(arrays["masks"][idx].copy()).long()
        return {
            "disease_labels": torch.zeros(71).float(),
            "patient_info": torch.zeros(4).float(),
            "seg_mask": mask,
        }

    if dataset == "ludb":
        info_min = torch.tensor([0.0, 11.0])
        info_max = torch.tensor([1.0, 90.0])
        label = torch.tensor(arrays["labels"][idx]).float().unsqueeze(0)
        pinfo = torch.from_numpy(arrays["p_info"][idx].copy()).float()
        pinfo = (pinfo - info_min) / (info_max - info_min + 1e-8)
        mask = torch.from_numpy(arrays["masks"][idx].copy()).long()
        if mask.shape[0] == 500 and mask.shape[1] == 12:
            mask = mask.transpose(0, 1)
        gen_len = cfg["generation_length"]
        if mask.shape[-1] < gen_len:
            mask = torch.nn.functional.pad(mask, (0, gen_len - mask.shape[-1]), value=0)
        elif mask.shape[-1] > gen_len:
            mask = mask[:, :gen_len]
        return {"disease_labels": label, "patient_info": pinfo, "seg_mask": mask}

    raise ValueError(f"Unknown dataset: {dataset}")


def generate_one_mode(dataset, fold, mode, model, device):
    cfg = DATASET_CONFIGS[dataset]

    arrays = load_conditioning_arrays(cfg, dataset, fold, mode)

    total_samples = next(arrays[k].shape[0] for k in ("labels", "masks") if k in arrays)

    fold_tag = f"_fold{fold}" if cfg["needs_fold"] else ""
    output_save_path = f"outputs/generated_{dataset}{fold_tag}_{mode}.npy"
    os.makedirs("outputs", exist_ok=True)

    print(f"Generating {dataset} {mode} split ({total_samples} samples)...")

    generated_results = []
    batch_size = cfg["batch_size"]
    gen_len = cfg["generation_length"]

    for i in tqdm(range(0, total_samples, batch_size)):
        upper_idx = min(i + batch_size, total_samples)

        batch_cond_list = []
        for idx in range(i, upper_idx):
            batch_cond_list.append(build_condition(dataset, cfg, arrays, idx))

        with torch.no_grad():
            conditioning_tensors = model.conditioner(batch_cond_list, device)

            output = generate_diffusion_cond(
                model,
                steps=50,
                cfg_scale=cfg["cfg_scale"],
                conditioning_tensors=conditioning_tensors,
                sample_size=gen_len,
                batch_size=len(batch_cond_list),
                device=device,
                sampler_type="dpmpp-2m-sde",
            )

            output = output[:, :, :cfg["target_length"]]
            generated_results.append(output.cpu().numpy())

            del output, conditioning_tensors
            if i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()

    final_data = np.concatenate(generated_results, axis=0)

    if dataset == "ludb":
        final_data = final_data.transpose(0, 2, 1)

    np.save(output_save_path, final_data)

    print("Generation finished")
    print(f"Final shape: {final_data.shape}")
    print(f"Saved to: {output_save_path}")


def run_inference(dataset, fold, mode):
    cfg = DATASET_CONFIGS[dataset]
    torch.set_float32_matmul_precision('high')

    model, device = load_model(cfg, dataset, fold)

    modes = ["test", "train"] if mode == "both" else [mode]
    for m in modes:
        print(f"\n>>> Processing mode: {m}")
        generate_one_mode(dataset, fold, m, model, device)


def run_inference_5fold(dataset, gpu_list):
    if not DATASET_CONFIGS[dataset]["needs_fold"]:
        raise ValueError(f"5-fold inference is only supported for fold-based datasets (got '{dataset}')")

    processes = []
    print(f"Starting 5-fold {dataset} generation (train+test) on {len(gpu_list)} GPUs...")

    for fold_idx in range(5):
        gpu_id = gpu_list[fold_idx]
        log_file = open(f"infer_fold_{fold_idx}.log", "w")

        cmd = [
            sys.executable, "inference.py",
            "--dataset", dataset,
            "--fold", str(fold_idx),
            "--mode", "both",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"   -> Starting fold {fold_idx} (GPU {gpu_id}), log: infer_fold_{fold_idx}.log")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        processes.append((p, log_file))

        time.sleep(3)

    print("\nAll inference processes started in the background!")
    print("Waiting for all tasks to finish...")

    for p, log_f in processes:
        p.wait()
        log_f.close()

    print("All 5-fold inference tasks finished!")


def main():
    parser = argparse.ArgumentParser(description="Generate ECGs with TriCondDiT")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to generate from")
    parser.add_argument("--fold", type=str, default=None,
                        help="Fold index (0-4) for fold-based datasets, or 'all' to run all folds in parallel")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["train", "test", "both"],
                        help="Which split to generate (train, test, or both for fold-based datasets)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (controls CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for the 5-fold parallel run (e.g. '0,1,2,3,4')")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.fold == "all":
        gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else DEFAULT_5FOLD_GPUS
        if len(gpu_list) < 5:
            raise ValueError("Need at least 5 GPU IDs for the 5-fold parallel run (use --gpus)")
        run_inference_5fold(args.dataset, gpu_list)
    else:
        run_inference(args.dataset, int(args.fold) if args.fold is not None else None, args.mode)


if __name__ == "__main__":
    main()
