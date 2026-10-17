# TriCondDiT

TriCondDiT, a diffusion-transformer framework for conditional ECG generation using disease labels, waveform segmentation masks, and patient information.

## Repository structure

```text
TriCondDiT/
├── configs/                 # Dataset-specific model and training configurations
├── data/                    # NumPy dataset loaders
├── inference/               # Sampling and generation utilities
├── models/                  # DiT backbone and conditioning modules
├── training/                # Training wrapper and optimization utilities
├── train.py                 # Training entry point
├── inference.py             # Conditional generation entry point
└── setup.sh                 # Reference environment setup script
```

## Environment

The reference implementation uses Python 3.10 and CUDA 11.8. A CUDA-capable GPU is required by the current training script.

The provided setup script creates the required Conda environment and installs the CUDA 11.8 PyTorch packages together with the necessary dependencies:

```bash
bash setup.sh
```

After installation, activate the Conda environment defined in `setup.sh` before running the training or inference scripts.

## Datasets

The datasets are not redistributed with this repository. Download them from their official PhysioNet pages and follow the license and citation requirements given on each page.

| Dataset | Official download page                                       | Configuration used here                                      |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| PTB-XL  | [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | 12 leads, 100 Hz, 1,000 samples, 71-label conditioning and patient information |
| LUDB    | [LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/)     | 12 leads, 500 Hz, 500-sample R-centered windows, binary labels, masks, and patient information |
| QTDB    | [QT Database v1.0.0](https://physionet.org/content/qtdb/1.0.0/) | Single lead, 125 Hz, 500 samples, waveform-mask conditioning |

The complete official datasets can also be downloaded with the commands published by PhysioNet:

```bash
# PTB-XL v1.0.3
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# LUDB v1.0.1
wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/

# QTDB v1.0.0
wget -r -N -c -np https://physionet.org/files/qtdb/1.0.0/
```

## Data preparation

The current release expects preprocessed NumPy arrays and does not include the scripts used to convert the original WFDB files. Prepare the arrays described below and place them under `data/`. ECG arrays should use `float32`; label arrays may be binary integer or floating-point arrays; mask arrays should contain integer class indices.

### PTB-XL

```text
data/ptbxl/
├── train_data.npy           # [N, 12, 1000]
├── train_label.npy          # [N, 71]
├── train_p_infor.npy        # [N, 4]: age, sex, height, weight
├── test_data.npy            # optional for evaluation; not read during generation
├── test_label.npy           # [N_test, 71]
└── test_p_infor.npy         # [N_test, 4]
```

Use the 100 Hz PTB-XL recordings. The loader normalizes patient information with the bounds `[100, 1, 210, 250]` for age, sex, height, and weight, respectively. Training uses `train_data.npy`, `train_label.npy`, and `train_p_infor.npy`. Conditional generation uses the label and patient-information arrays for the selected split.

### QTDB

```text
data/qtdb/
├── train_data.npy           # [N, 1, 500]
├── train_mask.npy           # [N, 1, 500]
└── test_mask.npy            # [N_test, 1, 500]
```

QTDB signals are represented as single-lead, 500-sample segments at 125 Hz. Mask values follow the class mapping `0 = background`, `1 = P wave`, `2 = T wave`, and `3 = QRS complex`. Training uses the signal and mask arrays; conditional generation uses the mask array for the selected split.

### LUDB

LUDB uses five patient-level folds:

```text
data/ludb/
├── fold_0/
│   ├── train_data.npy       # [N, 12, 500] or [N, 500, 12]
│   ├── train_labels.npy     # [N], binary Normal/AF labels
│   ├── train_pinfor.npy     # [N, 2]: sex, age
│   ├── train_masks.npy      
│   ├── test_labels.npy
│   ├── test_pinfor.npy
│   └── test_masks.npy
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

Repeat the same files for all five folds. The loader accepts either channel-first or time-first LUDB signals and masks and converts them internally when necessary. Patient information is ordered as sex and age. Mask values use the same `0/1/2/3` mapping as QTDB.

## Training

Run commands from the repository root. Training uses a fixed random seed of 42, a batch size of 32, and 100,000 optimization steps by default. Checkpoints are written every 5,000 steps under `outputs/<experiment>/checkpoints/`.

### PTB-XL

```bash
python train.py --dataset ptbxl --gpu_id 0
```

### QTDB

```bash
python train.py --dataset qtdb --gpu_id 0
```

### LUDB

Train one fold:

```bash
python train.py --dataset ludb --fold 0 --gpu_id 0
```

Train all five folds in parallel:

```bash
python train.py --dataset ludb --fold all --gpus 0,1,2,3,4
```

## Configuration

The main dataset-specific settings are stored in:

- `configs/ptbxl_diffusion.json`
- `configs/qtdb_diffusion.json`
- `configs/ludb_diffusion.json`

These files define the signal dimensions, conditioning modules, DiT architecture and learning rate, and classifier-free guidance dropout. Command-line options in `train.py` control the dataset, fold, GPU, checkpoint resumption, and maximum number of training iterations.
