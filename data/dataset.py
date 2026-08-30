# data/dataset.py
import numpy as np
import random
import torch


class PTBXLDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        label_path: str,
        patient_info_path: str,
        sample_size: int = 1000
    ):
        super().__init__()
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.patient_info = np.load(patient_info_path, mmap_mode='r')

        self.sample_size = sample_size

        assert len(self.data) == len(self.labels) == len(self.patient_info), \
            f"Length mismatch: Data={len(self.data)}, Label={len(self.labels)}, Info={len(self.patient_info)}"

        print(f'Loaded PTB-XL dataset: {len(self.data)} samples')

        # Patient info normalization bounds (4 dims: [Age, Sex, Height, Weight])
        self.info_min = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.info_max = torch.tensor([100.0, 1.0, 210.0, 250.0])

    def __len__(self):
        return len(self.data)

    def _normalize_info(self, info_tensor):
        """Normalize patient info to [0, 1]."""
        return (info_tensor - self.info_min) / (self.info_max - self.info_min + 1e-8)

    def __getitem__(self, idx):
        try:
            audio = torch.from_numpy(self.data[idx].copy()).float()

            orig_len = audio.shape[-1]
            if orig_len < self.sample_size:
                pad_len = self.sample_size - orig_len
                audio = torch.nn.functional.pad(audio, (0, pad_len), value=0.0)
                padding_mask = torch.cat([torch.ones(orig_len), torch.zeros(pad_len)], dim=0).bool()
            elif orig_len > self.sample_size:
                audio = audio[:, :self.sample_size]
                padding_mask = torch.ones(self.sample_size, dtype=torch.bool)
            else:
                padding_mask = torch.ones(self.sample_size, dtype=torch.bool)

            disease_labels = torch.from_numpy(self.labels[idx].copy()).float()

            raw_patient_info = torch.from_numpy(self.patient_info[idx].copy()).float()
            norm_patient_info = self._normalize_info(raw_patient_info)

            info = {
                "disease_labels": disease_labels,
                "patient_info": norm_patient_info,
                "sample_rate": 100,
                "seconds_total": self.sample_size / 100.0,
                "padding_mask": padding_mask
            }

            return (audio, info)

        except Exception as e:
            print(f'Failed to read PTB-XL index {idx}: {e}')
            return self[random.randrange(len(self))]

class QTDBDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        mask_path: str,
        sample_size: int = 512
    ):
        super().__init__()
        self.data = np.load(data_path, mmap_mode='r')
        self.masks = np.load(mask_path, mmap_mode='r')

        self.sample_size = sample_size

        assert len(self.data) == len(self.masks), "Data/mask length mismatch"

        self.original_len = self.data.shape[-1]
        self.pad_len = max(0, self.sample_size - self.original_len)

        print(f'Loaded QTDB dataset: {len(self.data)} samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            audio = torch.from_numpy(self.data[idx].copy()).float()

            mask = torch.from_numpy(self.masks[idx].copy()).long()

            if self.pad_len > 0:
                audio = torch.nn.functional.pad(audio, (0, self.pad_len), value=0.0)
                mask = torch.nn.functional.pad(mask, (0, self.pad_len), value=0)

            if audio.shape[-1] > self.sample_size:
                audio = audio[:, :self.sample_size]
                mask = mask[:, :self.sample_size]

            dummy_labels = torch.zeros(71).float()
            dummy_pinfo = torch.zeros(4).float()

            info = {
                "seg_mask": mask,
                "disease_labels": dummy_labels,
                "patient_info": dummy_pinfo,
                "sample_rate": 125,
                "seconds_total": self.sample_size / 125,
                "padding_mask": torch.cat([
                    torch.ones(self.original_len, dtype=torch.bool),
                    torch.zeros(self.pad_len, dtype=torch.bool)
                ], dim=0)
            }

            return (audio, info)

        except Exception as e:
            print(f'Failed to read QTDB index {idx}: {e}')
            return self[random.randrange(len(self))]

class LUDBDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path: str,
        label_path: str,
        patient_info_path: str,
        mask_path: str,
        sample_size: int = 500
    ):
        super().__init__()
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.patient_info = np.load(patient_info_path, mmap_mode='r')
        self.masks = np.load(mask_path, mmap_mode='r')

        self.sample_size = sample_size

        assert len(self.data) == len(self.labels) == len(self.patient_info) == len(self.masks), \
            f"LUDB length mismatch: data={len(self.data)}, labels={len(self.labels)}, info={len(self.patient_info)}, masks={len(self.masks)}"

        print(f'Loaded LUDB dataset: {len(self.data)} samples, sample size: {self.sample_size}')

        # Normalization bounds (patient info: [Sex, Age])
        self.info_min = torch.tensor([0.0, 11.0])
        self.info_max = torch.tensor([1.0, 90.0])

    def __len__(self):
        return len(self.data)

    def _normalize_info(self, info_tensor):
        """Simple min-max normalization."""
        if info_tensor.shape[0] != self.info_min.shape[0]:
            return info_tensor
        return (info_tensor - self.info_min) / (self.info_max - self.info_min + 1e-8)

    def __getitem__(self, idx):
        try:
            audio = torch.from_numpy(self.data[idx].copy()).float()

            if audio.shape[0] == 500 and audio.shape[1] == 12:
                audio = audio.transpose(0, 1)

            mask = torch.from_numpy(self.masks[idx].copy()).long()

            if mask.shape[0] == 500 and mask.shape[1] == 12:
                mask = mask.transpose(0, 1)

            orig_len = audio.shape[-1]

            if orig_len < self.sample_size:
                pad_len = self.sample_size - orig_len
                audio = torch.nn.functional.pad(audio, (0, pad_len), value=0.0)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                padding_mask = torch.cat([torch.ones(orig_len), torch.zeros(pad_len)], dim=0).bool()
            elif orig_len > self.sample_size:
                audio = audio[:, :self.sample_size]
                mask = mask[:, :self.sample_size]
                padding_mask = torch.ones(self.sample_size, dtype=torch.bool)
            else:
                padding_mask = torch.ones(self.sample_size, dtype=torch.bool)

            disease_labels = torch.tensor(self.labels[idx]).float().unsqueeze(0)

            raw_patient_info = torch.from_numpy(self.patient_info[idx].copy()).float()
            norm_patient_info = self._normalize_info(raw_patient_info)
            
            info = {
                "disease_labels": disease_labels,
                "patient_info": norm_patient_info,
                "seg_mask": mask,
                "sample_rate": 500,
                "seconds_total": self.sample_size / 500.0,
                "padding_mask": padding_mask
            }

            return (audio, info)

        except Exception as e:
            print(f'Failed to read LUDB index {idx}: {e}')
            return self[random.randrange(len(self))]


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=12, num_workers=4, shuffle = True):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    # PTB-XL
    if dataset_type == "ptbxl_npy":
        ptbxl_data_path = dataset_config.get("data_path", "")
        ptbxl_label_path = dataset_config.get("label_path", "")
        ptbxl_info_path = dataset_config.get("patient_info_path", "")

        train_set = PTBXLDataset(
            data_path=ptbxl_data_path,
            label_path=ptbxl_label_path,
            patient_info_path=ptbxl_info_path,
            sample_size=sample_size
        )

        return torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=dataset_config.get("drop_last", True),
            collate_fn=collation_fn
        )
    # QTDB
    if dataset_type == "qtdb_npy":
        qtdb_data_path = dataset_config.get("data_path", "")
        qtdb_mask_path = dataset_config.get("mask_path", "")

        train_set = QTDBDataset(
            data_path=qtdb_data_path,
            mask_path=qtdb_mask_path,
            sample_size=sample_size
        )

        return torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=dataset_config.get("drop_last", True),
            collate_fn=collation_fn
        )
    # LUDB
    if dataset_type == "ludb_npy":
        ludb_data_path = dataset_config.get("data_path", "")
        ludb_label_path = dataset_config.get("label_path", "")
        ludb_info_path = dataset_config.get("patient_info_path", "")
        ludb_mask_path = dataset_config.get("mask_path", "")

        train_set = LUDBDataset(
            data_path=ludb_data_path,
            label_path=ludb_label_path,
            patient_info_path=ludb_info_path,
            mask_path=ludb_mask_path,
            sample_size=sample_size
        )

        return torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=dataset_config.get("drop_last", True),
            collate_fn=collation_fn
        )

    raise NotImplementedError(f"Unknown dataset type: {dataset_type}")
