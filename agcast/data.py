import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class RadarDataset(Dataset):

    def __init__(self, config, data_path, split="test", group=None, keys_file=None):
        self.config = dict(config)
        self.name = config["dataname"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")
        self.split = split
        self.group = group or config["groups"][split]

        path = Path(data_path).expanduser()
        if path.is_dir():
            path = path / f"{self.name}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = str(path.resolve())

        self.input_len = int(config["pre_seq_length"])
        self.output_len = int(config["aft_seq_length"])
        self.layout = config.get("storage_layout", "samples")

        with h5py.File(self.path, "r") as hf:
            if self.group not in hf:
                raise KeyError(f"Group '{self.group}' not found in {self.path}.")
            node = hf[self.group]
            if self.layout == "array":
                self.keys = list(range(len(node["data"])))
            elif self.name == "meteonet":
                length_key = f"{self.group}_len"
                length = int(hf[length_key][()]) if length_key in hf else len(node)
                self.keys = [str(i) for i in range(length)]
            elif self.name == "shanghai":
                self.keys = sorted((k for k in node if k.isdigit()), key=int)
            elif self.name == "cikm":
                self.keys = list(node.keys())
                try:
                    self.keys.sort(key=lambda k: int(k.split("_")[-1]))
                except ValueError:
                    self.keys.sort()
            else:
                raise ValueError(f"Unsupported dataset: {self.name}")

            if keys_file:
                selected = json.loads(Path(keys_file).read_text(encoding="utf-8"))
                self.keys = selected if self.layout == "array" else [str(k) for k in selected]

        if not self.keys:
            raise ValueError(f"No samples found in group '{self.group}'.")

    def __len__(self):
        return len(self.keys)

    def sample_id(self, index):
        return f"{self.group}/{self.keys[index]}"

    def __getitem__(self, index):
        key = self.keys[index]
        with h5py.File(self.path, "r") as hf:
            if self.layout == "array":
                data = hf[self.group]["data"][key]
            else:
                data = hf[self.group][key][()]

        if self.name == "cikm":
            data = data.astype(np.float32) / 255.0
            frames = torch.from_numpy(data)[:, None]
            size = tuple(self.config["in_shape"][-2:])
            if tuple(frames.shape[-2:]) != size:
                frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
        elif self.name == "sevir":
            if data.ndim == 3:
                data = data[..., None]
            frames = torch.tensor(data).permute(0, 3, 1, 2).float()
            frames = (frames - 33.44) / 47.54
            if self.split == "train" and self.config.get("use_augment", False):
                if np.random.rand() > 0.5:
                    frames = torch.flip(frames, dims=[-1])
        else:
            frames = torch.from_numpy(data).float()
            if frames.ndim == 3:
                frames = frames[:, None]
            frames = frames / (70.0 if self.name == "meteonet" else 255.0)
            size = tuple(self.config["in_shape"][-2:])
            if self.name == "meteonet" or tuple(frames.shape[-2:]) != size:
                frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)

        required = self.input_len + self.output_len
        return frames[:self.input_len], frames[self.input_len:required]


def make_loader(dataset, batch_size=1, workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=(dataset.split == "train"),
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )