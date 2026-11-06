"""PyTorch Dataset 封装，按 seed 划分 train/val/test。"""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SheafFragmentDataset(Dataset):
    def __init__(self, index_path: str, split: str = "train",
                 val_frac: float = 0.2, test_frac: float = 0.2):
        idx = json.loads(Path(index_path).read_text())
        # 按 seed 划分（论文 Sec. 5.1.2："train/val/test split is by system instance"）
        seeds = sorted({int(Path(p).stem.split("seed")[-1]) for paths in idx.values() for p in paths})
        n = len(seeds)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        train_seeds = set(seeds[:n - n_val - n_test])
        val_seeds = set(seeds[n - n_val - n_test:n - n_test])
        test_seeds = set(seeds[n - n_test:])
        target = {"train": train_seeds, "val": val_seeds, "test": test_seeds}[split]
        self.paths = []
        for cls, paths in idx.items():
            for p in paths:
                seed = int(Path(p).stem.split("seed")[-1])
                if seed in target:
                    self.paths.append(p)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = torch.load(self.paths[i])
        return {
            "onsite": d["fragments"]["onsite"],
            "hopping": d["fragments"].get("hopping"),
            "observables": d["observables"],
            "meta": d["meta"],
        }