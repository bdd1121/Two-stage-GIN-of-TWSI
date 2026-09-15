# q_index.py
import csv
from pathlib import Path
from typing import Dict, Tuple, List
import torch

Triplet = Tuple[str, str, int]  # (label_str, bin_stem, graph_idx)

def load_q_index(index_csv: str) -> Dict[Triplet, str]:
    """
    讀取 s1 匯出的 index.csv，回傳 map: (label, bin_stem, gidx) -> embedding_path
    """
    mp: Dict[Triplet, str] = {}
    with open(index_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = row['label']
            stem = Path(row['bin_file']).stem
            gidx = int(row['graph_idx'])
            mp[(label, stem, gidx)] = row['embedding_path']
    return mp

def load_q_by_triplet(q_map: Dict[Triplet, str], label_str: str, bin_stem: str, gidx: int) -> torch.Tensor:
    """
    精確三元組取 Q（每個子圖各有 Q）
    """
    path = q_map[(label_str, bin_stem, gidx)]
    return torch.load(path)  # [hidden]

def aggregate_q_by_bin(q_map: Dict[Triplet, str], label_str: str, bin_stem: str, mode: str = "mean") -> torch.Tensor:
    """
    將同一個 .bin 的所有子圖 Q 聚合成單一 Q（適合二階段用一個 Q）
    mode: mean / max
    """
    paths: List[str] = [p for (lbl, stem, _), p in q_map.items() if lbl == label_str and stem == bin_stem]
    qs = [torch.load(p) for p in paths]  # [N, hidden]
    if len(qs) == 0:
        raise KeyError(f"No Q found for ({label_str}, {bin_stem}) in index.")
    Q = torch.stack(qs, 0)
    if mode == "mean":
        return Q.mean(0)
    elif mode == "max":
        return Q.max(0).values
    else:
        raise ValueError("mode must be 'mean' or 'max'")
