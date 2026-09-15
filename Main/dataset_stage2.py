# dataset_stage2.py
import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dgl.data.utils import load_graphs
from q_index import load_q_index, load_q_by_triplet, aggregate_q_by_bin

CANDIDATE_FEAT_KEYS = ["feat", "x", "h", "node_feat", "features"]

class Stage2GraphDataset(torch.utils.data.Dataset):
    """
    從 base_dir 讀取 Stage-2 的 .bin 圖，並附上一階 Q。
    - 若 aggregate_per_bin=True，則同一 .bin 下所有子圖共享一個聚合後的 Q（建議這個）
    - 否則就每個子圖取對應 gidx 的 Q
    """
    def __init__(self,
                 base_dir: str,
                 s1_index_csv: str,
                 aggregate_per_bin: bool = True,
                 agg_mode: str = "mean"):
        self.base_dir = base_dir
        self.q_map = load_q_index(s1_index_csv)
        self.aggregate_per_bin = aggregate_per_bin
        self.agg_mode = agg_mode

        self.records: List[Tuple[str, str, int]] = []  # (label_str, bin_file, gidx)
        self.data_list: List[Data] = []

        label_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for label_str in label_dirs:
            dpath = os.path.join(base_dir, label_str)
            try:
                _ = int(label_str)
            except:
                continue

            for bin_file in os.listdir(dpath):
                if not bin_file.endswith(".bin"):
                    continue
                graphs, _ = load_graphs(os.path.join(dpath, bin_file))

                # 先把聚合好的 Q 算好（若使用聚合）
                bin_stem = Path(bin_file).stem
                Q_agg: Optional[torch.Tensor] = None
                if self.aggregate_per_bin:
                    try:
                        Q_agg = aggregate_q_by_bin(self.q_map, label_str, bin_stem, mode=self.agg_mode)
                    except Exception as e:
                        print(f"[WARN] aggregate_q_by_bin failed for {label_str}/{bin_file}: {e}")
                        Q_agg = None  # 讓後面落回每圖個別對齊

                for gidx, g in enumerate(graphs):
                    if g.num_nodes() == 0:
                        continue
                    x = None
                    for k in CANDIDATE_FEAT_KEYS:
                        if k in g.ndata:
                            x = g.ndata[k]
                            break
                    if x is None:
                        print(f"[WARN] No feat in {label_str}/{bin_file} idx={gidx}; keys={list(g.ndata.keys())}")
                        continue
                    if torch.isinf(x).any() or torch.isnan(x).any():
                        print(f"[WARN] NaN/Inf in {label_str}/{bin_file} idx={gidx}; skip")
                        continue

                    src, dst = g.edges()
                    edge_index = torch.stack([src, dst], dim=0)
                    y = torch.tensor([int(label_str)], dtype=torch.long)

                    # 一階 Q
                    if Q_agg is not None:
                        q = Q_agg
                    else:
                        try:
                            q = load_q_by_triplet(self.q_map, label_str, bin_stem, gidx)
                        except Exception as e:
                            print(f"[WARN] load_q_by_triplet failed for {label_str}/{bin_file} gidx={gidx}: {e}")
                            continue

                    # 存到 Data 裡
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.q = q.view(1, -1)   # [1, hidden] 方便 batch 時 cat
                    data.bin_stem = bin_stem
                    data.rec_id = torch.tensor([len(self.records)], dtype=torch.long)
                    self.data_list.append(data)
                    self.records.append((label_str, bin_file, gidx))

        print(f"[S2] Loaded {len(self.data_list)} graphs from {self.base_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]
