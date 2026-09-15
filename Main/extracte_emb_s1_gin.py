import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINConv, GlobalAttention, global_mean_pool, BatchNorm
)
from dgl.data.utils import load_graphs
import warnings

warnings.filterwarnings("ignore", message="'nn.glob.GlobalAttention' is deprecated")


BASE_DIR   = "D:/DATASET_RE/STAGE_2"
WEIGHTS_TEMPLATE = "D:/M_114_駱沛葳/run/s1try/fold_{fold}_best_model.pth" #放一階段權重檔案的路徑
OUT_DIR    = "D:/DATASET_RE/stage1_Qtry"

BATCH_SIZE  = 32
DEVICE      = "cuda"   
USE_FP16    = True     
NUM_WORKERS = 0

FOLD_IDS = [1, 2, 3, 4, 5] # [0, 1, 2, 3, 4] 

CANDIDATE_FEAT_KEYS = ["feat", "x", "h", "node_feat", "features"]  


class GIN_S1(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 use_virtual_node: bool = False ):
        super().__init__()
        self.use_virtual_node = use_virtual_node

        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm(hidden_channels)

        self.bn_vn_inject = BatchNorm(hidden_channels)

        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm(hidden_channels)

        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        if self.use_virtual_node:
            self.vn_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        if self.use_virtual_node:
            g_pool = global_mean_pool(x, batch)  
            v = self.vn_mlp(g_pool)            
            x = x + v[batch]                     
            x = self.bn_vn_inject(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)

        x_pool = self.pool(x, batch)
        x_final = self.dropout(x_pool)
        return self.lin(x_final)

    def get_embedding(self, x, edge_index, batch):
        with torch.no_grad():

            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = torch.relu(x)

            if self.use_virtual_node:
                g_pool = global_mean_pool(x, batch)
                v = self.vn_mlp(g_pool)
                x = x + v[batch]
                x = self.bn_vn_inject(x)


            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = torch.relu(x)


            x_pool = self.pool(x, batch)
            return x_pool


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def infer_model_shapes_from_state_dict(sd: dict, in_channels_from_data: int):
    """
    根據 state_dict + 資料本身的節點特徵維度，推 in_channels / hidden_channels / num_classes。
    - in_channels：直接從資料 ds 取 (x 的最後一維)
    - hidden_channels / num_classes：從 lin.weight 取
    """
    if "lin.weight" not in sd:
        raise KeyError("state_dict 缺少 lin.weight，無法推斷 hidden/num_classes 維度")

    lin_w = sd["lin.weight"]         
    num_classes, hidden_channels = lin_w.shape

    in_channels = in_channels_from_data
    return in_channels, hidden_channels, num_classes



def pick_feat(g) -> Optional[torch.Tensor]:
    """嘗試多個鍵名抓節點特徵；若都沒有，回傳 None。"""
    for k in CANDIDATE_FEAT_KEYS:
        if k in g.ndata:
            return g.ndata[k]
    return None


class GraphDataset(torch.utils.data.Dataset):
    """
    只回傳 PyG Data；meta 放在 self.records，用 rec_id 回查。
    內建詳細診斷與統計。
    """
    def __init__(self, base_dir: str):
        self.records: List[Tuple[str, str, int]] = []  # (label_str, bin_file, graph_idx)
        self.data_list: List[Data] = []

        total_bins = 0
        total_loaded_graphs = 0
        skip_empty = 0
        skip_no_feat = 0
        skip_naninf = 0

        label_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"[SCAN] label dirs: {label_dirs}")

        for label_str in label_dirs:
            dir_path = os.path.join(base_dir, label_str)
            try:
                _ = int(label_str)
            except:
                print(f"[WARN] 非數字標籤資料夾，跳過: {label_str}")
                continue

            bin_files = [f for f in os.listdir(dir_path) if f.endswith(".bin")]
            print(f"[SCAN] label={label_str}, bin_files={len(bin_files)}")

            for bin_file in bin_files:
                bin_path = os.path.join(dir_path, bin_file)
                try:
                    graphs, _ = load_graphs(bin_path)
                except Exception as e:
                    print(f"[ERROR] 讀取 .bin 失敗: {bin_path} -> {e}")
                    continue

                total_bins += 1
                print(f"  [BIN] {bin_file}: graphs={len(graphs)}")

                for gidx, g in enumerate(graphs):
                    total_loaded_graphs += 1

                    if g.num_nodes() == 0:
                        skip_empty += 1
                        print(f"    [SKIP empty] {label_str}/{bin_file} idx={gidx}")
                        continue

                    x = pick_feat(g)
                    if x is None:
                        print(f"    [SKIP no_feat] {label_str}/{bin_file} idx={gidx} keys={list(g.ndata.keys())}")
                        skip_no_feat += 1
                        continue

                    if torch.isinf(x).any() or torch.isnan(x).any():
                        skip_naninf += 1
                        print(f"    [SKIP NaN/Inf] {label_str}/{bin_file} idx={gidx}")
                        continue

                    src, dst = g.edges()
                    edge_index = torch.stack([src, dst], dim=0)
                    y = torch.tensor([int(label_str)], dtype=torch.long)
                    rec_id = torch.tensor([len(self.records)], dtype=torch.long)

                    data = Data(x=x, edge_index=edge_index, y=y, rec_id=rec_id)
                    self.data_list.append(data)
                    self.records.append((label_str, bin_file, gidx))

        print(f"[INFO] 掃描完成：label_dirs={len(label_dirs)}, bin_files={total_bins}, "
              f"graphs_loaded(raw)={total_loaded_graphs}")
        print(f"[INFO] 跳過統計：empty={skip_empty}, no_feat={skip_no_feat}, NaN/Inf={skip_naninf}")
        print(f"[INFO] Total graphs (after filtering): {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def main():
    device = torch.device(DEVICE if (DEVICE != "cuda" or torch.cuda.is_available()) else "cpu")
    out_root = Path(OUT_DIR)
    safe_mkdir(out_root)

    ds = GraphDataset(BASE_DIR)
    if len(ds) == 0:
        print("[ERROR] 找不到任何可用圖，請根據上方 [SCAN]/[SKIP] 訊息檢查：")
        print("       1) label 子資料夾是否為純數字？")
        print("       2) .bin 檔是否存在？可讀？")
        print("       3) 節點特徵鍵名是不是 'feat/x/h/node_feat/features' 以外？")
        print("       4) 是否所有圖都是空圖或含 NaN/Inf？")
        return

    sample_in_channels = ds[0].x.size(1)
    print(f"[INFO] sample_in_channels from dataset = {sample_in_channels}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    amp_ctx = (torch.cuda.amp.autocast if (USE_FP16 and device.type == "cuda")
               else torch.cpu.amp.autocast)

    for fold in FOLD_IDS:
        weight_path = WEIGHTS_TEMPLATE.format(fold=fold)
        if not os.path.exists(weight_path):
            print(f"[WARN] fold {fold}: 找不到權重檔 {weight_path}，跳過這個 fold")
            continue

        print("=" * 80)
        print(f"[FOLD {fold}] 使用權重檔：{weight_path}")
        print("=" * 80)

        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = ckpt if (isinstance(ckpt, dict) and "state_dict" not in ckpt) else ckpt["state_dict"]
        cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}

        in_ch, hid_ch, num_cls = infer_model_shapes_from_state_dict(
            cleaned, in_channels_from_data=sample_in_channels
        )

        print(f"[FOLD {fold}] Inferred dims -> in={in_ch}, hidden={hid_ch}, num_classes={num_cls}")

        model = GIN_S1(in_ch, hid_ch, num_cls, use_virtual_node=False).to(device)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[FOLD {fold}] [WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[FOLD {fold}] [WARN] Unexpected keys: {unexpected}")
        model.eval()

        fold_out_dir = out_root / f"fold_{fold}"
        safe_mkdir(fold_out_dir)

        index_rows = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with amp_ctx():
                    embeds = model.get_embedding(batch.x, batch.edge_index, batch.batch)  # [B, hidden]

                rec_ids = batch.rec_id.view(-1).tolist()
                for i, rec in enumerate(rec_ids):
                    label_str, bin_file, gidx = ds.records[rec]
                    save_dir = fold_out_dir / label_str / Path(bin_file).stem
                    safe_mkdir(save_dir)
                    save_path = save_dir / f"embed_g{gidx}.pt"
                    torch.save(embeds[i].detach().cpu(), save_path)
                    index_rows.append([label_str, bin_file, gidx, str(save_path)])

        index_path = fold_out_dir / "index.csv"
        with open(index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "bin_file", "graph_idx", "embedding_path"])
            w.writerows(index_rows)

        print(f"[FOLD {fold}] Exported {len(index_rows)} embeddings.")
        print(f"[FOLD {fold}] Index saved to: {index_path}")

    print("[DONE] 所有 fold 處理完成。")


if __name__ == "__main__":
    main()

