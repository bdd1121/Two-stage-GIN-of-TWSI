# train_stage2_crossattn.py
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GlobalAttention, BatchNorm
from dgl.data.utils import load_graphs
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
from torch_geometric.data import Batch

warnings.filterwarnings("ignore", message="'nn.glob.GlobalAttention' is deprecated")


BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LR = 0.0001

# 二階段資料與一階 Q 索引
BASE_DIR_S2  = "D:/DATASET_RE/STAGE_2"   # 二階 .bin 根目錄

S1_INDEX_CSV_TEMPLATE = "D:/DATASET_RE/stage1_Qtry/fold_{fold}/index.csv"

SPLIT_DIR = "D:/final_method/splits_5foldtest"

LOG_ROOT  = "D:/M_114_駱沛葳/run/s2t2ry"  

AGG_PER_BIN = False # Q 聚合策略

AGG_MODE    = "mean"   # "mean" 或 "max"


CANDIDATE_FEAT_KEYS = ["feat", "x", "h", "node_feat", "features"]

Triplet = Tuple[str, str, int]  # (label_str, bin_stem, graph_idx)



def load_id_list(txt_path: str) -> set:
    """
    從 txt 載入要用的 sample id 集合。
    假設每一行格式類似：
      - "SS-001"
      - "SS-001.png"
      - "0/SS-001.bin"
      - "0 SS-001"
    這裡會取「最後一個 token」，再取檔名 stem，當成 bin_stem。
    """
    ids = set()
    if not os.path.exists(txt_path):
        print(f"[WARN] split file not found: {txt_path}")
        return ids
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            token = parts[-1]
            stem = Path(token).stem
            ids.add(stem)
    return ids



def get_Q_from_batch(batch, device=None):
    if hasattr(batch, 'Q') and isinstance(batch.Q, torch.Tensor):
        Q = batch.Q
    elif hasattr(batch, 'q') and isinstance(batch.q, torch.Tensor):
        Q = batch.q
    else:
        raise AttributeError("Batch has neither 'Q' nor 'q'. Check collate or dataset.")

    Q = Q.float()   
    return Q.to(device) if device is not None else Q


def load_q_index(index_csv: str) -> Dict[Triplet, str]:
    import csv
    mp: Dict[Triplet, str] = {}
    with open(index_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = row['label']
            stem = Path(row['bin_file']).stem
            gidx = int(row['graph_idx'])
            mp[(label, stem, gidx)] = row['embedding_path']
    return mp


def load_q_by_triplet(q_map: Dict[Triplet, str], label_str: str, bin_stem: str, gidx: int) -> torch.Tensor:
    path = q_map[(label_str, bin_stem, gidx)]
    q = torch.load(path, map_location="cpu")
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q)
    return q.float()  


def aggregate_q_by_bin(q_map: Dict[Triplet, str], label_str: str, bin_stem: str, mode: str = "mean") -> torch.Tensor:
    paths: List[str] = [p for (lbl, stem, _), p in q_map.items() if lbl == label_str and stem == bin_stem]
    qs = []
    for p in paths:
        q = torch.load(p, map_location="cpu")
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q)
        qs.append(q.float())   

    if len(qs) == 0:
        raise KeyError(f"No Q found for ({label_str}, {bin_stem}) in index.")

    Q = torch.stack(qs, 0)
    if mode == "mean":
        return Q.mean(0)
    elif mode == "max":
        return Q.max(0).values
    else:
        raise ValueError("mode must be 'mean' or 'max'")


def collate_q(batch_list):
    batch = Batch.from_data_list(batch_list)
    Q = torch.cat([d.q for d in batch_list], dim=0).contiguous()
    batch.Q = Q
    batch.q = Q
    return batch


# -------------------- Stage-2 Dataset --------------------
class Stage2GraphDataset(torch.utils.data.Dataset):
    """
    從 base_dir 讀取 Stage-2 的 .bin 圖，並附上一階 Q。
    - 若 aggregate_per_bin=True，則同一 .bin 下所有子圖共享一個聚合後的 Q（建議）
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
                bin_path = os.path.join(dpath, bin_file)
                graphs, _ = load_graphs(bin_path)

                bin_stem = Path(bin_file).stem
                Q_agg: Optional[torch.Tensor] = None
                if self.aggregate_per_bin:
                    try:
                        Q_agg = aggregate_q_by_bin(self.q_map, label_str, bin_stem, mode=self.agg_mode)
                    except Exception as e:
                        print(f"[WARN] aggregate_q_by_bin failed for {label_str}/{bin_file}: {e}")
                        Q_agg = None

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

                    data = Data(x=x, edge_index=edge_index, y=y)
                    data.q = q.view(1, -1)   # [1, hidden]
                    data.bin_stem = bin_stem
                    self.data_list.append(data)
                    self.records.append((label_str, bin_file, gidx))

        print(f"[S2] Loaded {len(self.data_list)} graphs from {self.base_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i]


class CrossAttention(nn.Module):
    """
    單頭 Cross-Attention：
    Q 來自 Stage1（[B, d]），K/V 來自 Stage2 的節點嵌入 H（[sum(N), d] + batch）
    """
    def __init__(self, d_model: int, d_k: int = None):
        super().__init__()
        if d_k is None:
            d_k = d_model
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, Q: torch.Tensor, H: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        """
        Q: [B, d]; H: [N_total, d]; batch_vec: 長度 N_total，指定每個節點屬於哪個 graph
        回傳: [B, d_k] 的 context（每個 graph 一個向量）
        """
        Qp = self.Wq(Q)        # [B, d_k]
        Kp = self.Wk(H)        # [N_total, d_k]
        Vp = self.Wv(H)        # [N_total, d_k]

        B = Q.size(0)
        out = []
        for b in range(B):
            mask = (batch_vec == b)
            Kb = Kp[mask]      # [N_b, d_k]
            Vb = Vp[mask]      # [N_b, d_k]
            if Kb.numel() == 0:
                out.append(torch.zeros(Qp.size(1), device=Q.device))
                continue
            attn = torch.matmul(Qp[b:b+1], Kb.t()) / self.scale  # [1, N_b]
            attn = torch.softmax(attn, dim=-1)                   # [1, N_b]
            ctx = torch.matmul(attn, Vb)                         # [1, d_k]
            out.append(ctx.squeeze(0))
        return torch.stack(out, dim=0)  # [B, d_k]


class Stage2_GIN_Cross(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm(hidden)

        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm(hidden)

        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Linear(hidden, num_classes)

        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        self.cross = CrossAttention(d_model=hidden, d_k=hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, Q):
        h = self.conv1(x, edge_index); h = self.bn1(h); h = torch.relu(h); h = self.dropout(h)
        h = self.conv2(h, edge_index); h = self.bn2(h); h = torch.relu(h)

        g_pool = self.pool(h, batch)              # [B, hidden] （Stage-2 自身圖表示）
        ctx = self.cross(Q, h, batch)             # [B, hidden] （Q 對 H 的 cross-attended）

        z = torch.cat([g_pool, ctx], dim=-1)      # [B, 2*hidden]
        out = self.fuse(z)                        # [B, num_classes]
        return out


# -------------------- 指標 --------------------
def metrics_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    sen  = tp / (tp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    f1   = 2 * prec * sen / (prec + sen + 1e-12)
    npv  = tn / (tn + fn + 1e-12)
    return acc, spec, sen, prec, f1, npv



def main():
    os.makedirs(LOG_ROOT, exist_ok=True)

    all_val_metrics = []
    test_results = []

    for fold in range(5):   
        print("=" * 80)
        print(f"[FOLD {fold+1}] 開始")
        print("=" * 80)

        s1_index_csv = S1_INDEX_CSV_TEMPLATE.format(fold=fold+1)
        if not os.path.exists(s1_index_csv):
            print(f"[ERROR] Stage1 index.csv not found for fold {fold}: {s1_index_csv}")
            continue

        ds = Stage2GraphDataset(
            base_dir=BASE_DIR_S2,
            s1_index_csv=s1_index_csv,
            aggregate_per_bin=AGG_PER_BIN,
            agg_mode=AGG_MODE
        )
        if len(ds) == 0:
            print(f"[ERROR] Stage-2 資料為 0，fold={fold}")
            continue

        labels = np.array([int(d.y.item()) for d in ds])
        sample0 = ds[0]
        in_dim = sample0.x.size(1)
        hidden = sample0.q.size(1)
        num_classes = len(np.unique(labels))
        print(f"[FOLD {fold+1}] in_channels={in_dim}, hidden={hidden}, num_classes={num_classes}")

        train_ids = load_id_list(os.path.join(SPLIT_DIR, f"fold_{fold}_train.txt"))
        val_ids   = load_id_list(os.path.join(SPLIT_DIR, f"fold_{fold}_val.txt"))
        test_ids  = load_id_list(os.path.join(SPLIT_DIR, f"fold_{fold}_test.txt"))

        if not train_ids or not val_ids or not test_ids:
            print(f"[WARN] fold_{fold} split txt 有空集合，請檢查 SPLIT_DIR")
        
        train_index, val_index, test_index = [], [], []
        for idx, (_, bin_file, _) in enumerate(ds.records):
            stem = Path(bin_file).stem
            if stem in train_ids:
                train_index.append(idx)
            if stem in val_ids:
                val_index.append(idx)
            if stem in test_ids:
                test_index.append(idx)

        print(f"[FOLD {fold+1}] Train={len(train_index)}, Val={len(val_index)}, Test={len(test_index)}")


        train_data = torch.utils.data.Subset(ds, train_index)
        val_data   = torch.utils.data.Subset(ds, val_index)
        test_data  = torch.utils.data.Subset(ds, test_index)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_q)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_q)
        test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_q)

        # ---- 模型、損失、優化器 ----
        model = Stage2_GIN_Cross(in_channels=in_dim, hidden=hidden, num_classes=num_classes, dropout=0.5).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=MODEL_LR, weight_decay=1e-4)

        fold_logdir = os.path.join(LOG_ROOT, f"fold_{fold+1}")
        writer = SummaryWriter(log_dir=fold_logdir)

        best_val_acc = 0
        best_wts = None
        fold_metrics = []
        since_fold = time.time()

        # ================== 訓練迴圈 ==================
        for epoch in range(EPOCHS):
            start_epoch = time.time()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            model.train()
            for batch in train_loader:
                Q = get_Q_from_batch(batch, device=DEVICE)
                batch = batch.to(DEVICE)

                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch, Q)
                loss = criterion(out, batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * batch.num_graphs
                preds = out.argmax(1)
                correct_train += (preds == batch.y.view(-1)).sum().item()
                total_train += batch.num_graphs

            train_loss /= (total_train + 1e-12)
            train_acc = correct_train / (total_train + 1e-12)
            print(f"[FOLD {fold+1}] Train Epoch {epoch+1}: Loss={train_loss:.6f} Acc={train_acc:.4f} "
                  f"Time={(time.time()-start_epoch):.0f}s")
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)

            # -------- 驗證 --------
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    Q = get_Q_from_batch(batch, device=DEVICE)
                    batch = batch.to(DEVICE)
                    out = model(batch.x, batch.edge_index, batch.batch, Q)
                    loss = criterion(out, batch.y.view(-1))
                    val_loss += loss.item() * batch.num_graphs
                    preds = out.argmax(1)
                    correct_val += (preds == batch.y.view(-1)).sum().item()
                    total_val += batch.num_graphs
                    y_true.extend(batch.y.view(-1).cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            val_acc, specificity, sensitivity, precision, f1, npv = metrics_binary(
                np.array(y_true), np.array(y_pred)
            )
            val_loss /= (total_val + 1e-12)
            FNR = 1 - sensitivity

            print(f"[FOLD {fold+1}] Val Epoch {epoch+1}: Loss={val_loss:.6f} Acc={val_acc:.4f} "
                  f"Spec={specificity:.4f} Sen={sensitivity:.4f} Prec={precision:.4f} "
                  f"F1={f1:.4f} NPV={npv:.4f} FNR={FNR:.4f}")

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)

            fold_metrics.append({
                'epoch': epoch+1,
                'loss': float(val_loss),
                'accuracy': float(val_acc),
                'specificity': float(specificity),
                'sensitivity': float(sensitivity),
                'precision': float(precision),
                'f1_score': float(f1),
                'NPV': float(npv),
                'FNR': float(FNR),
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_wts = copy.deepcopy(model.state_dict())
                print(f"[FOLD {fold+1}] New best model at Epoch {epoch+1}, Acc={best_val_acc:.4f}")

        # ---- 保存最佳權重 ----
        if best_wts is not None:
            torch.save(best_wts, os.path.join(LOG_ROOT, f"fold_{fold+1}_best_model.pth"))
        else:
            print(f"[FOLD {fold+1}] WARNING: best_wts is None，可能 val 沒有成功跑完")

        # -------- 測試 --------
        if best_wts is not None:
            model.load_state_dict(best_wts)
        model.eval()
        start_test = time.time()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                Q = get_Q_from_batch(batch, device=DEVICE)
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch, Q)
                loss = criterion(out, batch.y.view(-1))
                test_loss += loss.item() * batch.num_graphs
                preds = out.argmax(1)
                correct_test += (preds == batch.y.view(-1)).sum().item()
                total_test += batch.num_graphs
                y_true.extend(batch.y.view(-1).cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        test_acc, specificity, sensitivity, precision, f1, npv = metrics_binary(
            np.array(y_true), np.array(y_pred)
        )
        test_loss /= (total_test + 1e-12)
        FNR = 1 - sensitivity

        print(f"[FOLD {fold+1}] Test: Loss={test_loss:.6f} Acc={test_acc:.4f} "
              f"Spec={specificity:.4f} Sen={sensitivity:.4f} Prec={precision:.4f} "
              f"F1={f1:.4f} FNR={FNR:.4f} "
              f"Time={(time.time()-start_test):.0f}s")

        test_results.append({
            'fold': fold+1,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'precision': float(precision),
            'f1_score': float(f1),
            'NPV': float(npv),
            'FNR': float(FNR),
        })

        elapsed = time.time() - since_fold
        print(f"[FOLD {fold+1}] complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
        all_val_metrics.append(fold_metrics)
        writer.close()

    # ---- 平均驗證指標 ----
    avg_per_fold = {}
    for idx, metrics in enumerate(all_val_metrics, start=1):
        if not metrics:
            continue
        avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0] if k != 'epoch'}
        avg_per_fold[f"Fold {idx}"] = avg
    print("Average validation metrics per fold:")
    for fold_name, m in avg_per_fold.items():
        print(f"{fold_name}: " + ", ".join([f"{k}={v:.4f}" for k, v in m.items()]))

    # ---- 總平均測試指標 ----
    if test_results:
        average_test = {k: np.mean([r[k] for r in test_results]) for k in test_results[0] if k!='fold'}
        print("Average test metrics:")
        print(", ".join([f"{k}={v:.4f}" for k, v in average_test.items()]))

    # ---- 匯出到 Excel ----
    val_long = []
    for fold_id, fold_list in enumerate(all_val_metrics, start=1):
        for row in fold_list:
            row2 = dict(row)
            row2['fold'] = fold_id
            val_long.append(row2)

    val_df = pd.DataFrame(val_long)
    test_df = pd.DataFrame(test_results)

    val_df.to_excel(os.path.join(LOG_ROOT, "validation_metrics.xlsx"), index=False)
    test_df.to_excel(os.path.join(LOG_ROOT, "test_results.xlsx"), index=False)
    print("All metrics computed and saved.")


if __name__ == "__main__":
    main()
