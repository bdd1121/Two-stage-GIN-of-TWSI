import os
import time
import copy
import numpy as np
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
from dgl.data.utils import load_graphs
from torch_geometric.nn import GINConv, GlobalAttention, BatchNorm, global_mean_pool
import random

BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LR = 0.0001

# graph 根目錄，每個子資料夾名稱為標籤 (0/1/2/...)
base_dir = "D:/DATASET_RE/STAGE_1"

# 5-fold txt 檔的位置（裡面有 fold_0_train.txt / fold_0_val.txt / fold_0_test.txt ...）
SPLIT_DIR = "D:/final_method/splits_5foldtest"

# TensorBoard / model / Excel 輸出的根目錄
RUN_ROOT = "D:/M_114_駱沛葳/run/s1try"
os.makedirs(RUN_ROOT, exist_ok=True)


# ================== 1. 讀取 .bin 檔案並轉成 PyG Data ==================
data_list = []
file_keys = []   
max_val_all = -float('inf')
min_val_all = float('inf')

for label_str in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, label_str)
    if not os.path.isdir(dir_path):
        continue
    label = int(label_str)

    for bin_file in os.listdir(dir_path):
        if not bin_file.endswith('.bin'):
            continue

        graphs, _ = load_graphs(os.path.join(dir_path, bin_file))
        for g in graphs:
            if g.num_nodes() == 0:
                print(f"Warning: Skipping empty graph: {label_str}/{bin_file}")
                continue

            x = g.ndata['feat']
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"!!!!!!!!!!!!! BAD DATA DETECTED (NaN/Inf) !!!!!!!!!!!!!  {label_str}/{bin_file}")
                continue

            current_max = x.max().item()
            current_min = x.min().item()
            if current_max > max_val_all: max_val_all = current_max
            if current_min < min_val_all: min_val_all = current_min

            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0)
            y = torch.tensor([label], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

            # SS-xxx..._thumbnail_img.bin -> SS-xxx..._thumbnail_img.png
            stem = os.path.splitext(bin_file)[0]
            img_name = stem + ".png"
            file_keys.append(img_name)

print(f"Total graphs: {len(data_list)}")
print(f"======> Feature Value Range Check: Min={min_val_all}, Max={max_val_all}")

labels = np.array([d.y.item() for d in data_list])
file_keys = np.array(file_keys, dtype=str)


def load_name_set(txt_path):
    """讀取 fold_x_*.txt，回傳 set([檔名...])"""
    arr = np.loadtxt(txt_path, dtype=str)
    if arr.ndim == 0:   
        arr = np.array([arr])
    return set(arr.tolist())



class GIN_S1(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 use_virtual_node: bool = True):
        super().__init__()
        self.use_virtual_node = use_virtual_node

        # ---- GINConv layer 1 ----
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm(hidden_channels)

        # ---- VN 注入前的 BN ----
        self.bn_vn_inject = BatchNorm(hidden_channels)

        # ---- GINConv layer 2 ----
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm(hidden_channels)

        # ---- GlobalAttention Pooling ----
        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # ---- FC + Dropout ----
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # ---- Virtual Node MLP ----
        if self.use_virtual_node:
            self.vn_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

    def forward(self, x, edge_index, batch):
        # ----- Layer 1: GIN -----
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # ----- Virtual Node 注入 -----
        if self.use_virtual_node:
            g_pool = global_mean_pool(x, batch)  # [num_graphs, hidden]
            v = self.vn_mlp(g_pool)              # [num_graphs, hidden]
            x = x + v[batch]                     # 每個 node 加上對應圖的 VN
            x = self.bn_vn_inject(x)

        # ----- Layer 2: GIN -----
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)

        # ----- GlobalAttention Pooling + FC -----
        x_pool = self.pool(x, batch)
        x_final = self.dropout(x_pool)
        return self.lin(x_final)

    def get_embedding(self, x, edge_index, batch):
        with torch.no_grad():
            # ----- Layer 1 -----
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = torch.relu(x)

            if self.use_virtual_node:
                g_pool = global_mean_pool(x, batch)
                v = self.vn_mlp(g_pool)
                x = x + v[batch]
                x = self.bn_vn_inject(x)

            # ----- Layer 2 -----
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = torch.relu(x)

            # ----- GlobalAttention Pooling -----
            x_pool = self.pool(x, batch)
            return x_pool
  


all_val_metrics = []
test_results = []

in_dim = data_list[0].num_node_features
num_classes = len(np.unique(labels))

for i in range(5):
    print(f"\n================ Processing Fold {i+1} ================")
    since_fold = time.time()
    writer = SummaryWriter(log_dir=os.path.join(RUN_ROOT, f"fold_{i+1}"))

    # --- 用 txt 檔決定 train / val / test ---
    train_txt = os.path.join(SPLIT_DIR, f"fold_{i}_train.txt")
    val_txt   = os.path.join(SPLIT_DIR, f"fold_{i}_val.txt")
    test_txt  = os.path.join(SPLIT_DIR, f"fold_{i}_test.txt")

    train_names = load_name_set(train_txt)
    val_names   = load_name_set(val_txt)
    test_names  = load_name_set(test_txt)

    # 根據 file_keys（每個樣本的 png 名稱）挑 index
    train_index = [idx for idx, name in enumerate(file_keys) if name in train_names]
    val_index   = [idx for idx, name in enumerate(file_keys) if name in val_names]
    test_index  = [idx for idx, name in enumerate(file_keys) if name in test_names]

    print(
        f"Fold {i + 1}: "
        f"Train length: {len(train_index)}, "
        f"Validation length: {len(val_index)}, "
        f"Test length: {len(test_index)}"
    )

    train_data = [data_list[j] for j in train_index]
    val_data   = [data_list[j] for j in val_index]
    test_data  = [data_list[j] for j in test_index]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 模型、損失、優化器
    model = GIN_S1(in_channels=in_dim, hidden_channels=64, num_classes=num_classes,use_virtual_node=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_LR)
    
    best_val_acc = 0
    best_wts = None
    fold_metrics = []

    # ----------- 訓練 + 驗證迴圈 -----------
    for epoch in range(EPOCHS):
        start_epoch = time.time()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 訓練
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            preds = out.argmax(1)
            correct_train += (preds == batch.y).sum().item()
            total_train += batch.num_graphs

        train_loss /= total_train
        train_acc = correct_train / total_train
        print(f"Fold {i+1} | Train Epoch {epoch+1}: Loss={train_loss:.6f} Acc={train_acc:.4f} "
              f"Time={(time.time()-start_epoch):.0f}s")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)

        # 驗證
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds = out.argmax(1)
                correct_val += (preds == batch.y).sum().item()
                total_val += batch.num_graphs
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= total_val
        val_acc = correct_val / total_val

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn/(tn+fp) if (tn+fp) > 0 else 0.0
        sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        precision   = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        f1 = 2*(precision*sensitivity)/(precision+sensitivity) if (precision + sensitivity) > 0 else 0.0
        FNR = 1 - sensitivity
        PPV = precision
        NPV = tn/(tn+fn) if (tn+fn) > 0 else 0.0

        print(f"Fold {i+1} | Val Epoch {epoch+1}: Loss={val_loss:.6f} Acc={val_acc:.4f} "
              f"Spec={specificity:.4f} Sen={sensitivity:.4f} Prec={precision:.4f} "
              f"F1={f1:.4f} FNR={FNR:.4f} PPV={PPV:.4f} NPV={NPV:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        fold_metrics.append({
            'epoch': epoch+1,
            'loss': val_loss,
            'accuracy': val_acc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1,
            'FNR': FNR,
            'PPV': PPV,
            'NPV': NPV,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            print(f"New best model: Fold {i+1}, Epoch {epoch+1}, Acc={best_val_acc:.4f}")

    # 保存最佳權重
    best_model_path = os.path.join(RUN_ROOT, f"fold_{i+1}_best_model.pth")
    torch.save(best_wts, best_model_path)
    print(f"Best model for Fold {i+1} saved to {best_model_path}")

    # ----------- 測試 -----------
    start_test = time.time()
    model.load_state_dict(best_wts)
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            test_loss += loss.item() * batch.num_graphs
            preds = out.argmax(1)
            correct_test += (preds == batch.y).sum().item()
            total_test += batch.num_graphs
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss /= total_test
    test_acc = correct_test / total_test
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0.0
    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    precision   = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    f1 = 2*(precision*sensitivity)/(precision+sensitivity) if (precision + sensitivity) > 0 else 0.0
    FNR = 1 - sensitivity
    PPV = precision
    NPV = tn/(tn+fn) if (tn+fn) > 0 else 0.0

    print(f"Fold {i+1} | Test: Loss={test_loss:.6f} Acc={test_acc:.4f} "
          f"Spec={specificity:.4f} Sen={sensitivity:.4f} Prec={precision:.4f} "
          f"F1={f1:.4f} FNR={FNR:.4f} PPV={PPV:.4f} NPV={NPV:.4f} "
          f"Time={(time.time()-start_test):.0f}s")
    test_results.append({
        'fold': i+1,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': f1,
        'FNR': FNR,
        'PPV': PPV,
        'NPV': NPV,
    })

    print(f"Fold {i+1} complete in {(time.time()-since_fold)//60:.0f}m {(time.time()-since_fold)%60:.0f}s")
    all_val_metrics.append(fold_metrics)
    writer.close()

# ================== 4. 平均指標 & 匯出 ==================
average_fold_metrics = {}
for idx, metrics in enumerate(all_val_metrics, start=1):
    avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0] if k != 'epoch'}
    average_fold_metrics[f"Fold {idx}"] = avg

print("Average validation metrics per fold:")
for fold, m in average_fold_metrics.items():
    print(f"{fold}: " + ", ".join([f"{k}={v:.4f}" for k, v in m.items()]))

average_test = {k: np.mean([r[k] for r in test_results]) for k in test_results[0] if k!='fold'}
print("Average test metrics:")
print(", ".join([f"{k}={v:.4f}" for k, v in average_test.items()]))

# 匯出到 Excel
val_df = pd.DataFrame(all_val_metrics)
val_df.to_excel(os.path.join(RUN_ROOT, "validation_metrics.xlsx"), index=False)
test_df = pd.DataFrame(test_results)
test_df.to_excel(os.path.join(RUN_ROOT, "test_results.xlsx"), index=False)

print("All metrics computed and saved.")
