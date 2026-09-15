
import skimage.segmentation as _seg
import skimage.morphology as _morph
_morph.watershed = _seg.watershed
import os
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from dgl.data.utils import save_graphs
import torch
from histocartography.preprocessing import (
    DeepFeatureExtractor,
    ColorMergedSuperpixelExtractor,
    KNNGraphBuilder 
)
#限定python3.8
import cv2 

IMAGE_PATH = "D:/M_114_駱沛葳/Dataset/Origin/2"       # 影像資料夾 
MASK_PATH  = "D:/M_114_駱沛葳/Dataset/Tissue_Mask/2"  # tissue masks 資料夾（檔名與影像相同）
SAVE_PATH  = "D:/M_114_駱沛葳/Dataset/Tissue_Graph/2" # 輸出 tissue_graphs 的根目錄

class TgBuilding:
    def __init__(self):
                
        # 2) 超像素 / 特徵 / RAG
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            nr_superpixels=1500,
            compactness=10,            
            
        )
        self.tissue_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=224
            
        )
        
        self.knn_graph_builder = KNNGraphBuilder(k = 2)
        self.image_ids_failing = []

    @staticmethod
    def _load_mask_like(mask_path, H, W):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask

    def _build_tg(self, image, tissue_mask, img_fp_debug=""): 
        superpixels, _ = self.tissue_detector.process(image, tissue_mask=tissue_mask)
        features = self.tissue_feature_extractor.process(image, superpixels)  # shape: [N, D]


        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        # --- 檢查並處理 NaN / Inf 特徵 ---
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"\n!!! WARNING: NaN/Inf detected in features for image: {img_fp_debug}")
            
            nan_nodes = np.where(np.isnan(features).any(axis=1))[0]
            inf_nodes = np.where(np.isinf(features).any(axis=1))[0]

            if len(nan_nodes) > 0:
                print(f"    - NaN feature detected in node indices: {nan_nodes}")
            if len(inf_nodes) > 0:
                print(f"    - Inf feature detected in node indices: {inf_nodes}")

            feats_clean = features.copy()
            feats_clean[~np.isfinite(feats_clean)] = np.nan  

            col_means = np.nanmean(feats_clean, axis=0)

            col_means = np.where(np.isnan(col_means), 0.0, col_means)

            bad_mask = ~np.isfinite(features)  
            if bad_mask.any():
                bad_rows, bad_cols = np.where(bad_mask)
                print(f"    - Replacing {len(bad_rows)} NaN/Inf entries with column means.")

                features[bad_rows, bad_cols] = col_means[bad_cols]

            if np.isnan(features).any() or np.isinf(features).any():
                print("    - After replacement, still found NaN/Inf. Setting all remaining to 0.")
                features[~np.isfinite(features)] = 0.0

        graph = self.knn_graph_builder.process(superpixels, features)
        return graph, superpixels


    def process(self, image_path, mask_path, save_path):

        subdirs = os.listdir(image_path)
        image_fnames = []
        for subdir in (subdirs + ['']):
            image_fnames += glob(os.path.join(image_path, subdir, '*.png'))

        out_dir = os.path.join(save_path, 'tissue_graphs')
        os.makedirs(out_dir, exist_ok=True)

        print(f'*** Start analysing {len(image_fnames)} images ***')
        for img_fp in tqdm(image_fnames):
            try:
                image_name = Path(img_fp).name
                image = np.array(Image.open(img_fp).convert('RGB'))
                H, W = image.shape[:2]

                mask_fp = os.path.join(mask_path, image_name)
                if not os.path.exists(mask_fp):
                    print(f'Warning: missing mask for {img_fp} -> {mask_fp}')
                    self.image_ids_failing.append(img_fp)
                    continue
                tissue_mask = self._load_mask_like(mask_fp, H, W)

                try:

                    tissue_graph, _ = self._build_tg(image, tissue_mask, img_fp_debug=img_fp)
                except Exception as e:

                    print(f'Warning: {img_fp} failed during tissue graph generation. {e}')
                    self.image_ids_failing.append(img_fp)
                    continue

                tg_out = os.path.join(out_dir, image_name.replace('.png', '.bin'))
                save_graphs(filename=tg_out, g_list=[tissue_graph])

            except Exception as e:
                print(f'Warning: {img_fp} unexpected failure. {e}')
                self.image_ids_failing.append(img_fp)
                continue

        print('Out of {} images, {} successful graph generations.'.format(
            len(image_fnames),
            len(image_fnames) - len(self.image_ids_failing)
        ))
        print('Failing IDs are:', self.image_ids_failing)

if __name__ == "__main__":

    if not os.path.isdir(IMAGE_PATH) or not os.listdir(IMAGE_PATH):
        raise ValueError("Data directory is either empty or does not exist.")

    tg_builder = TgBuilding()
    tg_builder.process(IMAGE_PATH, MASK_PATH, SAVE_PATH)