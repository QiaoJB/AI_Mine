import rasterio
import numpy as np
import os
import yaml
from rasterio.warp import reproject, Resampling
from sklearn.decomposition import PCA

CONFIG_PATH = "configs/gold.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = "data/sentinel2"
OUT_PATH = "data/sentinel2/s2_stack.npy"
BANDS = cfg["bands"]

REF_BAND = BANDS[0] + ".tiff"

def normalize(img):
    img = img.astype(np.float32) / 10000.0
    return np.clip(img, 0, 1)

ref_path = os.path.join(DATA_DIR, REF_BAND)
with rasterio.open(ref_path) as ref:
    ref_img = ref.read(1)
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width

imgs = []

for band in BANDS:
    path = os.path.join(DATA_DIR, band + ".tiff")
    with rasterio.open(path) as src:
        img = src.read(1)

        if (src.height != H or src.width != W or src.transform != ref_transform):
            dst = np.zeros((H, W), dtype=np.float32)
            reproject(img, dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.bilinear)
            img = dst

        imgs.append(normalize(img))

stack = np.stack(imgs, axis=0)
np.save(OUT_PATH, stack)

print("✅ s2_stack:", stack.shape)

# ===============================
# ③ 生成光谱异常融合图（Spectral Anomaly Map）
# ===============================
print("🟢 Generating spectral anomaly map...")

C, H, W = stack.shape
flat = stack.reshape(C, -1).T   # (N_pixels, C)

# PCA提取主异常方向
pca = PCA(n_components=1)
pc1 = pca.fit_transform(flat)

spectral_map = pc1.reshape(H, W)

# 归一化到0-1
spectral_map = (spectral_map - spectral_map.min()) / (spectral_map.max() - spectral_map.min() + 1e-6)

out_tif = "data/sentinel2/spectral_anomaly_map.tif"

with rasterio.open(ref_path) as ref:
    profile = ref.profile
    profile.update(dtype="float32", count=1)

with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(spectral_map.astype(np.float32), 1)

print("✅ Spectral anomaly map saved:", out_tif)