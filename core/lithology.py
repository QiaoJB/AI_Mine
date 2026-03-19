import rasterio
import numpy as np
import os
import yaml
from rasterio.warp import reproject, Resampling
from sklearn.cluster import KMeans

def generate_lithology(image_path, mineral_type):
    CONFIG_PATH = f"M:/mine_predict/configs/{mineral_type}.yaml"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    DATA_DIR = os.path.join(image_path, "npy")

    os.makedirs(DATA_DIR, exist_ok=True)

    BANDS = cfg["bands"]

    REF_BAND = BANDS[0] + ".tiff"

    def normalize(img):
        img = img.astype(np.float32) / 10000.0
        return np.clip(img, 0, 1)

    ref_path = os.path.join(image_path, REF_BAND)
    with rasterio.open(ref_path) as ref:
        ref_img = ref.read(1)
        ref_transform = ref.transform
        ref_crs = ref.crs
        H, W = ref.height, ref.width

    imgs = []

    for band in BANDS:
        path = os.path.join(image_path, band + ".tiff")
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
    
    C, H, W = stack.shape

    flat = stack.reshape(C, -1).T

    kmeans = KMeans(n_clusters=cfg["lithology_clusters"], random_state=0)
    labels = kmeans.fit_predict(flat)

    # One-hot
    onehot = np.eye(cfg["lithology_clusters"])[labels]
    out_file = os.path.join(DATA_DIR, "lithology_features.npy")
    np.save(out_file, onehot)

    return f"{out_file} generated"


