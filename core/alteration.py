import numpy as np
import rasterio
import yaml
import os
from scipy.ndimage import gaussian_filter

def generate_alteration(image_path, mineral_type):
    CONFIG_PATH = f"M:/mine_predict/configs/{mineral_type}.yaml"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bands = cfg["bands"]
    indices_cfg = cfg["alteration_indices"]
    
    OUT_DIR = os.path.join(image_path, "npy")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PATH = os.path.join(OUT_DIR, "alter_stack.npy")

    eps = 1e-6


    # ===============================
    # 读取波段
    # ===============================

    band_data = {}

    for b in bands:
        
        with rasterio.open(os.path.join(image_path, f"{b}.tiff")) as src:

            band_data[b] = src.read(1).astype(np.float32)

            transform = src.transform
            crs = src.crs
            height, width = src.shape


    # ===============================
    # 工具函数
    # ===============================

    def norm_diff(a, b):
        return (a - b) / (a + b + eps)


    def robust_norm(data):

        med = np.nanmedian(data)
        std = np.nanstd(data)

        return (data - med) / (std + eps)


    # ===============================
    # 环境抑制因子
    # ===============================

    B = band_data

    ndvi = norm_diff(B["B08"], B["B04"])

    moisture = norm_diff(B["B11"], B["B08"])

    # normalize to 0-1
    ndvi_norm = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi) + eps)
    moisture_norm = (moisture - np.nanmin(moisture)) / (np.nanmax(moisture) - np.nanmin(moisture) + eps)

    # penalty factor
    penalty = (1 - ndvi_norm) * (1 - moisture_norm)

    penalty = np.clip(penalty, 0, 1)


    # ===============================
    # 指数计算
    # ===============================

    def idx(name):

        B = band_data

        # ---------------------------
        # Fe oxide
        # ---------------------------

        if name == "fe_oxide_nd":
            return norm_diff(B["B04"], B["B02"])

        if name == "hematite_nd":
            return norm_diff(B["B04"], B["B03"])

        if name == "redness":
            return B["B04"] / (B["B03"] + eps)

        # ---------------------------
        # Clay / Al-OH
        # ---------------------------

        if name == "al_oh_nd":
            return norm_diff(B["B11"], B["B12"])

        if name == "clay_ratio":
            return (B["B11"] + B["B12"]) / (B["B08"] + eps)

        if name == "chlorite_nd":
            return norm_diff(B["B11"], B["B08"])

        # ---------------------------
        # Silica
        # ---------------------------

        if name == "silica_proxy":
            return B["B08"] / (B["B11"] + eps)

        # ---------------------------
        # Dark minerals
        # ---------------------------

        if name == "darkness":
            return (B["B02"] + B["B03"] + B["B04"]) / 3.0

        raise ValueError(f"Unknown index: {name}")


    # ===============================
    # 计算蚀变指数
    # ===============================

    stack = []

    for name in indices_cfg:

        data = idx(name)

        data = np.where(np.isfinite(data), data, np.nan)

        data = np.clip(data, -2, 2)

        # ===============================
        # 环境抑制
        # ===============================

        data = data * penalty

        # ===============================
        # robust normalization
        # ===============================

        data = robust_norm(data)

        # ===============================
        # 轻微平滑
        # ===============================

        data = gaussian_filter(data, sigma=0.6)

        stack.append(data)


    stack = np.stack(stack, axis=0)

    np.save(OUT_PATH, stack)

    return f"{OUT_PATH} generated"
