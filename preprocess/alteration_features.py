import numpy as np
import rasterio
import yaml
import os

CONFIG_PATH = "configs/gold.yaml"  # 可切换矿种

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

bands = cfg["bands"]
indices_cfg = cfg["alteration_indices"]

S2_DIR = "data/sentinel2/"
OUT_PATH = "data/alter_stack.npy"
os.makedirs("data/alteration_indices", exist_ok=True)

band_data = {}
for b in bands:
    with rasterio.open(f"{S2_DIR}{b}.tiff") as src:
        band_data[b] = src.read(1).astype(np.float32)

eps = 1e-6

def norm_diff(a, b):
    return (a - b) / (a + b + eps)

def idx(name):
    B = band_data

    # 🟥 铁氧化（Fe³⁺）
    if name == "fe_oxide_nd": return norm_diff(B["B04"], B["B02"])
    if name == "redness":     return B["B04"] / (B["B03"] + eps)

    # 🟦 羟基 / 粘土蚀变（Al-OH）
    if name == "al_oh_nd":    return norm_diff(B["B11"], B["B12"])
    if name == "clay_ratio":  return (B["B11"] + B["B12"]) / (B["B08"] + eps)

    # 🟨 硅化（石英）
    if name == "silica_proxy": return B["B08"] / (B["B11"] + eps)

    # 🌿 植被
    if name == "ndvi": return norm_diff(B["B08"], B["B04"])

    # 🌫 水分
    if name == "moisture_nd": return norm_diff(B["B11"], B["B08"])

    # ⬛ 暗色物质（煤/碳质）
    if name == "darkness": return (B["B02"] + B["B03"] + B["B04"]) / 3.0

    raise ValueError(f"Unknown index: {name}")


# ===============================
# 计算指数
# ===============================
stack = []
for name in indices_cfg:
    data = idx(name)

    # NDVI 作为抑制因子，不截断
    if name != "ndvi":
        data = np.clip(data, -1, 1)

    stack.append(data)

stack = np.stack(stack, axis=0)
np.save(OUT_PATH, stack)
print("✅ alter_stack saved:", stack.shape)

# ===============================
# 保存 GeoTIFF
# ===============================
with rasterio.open(f"{S2_DIR}{bands[0]}.tiff") as src:
    transform = src.transform
    crs = src.crs
    height, width = src.shape

for name, data in zip(indices_cfg, stack):
    out_path = f"data/alteration_indices/{name}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data.astype(np.float32), 1)

    print(f"✅ {name} saved")


# ===============================
# 计算融合蚀变强度图
# ===============================
indices_dict = {name: stack[i] for i, name in enumerate(indices_cfg)}

# 自动识别正向指标
positive_keys = [k for k in indices_cfg if k != "ndvi"]

fusion = np.zeros_like(stack[0])

for k in positive_keys:
    fusion += indices_dict[k]

fusion /= len(positive_keys)

# NDVI 抑制机制
if "ndvi" in indices_dict:
    ndvi = indices_dict["ndvi"]
    ndvi_norm = (ndvi + 1) / 2
    fusion *= (1 - ndvi_norm)

# 归一化
fusion = (fusion - fusion.min()) / (fusion.max() - fusion.min() + 1e-6)

np.save("data/alteration_fusion.npy", fusion)
print("🔥 Fusion alteration map saved")

out_path_fusion = "data/alteration_indices/fusion_alteration_map.tif"
with rasterio.open(
    out_path_fusion,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=crs,
    transform=transform,
) as dst:
    dst.write(fusion.astype(np.float32), 1)

print("🗺 Fusion GeoTIFF saved")
