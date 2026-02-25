import rasterio
import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from rasterio.warp import reproject, Resampling

DEM_PATH = "data/dem/dem.tif"
REF_PATH = "data/sentinel2/B04.tiff"   # 👈 新增：S2参考影像
OUT_PATH = "data/dem/dem_stack.npy"

def compute_slope_aspect(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(dy, dx)
    return slope, aspect

# ===============================
# ① 读取参考影像（S2网格）
# ===============================
with rasterio.open(REF_PATH) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width

# ===============================
# ② 读取DEM并重采样到S2网格
# ===============================
with rasterio.open(DEM_PATH) as src:
    dem_raw = src.read(1).astype(np.float32)

    dem = np.zeros((H, W), dtype=np.float32)

    reproject(
        dem_raw,
        dem,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

# ===============================
# ③ 后续计算全部基于对齐后的DEM
# ===============================
dem = np.nan_to_num(dem)
dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

slope, aspect = compute_slope_aspect(dem)
curvature = gaussian_filter(dem, sigma=3) - dem
roughness = gaussian_filter(np.abs(slope), sigma=3)

def norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)

stack = np.stack([
    norm(dem),
    norm(slope),
    np.sin(aspect),
    np.cos(aspect),
    norm(curvature),
    norm(roughness)
], axis=0)

np.save(OUT_PATH, stack)
print("✅ DEM features saved:", stack.shape)

# ===============================
# ④ 生成地形异常图（Terrain Anomaly Map）
# ===============================
print("🟤 Generating terrain anomaly map...")

C, H, W = stack.shape
flat = stack.reshape(C, -1).T

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pc1 = pca.fit_transform(flat)

terrain_map = pc1.reshape(H, W)
terrain_map = (terrain_map - terrain_map.min()) / (terrain_map.max() - terrain_map.min() + 1e-6)

out_tif = "data/dem/terrain_anomaly_map.tif"

with rasterio.open(REF_PATH) as ref:
    profile = ref.profile
    profile.update(dtype="float32", count=1)

with rasterio.open(out_tif, "w", **profile) as dst:
    dst.write(terrain_map.astype(np.float32), 1)

print("✅ Terrain anomaly map saved:", out_tif)
