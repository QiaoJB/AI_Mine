import os
import numpy as np
import rasterio
from scipy.ndimage import sobel, gaussian_filter

DEM_STACK_PATH = "data/dem/dem_stack.npy"
OUT_STACK = "data/structure/structure_stack.npy"
OUT_LINEAMENT = "data/structure/lineament_likelihood.tif"
OUT_DIRECTION = "data/structure/structure_direction.tif"
REF_PATH = "data/sentinel2/B04.tiff"


def _norm01(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


if not os.path.exists(DEM_STACK_PATH):
    raise FileNotFoundError("请先运行 preprocess/dem_features.py 生成 data/dem/dem_stack.npy")

os.makedirs("data/structure", exist_ok=True)

dem_stack = np.load(DEM_STACK_PATH).astype(np.float32)
# dem_stack[1] 近似坡度信息，适合构造边界增强
slope = dem_stack[1]

# 利用梯度提取线性构造（断裂/节理）候选
gx = sobel(slope, axis=1)
gy = sobel(slope, axis=0)
grad_mag = np.sqrt(gx**2 + gy**2)
lineament = _norm01(gaussian_filter(grad_mag, sigma=1.2))

# 主方向（弧度），映射到[0,1]
direction = np.arctan2(gy, gx)
direction_norm = (direction + np.pi) / (2 * np.pi)

# 构造复杂度：局部方向离散程度（越高越复杂）
dir_sin = np.sin(direction)
dir_cos = np.cos(direction)
coherence = np.sqrt(gaussian_filter(dir_sin, 2.0) ** 2 + gaussian_filter(dir_cos, 2.0) ** 2)
complexity = 1.0 - _norm01(coherence)

stack = np.stack([
    lineament,
    direction_norm.astype(np.float32),
    complexity.astype(np.float32),
], axis=0).astype(np.float32)

np.save(OUT_STACK, stack)
print("✅ structure stack saved:", stack.shape)

with rasterio.open(REF_PATH) as ref:
    profile = ref.profile
    profile.update(dtype="float32", count=1)

with rasterio.open(OUT_LINEAMENT, "w", **profile) as dst:
    dst.write(lineament.astype(np.float32), 1)

with rasterio.open(OUT_DIRECTION, "w", **profile) as dst:
    dst.write(direction_norm.astype(np.float32), 1)

print("✅ lineament likelihood saved:", OUT_LINEAMENT)
print("✅ structure direction saved:", OUT_DIRECTION)
