import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from skimage.feature import canny
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

DEM_PATH = "data/dem/dem.tif"
REF_PATH = "data/sentinel2/B04.tiff"

print("📦 读取参考影像...")

with rasterio.open(REF_PATH) as ref:
    H, W = ref.height, ref.width
    dst_transform = ref.transform
    dst_crs = ref.crs
    profile = ref.profile

print("📦 重采样 DEM 到 Sentinel 分辨率...")

with rasterio.open(DEM_PATH) as src:
    dem = np.full((H, W), np.nan, dtype=np.float32)

    reproject(
        source=rasterio.band(src, 1),
        destination=dem,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=src.nodata,
        dst_nodata=np.nan
    )

valid_mask = np.isfinite(dem)
dem_filled = dem.copy()
dem_filled[~valid_mask] = np.nanmean(dem)

print("🔹 计算 DEM 坡度...")

dx = sobel(dem_filled, axis=1)
dy = sobel(dem_filled, axis=0)
slope = np.sqrt(dx**2 + dy**2)

print("🔹 计算 DEM 曲率...")

curvature = gaussian_filter(dem_filled, 3) - dem_filled

print("🔹 DEM 多尺度高通梯度增强...")

dem_small = gaussian_filter(dem_filled, 1)
dem_large = gaussian_filter(dem_filled, 10)
dem_hp = dem_small - dem_large

gx = sobel(dem_hp, axis=0)
gy = sobel(dem_hp, axis=1)
dem_grad = np.hypot(gx, gy)

dem_grad = (dem_grad - dem_grad.min()) / (dem_grad.max() - dem_grad.min() + 1e-6)

print("📦 读取 Sentinel 多波段进行 PCA...")

# 读取多个波段（你可自行扩展）
band_list = ["B02.tiff", "B03.tiff", "B04.tiff", "B08.tiff"]

stack_img = []

for b in band_list:
    path = f"data/sentinel2/{b}"
    with rasterio.open(path) as src:
        img = src.read(1).astype(np.float32)
        stack_img.append(img)

stack_img = np.stack(stack_img, axis=-1)
H, W, C = stack_img.shape

# reshape for PCA
reshaped = stack_img.reshape(-1, C)
reshaped[np.isnan(reshaped)] = 0

pca = PCA(n_components=1)
pca_img = pca.fit_transform(reshaped)
pca_img = pca_img.reshape(H, W)

pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min() + 1e-6)

print("🔹 Sentinel 结构梯度提取...")

sentinel_edges = canny(pca_img, sigma=2)

sentinel_grad = sobel(pca_img)
sentinel_grad = (sentinel_grad - sentinel_grad.min()) / (
    sentinel_grad.max() - sentinel_grad.min() + 1e-6
)

print("🔹 DEM + Sentinel 融合构造提取...")

threshold = np.percentile(dem_grad, 70)

fusion_edges = sentinel_edges * (dem_grad > threshold)

print("🔹 构造密度计算...")

fusion_density = uniform_filter(fusion_edges.astype(float), size=21)

fusion_density = (fusion_density - fusion_density.min()) / (
    fusion_density.max() - fusion_density.min() + 1e-6
)

# 恢复 NaN
for arr in [slope, curvature, dem_grad, sentinel_grad, fusion_density]:
    arr[~valid_mask] = np.nan

print("💾 保存 structural_stack.npy")

stack = np.stack([
    slope,
    curvature,
    dem_grad,
    sentinel_grad,
    fusion_density
], axis=0)

np.save("data/dem/structural_stack.npy", stack)

print("✅ 融合结构特征生成完成")