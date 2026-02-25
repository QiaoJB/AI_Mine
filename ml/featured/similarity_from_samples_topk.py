import numpy as np
import geopandas as gpd
import rasterio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ===============================
# 参数
# ===============================
K = 5          # 取最像的K个矿点（3~10都可以试）
SIGMA_SCALE = 1.0   # 控制相似度衰减快慢（越大 → 预测范围越广）

# ===============================
# 读取特征
# ===============================
X = np.load("ml/feature_cube.npy")     # [N, D]
mean, std = np.load("ml/feature_norm.npy", allow_pickle=True)
X = (X - mean) / (std + 1e-6)

# ===============================
# 栅格信息
# ===============================
ref = rasterio.open("data/sentinel2/B04.tiff")
H, W = ref.height, ref.width
transform = ref.transform
crs = ref.crs

# ===============================
# 读取样本
# ===============================
shp = gpd.read_file("data/labels/gold_points.shp").to_crs(crs)

samples = []
for pt in shp.geometry:
    r, c = ~transform * (pt.x, pt.y)
    r, c = int(r), int(c)
    idx = r * W + c
    if 0 <= idx < X.shape[0]:
        samples.append(X[idx])

samples = np.array(samples)

if len(samples) == 0:
    raise ValueError("❌ 没有匹配到样本像素")

print("✅ 矿样本数:", len(samples))

# ===============================
# 计算相似度（Top-K矿匹配）
# ===============================
print("🔍 计算全图矿相似度...")

# 距离标准差用于控制衰减尺度
global_std = X.std()

sim_all = np.zeros(len(X), dtype=np.float32)

for i in tqdm(range(len(X))):
    px = X[i]

    # 与所有矿点的欧氏距离
    dist = np.linalg.norm(samples - px, axis=1)

    # 转相似度（高斯核）
    sim = np.exp(-(dist**2) / (2 * (global_std * SIGMA_SCALE)**2))

    # 取最像的K个矿点
    topk = np.partition(sim, -K)[-K:]

    sim_all[i] = topk.mean()

# ===============================
# 归一化
# ===============================
sim_all = (sim_all - sim_all.min()) / (sim_all.max() - sim_all.min() + 1e-6)

# ===============================
# 变为栅格
# ===============================
heat = sim_all.reshape(H, W)
heat = gaussian_filter(heat, 1.2)

# 百分位增强（突出异常）
v = heat[heat > 0]
low, high = np.percentile(v, [92, 99])
heat = (heat - low) / (high - low + 1e-6)
heat = np.clip(heat, 0, 1)

# ===============================
# 保存
# ===============================
out = "output/mineral_similarity_topk.tif"
with rasterio.open(out, "w",
                   driver="GTiff",
                   height=H,
                   width=W,
                   count=1,
                   dtype="float32",
                   crs=crs,
                   transform=transform) as dst:
    dst.write(heat.astype(np.float32), 1)

print("🏆 Top-K矿相似度图生成:", out)