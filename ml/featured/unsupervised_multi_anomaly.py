import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

print("📦 读取特征...")
X = np.load("ml/feature_cube.npy")   # [N, D]

# ===============================
# 1️⃣ Robust Mahalanobis
# ===============================
print("🔹 Robust Mahalanobis...")
mcd = MinCovDet().fit(X)
md = mcd.mahalanobis(X)
md = (md - md.min()) / (md.max() - md.min() + 1e-6)

# ===============================
# 2️⃣ Isolation Forest
# ===============================
print("🔹 Isolation Forest...")
iso = IsolationForest(
    n_estimators=300,
    contamination=0.03,
    random_state=0,
    n_jobs=-1
)
iso.fit(X)
iso_score = -iso.decision_function(X)
iso_score = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min() + 1e-6)

# ===============================
# 3️⃣ GMM 概率异常
# ===============================
print("🔹 GMM 异常...")
gmm = GaussianMixture(8, covariance_type='full', random_state=0).fit(X)
prob = gmm.score_samples(X)
gmm_score = -prob
gmm_score = (gmm_score - gmm_score.min()) / (gmm_score.max() - gmm_score.min() + 1e-6)

# ===============================
# 4️⃣ 综合异常分数（核心）
# ===============================
print("🔹 融合异常...")
score = (
    0.4 * iso_score +
    0.3 * gmm_score +
    0.3 * md
)

score = (score - score.min()) / (score.max() - score.min() + 1e-6)

# ===============================
# 5️⃣ 恢复为图像
# ===============================
with rasterio.open("data/sentinel2/B04.tiff") as src:
    H, W = src.height, src.width
    transform, crs = src.transform, src.crs

heat = score.reshape(H, W)

# 平滑让地质异常更连续
heat = gaussian_filter(heat, 1.2)

# 百分位增强（突出异常区）
v = heat.flatten()
low, high = np.percentile(v, [92, 99])
heat = (heat - low) / (high - low + 1e-6)
heat = np.clip(heat, 0, 1)

# ===============================
# 6️⃣ 输出
# ===============================
out = "output/anomaly_multi_model.tif"

with rasterio.open(out, "w", driver="GTiff",
                   height=H, width=W, count=1, dtype="float32",
                   crs=crs, transform=transform) as dst:
    dst.write(heat.astype(np.float32), 1)

print("=======================================")
print("✅ 多模型无样本异常图生成完成：", out)
print("值越高 = 地质组合越异常 = 潜在矿化异常")
print("=======================================")