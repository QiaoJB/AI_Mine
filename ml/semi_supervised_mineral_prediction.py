import numpy as np
import geopandas as gpd
import rasterio

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

from scipy.ndimage import gaussian_filter

print("📦 读取特征...")

X = np.load("ml/feature_cube.npy")

# ===============================
# NaN处理
# ===============================

nan_mask = np.any(np.isnan(X), axis=1)
X[nan_mask] = np.nanmean(X, axis=0)

# ===============================
# 标准化
# ===============================

print("🔹 特征标准化")

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# PCA降维（稳定聚类）
# ===============================

print("🔹 PCA降维")

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

print("PCA variance:", pca.explained_variance_ratio_.sum())

# ===============================
# 读取影像
# ===============================

with rasterio.open("data/sentinel2/B04.tiff") as ref:

    H = ref.height
    W = ref.width

    transform = ref.transform
    crs = ref.crs
    profile = ref.profile

print("pixels:", H * W)

# ===============================
# 空间坐标特征（避免碎片）
# ===============================

print("🔹 加入空间坐标")

ys, xs = np.indices((H, W))

coords = np.stack([ys.flatten(), xs.flatten()], axis=1)

coords = coords / max(H, W)

coords = coords * 0.05

X_aug = np.concatenate([X_pca, coords], axis=1)

# ===============================
# GMM聚类
# ===============================

print("🔹 GMM聚类")

n_cluster = 15

gmm = GaussianMixture(
    n_components=n_cluster,
    covariance_type="full",
    random_state=0
)

gmm.fit(X_aug)

prob_cluster = gmm.predict_proba(X_aug)

# ===============================
# IsolationForest异常检测
# ===============================

print("🔹 Isolation Forest异常检测")

iso = IsolationForest(
    n_estimators=400,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_pca)

anomaly_score = -iso.score_samples(X_pca)

# 归一化
anomaly_score = (anomaly_score - anomaly_score.min()) / (
    anomaly_score.max() - anomaly_score.min() + 1e-6
)

# ===============================
# 读取矿点
# ===============================

print("🔹 读取矿点")

shp = gpd.read_file("data/labels/gold_points.shp").to_crs(crs)

cluster_weight = np.zeros(n_cluster)

buffer = 4

# ===============================
# 矿点cluster统计
# ===============================

for pt in shp.geometry:

    col, row = ~transform * (pt.x, pt.y)

    row = int(row)
    col = int(col)

    for dr in range(-buffer, buffer + 1):
        for dc in range(-buffer, buffer + 1):

            rr = row + dr
            cc = col + dc

            if 0 <= rr < H and 0 <= cc < W:

                idx = rr * W + cc

                cluster_weight += prob_cluster[idx]

# 归一化
cluster_weight = cluster_weight / (cluster_weight.sum() + 1e-6)

print("Cluster weights:", cluster_weight)

# ===============================
# GMM概率
# ===============================

print("🔹 生成GMM概率图")

gmm_prob = np.dot(prob_cluster, cluster_weight)

# ===============================
# 融合异常指数
# ===============================

print("🔹 融合异常指数")

final_prob = 0.7 * gmm_prob + 0.3 * anomaly_score

# ===============================
# 空间平滑
# ===============================

heat = final_prob.reshape(H, W)

heat = gaussian_filter(heat, sigma=2)

# 概率增强
heat = heat ** 1.6

# 归一化
heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

# ===============================
# 输出结果
# ===============================

profile.update(dtype="float32", count=1)

with rasterio.open(
    "output/gold_prediction_small_samples.tif",
    "w",
    **profile
) as dst:

    dst.write(heat.astype(np.float32), 1)

print("🏆 小样本成矿预测完成")