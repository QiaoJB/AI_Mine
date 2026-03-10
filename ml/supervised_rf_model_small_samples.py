import numpy as np
import geopandas as gpd
import rasterio

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import grey_closing

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

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# PCA降维
# ===============================

print("🔹 PCA降维")

pca = PCA(n_components=8)
X = pca.fit_transform(X)

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
# 多尺度邻域特征
# ===============================

print("🔹 添加多尺度邻域特征")

X_map = X.reshape(H, W, -1)

neighbor_features = []

for i in range(X_map.shape[2]):

    smooth3 = uniform_filter(X_map[:,:,i], size=3)
    smooth7 = uniform_filter(X_map[:,:,i], size=7)
    smooth11 = uniform_filter(X_map[:,:,i], size=11)

    neighbor_features.append(smooth3.reshape(-1))
    neighbor_features.append(smooth7.reshape(-1))
    neighbor_features.append(smooth11.reshape(-1))

neighbor_features = np.stack(neighbor_features, axis=1)

X = np.concatenate([X, neighbor_features], axis=1)

print("feature dim:", X.shape[1])

# ===============================
# 构建标签
# ===============================

y = np.zeros(len(X))

shp = gpd.read_file("data/labels/gold_points.shp").to_crs(crs)

buffer_radius = 4   # ⭐扩大buffer

for pt in shp.geometry:

    col, row = ~transform * (pt.x, pt.y)

    row = int(row)
    col = int(col)

    for dr in range(-buffer_radius, buffer_radius + 1):
        for dc in range(-buffer_radius, buffer_radius + 1):

            rr = row + dr
            cc = col + dc

            if 0 <= rr < H and 0 <= cc < W:

                idx = rr * W + cc
                y[idx] = 1

print("正样本:", int(y.sum()))

# ===============================
# 负样本采样（远离矿点）
# ===============================

pos_idx = np.where(y == 1)[0]
neg_idx_all = np.where(y == 0)[0]

neg_sample_size = min(len(pos_idx) * 3, len(neg_idx_all))

neg_idx = np.random.choice(
    neg_idx_all,
    size=neg_sample_size,
    replace=False
)

train_idx = np.concatenate([pos_idx, neg_idx])

X_train = X[train_idx]
y_train = y[train_idx]

print("训练样本:", len(train_idx))

# ===============================
# RF模型
# ===============================

rf = RandomForestClassifier(

    n_estimators=600,
    max_depth=12,
    min_samples_leaf=3,

    class_weight="balanced",

    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# ===============================
# 全图预测
# ===============================

print("🔹 RF预测")

prob = rf.predict_proba(X)[:, 1]

heat = prob.reshape(H, W)

# ===============================
# 空间平滑
# ===============================

heat = gaussian_filter(heat, sigma=3)

# 形态学闭运算（消除碎斑）
heat = grey_closing(heat, size=(7,7))

# 概率增强
heat = heat ** 1.6

# 归一化
heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

# ===============================
# 保存结果
# ===============================

profile.update(dtype="float32", count=1)

with rasterio.open(
    "output/gold_rf_prediction_optimized.tif",
    "w",
    **profile
) as dst:

    dst.write(heat.astype(np.float32), 1)

print("🏆 RF 成矿预测完成")