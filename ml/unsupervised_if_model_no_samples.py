import numpy as np
import rasterio
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

print("📦 读取特征空间...")
X = np.load("ml/feature_cube.npy")   # shape: (N, F)

# ===============================
# 1️⃣ 清理 NaN
# ===============================
print("🔹 清理无效值...")
nan_mask = np.any(np.isnan(X), axis=1)
X_clean = X.copy()
X_clean[nan_mask] = np.nanmean(X_clean, axis=0)

# ===============================
# 2️⃣ 标准化
# ===============================
print("🔹 特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ===============================
# 3️⃣ PCA 降维（去噪增强）
# ===============================
print("🔹 PCA 降维增强异常可分性...")

pca = PCA(n_components=0.95)  # 保留95%信息
X_pca = pca.fit_transform(X_scaled)

print(f"降维后维度: {X_pca.shape[1]}")

# ===============================
# 4️⃣ Isolation Forest（全局异常）
# ===============================
print("🔹 Isolation Forest...")

iso = IsolationForest(
    n_estimators=400,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

iso.fit(X_pca)
iso_score = -iso.score_samples(X_pca)

iso_score = (iso_score - iso_score.min()) / (
    iso_score.max() - iso_score.min() + 1e-6
)

# ===============================
# 5️⃣ LOF（局部异常）
# ===============================
print("🔹 LOF 局部异常检测...")

# 加入微扰打破重复点
X_lof = X_pca + np.random.normal(0, 1e-6, X_pca.shape)

lof = LocalOutlierFactor(
    n_neighbors=60,        # 稍微调大
    contamination=0.05,
    n_jobs=-1
)

lof_pred = lof.fit_predict(X_lof)
lof_score = -lof.negative_outlier_factor_

lof_score = (lof_score - lof_score.min()) / (
    lof_score.max() - lof_score.min() + 1e-6
)

# ===============================
# 6️⃣ 融合双异常模型（纯数据驱动）
# ===============================
print("🔹 融合异常指数...")

final_score = 0.7 * iso_score + 0.3 * lof_score

# 也可以试：
# final_score = iso_score * lof_score

# ===============================
# 7️⃣ 转回空间
# ===============================
print("🔹 生成空间概率图...")

# 获取空间尺寸
with rasterio.open("data/sentinel2/B04.tiff") as ref:
    H, W = ref.height, ref.width
    profile = ref.profile
    profile.update(dtype="float32", count=1)

prob_img = final_score.reshape(H, W)

# 空间平滑
prob_img = gaussian_filter(prob_img, sigma=1.2)

# 归一化
prob_img = (prob_img - prob_img.min()) / (
    prob_img.max() - prob_img.min() + 1e-6
)

# ===============================
# 8️⃣ 输出结果
# ===============================
with rasterio.open("output/gold_unsupervised_pure_feature.tif", "w", **profile) as dst:
    dst.write(prob_img.astype(np.float32), 1)

print("✅ 纯 feature_cube 无监督异常图生成完成")