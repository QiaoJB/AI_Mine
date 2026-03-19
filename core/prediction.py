import numpy as np
import rasterio
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import os

def generate_prediction(image_path: str):
    NPY_PATH = os.path.join(image_path, "npy")
    os.makedirs(NPY_PATH, exist_ok=True)

    OUT_PATH = os.path.join(image_path, "output")
    os.makedirs(OUT_PATH, exist_ok=True)
    # 读取特征空间..."
    X = np.load(os.path.join(NPY_PATH, "feature_cube.npy"))   # shape: (N, F)

    # ===============================
    # 清理 NaN
    # ===============================
    nan_mask = np.any(np.isnan(X), axis=1)
    X_clean = X.copy()
    X_clean[nan_mask] = np.nanmean(X_clean, axis=0)

    # ===============================
    # 标准化
    # ===============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # ===============================
    # PCA 降维（去噪增强）
    # ===============================

    pca = PCA(n_components=0.95)  # 保留95%信息
    X_pca = pca.fit_transform(X_scaled)

    # ===============================
    # Isolation Forest（全局异常）
    # ===============================

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
    # LOF（局部异常）
    # ===============================

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
    # 融合双异常模型（纯数据驱动）
    # ===============================

    final_score = 0.7 * iso_score + 0.3 * lof_score

    # 也可以试：
    # final_score = iso_score * lof_score

    # ===============================
    # 转回空间
    # ===============================

    # 获取空间尺寸
    with rasterio.open(os.path.join(image_path, "B04.tiff")) as ref:
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
    # 输出结果
    # ===============================
    out_file = os.path.join(OUT_PATH, "unsupervised_prediction.tif")
    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(prob_img.astype(np.float32), 1)

    return f"{out_file} generated"