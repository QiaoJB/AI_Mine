import os
import numpy as np
import rasterio
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

CONFIG_PATH = "configs/gold.yaml"
S2_STACK_PATH = "data/sentinel2/s2_stack.npy"
OUT_STACK = "data/lithology/lithology_stack.npy"
OUT_LABEL = "data/lithology/lithology_classes.tif"
OUT_SCORE = "data/lithology/lithology_confidence.tif"


def _normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + 1e-6)


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

bands = cfg["bands"]
os.makedirs("data/lithology", exist_ok=True)

# 优先复用已有S2栈，避免重复IO
if os.path.exists(S2_STACK_PATH):
    stack = np.load(S2_STACK_PATH).astype(np.float32)
else:
    imgs = []
    for band in bands:
        with rasterio.open(f"data/sentinel2/{band}.tiff") as src:
            img = src.read(1).astype(np.float32) / 10000.0
            imgs.append(np.clip(img, 0, 1))
    stack = np.stack(imgs, axis=0)

C, H, W = stack.shape
X = stack.reshape(C, -1).T

# 岩性光谱空间：PCA压缩 + 无监督聚类
pca = PCA(n_components=min(4, C), random_state=0)
X_pca = pca.fit_transform(X)

kmeans = MiniBatchKMeans(n_clusters=7, random_state=0, batch_size=4096, n_init=10)
labels = kmeans.fit_predict(X_pca).astype(np.float32)

# 置信度：与所属聚类中心的距离（距离越近置信越高）
centers = kmeans.cluster_centers_
dist = np.linalg.norm(X_pca - centers[labels.astype(np.int32)], axis=1)
dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
confidence = 1.0 - dist

# 岩性特征栈：
# 1) 聚类标签（离散）
# 2) 置信度（连续）
# 3+) PCA主分量（连续）
features = [labels.reshape(H, W), confidence.reshape(H, W)]
for i in range(X_pca.shape[1]):
    features.append(_normalize(X_pca[:, i]).reshape(H, W))

lith_stack = np.stack(features, axis=0).astype(np.float32)
np.save(OUT_STACK, lith_stack)
print("✅ lithology stack saved:", lith_stack.shape)

ref_path = f"data/sentinel2/{bands[0]}.tiff"
with rasterio.open(ref_path) as ref:
    profile = ref.profile
    profile.update(dtype="float32", count=1)

with rasterio.open(OUT_LABEL, "w", **profile) as dst:
    dst.write(labels.reshape(H, W), 1)

with rasterio.open(OUT_SCORE, "w", **profile) as dst:
    dst.write(confidence.reshape(H, W).astype(np.float32), 1)

print("✅ lithology classes saved:", OUT_LABEL)
print("✅ lithology confidence saved:", OUT_SCORE)
