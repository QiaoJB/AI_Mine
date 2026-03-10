import numpy as np
import yaml
from sklearn.cluster import KMeans
# import rasterio

with open("configs/gold.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

s2 = np.load("data/sentinel2/s2_stack.npy")
C, H, W = s2.shape

flat = s2.reshape(C, -1).T

kmeans = KMeans(n_clusters=cfg["lithology_clusters"], random_state=0)
labels = kmeans.fit_predict(flat)

lith_map = labels.reshape(H, W)

np.save("data/lithology_map.npy", lith_map)

# One-hot
onehot = np.eye(cfg["lithology_clusters"])[labels]
np.save("data/lithology_onehot.npy", onehot)

print("✅ Lithology clustering done")