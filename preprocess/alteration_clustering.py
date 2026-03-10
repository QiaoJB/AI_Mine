import numpy as np
import yaml
from sklearn.mixture import GaussianMixture

with open("configs/gold.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

alt = np.load("data/alter_stack.npy")
C, H, W = alt.shape

flat = alt.reshape(C, -1).T

gmm = GaussianMixture(cfg["alteration_clusters"], random_state=0)
labels = gmm.fit_predict(flat)

alt_map = labels.reshape(H, W)

np.save("data/alteration_pattern_map.npy", alt_map)

onehot = np.eye(cfg["alteration_clusters"])[labels]
np.save("data/alteration_onehot.npy", onehot)

print("✅ Alteration pattern clustering done")