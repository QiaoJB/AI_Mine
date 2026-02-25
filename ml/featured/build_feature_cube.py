import numpy as np

# 已有数据
s2 = np.load("data/sentinel2/s2_stack.npy")        # [Bs, H, W]
dem = np.load("data/dem/dem_stack.npy")            # [Bd, H, W]
alt = np.load("data/alter_stack.npy")              # [Ba, H, W]

# 合并为像素特征
cube = np.concatenate([s2, dem, alt], axis=0)      # [D, H, W]
D, H, W = cube.shape

# reshape为 [N, D]
features = cube.reshape(D, -1).T                   # [H*W, D]

# 标准化（全图背景统计）
mean = features.mean(axis=0)
std = features.std(axis=0) + 1e-6
features = (features - mean) / std

np.save("ml/feature_cube.npy", features)
np.save("ml/feature_norm.npy", (mean, std))
print("✅ 特征空间:", features.shape)
