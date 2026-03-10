import numpy as np

# ===============================
# 1️⃣ 读取原始特征
# ===============================
lith = np.load("data/lithology_onehot.npy")
alt = np.load("data/alter_stack.npy")   
struct = np.load("data/dem/structural_stack.npy")

H, W = struct.shape[1:]

# ===============================
# 2️⃣ reshape
# lith & alt: [H*W, feature]
# struct: [H*W, channels]
# ===============================
lith = lith.reshape(H*W, -1)
alt = alt.reshape(H*W, -1)
struct = struct.reshape(struct.shape[0], -1).T  # [H*W, channels]

# ===============================
# 3️⃣ 合并
# ===============================
X = np.concatenate([lith, alt, struct], axis=1)
print("拼接后维度:", X.shape)

# ===============================
# 4️⃣ 处理 NaN
# 填充每列的中位数
# ===============================
for i in range(X.shape[1]):
    col = X[:, i]
    if np.isnan(col).any():
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        X[:, i] = col

# ===============================
# 5️⃣ 标准化
# ===============================
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-6
X = (X - mean) / std

print("最终NaN比例:", np.isnan(X).mean())

# ===============================
# 6️⃣ 保存
# ===============================
np.save("ml/feature_cube.npy", X)
np.save("ml/feature_norm.npy", (mean, std))

print("✅ Feature cube 构建完成:", X.shape)
print("lith nan比例:", np.isnan(lith).mean())
print("alt nan比例:", np.isnan(alt).mean())
print("struct nan比例:", np.isnan(struct).mean())