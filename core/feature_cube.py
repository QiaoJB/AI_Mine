import numpy as np
import os

def build_feature_cube(image_path):
    # ===============================
    # 1️⃣ 读取原始特征
    # ===============================
    NPY_PATH = os.path.join(image_path, "npy")
    os.makedirs(NPY_PATH, exist_ok=True)

    lith = np.load(os.path.join(NPY_PATH, "lithology_features.npy"))
    alt = np.load(os.path.join(NPY_PATH, "alter_stack.npy"))   
    struct = np.load(os.path.join(NPY_PATH, "structure_features.npy"))

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

    # ===============================
    # 6️⃣ 保存
    # ===============================
    out_file = os.path.join(NPY_PATH, "feature_cube.npy")
    np.save(out_file, X)
    
    return f"{out_file} generated"