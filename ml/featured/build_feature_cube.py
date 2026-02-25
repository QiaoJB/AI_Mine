import os
import numpy as np


def load_stack(path: str, name: str):
    if not os.path.exists(path):
        print(f"⚠️ {name} not found: {path} (skip)")
        return None
    arr = np.load(path)
    print(f"✅ loaded {name}: {arr.shape}")
    return arr


# 基础特征
s2 = load_stack("data/sentinel2/s2_stack.npy", "sentinel2")
dem = load_stack("data/dem/dem_stack.npy", "dem")
alt = load_stack("data/alter_stack.npy", "alteration")

required = [s2, dem, alt]
if any(x is None for x in required):
    raise FileNotFoundError("基础特征缺失，请先生成 s2_stack / dem_stack / alter_stack")

stacks = [s2, dem, alt]

# 新增：岩性识别 / 构造识别特征（可选）
lith = load_stack("data/lithology/lithology_stack.npy", "lithology")
if lith is not None:
    stacks.append(lith)

struct = load_stack("data/structure/structure_stack.npy", "structure")
if struct is not None:
    stacks.append(struct)

# 合并为像素特征
cube = np.concatenate(stacks, axis=0)
D, H, W = cube.shape

# reshape为 [N, D]
features = cube.reshape(D, -1).T

# 标准化（全图背景统计）
mean = features.mean(axis=0)
std = features.std(axis=0) + 1e-6
features = (features - mean) / std

np.save("ml/feature_cube.npy", features)
np.save("ml/feature_norm.npy", np.array([mean, std], dtype=object), allow_pickle=True)
print("✅ 特征空间:", features.shape)
