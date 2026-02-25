# AI_Mine

面向智能探矿的遥感+地形多源特征工程与异常识别流程。

## 当前能力
- Sentinel-2 预处理与光谱异常图生成。
- 蚀变指数计算与融合蚀变图生成。
- DEM 地形特征与地形异常图生成。
- **新增：地质岩性识别（无监督聚类）**。
- **新增：地质构造识别（线性构造概率 + 主方向）**。
- 多源特征立方构建与无监督异常找矿。

## 推荐流程
> 先准备 `data/sentinel2/*.tiff` 与 `data/dem/dem.tif`。

1. 光谱栈与光谱异常：
```bash
python preprocess/s2_preprocess.py
```

2. 蚀变特征：
```bash
python preprocess/alteration_features.py
```

3. DEM 特征：
```bash
python preprocess/dem_features.py
```

4. **岩性识别（新增）**：
```bash
python preprocess/lithology_features.py
```
输出：
- `data/lithology/lithology_stack.npy`
- `data/lithology/lithology_classes.tif`
- `data/lithology/lithology_confidence.tif`

5. **构造识别（新增）**：
```bash
python preprocess/structure_features.py
```
输出：
- `data/structure/structure_stack.npy`
- `data/structure/lineament_likelihood.tif`
- `data/structure/structure_direction.tif`

6. 构建统一特征空间（会自动拼接岩性/构造特征，若存在）：
```bash
python ml/featured/build_feature_cube.py
```

7. 无监督多模型异常图：
```bash
python ml/featured/unsupervised_multi_anomaly.py
```

## 说明
- 岩性识别使用 PCA + MiniBatchKMeans 做像素级无监督分类。
- 构造识别基于坡度梯度场提取线性构造概率与方向信息。
- `ml/featured/build_feature_cube.py` 对岩性/构造特征为可选输入，不会破坏现有流程。
