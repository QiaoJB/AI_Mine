# generate_geobody_topz_layered.py
# ------------------------------------------------------------
# 功能：
# 1) 每口钻孔的地层，按层序约束匹配全局层序
# 2) 遇到异常点（与拟合平面/曲面偏离较大）单独生成新的层
# 3) 输出：
#    - 每层 OBJ/MTL
#    - 合并 OBJ/MTL
#    - 异常点 CSV
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay
from collections import defaultdict
from scipy.interpolate import Rbf


# =========================
# 配置区（按需修改）
# =========================
SHP_PATH = "MHWE1/Export_Output.shp"
OUT_DIR = "output_layered_match2"

COMBINED_OBJ = os.path.join(OUT_DIR, "geobody_combined.obj")
COMBINED_MTL = os.path.join(OUT_DIR, "geobody_combined.mtl")
ANOMALY_CSV = os.path.join(OUT_DIR, "anomalies.csv")
layer_spacing = 5.0  # 合并 OBJ 层间爆炸偏移

# 颜色/贴图（按你的字典）
layer_colors = { 
    "clay": [255, 200, 150], 
    "limestone": [100, 100, 200], 
    "gravel": [200, 200, 200], 
    "nurrock": [180, 100, 50], 
    "nurmudstone": [150, 150, 255], 
    "mudstone": [255, 255, 100], 
    "gralevel": [255, 200, 150], 
    "shale": [100, 200, 100], 
    "loosesedim": [200, 255, 150], 
    "coal": [0, 0, 0], 
}

color_default = [150, 150, 150]

layer_textures = {
    "clay": "clay.jpg", "limestone": "limestone.jpg",
    "gravel": "gravel.jpg", "nurrock": "nurrock.jpg",
    "nurmudstone": "nurmudstone.jpg", "mudstone": "mudstone.jpg",
    "gralevel": "gralevel.jpg", "shale": "shale.jpg",
    "loosesedim": "loosesedim.jpg", "coal": "coal.jpg",
}

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# OBJ / MTL 工具函数
# =========================

def write_mtl(mtl_path, materials):
    with open(mtl_path, "w", encoding="utf-8") as f:
        for name, info in materials.items():
            r, g, b = info.get("rgb", color_default)
            tex = info.get("texture")

            base = name.split("_")[0]

            # ---------- 透明度规则 ----------
            if base == "coal":
                opacity = 1.0      # 煤层不透明
            else:
                opacity = 1.0      # 其它地层 50% 透明
            # --------------------------------

            f.write(f"newmtl {name}\n")
            f.write(f"Ka {r/255:.6f} {g/255:.6f} {b/255:.6f}\n")
            f.write(f"Kd {r/255:.6f} {g/255:.6f} {b/255:.6f}\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")

            # 👇 关键行
            f.write(f"d {opacity:.3f}\n")

            f.write("illum 2\n")
            if tex:
                f.write(f"map_Kd {tex}\n")
            f.write("\n")


def write_obj_with_uv(obj_path, mtl_path, vertices, uvs, faces_by_material):
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for mat, faces in faces_by_material:
            f.write(f"usemtl {mat}\n")
            for a, b, c in faces:
                f.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")

def write_single_layer_obj(obj_path, mtl_path, verts, uvs, faces, mat, rgb, tex):
    write_mtl(mtl_path, {mat: {"rgb": rgb, "texture": tex}})
    write_obj_with_uv(obj_path, mtl_path, verts, uvs, [(mat, faces)])

# =========================
# 几何 / UV 工具
# =========================

def gis_points_to_obj(pts):
    pts = np.asarray(pts, float)
    return np.column_stack([pts[:, 0], pts[:, 2], pts[:, 1]])  # X, Z, Y

def uv_from_vertices(verts):
    if verts.shape[0] == 0:
        return np.zeros((0, 2))
    xz = verts[:, [0, 2]]
    mn = xz.min(axis=0)
    mx = xz.max(axis=0)
    d = mx - mn
    d[d == 0] = 1.0
    return (xz - mn) / d

def flip_faces(faces):
    return [(a, c, b) for (a, b, c) in faces]

# =========================
# 读取 SHP & 预处理
# =========================
print("Loading SHP ...")
gdf = gpd.read_file(SHP_PATH)

required_cols = {"ID", "Lithology", "Z_surface", "Depth_top", "D_bottom"}
if not required_cols.issubset(gdf.columns):
    raise ValueError(f"SHP 缺少字段: {required_cols}")

# X/Y 调换（与旧 OBJ 约定一致）
gdf["X"] = gdf.geometry.y
gdf["Y"] = gdf.geometry.x

df = gdf.sort_values(["ID", "Depth_top"]).reset_index(drop=True)
df = df[["ID","X","Y","Z_surface","Depth_top","D_bottom","Lithology"]].copy()

# 设置第一个钻孔为原点
first_id = df.sort_values("ID")["ID"].iloc[0]
x0 = df[df.ID==first_id]["X"].iloc[0]
y0 = df[df.ID==first_id]["Y"].iloc[0]
z0 = df[df.ID==first_id]["Z_surface"].iloc[0]

df["X"] -= x0
df["Y"] -= y0
df["Z_surface"] -= z0
df["topZ"] = df["Z_surface"] - df["Depth_top"]

# =========================
# 每孔内部 LayerKey 编号（按顺序）
# =========================
df["LayerKey"] = None
for bid, grp in df.groupby("ID"):
    cnt = {}
    keys = []
    for lith in grp["Lithology"]:
        cnt[lith] = cnt.get(lith, 0)+1
        keys.append(f"{lith}_{cnt[lith]}")
    df.loc[grp.index, "LayerKey"] = keys

# =========================
# 构建全局层序（按层序约束）
# =========================
layer_order = []
litho_layerkeys = defaultdict(list)
for bid, grp in df.groupby("ID"):
    for lk in grp["LayerKey"]:
        base = lk.split("_")[0]
        if lk not in layer_order:
            layer_order.append(lk)
            litho_layerkeys[base].append(lk)
print(layer_order)

# =========================
# 构建每个 LayerKey 的趋势面（RBF）
# =========================
layer_trend_models = {}

for lk in layer_order:
    pts = []
    tops = []

    for bid, grp in df.groupby("ID"):
        rec = grp[grp.LayerKey == lk]
        if len(rec) == 0:
            continue
        r = rec.iloc[0]
        pts.append([r.X, r.Y])
        tops.append(r.topZ)

    # 点数太少不建趋势面
    if len(pts) < 4:
        continue

    pts = np.array(pts)
    tops = np.array(tops)

    try:
        rbf = Rbf(
            pts[:, 0],
            pts[:, 1],
            tops,
            function="multiquadric",
            smooth=8.0
        )
        layer_trend_models[lk] = rbf
    except Exception as e:
        print(f"[WARN] Trend surface build failed for {lk}: {e}")


# =========================
# 层序约束 + 趋势面异常识别 + 残差最小向下匹配
# =========================
borehole_assignments = defaultdict(dict)
anomalies = []
new_layer_counter = defaultdict(int)

# ---------- 初始化（严格按 LayerKey） ----------
for bid, grp in df.groupby("ID"):
    for _, r in grp.iterrows():
        borehole_assignments[bid][r.LayerKey] = {
            "top": r.topZ,
            "X": r.X,
            "Y": r.Y
        }

# ---------- 按岩性处理 ----------
for lith, lk_list in litho_layerkeys.items():

    for i, lk in enumerate(lk_list):

        rbf = layer_trend_models.get(lk)
        if rbf is None:
            continue

        pts = []
        tops = []
        bids = []

        for bid, layers in borehole_assignments.items():
            rec = layers.get(lk)
            if rec is None:
                continue
            pts.append([rec["X"], rec["Y"]])
            tops.append(rec["top"])
            bids.append(bid)

        if len(pts) < 4:
            continue

        pts = np.array(pts)
        tops = np.array(tops)

        # ---------- 趋势面残差 ----------
        z_fit_all = rbf(pts[:, 0], pts[:, 1])
        residuals = tops - z_fit_all

        local_th = max(
            np.percentile(np.abs(residuals), 95),
            15.0
        )
        # print(local_th)

        # ---------- 逐点判断 ----------
        for j, bid in enumerate(bids):

            dz = residuals[j]
            if abs(dz) <= local_th * 1.2:
                continue  # 正常点

            # ---------- 残差最小原则向下匹配 ----------
            best_lk = None
            best_score = np.inf

            x, y = pts[j]
            z_obs = tops[j]

            for cand_lk in lk_list[i + 1:]:
                rec2 = borehole_assignments[bid].get(cand_lk)
                if rec2 is None:
                    continue

                rbf2 = layer_trend_models.get(cand_lk)
                if rbf2 is None:
                    continue

                z_pred = rbf2(x, y)
                score = abs(z_obs - z_pred)

                if score < best_score:
                    best_score = score
                    best_lk = cand_lk

            match_th = local_th * 1.2
            reassigned = False

            if best_lk is not None and best_score < match_th:
                borehole_assignments[bid][best_lk] = borehole_assignments[bid][lk]
                borehole_assignments[bid][lk] = None
                reassigned = True

            # ---------- 无法匹配 → 新建异常层 ----------
            if not reassigned:
                new_layer_counter[lith] += 1
                new_lk = f"{lith}_{len(lk_list) + new_layer_counter[lith]}"

                borehole_assignments[bid][new_lk] = borehole_assignments[bid][lk]
                borehole_assignments[bid][lk] = None

                layer_order.append(new_lk)
                litho_layerkeys[lith].append(new_lk)

                anomalies.append({
                    "ID": bid,
                    "old_layer": lk,
                    "new_layer": new_lk,
                    "topZ": z_obs,
                    "residual": dz
                })



# =========================
# 构建 borehole_map
# =========================
borehole_map = {}
for bid in sorted(df.ID.unique()):
    x = df[df.ID==bid]["X"].iloc[0]
    y = df[df.ID==bid]["Y"].iloc[0]
    borehole_map[bid] = {"XY": (x, y), "layers": borehole_assignments[bid]}

XY = np.array([borehole_map[bid]["XY"] for bid in borehole_map])
extent = max(np.ptp(XY[:,0]), np.ptp(XY[:,1]), 1.0)
auto_radius = max(extent*0.02, 0.2)*0.5
two_pt_off = auto_radius*0.9

# =========================
# 构建每个全局层的 top-surface
# =========================
layer_meshes = {}
materials = {}

for lk in layer_order:
    pts, tops = [], []
    for bid, info in borehole_map.items():
        rec = info["layers"].get(lk)
        if rec: pts.append(info["XY"]); tops.append(rec["top"])
    if len(pts)==0: continue
    pts = np.array(pts); tops = np.array(tops)
    n = len(pts)
    if n == 1:
        c = pts[0]
        z = tops[0]

        # 与 n==2 模式保持一致
        half_size = two_pt_off

        verts = np.array([
            [c[0] - half_size, z, c[1] - half_size],
            [c[0] + half_size, z, c[1] - half_size],
            [c[0] + half_size, z, c[1] + half_size],
            [c[0] - half_size, z, c[1] + half_size],
        ])

        faces = flip_faces([
            (0, 1, 2),
            (0, 2, 3)
        ])
    elif n==2:
        A,B=pts; zA,zB=tops; d=B-A; perp=np.array([-d[1],d[0]]); perp/=np.linalg.norm(perp)+1e-9; o=two_pt_off
        verts=np.array([[A[0]+perp[0]*o,zA,A[1]+perp[1]*o],
                        [B[0]+perp[0]*o,zB,B[1]+perp[1]*o],
                        [B[0]-perp[0]*o,zB,B[1]-perp[1]*o],
                        [A[0]-perp[0]*o,zA,A[1]-perp[1]*o]])
        faces = flip_faces([(0,1,2),(0,2,3)])
    else:
        verts = gis_points_to_obj(np.column_stack([pts[:,0], pts[:,1], tops]))
        tri = Delaunay(pts)
        faces = flip_faces([tuple(t) for t in tri.simplices])
    uvs = uv_from_vertices(verts)
    base = lk.split("_")[0]; rgb=layer_colors.get(base,color_default); tex=layer_textures.get(base)
    layer_meshes[lk] = {"verts":verts,"uvs":uvs,"faces":faces,"material":lk,"rgb":rgb,"texture":tex}
    materials[lk] = {"rgb":rgb,"texture":tex}

# =========================
# 输出每层 OBJ
# =========================
for lk,m in layer_meshes.items():
    obj = os.path.join(OUT_DIR,f"{lk}.obj")
    mtl = os.path.join(OUT_DIR,f"{lk}.mtl")
    write_single_layer_obj(obj, mtl, m["verts"], m["uvs"], m["faces"], lk, m["rgb"], m["texture"])

# =========================
# 合并 OBJ
# =========================
write_mtl(COMBINED_MTL, materials)
g_verts,g_uvs,g_faces=[],[],[]; voff=0
for i,lk in enumerate(layer_order):
    if lk not in layer_meshes: continue
    m=layer_meshes[lk]; v=m["verts"].copy(); v[:,1]+=(len(layer_order)-i-1)*layer_spacing
    f=[(a+voff,b+voff,c+voff) for a,b,c in m["faces"]]
    g_verts.append(v); g_uvs.append(m["uvs"]); g_faces.append((lk,f))
    voff+=len(v)
g_verts=np.vstack(g_verts); g_uvs=np.vstack(g_uvs)
write_obj_with_uv(COMBINED_OBJ,COMBINED_MTL,g_verts,g_uvs,g_faces)

# =========================
# 输出异常点 CSV
# =========================
if anomalies:
    pd.DataFrame(anomalies).to_csv(ANOMALY_CSV, index=False)

# =========================
# 合并所有 coal 层生成一个 OBJ（支持爆炸）
# =========================

def merge_coal_layers(layer_meshes, out_obj, out_mtl, layer_spacing=0.0):
    """
    合并所有 coal 层（coal开头的 LayerKey），生成一个 OBJ 和 MTL。
    支持爆炸效果：layer_spacing>0 时按层间偏移堆叠。
    """
    coal_layers = [lk for lk in layer_meshes if lk.startswith("coal")]
    if not coal_layers:
        print("No coal layers found.")
        return

    materials = {lk: {"rgb": layer_meshes[lk]["rgb"], "texture": layer_meshes[lk]["texture"]} 
                 for lk in coal_layers}

    write_mtl(out_mtl, materials)

    g_verts, g_uvs, g_faces = [], [], []
    voff = 0

    for i, lk in enumerate(coal_layers):
        m = layer_meshes[lk]
        v = m["verts"].copy()
        # 爆炸偏移：每层沿 Y 方向堆叠
        v[:, 1] += (len(coal_layers) - i - 1) * layer_spacing
        f = [(a+voff, b+voff, c+voff) for a, b, c in m["faces"]]
        g_verts.append(v)
        g_uvs.append(m["uvs"])
        g_faces.append((lk, f))
        voff += len(v)

    g_verts = np.vstack(g_verts)
    g_uvs = np.vstack(g_uvs)
    write_obj_with_uv(out_obj, out_mtl, g_verts, g_uvs, g_faces)

    print(f"Coal layers merged OBJ: {out_obj}")


# 使用示例
coal_obj = os.path.join(OUT_DIR, "coal_combined.obj")
coal_mtl = os.path.join(OUT_DIR, "coal_combined.mtl")
merge_coal_layers(layer_meshes, coal_obj, coal_mtl, layer_spacing=layer_spacing)



# =========================
# 钻孔柱状图 OBJ 生成（最终版）
# =========================

print("Generating borehole columns OBJ (final version)...")

BOREHOLE_OBJ = os.path.join(OUT_DIR, "borehole_columns.obj")
BOREHOLE_MTL = os.path.join(OUT_DIR, "borehole_columns.mtl")

N_SIDE = 32   # 圆柱精度（24~40 都可以）
bh_vertices = []
bh_uvs = []
bh_faces_by_mat = []
col_materials = {}
vertex_offset = 0


def cylinder_uvs(verts, x0, z0, y_bot, y_top):
    """
    圆柱侧面 UV（防止拉花）
    U: 圆周方向
    V: 高度方向
    """
    uvs = []
    h = max(y_top - y_bot, 1e-6)
    for x, y, z in verts:
        theta = np.arctan2(z - z0, x - x0)
        u = (theta + np.pi) / (2 * np.pi)
        v = (y - y_bot) / h
        uvs.append([u, v])
    return np.array(uvs)


for bid, grp in df.groupby("ID"):
    # 钻孔 XY（已做过原点平移）
    x0 = grp["X"].iloc[0]
    z0 = grp["Y"].iloc[0]

    # 严格按 Depth_top 排序（由浅到深）
    grp = grp.sort_values("Depth_top")

    for _, r in grp.iterrows():
        lith = str(r["Lithology"])
        base = lith

        rgb = layer_colors.get(base, color_default)
        tex = layer_textures.get(base)

        # Z 值
        y_top = float(r["Z_surface"] - r["Depth_top"])
        y_bot = float(r["Z_surface"] - r["D_bottom"])

        if abs(y_top - y_bot) < 1e-6:
            continue

        # ---------- 生成圆柱顶点 ----------
        verts_seg = []

        # 底圈
        for i in range(N_SIDE):
            ang = 2 * np.pi * i / N_SIDE
            vx = x0 + auto_radius * np.cos(ang)
            vz = z0 + auto_radius * np.sin(ang)
            verts_seg.append([vx, y_bot, vz])

        # 顶圈
        for i in range(N_SIDE):
            ang = 2 * np.pi * i / N_SIDE
            vx = x0 + auto_radius * np.cos(ang)
            vz = z0 + auto_radius * np.sin(ang)
            verts_seg.append([vx, y_top, vz])

        verts_seg = np.array(verts_seg)

        # ---------- UV（圆柱展开） ----------
        uvs_seg = cylinder_uvs(
            verts_seg, x0, z0, y_bot, y_top
        )

        # ---------- 面（法线朝外） ----------
        faces_seg = []
        for i in range(N_SIDE):
            j = (i + 1) % N_SIDE

            bi = i
            bj = j
            ti = i + N_SIDE
            tj = j + N_SIDE

            # 外法线顺序（右手系）
            faces_seg.append((
                bi + vertex_offset,
                ti + vertex_offset,
                bj + vertex_offset
            ))
            faces_seg.append((
                bj + vertex_offset,
                ti + vertex_offset,
                tj + vertex_offset
            ))

        bh_vertices.append(verts_seg)
        bh_uvs.append(uvs_seg)
        bh_faces_by_mat.append((base, faces_seg))

        col_materials[base] = {
            "rgb": rgb,
            "texture": tex
        }

        vertex_offset += verts_seg.shape[0]

# ---------- 输出 OBJ ----------
if bh_vertices:
    bh_vertices = np.vstack(bh_vertices)
    bh_uvs = np.vstack(bh_uvs)

    write_mtl(BOREHOLE_MTL, col_materials)
    write_obj_with_uv(
        BOREHOLE_OBJ,
        BOREHOLE_MTL,
        bh_vertices,
        bh_uvs,
        bh_faces_by_mat
    )

    print("✔ Borehole columns OBJ written:", BOREHOLE_OBJ)
else:
    print("⚠ No borehole columns generated.")


print("All done. Output ->", OUT_DIR)
