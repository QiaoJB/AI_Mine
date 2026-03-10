# visual_feature_interactive.py

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap


class FeatureVisualizer:

    def __init__(self,
                 feature_data,
                 ref_tif,
                 base_img="data/sentinel2/TRUE.tiff",
                 sample_shp=None,
                 channel_names=None,
                 cmap='viridis'):

        self.ref = rasterio.open(ref_tif)
        self.H, self.W = self.ref.height, self.ref.width

        # -----------------------
        # feature data
        # -----------------------

        if isinstance(feature_data, str):
            X = np.load(feature_data)
        else:
            X = feature_data

        if X.ndim == 2:

            N, C = X.shape
            self.C = C
            self.X_img = X.reshape(self.H, self.W, C).transpose(2,0,1)

        elif X.ndim == 3:

            self.C = X.shape[0]
            self.X_img = X

        else:
            raise ValueError("Unsupported feature shape")

        # -----------------------
        # channel names
        # -----------------------

        if channel_names is None:
            self.channel_names = [f'Channel {i}' for i in range(self.C)]
        else:
            self.channel_names = channel_names

        # -----------------------
        # base image
        # -----------------------

        self.base = None

        if base_img is not None:

            with rasterio.open(base_img) as src:
                img = src.read()

            if img.shape[0] >= 3:
                rgb = np.stack([img[0], img[1], img[2]], axis=-1)
            else:
                rgb = img[0]

            rgb = rgb.astype(np.float32)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

            self.base = rgb

        # -----------------------
        # samples
        # -----------------------

        self.samples = None

        if sample_shp is not None:

            gdf = gpd.read_file(sample_shp).to_crs(self.ref.crs)

            xs = []
            ys = []

            for pt in gdf.geometry:

                r, c = ~self.ref.transform * (pt.x, pt.y)

                xs.append(int(c))
                ys.append(int(r))

            self.samples = (xs, ys)

        # -----------------------
        # visualization
        # -----------------------

        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(9,9))
        plt.subplots_adjust(bottom=0.15)

        if self.base is not None:
            self.ax.imshow(self.base)

        self.overlay = self.ax.imshow(
            self.X_img[self.idx],
            cmap=cmap,
            alpha=0.6
        )

        if self.samples is not None:

            xs, ys = self.samples

            self.ax.scatter(
                xs,
                ys,
                c='red',
                s=40,
                edgecolors='black',
                label='Samples'
            )

            self.ax.legend()

        self.ax.set_title(self.channel_names[self.idx])

        self.fig.colorbar(self.overlay, ax=self.ax)

        # -----------------------
        # slider
        # -----------------------

        ax_alpha = plt.axes([0.25, 0.03, 0.5, 0.03])

        self.slider = Slider(
            ax_alpha,
            'Opacity',
            0.0,
            1.0,
            valinit=0.6
        )

        self.slider.on_changed(self.update_alpha)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def update_alpha(self, val):

        self.overlay.set_alpha(val)
        self.fig.canvas.draw_idle()

    def on_key(self, event):

        if event.key == 'right':
            self.idx = (self.idx + 1) % self.C

        elif event.key == 'left':
            self.idx = (self.idx - 1) % self.C

        else:
            return

        self.update()

    def update(self):

        self.overlay.set_data(self.X_img[self.idx])

        self.ax.set_title(self.channel_names[self.idx])

        self.overlay.set_clim(
            vmin=np.nanmin(self.X_img[self.idx]),
            vmax=np.nanmax(self.X_img[self.idx])
        )

        self.fig.canvas.draw_idle()


# ===============================
# 岩性 one-hot → 单通道
# ===============================

def build_lithology_map(lith_file, ref_tif):

    X = np.load(lith_file)

    with rasterio.open(ref_tif) as src:
        H = src.height
        W = src.width

    N, C = X.shape

    lith_map = np.argmax(X, axis=1).reshape(H, W)

    lith_map = lith_map[np.newaxis,:,:]

    return lith_map


# ===============================
# 读取预测结果
# ===============================

def load_prediction_stack(tif_list):

    stack = []

    for tif in tif_list:

        with rasterio.open(tif) as src:
            stack.append(src.read(1))

    return np.stack(stack)


# ===============================
# main
# ===============================

if __name__ == "__main__":

    ref = "data/sentinel2/B04.tiff"

    # ===============================
    # 岩性（单通道）
    # ===============================

    lith_map = build_lithology_map(
        "data/lithology_onehot.npy",
        ref
    )

    lith_cmap = ListedColormap([
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf"
    ])

    FeatureVisualizer(
        lith_map,
        ref,
        channel_names=["Lithology"],
        cmap=lith_cmap
    )

    # ===============================
    # 蚀变
    # ===============================

    alter_names = [
        "fe_oxide_nd",
        "hematite_nd",
        "al_oh_nd",
        "chlorite_nd",
        "silica_proxy"
    ]

    FeatureVisualizer(
        "data/alter_stack.npy",
        ref,
        channel_names=alter_names,
        cmap='plasma'
    )

    # ===============================
    # 结构
    # ===============================

    struct_names = [
        "Slope",
        "Curvature",
        "DEM_grad",
        "Sentinel_grad",
        "Fusion_density"
    ]

    FeatureVisualizer(
        "data/dem/structural_stack.npy",
        ref,
        channel_names=struct_names,
        cmap='terrain'
    )

    # ===============================
    # 预测结果（通道）
    # ===============================

    pred_stack = load_prediction_stack([
        "output/gold_unsupervised_pure_feature.tif",
        "output/gold_prediction_small_samples.tif",
        "output/gold_rf_prediction_optimized.tif"
    ])

    pred_names = [
        "Unsupervised IF",
        "Small Sample GMM+IF",
        "Random Forest"
    ]

    FeatureVisualizer(
        pred_stack,
        ref,
        sample_shp="data/labels/gold_points.shp",
        channel_names=pred_names,
        cmap="turbo"
    )