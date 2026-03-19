import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def view_npy_features(npy_path):

    parent_dir = os.path.dirname(os.path.dirname(npy_path))
    
    ref_tif = os.path.join(parent_dir, "B04.tiff")

    X = np.load(npy_path)

    print("Feature shape:", X.shape)
    print("Dtype:", X.dtype)

    # 如果是 [N_pixels, C]
    if X.ndim == 2:

        if ref_tif is None:
            return "Flattened feature detected. Please provide ref_tif."

        with rasterio.open(ref_tif) as src:
            H, W = src.height, src.width

        N, C = X.shape

        if N != H * W:
            return "Pixel count mismatch with reference raster."

        X = X.reshape(H, W, C).transpose(2, 0, 1)

    # 如果是 [C,H,W]
    if X.ndim == 3:
        C = X.shape[0]

        plt.figure(figsize=(10, 6))

        for i in range(min(C, 6)):

            plt.subplot(2, 3, i + 1)

            img = X[i]
            vmin, vmax = np.nanpercentile(img, [2, 98])

            plt.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
            plt.title(f"Channel {i}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        return f"""
Feature file loaded

Path: {npy_path}
Channels: {C}
Shape: {X.shape}
"""

    return "Unsupported feature format"