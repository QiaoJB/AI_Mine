import rasterio
import numpy as np
import matplotlib.pyplot as plt

def view_raster_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:

        data = src.read()
        bands, H, W = data.shape

        plt.figure(figsize=(8, 8))

        # 单波段
        if bands == 1:
            img = data[0]
            vmin, vmax = np.nanpercentile(img, [2, 98])
            plt.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)

        # RGB
        elif bands >= 3:
            rgb = data[:3].astype(np.float32)

            for i in range(3):
                vmin, vmax = np.nanpercentile(rgb[i], [2, 98])
                rgb[i] = np.clip((rgb[i] - vmin) / (vmax - vmin + 1e-6), 0, 1)

            rgb = np.transpose(rgb, (1, 2, 0))
            plt.imshow(rgb)

        else:
            img = data[0]
            plt.imshow(img, cmap="gray")

        plt.title(tiff_path)
        plt.axis("off")
        plt.show()

    return f"""
Raster loaded successfully

File: {tiff_path}
Bands: {bands}
Height: {H}
Width: {W}
CRS: {src.crs}
"""
