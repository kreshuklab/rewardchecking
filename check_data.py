import os
from glob import glob

import imageio
import napari


def check_data(folder):
    sp = imageio.imread(os.path.join(folder, "superpixel.tif"))
    merged_paths = glob(os.path.join(folder, "merged*.tif"))
    merged_paths.sort()
    merged_ims = {}
    for ii, p in enumerate(merged_paths):
        merged = imageio.imread(p)
        merged_ims[f"merged-{ii}"] = merged

    v = napari.Viewer()
    v.add_labels(sp, name="superpixel")
    for name, merged in merged_ims.items():
        v.add_labels(merged, name=name)
    napari.run()


check_data("./line_data")
check_data("./circle_data")
