import os
from glob import glob

import vigra
import imageio


def postprocess(folder):
    files = glob(os.path.join(folder, "*.tif"))
    for ff in files:
        im = imageio.imread(ff)
        out, _, _ = vigra.analysis.relabelConsecutive(im.astype("uint32"))
        imageio.imwrite(ff, out.astype(im.dtype))


postprocess("./line_data")
postprocess("./circle_data")
