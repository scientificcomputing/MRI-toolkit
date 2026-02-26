"""Masking utils

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np
import skimage

def largest_island(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    newmask = skimage.measure.label(mask, connectivity=connectivity)
    regions = skimage.measure.regionprops(newmask)
    regions.sort(key=lambda x: x.num_pixels, reverse=True)
    return newmask == regions[0].label