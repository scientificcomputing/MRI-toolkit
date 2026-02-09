"""MRI Data Base class and functions Module

Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
Copyright (C) 2026   Simula Research Laboratory
"""

import numpy as np


class MRIData:
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        self.data = data
        self.affine = affine

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.affine

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
