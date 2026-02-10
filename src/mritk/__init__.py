# Copyright (C) 2026   Jørgen Riseth (jnriseth@gmail.com)
# Copyright (C) 2026   Cécile Daversin-Catty (cecile@simula.no)
# Copyright (C) 2026   Simula Research Laboratory


from importlib.metadata import metadata

from . import data, segmentation, statistics


meta = metadata("mritk")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["license-expression"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = [
    "data",
    "segmentation",
    "statistics",
]
