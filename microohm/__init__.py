"""microohm - SI unit system in Numpy using new DTypeMeta"""
from importlib.metadata import version
from microohm import units

__version__ = version("microohm")
__all__ = ["__version__", "units"]
