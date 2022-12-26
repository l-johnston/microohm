"""microohm - SI unit system in Numpy using new DTypeMeta"""
from importlib.metadata import version
from microohm.quantity import Quantity
from microohm.quantitydtype import QuantityDType

__version__ = version("microohm")
__all__ = ["__version__", "Quantity", "QuantityDType"]
