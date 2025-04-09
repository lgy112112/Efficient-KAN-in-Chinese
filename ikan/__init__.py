__version__ = "1.3.0"

from .KAN import KANLinear, KAN
from .ChebyKAN import ChebyKANLinear, ChebyKAN
from .FourierKAN import FourierKANLinear, FourierKAN
from .JacobiKAN import JacobiKANLinear, JacobiKAN
from .TaylorKAN import TaylorKANLinear, TaylorKAN
from .WaveletKAN import WaveletKANLinear, WaveletKAN
from .GroupKAN import GroupKANLinear, GroupKAN
from .kat_1dgroup_triton import RationalTriton1DGroup, KAT_Group
from .kat_2dgroup_triton import KAT_Group2D
from .kat_1dgroup_torch import KAT_Group_Torch

__all__ = [
    "KANLinear",
    "KAN",
    "ChebyKANLinear",
    "ChebyKAN",
    "FourierKANLinear",
    "FourierKAN",
    "JacobiKANLinear",
    "JacobiKAN",
    "TaylorKANLinear",
    "TaylorKAN",
    "WaveletKANLinear",
    "WaveletKAN",
    "GroupKANLinear",
    "GroupKAN",
]
