__version__ = "0.2.0"

from .KAN import KANLinear, KAN
from .ChebyKAN import ChebyKANLinear, ChebyKAN
from .FourierKAN import FourierKANLinear, FourierKAN
from .JacobiKAN import JacobiKANLinear, JacobiKAN
from .TaylorKAN import TaylorKANLinear, TaylorKAN
from .WaveletKAN import WaveletKANLinear, WaveletKAN

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
]
