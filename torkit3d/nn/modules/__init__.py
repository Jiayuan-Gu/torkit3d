"""Wrappers of builtin modules.

Notes:
    1. Builtin modules usually have builtin initializations.
    2. Default initialization of BN has been fixed since pytorch v1.2.0.
    3. If BN is applied after convolution, bias is unnecessary.
"""

from .conv import *
from .linear import *
from .mlp import *
