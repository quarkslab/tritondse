from pathlib import Path
from typing import Union
from triton import ARCH
from enum import IntEnum

# Type representing file either as a file, either as a Path object
PathLike = Union[str, Path]

# Integer representing an address
Addr = int

# Integer representing a relative address
rAddr = int

# Integer representing a value in bits
Bits = int

# Integer representing a value in bytes
Bytes = int


class Architecture(IntEnum):
    """ Common architecture Enum fully compatible with Triton ARCH """
    X86 = ARCH.X86
    X86_64 = ARCH.X86_64
    ARM32 = ARCH.ARM32
    AARCH64 = ARCH.AARCH64
