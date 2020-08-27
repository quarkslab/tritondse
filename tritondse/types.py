from enum    import IntEnum
from pathlib import Path
from triton  import ARCH
from typing  import Union

# Type representing file either as a file, either as a Path object
PathLike = Union[str, Path]

# Integer representing an address
Addr = int

# Integer representing a relative address
rAddr = int

# Integer representing a value in bits
BitSize = int

# Integer representing a value in bytes
ByteSize = int


class Architecture(IntEnum):
    """ Common architecture Enum fully compatible with Triton ARCH """
    AARCH64 = ARCH.AARCH64
    ARM32   = ARCH.ARM32
    X86     = ARCH.X86
    X86_64  = ARCH.X86_64
