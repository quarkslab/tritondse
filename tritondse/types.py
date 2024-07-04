# built-in imports
from __future__ import annotations
import sys
from enum import IntEnum, Enum, auto, IntFlag
from pathlib import Path
from typing import Union, TypeVar, Tuple
import io
import enum_tools.documentation
from dataclasses import dataclass

# third-party imports
from triton import ARCH, SOLVER_STATE, SOLVER


PathLike = Union[str, Path]
"""Type representing file either as a file, either as a Path object"""

Addr = int
"""Integer representing an address"""

rAddr = int
"""Integer representing a relative address"""

BitSize = int
"""Integer representing a value in bits"""

ByteSize = int
"""Integer representing a value in bytes"""

Input = bytes
""" Type representing an Input (which is bytes) """

Register = TypeVar('Register')
"""Register identifier as used by Triton *(not the Register object itself)*"""

Registers = TypeVar('Registers')
"""Set of registers as used by Triton"""

PathConstraint = TypeVar('PathConstraint')
""" `PathConstraint <https://triton.quarkslab.com/documentation/doxygen/py_PathConstraint_page.html>`_ object as returned by Triton"""

AstNode = TypeVar('AstNode')
""" SMT logic formula as returned by Triton (`AstNode <https://triton.quarkslab.com/documentation/doxygen/py_AstNode_page.html>`_) """

Model = TypeVar('Model')
""" Solver `Model <https://triton.quarkslab.com/documentation/doxygen/py_SolverModel_page.html>`_ as returned by Triton """

Expression = TypeVar('Expression')
""" Symbolic Expression as returned by Triton (`SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_) """

SymbolicVariable = TypeVar('SymbolicVariable')
""" Symbolic Variable as returned by Triton (`SymbolicVariable <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicVariable_page.html>`_) """

Edge = Tuple[Addr, Addr]
""" Type representing a edge in the program """

PathHash = str
"""Type representing the hash of path to uniquely identify any path """


@enum_tools.documentation.document_enum
class SymExType(str, Enum):
    """
    Symbolic Expression type enum. (internal usage only)
    """

    CONDITIONAL_JMP = 'cond-jcc'  # doc: symbolic expression is a conditional jump
    DYNAMIC_JMP = 'dyn-jmp'       # doc: symbolic expression is a dynamic jump
    SYMBOLIC_READ = 'sym-read'    # doc: symbolic expression is a symbolic memory read
    SYMBOLIC_WRITE = 'sym-write'  # doc: symbolic expression is a symbolic memory write


if sys.version_info.minor >= 8:
    from typing import TypedDict

    class PathBranch(TypedDict):
        """
        Typed dictionary describing the branch information
        returned by Triton (with getBranchConstraints())
        """
        isTaken: bool
        srcAddr: Addr
        dstAddr: Addr
        constraint: AstNode
else:
    PathBranch = TypeVar('PathBranch')
    """ PathBranchobject as returned by Triton.
    Thus it is a dictionary with the keys:
    """


@enum_tools.documentation.document_enum
class Architecture(IntEnum):
    """
    Common architecture Enum fully compatible with Triton
    `ARCH <https://triton.quarkslab.com/documentation/doxygen/py_ARCH_page.html>`_
    """
    AARCH64 = ARCH.AARCH64  # doc: Aarch64 architecture
    ARM32 = ARCH.ARM32      # doc: ARM architecture (32 bits)
    X86 = ARCH.X86          # doc: x86 architecture (32 bits)
    X86_64 = ARCH.X86_64    # doc: x86-64 architecture (64 bits)


@enum_tools.documentation.document_enum
class ArchMode(IntFlag):
    """
    Various architecture specific modes that can be enabled or disabled.
    (meant to be fulfilled)
    """
    THUMB = 1   # doc: set thumb mode for ARM32 architecture


@enum_tools.documentation.document_enum
class Platform(IntEnum):
    """
    Platform associated to a binary
    """
    LINUX = auto()    # doc: Linux platform
    WINDOWS = auto()  # doc: Windows platform
    MACOS = auto()    # doc: Mac OS platform
    ANDROID = auto()  # doc: Android platform
    IOS = auto()      # doc: IOS platform


@enum_tools.documentation.document_enum
class SmtSolver(IntEnum):
    """ Common SMT Solver Enum fully compatible with Triton """
    Z3 = SOLVER.Z3              # doc: Z3 SMT solver
    BITWUZLA = SOLVER.BITWUZLA  # doc: bitwuzla solver


@enum_tools.documentation.document_enum
class SolverStatus(IntEnum):
    """ Common Solver Enum fully compatible with Triton ARCH """
    SAT     = SOLVER_STATE.SAT      # doc: Formula is satisfiable (SAT)
    UNSAT   = SOLVER_STATE.UNSAT    # doc: Formula is unsatisfiable (UNSAT)
    TIMEOUT = SOLVER_STATE.TIMEOUT  # doc: Formula solving did timeout
    UNKNOWN = SOLVER_STATE.UNKNOWN  # doc: Formula solving failed


@enum_tools.documentation.document_enum
class Perm(IntFlag):
    """
    Flags encoding permissions. Used for memory pages.
    They can be combined as flags. e.g:

    .. code-block:: python

        rw = Perm.R | Perm.W
    """
    R = 4  # doc: Read
    W = 2  # doc: Write
    X = 1  # doc: Execute


@enum_tools.documentation.document_enum
class Endian(IntEnum):
    """
    Endianness of the binary.
    """
    LITTLE = 1  # doc: Little-endian
    BIG = 2     # doc: Big-endian


@enum_tools.documentation.document_enum
class Format(IntEnum):
    """
    Executable File Format
    """
    ELF = auto()    # doc: ELF file format
    PE = auto()     # doc: PE file format
    MACHO = auto()  # doc: Mach-O file format


@dataclass
class FileDesc:
    """
    Type representing a file descriptor
    """
    """ The target program's file descriptor """
    id: int
    """ Name of the file """
    name: str
    """ The python file stream object """
    fd: io.IOBase

    @property
    def offset(self) -> int:
        return self.fd.tell()

    def is_real_fd(self) -> bool:
        return isinstance(self.fd, io.TextIOWrapper)

    def is_input_fd(self) -> bool:
        return isinstance(self.fd, io.BytesIO)

    def fgets(self, max_size: int) -> bytes:
        s = b""
        for i in range(max_size):
            c = self.fd.read(1)
            if not c:  # EOF
                break
            c = c if isinstance(c, bytes) else c.encode()
            s += c
            if c == b"\x00":
                return s
            elif c == b"\n":
                break
        # If get there read max_size
        return s+b"\x00"

    def read(self, size: int) -> bytes:
        data = self.fd.read(size)
        return data if isinstance(data, bytes) else data.encode()
