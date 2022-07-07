from __future__ import annotations

import sys
from enum import IntEnum, Enum, auto, IntFlag
from pathlib import Path
from triton import ARCH, SOLVER_STATE, SOLVER
from typing import Union, TypeVar, Tuple


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
""" Symbolic Variable as returned by Triton (`SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicVariable_page.html>`_) """

Edge = Tuple[Addr, Addr]
""" Type representing a edge in the program """

PathHash = str
"""Type representing the hash of path to uniquely identify any path """


class SymExType(str, Enum):
    """
    Symobolic Expression type enum.
    """

    CONDITIONAL_JMP = 'cond-jcc'
    DYNAMIC_JMP = 'dyn-jmp'
    SYMBOLIC_READ = 'sym-read'
    SYMBOLIC_WRITE = 'sym-write'


if sys.version_info.minor >= 8:
    from typing import TypedDict

    class PathBranch(TypedDict):
        """
        Typed dictionnary describing the branch information
        returned by Triton (with getBranchConstraints())
        """
        isTaken: bool
        srcAddr: Addr
        dstAddr: Addr
        constraint: AstNode
else:
    PathBranch = TypeVar('PathBranch')
    """ PathBranchobject as returned by Triton.
    Thus it is a dictionnary with the keys:
    """


class Architecture(IntEnum):
    """ Common architecture Enum fully compatible with Triton `ARCH <https://triton.quarkslab.com/documentation/doxygen/py_ARCH_page.html>`_ """
    AARCH64 = ARCH.AARCH64
    ARM32   = ARCH.ARM32
    X86     = ARCH.X86
    X86_64  = ARCH.X86_64


class Platform(IntEnum):
    """ Enum to manipulate the platform associated to a binary"""
    LINUX = auto()
    WINDOWS = auto()
    MACOS = auto()
    ANDROID = auto()
    IOS = auto()

class SmtSolver(IntEnum):
    """ Common SMT Solver Enum fully compatible with Triton """
    Z3 = SOLVER.Z3
    BITWUZLA = SOLVER.BITWUZLA


class SolverStatus(IntEnum):
    """ Common Solver Enum fully compatible with Triton ARCH """
    SAT     = SOLVER_STATE.SAT
    UNSAT   = SOLVER_STATE.UNSAT
    TIMEOUT = SOLVER_STATE.TIMEOUT
    UNKNOWN = SOLVER_STATE.UNKNOWN

class Perm(IntFlag):
    R = 4
    W = 2
    X = 1

class Endian(IntEnum):
    LITTLE = 1
    BIG = 2
