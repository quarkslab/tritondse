# built-in modules
import platform
from collections import namedtuple

# third-party module
from triton import OPCODE, TritonContext

# local imports
from tritondse.types import Architecture, Addr

Arch = namedtuple("Arch", "ret_reg pc_reg bp_reg sp_reg sys_reg reg_args halt_inst syscall_inst")

ARCHS = {
    Architecture.X86:     Arch('eax', 'eip', 'ebp', 'esp', 'eax',
                               [],
                               OPCODE.X86.HLT,
                               [OPCODE.X86.SYSCALL, OPCODE.X86.SYSENTER]),  # ignore int 80
    Architecture.X86_64:  Arch('rax', 'rip', 'rbp', 'rsp', 'rax',
                               ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'],
                               OPCODE.X86.HLT,
                               [OPCODE.X86.SYSCALL, OPCODE.X86.SYSENTER]),  # ignore int 80
    Architecture.AARCH64: Arch('x0', 'pc', 'sp', 'sp', 'x8',
                               ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
                               OPCODE.AARCH64.HLT,
                               [OPCODE.AARCH64.SVC]),
    Architecture.ARM32:   Arch('r0', 'pc', 'r11', 'sp', 'r7',
                               ['r0', 'r1', 'r2', 'r3'],
                               OPCODE.ARM32.HLT,
                               [OPCODE.ARM32.SVC])
}


class CpuState(dict):
    """
    Thin wrapper on a TritonContext, to allow accessing
    and modifying registers in a Pythonic way. It also
    abstract base, stack, and program counter for architecture
    agnostic operations. This class performs all actions
    on the TritonContext, and does not hold any information.
    It is just acting as a proxy

    .. note:: This class adds dynamically attributes corresponding
              to register. Thus attributes will vary from an architecture
              to the other.

    >>> cpu.rax
    12
    >>> cpu.rax += 1
    >>> cpu.rax
    13

    No data is stored, all operations are performed on the
    TritonContext:

    >>> cpu.__ctx.getConcreteRegisterValue(cpu.rsp)
    0x7ff6540
    >>> cpu.stack_pointer += 8
    >>> cpu.__ctx.getConcreteRegisterValue(cpu.rsp)
    0x7ff6548

    .. note:: The user is not meant to instanciate it manually, and must
              use it through :py:obj:`ProcessState`.
    """

    def __init__(self, ctx: TritonContext, arch_info: Arch):
        super(CpuState, self).__init__()
        self.__ctx = ctx
        self.__archinfo = arch_info
        for r in ctx.getAllRegisters():
            self[r.getName()] = r

    def __getattr__(self, name: str):
        """
        Return the concrete value of a given register name

        :param name: The name of the register
        :return: the concrete value of a given register
        """
        if name in self:
            return self.__ctx.getConcreteRegisterValue(self[name])
        else:
            super().__getattr__(name)

    def __setattr__(self, name: str, value: int):
        """
        Set a concrete value to a given register name

        :param name: The name of the register
        :param value: The concrete value to set
        """
        if name in self:
            self.__ctx.setConcreteRegisterValue(self[name], value)
        else:
            super().__setattr__(name, value)

    @property
    def program_counter(self) -> int:
        """
        :return: The value of the program counter (RIP for x86, PC for ARM ..)
        :rtype: int
        """
        return getattr(self, self.__archinfo.pc_reg)

    @program_counter.setter
    def program_counter(self, value: int) -> None:
        """
        Set a value to the program counter

        :param value: Value to set
        :type value: int
        """
        setattr(self, self.__archinfo.pc_reg, value)

    @property
    def base_pointer(self) -> int:
        """
        :return: The value of the base pointer register
        """
        return getattr(self, self.__archinfo.bp_reg)

    @base_pointer.setter
    def base_pointer(self, value: int) -> None:
        """
        Set a value to the base pointer register

        :param value: Value to set
        :return: None
        """
        setattr(self, self.__archinfo.bp_reg, value)

    @property
    def stack_pointer(self) -> int:
        """
        :return: The value of the stack pointer register
        """
        return getattr(self, self.__archinfo.sp_reg)

    @stack_pointer.setter
    def stack_pointer(self, value: int) -> None:
        """
        Set a value to the stack pointer register

        :param value: Value to set
        :type value: int
        """
        setattr(self, self.__archinfo.sp_reg, value)


def local_architecture() -> Architecture:
    """
    Returns the architecture of the local machine.

    :return: local architecture
    """
    arch_m = {"i386": Architecture.X86,
              "x86_64": Architecture.X86_64,
              "armv7l": Architecture.ARM32,
              "aarch64": Architecture.AARCH64}
    return arch_m[platform.machine()]
