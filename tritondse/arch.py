from collections     import namedtuple
from tritondse.types import Architecture, Addr

Arch = namedtuple("Arch", "ret_reg pc_reg bp_reg sp_reg sys_reg reg_args halt_inst")

ARCHS = {
    Architecture.X86:     Arch('eax', 'eip', 'ebp', 'esp', 'eax', [], b"\xf4"),
    Architecture.X86_64:  Arch('rax', 'rip', 'rbp', 'rsp', 'rax', ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'], b"\xf4"),
    Architecture.AARCH64: Arch('x0', 'pc', 'sp', 'sp', 'x8', ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], b"")
    # ARM ?
}


class CpuState(dict):
    def __init__(self, ctx, arch_info):
        super(CpuState, self).__init__()
        self.__ctx = ctx
        self.__archinfo = arch_info
        for r in ctx.getAllRegisters():
            self[r.getName()] = r

    def __getattr__(self, name):
        if name in self:
            return self.__ctx.getConcreteRegisterValue(self[name])
        else:
            super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in self:
            self.__ctx.setConcreteRegisterValue(self[name], value)
        else:
            super().__setattr__(name, value)

    @property
    def program_counter(self) -> Addr:
        return getattr(self, self.__archinfo.pc_reg)

    @program_counter.setter
    def program_counter(self, value) -> None:
        setattr(self, self.__archinfo.pc_reg, value)

    @property
    def base_pointer(self) -> Addr:
        return getattr(self, self.__archinfo.bp_reg)

    @base_pointer.setter
    def base_pointer(self, value) -> None:
        setattr(self, self.__archinfo.bp_reg, value)

    @property
    def stack_pointer(self) -> Addr:
        return getattr(self, self.__archinfo.sp_reg)

    @stack_pointer.setter
    def stack_pointer(self, value) -> None:
        setattr(self, self.__archinfo.sp_reg, value)
