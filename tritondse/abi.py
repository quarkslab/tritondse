# built-in imports
from collections             import namedtuple
from typing                  import Tuple

# local imports
from tritondse.process_state import ProcessState
from tritondse.types         import Architecture, Register, Registers, Addr, Expression

Arch = namedtuple("Arch", "ret_reg pc_reg bp_reg sp_reg sys_reg reg_args")

ARCHS = {
    Architecture.X86:     Arch('eax', 'eip', 'ebp', 'esp', 'eax', []),
    Architecture.X86_64:  Arch('rax', 'rip', 'rbp', 'rsp', 'rax', ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']),
    Architecture.AARCH64: Arch('x0', 'pc', 'sp', 'sp', 'x8', ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
    # ARM ?
}


class ABI(object):
    """
    This class is used to represent the ABI.
    """
    def __init__(self, pstate: ProcessState):
        self.pstate = pstate

    @property
    def arch(self) -> Arch:
        try:
            return ARCHS[self.pstate.architecture]
        except KeyError:
            raise Exception(f"Architecture {self.pstate.architecture} not supported")


    @property
    def registers(self) -> Registers:
        return self.pstate.tt_ctx.registers


    def get_ret_register(self) -> Register:
        """ Return the appropriate return register according to the arch """
        return getattr(self.registers, self.arch.ret_reg)


    def get_pc_register(self) -> Register:
        """ Return the appropriate pc register according to the arch """
        return getattr(self.registers, self.arch.pc_reg)


    def get_bp_register(self) -> Register:
        """ Return the appropriate base pointer register according to the arch """
        return getattr(self.registers, self.arch.bp_reg)


    def get_sp_register(self) -> Register:
        """ Return the appropriate stack pointer register according to the arch """
        return getattr(self.registers, self.arch.sp_reg)


    def get_sys_register(self) -> Register:
        """ Return the appropriate syscall id register according to the arch """
        return getattr(self.registers, self.arch.sys_reg)


    def get_arg_register(self, i: int) -> Register:
        """
        Return the appropriate register according to the arch.

        .. FIXME: That method should be protected (with _)

        :raise: IndexError If the index is out of arguments bound
        :return: Register
        """
        return getattr(self.registers, self.arch.reg_args[i])

    @property
    def stack_pointer_value(self) -> Addr:
        return self.pstate.read_register(self.get_sp_register())

    def get_argument_value(self, i: int) -> int:
        """
        Return the integer value of parameters following the call convention.
        Thus the value originate either from a register or the stack.

        :param i: Ith argument of the function
        :return: integer value of the parameter
        """
        try:
            return self.pstate.read_register(self.get_arg_register(i))
        except IndexError:
            len_args = len(self.arch.reg_args)
            return self.get_stack_value(i-len_args)

    def get_symbolic_argument(self, i: int) -> Expression:
        """
        Return the symbolic expression associated with the given ith parameter.

        :param i: Ith function parameter
        :return: Symbolic expression associated
        """
        try:
            return self.pstate.read_symbolic_register(self.get_arg_register(i))
        except IndexError:
            len_args = len(self.arch.reg_args)
            addr = self.stack_pointer_value + ((i-len_args) * self.pstate.ptr_size)
            return self.pstate.read_symbolic_memory_int(addr, self.pstate.ptr_size)


    def get_full_argument(self, i: int) -> Tuple[int, Expression]:
        """
        Get both the concrete argument value along with its symbolic expression.

        :return: Tuple containing concrete value and symbolic expression
        """
        return self.get_argument_value(i), self.get_symbolic_argument(i)


    def get_string_argument(self, i: int) -> str:
        """ Return the string on the given function parameter """
        return self.pstate.get_memory_string(self.get_argument_value(i))


    def get_format_string(self, addr: Addr) -> str:
        """ Returns a formatted string from a memory address """
        return self.pstate.get_memory_string(addr)                                             \
               .replace("%s", "{}").replace("%d", "{}").replace("%#02x", "{:#02x}")     \
               .replace("%#x", "{:#x}").replace("%x", "{:x}").replace("%02X", "{:02X}") \
               .replace("%c", "{:c}").replace("%02x", "{:02x}").replace("%ld", "{}")    \
               .replace("%*s", "").replace("%lX", "{:X}").replace("%08x", "{:08x}")     \
               .replace("%u", "{}").replace("%lu", "{}").replace("%zu", "{}")           \
               .replace("%02u", "{:02d}").replace("%03u", "{:03d}")                     \
               .replace("%03d", "{:03d}").replace("%p", "{:#x}").replace("%i", "{}")


    def get_format_arguments(self, s, args):
        s_str = self.pstate.get_memory_string(s)
        postString = [i for i, x in enumerate([i for i, c in enumerate(s_str) if c == '%']) if s_str[x+1] == "s"]
        for p in postString:
            args[p] = self.pstate.get_memory_string(args[p])
        return args


    def get_stack_value(self, index: int) -> int:
        addr = self.stack_pointer_value + (index * self.pstate.ptr_size)
        return self.pstate.read_memory_int(addr, self.pstate.ptr_size)


