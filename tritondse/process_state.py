# built-ins
import sys
import time
import logging
from typing import Union, Callable, Tuple, Optional


# third-party
from triton import TritonContext, MemoryAccess, CALLBACK, CPUSIZE, Instruction

# local imports
from tritondse.thread_context import ThreadContext
from tritondse.program        import Program
from tritondse.heap_allocator import HeapAllocator
from tritondse.types          import Architecture, Addr, ByteSize, BitSize, PathConstraint, Register, Expression, AstNode, Registers
from tritondse.arch           import ARCHS, CpuState




class ProcessState(object):
    """
    This class is used to represent the state of a process.
    """
    def __init__(self, thread_scheduling: int, time_coefficient: int):
        # Memory mapping
        self.BASE_PLT   = 0x01000000
        self.BASE_ARGV  = 0x02000000
        self.BASE_CTYPE = 0x03000000
        self.ERRNO_PTR  = 0x04000000
        self.BASE_HEAP  = 0x10000000
        self.END_HEAP   = 0x6fffffff
        self.BASE_STACK = 0xefffffff
        self.END_STACK  = 0x70000000
        self.START_MAP  = 0x01000000
        self.END_MAP    = 0xf0000000

        # The Triton's context
        self.tt_ctx = TritonContext()
        self.actx = self.tt_ctx.getAstContext()

        # Cpu object wrapping registers values
        self.cpu = None
        self._archinfo = None

        # Used to define that the process must exist
        self.stop = False

        # Signals table used by raise(), signal(), etc.
        self.signals_table = dict()

        # File descriptors table used by fopen(), fprintf(), etc.
        self.fd_table = {
            0: sys.stdin,
            1: sys.stdout,
            2: sys.stderr,
        }
        # Unique file id incrementation
        self.fd_id = len(self.fd_table)

        # Allocation information used by malloc()
        self.heap_allocator = HeapAllocator(self.BASE_HEAP, self.END_HEAP)

        # Unique thread id incrementation
        self.utid = 0

        # Current thread id
        self.tid = self.utid

        # Threads contexts
        self.threads = {
            self.utid: ThreadContext(self.tid, thread_scheduling)
        }

        # Thread mutext init magic number
        self.PTHREAD_MUTEX_INIT_MAGIC = 0xdead

        # Mutex and semaphore
        self.mutex_locked = False
        self.semaphore_locked = False

        # The time when the ProcessState is instancied.
        # It's used to provide a deterministic behavior when calling functions
        # like gettimeofday(), clock_gettime(), etc.
        self.time = time.time()

        # Hold the loading address where the main program has been loaded
        self.load_addr = 0

        # Configuration values
        self.thread_scheduling_count = thread_scheduling
        self.time_inc_coefficient = time_coefficient

        # Runtime temporary variables
        self.__pcs_updated = False

        # The current instruction executed
        self.__current_inst = None

        # The memory mapping of the program ({vaddr_s : vaddr_e})
        self.__program_segments_mapping = {}


    def get_unique_thread_id(self):
        """ Return an unique thread id """
        self.utid += 1
        return self.utid


    def get_unique_file_id(self):
        """ Return an unique file descriptor """
        self.fd_id += 1
        return self.fd_id


    @property
    def architecture(self) -> Architecture:
        """ Return architecture of the current process state """
        return Architecture(self.tt_ctx.getArchitecture())


    @architecture.setter
    def architecture(self, arch: Architecture) -> None:
        """
        Set the architecture of the process state.
        Internal set it in the TritonContext
        """
        self.tt_ctx.setArchitecture(arch)


    @property
    def ptr_size(self) -> ByteSize:
        """ Size of a pointer in bytes """
        return self.tt_ctx.getGprSize()


    @property
    def ptr_bit_size(self) -> BitSize:
        """ Size of a pointer in bits """
        return self.tt_ctx.getGprBitSize()


    @property
    def minus_one(self) -> int:
        """ -1 according to the architecture size """
        return (1 << self.ptr_bit_size) - 1


    @property
    def registers(self) -> Registers:
        """ All registers according to the current architecture defined """
        return self.tt_ctx.registers


    @property
    def return_register(self) -> Register:
        """ Return the appropriate return register according to the arch """
        return getattr(self.registers, self._archinfo.ret_reg)

    @property
    def program_counter_register(self) -> Register:
        """ Return the appropriate pc register according to the arch """
        return getattr(self.registers, self._archinfo.pc_reg)

    @property
    def base_pointer_register(self) -> Register:
        """ Return the appropriate base pointer register according to the arch """
        return getattr(self.registers, self._archinfo.bp_reg)

    @property
    def stack_pointer_register(self) -> Register:
        """ Return the appropriate stack pointer register according to the arch """
        return getattr(self.registers, self._archinfo.sp_reg)

    @property
    def _syscall_register(self) -> Register:
        """ Return the appropriate syscall id register according to the arch """
        return getattr(self.registers, self._archinfo.sys_reg)


    def _get_argument_register(self, i: int) -> Register:
        """
        Return the appropriate register according to the arch.

        :raise: IndexError If the index is out of arguments bound
        :return: Register
        """
        return getattr(self.registers, self._archinfo.reg_args[i])


    def initialize_context(self, arch: Architecture):
        """
        Initialize the context

        :param arch: The architecture to initialize
        :return: None
        """
        self.architecture = arch
        self._archinfo = ARCHS[self.architecture]
        self.cpu = CpuState(self.tt_ctx, self._archinfo)
        self.write_register(self.base_pointer_register, self.BASE_STACK)
        self.write_register(self.stack_pointer_register, self.BASE_STACK)


    def load_program(self, p: Program, base_addr: Addr = 0) -> None:
        """
        Load the given program in the process state memory

        :param p: Program to load in the process memory
        :param base_addr: Base address where to load the program (if PIE)
        :return: True on whether loading succeeded or not
        """
        # Set the program counter to points to entrypoint
        self.cpu.program_counter = p.entry_point

        # Set loading address
        self.load_addr = base_addr
        # TODO: If PIE use this address, if not set absolute address (from binary)

        # Load memory areas in memory
        for vaddr, data in p.memory_segments():
            logging.debug(f"Loading {vaddr:#08x} - {vaddr+len(data):#08x}")
            self.tt_ctx.setConcreteMemoryAreaValue(vaddr, data)
            size = len(data)
            if vaddr in self.__program_segments_mapping and self.__program_segments_mapping[vaddr] > vaddr + size:
                # If we already have a vaddr entry, keep the larger one
                pass
            else:
                self.__program_segments_mapping.update({vaddr : vaddr + size})


    def is_valid_memory_mapping(self, ptr, padding_segment=0) -> bool:
        """
        Check if a given address is mapped into our memory areas

        :param ptr: The pointer to check
        :param padding_segment: A padding to add at the end of segment if necessary
        :return: True if ptr is mapped otherwise returns False
        """
        valid_access = False

        # Check stack area
        if ptr <= self.BASE_STACK and ptr >= self.END_STACK:
            valid_access = True

        # Check heap area
        if ptr >= self.BASE_HEAP and ptr <= self.END_HEAP:
            if self.is_heap_ptr(ptr) and self.heap_allocator.is_ptr_allocated(ptr):
                valid_access = True

        # Check other areas
        if ptr >= self.BASE_PLT and ptr <= self.ERRNO_PTR + CPUSIZE.QWORD:
            valid_access = True

        # Check segments mapping
        for vaddr_s, vaddr_e in self.__program_segments_mapping.items():
            if ptr >= vaddr_s and ptr < vaddr_e + padding_segment:
                valid_access = True

        return valid_access


    def read_register(self, register: Union[str, Register]) -> int:
        """
        Read the current concrete value of the given register.

        :param register: string of the register or Register object
        :return: Integer value
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        return self.tt_ctx.getConcreteRegisterValue(reg)


    def write_register(self, register: Union[str, Register], value: int) -> None:
        """
        Read the current concrete value of the given register.

        :param register: string of the register or Register object
        :param value: integer value to assign in the register
        :return: Integer value
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        return self.tt_ctx.setConcreteRegisterValue(reg, value)


    def read_memory_int(self, addr: Addr, size: ByteSize) -> int:
        """
        Read in the process memory a little-endian integer of the ``size`` at ``addr``.

        :param addr: Address at which to read data
        :param size: Number of bytes to read
        :return: Integer value read
        """
        return self.tt_ctx.getConcreteMemoryValue(MemoryAccess(addr, size))


    def read_memory_ptr(self, addr: Addr) -> int:
        """
        Read in the process memory a little-endian integer of the
         ptr_size size

        :param addr: Address at which to read data
        :return: Integer value read
        """
        return self.tt_ctx.getConcreteMemoryValue(MemoryAccess(addr, self.ptr_size))


    def read_memory_bytes(self, addr: Addr, size: ByteSize) -> bytes:
        """
        Read in the process memory ``size`` amount of bytes at ``addr``.
        Data read is returned as bytes.

        :param addr: Address at which to read data
        :param size: Number of bytes to read
        :return: Data read
        """
        return self.tt_ctx.getConcreteMemoryAreaValue(addr, size)


    def write_memory_int(self, addr: Addr, size: ByteSize, value: int) -> None:
        """
        Write in the process memory the given integer value of the given size at
        a specific address.

        :param addr: address where to write data
        :param size: size of data to write in bytes
        :param value: data to write represented as an integer
        :return: None

        .. todo:: Adding a parameter to specify endianess if needed
        """
        self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, size), value)


    def write_memory_bytes(self, addr: Addr, data: bytes) -> None:
        """
        Write the given bytes in the process memory. Size is automatically
        deduced with data size.

        :param addr: address where to write data
        :param data: bytes data to write
        :return: None
        """
        self.tt_ctx.setConcreteMemoryAreaValue(addr, data)


    def write_memory_byte(self, addr: Addr, data: Union[bytes, int]) -> None:
        """
        Write the given bytes in the process memory. Size is automatically
        deduced with data size.

        :param addr: address where to write data
        :param data: bytes data to write
        :return: None
        """
        if isinstance(data, int):
            self.write_memory_int(addr, CPUSIZE.BYTE, data)
        else:
            self.write_memory_bytes(addr, data)


    def write_memory_ptr(self, addr: Addr, value: int) -> None:
        """
        Similar to write_memory_int but the size is automatically adjusted
        to be ptr_size.

        :param addr: address where to write data
        :param value: pointer value to write
        """
        self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, self.ptr_size), value)


    def register_triton_callback(self, cb_type: CALLBACK, callback: Callable):
        """
        Register the given ``callback`` as triton callback to hook memory/registers
        read/writes.

        :param cb_type: CALLBACK type as defined by Triton
        :param callback: routines to call on the given event
        :return: None
        """
        self.tt_ctx.addCallback(cb_type, callback)


    def is_heap_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is coming from the heap area.

        :param ptr: Address to check
        :return: True if pointer points to the heap area (allocated or not).
        """
        if self.BASE_HEAP <= ptr < self.END_HEAP:
            return True
        return False


    def is_stack_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is coming from the stack area.

        :param ptr: Address to check
        :return: True if pointer points to the stack area (allocated or not).
        """
        if self.BASE_STACK <= ptr < self.END_STACK:
            return True
        return False


    def process_instruction(self, instruction: Instruction) -> bool:
        """
        Process the given triton instruction on this process state.
        :param instruction:
        :return:
        """
        self.__pcs_updated = False
        __len_pcs = self.tt_ctx.getPathPredicateSize()

        self.tt_ctx.disassembly(instruction)
        self.__current_inst = instruction
        ret = self.tt_ctx.buildSemantics(instruction)

        # Simulate that the time of an executed instruction is time_inc_coefficient.
        # For example, if time_inc_coefficient is 0.0001, it means that an instruction
        # takes 100us to be executed. Used to provide a deterministic behavior when
        # calling time functions (e.g gettimeofday(), clock_gettime(), ...).
        self.time += self.time_inc_coefficient

        if self.tt_ctx.getPathPredicateSize() > __len_pcs:
            self.__pcs_updated = True

        return ret


    def is_path_predicate_updated(self) -> bool:
        """ Return whether or not the path predicate has been updated """
        return self.__pcs_updated


    @property
    def last_branch_constraint(self) -> PathConstraint:
        """
        Return the last PathConstraint object added in the path predicate.
        Should be called after ``is_path_predicate_updated``.

        :raise: IndexError if the path predicate is empty
        :return:
        """
        return self.tt_ctx.getPathConstraints()[-1]


    @property
    def current_instruction(self) -> Optional[Instruction]:
        """
        Return the current Instruction.
        """
        return self.__current_inst


    def get_memory_string(self, addr: Addr) -> str:
        """ Returns a string from a memory address """
        s = ""
        index = 0
        while True:
            val = self.tt_ctx.getConcreteMemoryValue(addr + index)
            if not val:
                return s
            s += chr(val)
            index += 1


    def read_symbolic_register(self, register: Union[str, Register]) -> Expression:
        """
        Get the symbolic expression associated with the given register.

        :param register: register string, or Register object
        :return: Symbolic Expression of the register
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        sym_reg = self.tt_ctx.getSymbolicRegister(reg)
        if sym_reg is None:
            return self.tt_ctx.newSymbolicExpression(self.tt_ctx.getRegisterAst(reg))
        else:
            return sym_reg


    def write_symbolic_register(self, register: Union[str, Register], expr: Union[AstNode, Expression], comment = "") -> None:
        """
        Assign the given symbolic expression to the register. The given expression can either be a SMT AST node
        or directly an Expression (SymbolicExpression).

        :param register: register identifier (str or Register)
        :param expr: expression to assign (AstNode or Expression)
        :param comment: Comment to add on the symbolic expression
        :return: None
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        exp = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign {reg.getName()}: {comment}")
        self.write_register(reg, exp.getAst().evaluate())  # Update concrete state to keep sync
        self.tt_ctx.assignSymbolicExpressionToRegister(exp, reg)


    def read_symbolic_memory_int(self, addr: Addr, size: ByteSize) -> Expression:
        """
        Return a new Symbolic Expression representing the whole memory range given in parameter.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :raise: RuntimeError If the size if not aligned
        :return: Symbolic Expression associated with the memory
        """
        if size == 1:
            return self.read_symbolic_memory_byte(addr)
        elif size in [2, 4, 8, 16, 32, 64]:
            ast = self.tt_ctx.getMemoryAst(MemoryAccess(addr, size))
            return self.tt_ctx.newSymbolicExpression(ast)
        else:
            raise RuntimeError("size should be aligned [1, 2, 4, 8, 16, 32, 64] (bytes)")


    def read_symbolic_memory_byte(self, addr: Addr) -> Expression:
        """
        Thin wrapper to retrieve the symbolic expression of a single bytes in memory.

        :param addr: Memory address
        :return: Symbolic Expression associated with the memory
        """
        res = self.tt_ctx.getSymbolicMemory(addr)
        if res is None:
            return self.tt_ctx.newSymbolicExpression(self.tt_ctx.getMemoryAst(MemoryAccess(addr, 1)))
        else:
            return res

    def read_symbolic_memory_bytes(self, addr: Addr, size: ByteSize) -> Expression:
        """
        Return a new Symbolic Expression representing the whole memory range given in parameter.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :return: Symbolic Expression associated with the memory
        """
        if size == 1:
            return self.read_symbolic_memory_byte(addr)
        else:  # Need to create a per-byte expression with concat
            asts = [self.tt_ctx.getMemoryAst(MemoryAccess(addr+i, CPUSIZE.BYTE)) for i in range(size)]
            concat_expr = self.actx.concat(asts)
            return self.tt_ctx.newSymbolicExpression(concat_expr)


    def write_symbolic_memory_int(self, addr: Addr, size: ByteSize, expr: Union[AstNode, Expression]) -> None:
        """
        Assign the given symbolic expression representing an integer to the given address.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :param expr: Symbolic Expression or AST to assign
        :raise: RuntimeError If the size if not aligned
        :return: None
        """
        expr = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign memory")
        if size in [1, 2, 4, 8, 16, 32, 64]:
            self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, size), expr.getAst().evaluate())  # To keep the concrete state synchronized
            self.tt_ctx.assignSymbolicExpressionToMemory(expr, MemoryAccess(addr, size))
        else:
            raise RuntimeError("size should be aligned [1, 2, 4, 8, 16, 32, 64] (bytes)")


    def write_symbolic_memory_byte(self, addr: Addr, expr: Union[AstNode, Expression]) -> None:
        """
        Set a single bytes symbolic at the given address

        .. NOTE: We purposefully not provide a way to assign in memory a symbolic expression of
           arbitrary size as it would imply doing many extract on the given expression. For buffer
           you should do it in a per-byte manner with this method.

        :param addr: Memory address
        :param expr: Byte Expression or AST to assign
        :return: None
        """
        expr = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign memory")
        ast = expr.getAst()
        assert ast.getBitvectorSize() == 8
        self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, CPUSIZE.BYTE), ast.evaluate())  # Keep concrete state synced
        self.tt_ctx.assignSymbolicExpressionToMemory(expr, MemoryAccess(addr, CPUSIZE.BYTE))


    def is_memory_symbolic(self, addr: Addr, size: ByteSize) -> bool:
        """
        Iterate the symbolic memory and returns whether or not at least one byte of the buffer
        is symbolic

        :param addr: Memory address
        :param size: size of the memory range to check
        :return: True if at least one byte of the memory is symbolic, false otherwise
        """
        for i in range(addr, addr+size):
            if self.tt_ctx.isMemorySymbolized(MemoryAccess(i, 1)):
                return True
        return False


    def push_constraint(self, constraint: AstNode) -> None:
        """
        Thin wrapper on underneath triton context function to add a path
        constraint.

        :param constraint: Constraint expression to add
        :return: None
        """
        self.tt_ctx.pushPathConstraint(constraint)


    def concretize_register(self, register: Union[str, Register]) -> None:
        """
        Concretize the given register with the given value. If no value is provided the
        current register concrete value is used. This operation allows given the register
        a constant and staying sound!

        :param register: Register identifier (str or Register)
        :return: None
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        if self.tt_ctx.isRegisterSymbolized(reg):
            value = self.read_register(reg)
            self.push_constraint(self.read_symbolic_register(reg).getAst() == value)
        # Else do not even push the constraint


    def concretize_memory_int(self, addr: Addr, size: ByteSize) -> None:
        """
        Concretize the given memory with its current integer value. This operation is sound and
        allows restraining the memory value to its constant value.

        :param addr: Address to concretize
        :param size: Size of the integer to concretize
        :return: None
        """
        value = self.read_memory_int(addr, size)
        if self.tt_ctx.isMemorySymbolized(MemoryAccess(addr, size)):
            self.push_constraint(self.read_symbolic_memory_int(addr, size).getAst() == value)
        # else do not even push the constraint


    def concretize_memory_bytes(self, addr: Addr, size: ByteSize) -> None:
        """
        Concretize the given range of memory with its current value.

        :param addr: Address to concretize
        :param size: Size of the memory buffer to concretize
        :return: None
        """
        data = self.read_memory_bytes(addr, size)
        if self.is_memory_symbolic(addr, size):
            for i in range(size):
                self.push_constraint(self.read_symbolic_memory_byte(addr+i).getAst() == data[i])


    def concretize_argument(self, index: int) -> None:
        """
        Concretize the given function parameter according to the calling convention.
        This operation is sound !

        :param index: Argument index
        :return: None
        """
        try:
            self.concretize_register(self._get_argument_register(index))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            addr = self.cpu.stack_pointer + ((index-len_args) * self.ptr_size)  # Retrieve stack address
            self.concretize_memory_int(addr, self.ptr_size)                     # Concretize the value at this addr


    def get_argument_value(self, i: int) -> int:
        """
        Return the integer value of parameters following the call convention.
        Thus the value originate either from a register or the stack.

        :param i: Ith argument of the function
        :return: integer value of the parameter
        """
        try:
            return self.read_register(self._get_argument_register(i))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            return self.get_stack_value(i-len_args)

    def get_argument_symbolic(self, i: int) -> Expression:
        """
        Return the symbolic expression associated with the given ith parameter.

        :param i: Ith function parameter
        :return: Symbolic expression associated
        """
        try:
            return self.read_symbolic_register(self._get_argument_register(i))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            addr = self.cpu.stack_pointer + ((i-len_args) * self.ptr_size)
            return self.read_symbolic_memory_int(addr, self.ptr_size)


    def get_full_argument(self, i: int) -> Tuple[int, Expression]:
        """
        Get both the concrete argument value along with its symbolic expression.

        :return: Tuple containing concrete value and symbolic expression
        """
        return self.get_argument_value(i), self.get_argument_symbolic(i)


    def get_string_argument(self, i: int) -> str:
        """ Return the string on the given function parameter """
        return self.get_memory_string(self.get_argument_value(i))


    def get_format_string(self, addr: Addr) -> str:
        """ Returns a formatted string from a memory address """
        return self.get_memory_string(addr)                                             \
               .replace("%s", "{}").replace("%d", "{}").replace("%#02x", "{:#02x}")     \
               .replace("%#x", "{:#x}").replace("%x", "{:x}").replace("%02X", "{:02X}") \
               .replace("%c", "{:c}").replace("%02x", "{:02x}").replace("%ld", "{}")    \
               .replace("%*s", "").replace("%lX", "{:X}").replace("%08x", "{:08x}")     \
               .replace("%u", "{}").replace("%lu", "{}").replace("%zu", "{}")           \
               .replace("%02u", "{:02d}").replace("%03u", "{:03d}")                     \
               .replace("%03d", "{:03d}").replace("%p", "{:#x}").replace("%i", "{}")


    def get_format_arguments(self, s: str, args: list):
        """ Returns the formated arguments """
        s_str = self.get_memory_string(s)
        postString = [i for i, x in enumerate([i for i, c in enumerate(s_str) if c == '%']) if s_str[x+1] == "s"]
        for p in postString:
            args[p] = self.get_memory_string(args[p])
        return args


    def get_stack_value(self, index: int) -> int:
        """
        Returns the value at the index position of the stack
        :param index: The index position from the top of the stack
        :return: the value got
        """
        addr = self.cpu.stack_pointer + (index * self.ptr_size)
        return self.read_memory_int(addr, self.ptr_size)


    def pop_stack_value(self) -> int:
        """
        Pop a stack value
        :return: int
        """
        val = self.read_memory_ptr(self.cpu.stack_pointer)
        self.cpu.stack_pointer += self.ptr_size
        return val

    def push_stack_value(self, value: int) -> None:
        """
        Push a stack value

        :param value: The value to push
        :return: None
        """
        self.write_memory_ptr(self.cpu.stack_pointer-self.ptr_size, value)
        self.cpu.stack_pointer -= self.ptr_size

    def is_halt_instruction(self) -> bool:
        """ Return true if on halt instruction architecture independent (in theory) """
        halt_opc = self._archinfo.halt_inst
        return self.__current_inst.getType() == halt_opc
