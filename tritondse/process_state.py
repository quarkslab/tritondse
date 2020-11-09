# built-ins
import sys
import time
import logging
from typing import Union, Callable

# third-party
from triton import TritonContext, MemoryAccess, CALLBACK, CPUSIZE, Instruction

# local imports
from tritondse.thread_context import ThreadContext
from tritondse.config         import Config
from tritondse.program        import Program
from tritondse.heap_allocator import HeapAllocator
from tritondse.types          import Architecture, Addr, ByteSize, BitSize, PathConstraint, Register, Expression, AstNode


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

        # runtime temporary variables
        self.__pcs_updated = False


    def get_unique_thread_id(self):
        self.utid += 1
        return self.utid


    def get_unique_file_id(self):
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
        return ((1 << self.ptr_bit_size) - 1)


    def load_program(self, p: Program, base_addr: Addr = 0) -> None:
        """
        Load the given program in the process state memory
        :param p: Program to load in the process memory
        :param base_addr: Base address where to load the program (if PIE)
        :return: True on whether loading succeeded or not
        """
        # Set loading address
        self.load_addr = base_addr
        # TODO: If PIE use this address, if not set absolute address (from binary)

        # Load memory areas in memory
        for vaddr, data in p.memory_segments():
            logging.debug(f"Loading {vaddr:#08x} - {vaddr+len(data):#08x}")
            self.tt_ctx.setConcreteMemoryAreaValue(vaddr, data)


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


    def read_memory_bytes(self, addr: Addr, size: ByteSize) -> bytes:
        """
        Read in the process memory ``size`` amount of bytes at ``addr``.
        Data read is returned as bytes.

        :param addr: Address at which to read data
        :param size: Number of bytes to read
        :return: Data read
        """
        return self.tt_ctx.getConcreteMemoryAreaValue(addr, size)


    def write_memory(self, addr: Addr, size: ByteSize, data: Union[int, bytes]) -> None:
        """
        Write in the process memory the given data of the given size at
        a specific address.

        :param addr: address where to write data
        :param size: size of data to write in bytes
        :param data: data to write represented as an integer
        :return: None

        .. todo:: Adding a parameter to specify endianess if needed
        """
        assert type(data) is int or type(data) is bytes

        if type(data) is int:
            self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, size), data)
        else:
            self.tt_ctx.setConcreteMemoryAreaValue(addr, data)


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

        ret = self.tt_ctx.processing(instruction)

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
        return self.tt_ctx.getSymbolicRegister(reg)


    def write_symbolic_register(self, register: Union[str, Register], expr: Union[AstNode, Expression]) -> None:
        """
        Assign the given symbolic expression to the register. The given expression can either be a SMT AST node
        or directly an Expression (SymbolicExpression).

        :param register: register identifier (str or Register)
        :param expr: expression to assign (AstNode or Expression)
        :return: None
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        exp = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign {reg.getName()}")
        self.tt_ctx.assignSymbolicExpressionToRegister(exp, reg)


    def read_symbolic_memory_int(self, addr: Addr, size: ByteSize) -> Expression:
        """
        Return a new Symbolic Expression representing the whole memory range given in parameter.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :return: Symbolic Expression associated with the memory
        """
        if size == 1:
            return self.tt_ctx.getSymbolicMemory(addr)
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
        return self.tt_ctx.getSymbolicMemory(addr)


    def read_symbolic_memory_bytes(self, addr: Addr, size: ByteSize) -> Expression:
        """
        Return a new Symbolic Expression representing the whole memory range given in parameter.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :return: Symbolic Expression associated with the memory
        """
        if size == 1:
            return self.tt_ctx.getSymbolicMemory(addr)
        else:  # Need to create a per-byte expression with concat
            asts = [self.tt_ctx.getMemoryAst(MemoryAccess(addr+i, CPUSIZE.BYTE)) for i in range(size)]
            concat_expr = self.actx.concat(asts)
            return self.tt_ctx.newSymbolicExpression(concat_expr)


    def write_symbolic_memory_int(self, addr: Addr, size: ByteSize, expr: Expression) -> None:
        """
        Assign the given symbolic expression representing an integer to the given address.
        That function should not be used on big memory chunks.

        :param addr: Memory address
        :param size: memory size in bytes
        :param expr: Symbolic Expression to assign
        """
        if size in [1, 2, 4, 8, 16, 32, 64]:
            self.tt_ctx.assignSymbolicExpressionToMemory(expr, MemoryAccess(addr, size))
        else:
            raise RuntimeError("size should be aligned [1, 2, 4, 8, 16, 32, 64] (bytes)")


    def write_symbolic_memory_byte(self, addr: Addr, expr: Expression) -> None:
        """
        Set a single bytes symbolic at the given address

        .. NOTE: We purposefully not provide a way to assign in memory a symbolic expression of
           arbitrary size as it would imply doing many extract on the given expression. For buffer
           you should do it in a per-byte manner with this method.

        :param addr: Memory address
        :param expr: Byte expression to assign
        :return: None
        """
        ast = expr.getAst()
        assert ast.getBitVectorSize() == 8
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
            s = self.tt_ctx.getSymbolicMemory()
            if s.isSymbolized():
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


    def concretize_register(self, register: Union[str, Register], value: int = None) -> None:
        """
        Concretize the given register with the given value. If no value is provided the
        current register concrete value is used. This operation allows given the register
        a constant and staying sound!

        :param register: Register identifier (str or Register)
        :param value: Int
        :return: None
        """
        # FIXME: If a value is provided should we do ctx.concretizeRegister, setConcreteRegisterValue ?
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        value = self.read_register(reg) if value is None else value
        self.push_constraint(self.read_symbolic_register(reg).getAst() == value)


    def concretize_memory_int(self, addr: Addr, size: ByteSize, value: int = None) -> None:
        """
        Concretize the given memory with the given integer value. If no value is provided
        the current concrete value is used. This operation is sound and allows restraining
        the memory value.

        :param addr: Address to concretize
        :param size: Size of the integer to concretize
        :param value: Integer value to concretize
        :return: None
        """
        # FIXME: If a value is provided should we do ctx.concretizeMemory ? setConcreteRegisterValue ?
        value = self.read_memory_int(addr, size) if value is None else value
        self.push_constraint(self.read_symbolic_memory_int(addr, size) == value)


    def concretize_memory_bytes(self, addr: Addr, size: ByteSize, data: bytes = None) -> None:
        """
        Concretize the given range of memory with the given value. If no value is provided
        uses the current concrete value.

        :param addr: Address to concretize
        :param size: Size of the memory buffer to concretize
        :param data: Data to concretize
        :return: None
        """
        if data:
            assert len(data) == size
        data = self.read_memory_bytes(addr, size) if data is None else data
        for i in range(size):
            self.push_constraint(self.read_symbolic_memory_byte(addr+i) == data[i])
