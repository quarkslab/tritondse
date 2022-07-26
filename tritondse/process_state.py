# built-ins
from __future__ import annotations
import sys
import time
import logging
import re
from typing import Union, Callable, Tuple, Optional, List, Dict


# third-party
# import z3  # For direct value enumeration
from triton import TritonContext, MemoryAccess, CALLBACK, CPUSIZE, Instruction, MODE, AST_NODE, SOLVER, EXCEPTION

# local imports
from tritondse.thread_context import ThreadContext
from tritondse.program import Program
from tritondse.heap_allocator import HeapAllocator
from tritondse.types import Architecture, Addr, ByteSize, BitSize, PathConstraint, Register, \
        Expression, AstNode, Registers, SolverStatus, Model, SymbolicVariable, ArchMode, Perm, FileDesc
from tritondse.arch import ARCHS, CpuState
from tritondse.loader import Loader
from tritondse.memory import Memory, MemoryAccessViolation

class ProcessState(object):
    """
    Current process state. This class keeps all the runtime related to a running
    process, namely current, instruction, thread, memory maps, file descriptors etc.
    It also wraps Triton execution and thus hold its context. At the top of this,
    it provides a user-friendly API to access data in both the concrete and symbolic
    state of Triton.
    """
    def __init__(self, time_inc_coefficient: float = 0.0001):
        """
        :param time_inc_coefficient: Time coefficient to represent execution time of an instruction see: :py:attr:`tritondse.Config.time_inc_coefficient`
        """
        # EXTERN_BASE is a "fake" memory area (not mapped) that will
        # which addresses will be used for external symbols
        self.EXTERN_FUNC_BASE = 0x01000000  # Not really PLT but a dummy address space meant to contain some pointers to external symbols

        # Memory maps that will contains external symbols
        # used
        self.EXTERN_SYM_BASE = 0x01001000
        self.EXTERN_SYM_SIZE = 0x1000
        self.ERRNO_PTR = self.EXTERN_SYM_BASE+self.EXTERN_SYM_SIZE-4  # put errno at the end

        # This range will be dynamically allocated
        # upon request.
        self.BASE_HEAP = 0x10000000
        self.END_HEAP = 0x6fffffff

        # The stack size is 0x80000000 bytes
        self.BASE_STACK = 0xf0000000
        self.END_STACK  = 0x70000001 # This is inclusive

        # The Triton's context
        self.tt_ctx = TritonContext()
        self.actx = self.tt_ctx.getAstContext()

        # Cpu object wrapping registers values
        self.cpu: CpuState = None  #: CpuState holding concrete values of registers *(initialized when calling load)*
        self._archinfo = None

        # Memory object
        self.memory = Memory(self.tt_ctx)

        # Used to define that the process must exist
        self.stop = False

        # Signals table used by raise(), signal(), etc.
        self.signals_table = dict()

        # Dynamic symbols name -> addr (where they are mapped)
        self.dynamic_symbol_table = {}

        # File descriptors table used by fopen(), fprintf(), etc.
        self.fd_table = {
            0: FileDesc(name="stdin", offset=0, fd=sys.stdin),
            1: FileDesc("stdout", 0, sys.stdout),
            2: FileDesc("stderr", 0, sys.stderr),
        }
        # Unique file id incrementation
        self.fd_id = len(self.fd_table)

        # Allocation information used by malloc()
        self.heap_allocator = HeapAllocator(self.BASE_HEAP, self.END_HEAP, self.memory)

        # Unique thread id incrementation
        self._utid = 0

        # Current thread id
        self._tid = self._utid

        # Threads contexts
        self._threads = {
            self._tid: ThreadContext(self._tid)
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
        self.time_inc_coefficient = time_inc_coefficient

        # Runtime temporary variables
        self.__pcs_updated = False

        # The current instruction executed
        self.__current_inst = None

        # The memory mapping of the program ({vaddr_s : vaddr_e})
        self.__program_segments_mapping = {}

    @property
    def threads(self) -> List[ThreadContext]:
        """
        Gives a list of all threads currently active
        :return:
        """
        return list(self._threads.values())

    @property
    def current_thread(self) -> ThreadContext:
        """
        Gives the current thread selected
        :return: current thread
        :rtype: ThreadContext
        """
        return self._threads[self._tid]

    def switch_thread(self, thread: ThreadContext) -> bool:
        """
        Change the current thread to the one given in parameter.
        Thus save the current context, and restore the one of the
        thread given in parameter. It also resets the counter of
        the thread restored. If the current_thread is dead, it will
        also remove it !

        :param thread: thread to restore ThreadContext
        :return: True if the switch worked fine
        """
        assert (thread.tid in self._threads)

        try:
            if self.current_thread.is_dead():
                del self._threads[self._tid]
                # TODO: Finding all other threads joining it / (or locked by it ?) to unlock them
            else:  # Do a normal switch
                # Reset the counter and save its context
                self.current_thread.save(self.tt_ctx)

            # Schedule to the next thread
            thread.count = 0  # Reset the counter
            thread.restore(self.tt_ctx)
            self._tid = thread.tid
            return True

        except Exception as e:
            logging.error(f"Error while doing context switch: {e}")
            return False

    def spawn_new_thread(self, new_pc: Addr, args: Addr) -> ThreadContext:
        """
        Create a new thread in the process state. Parameters are the
        new program counter and a pointer to arguments to provide the thread.

        :param new_pc: new program counter (function to execution)
        :param args: arguments
        :return: thread context newly created
        """
        tid = self._get_unique_thread_id()
        thread = ThreadContext(tid)
        thread.save(self.tt_ctx)

        # Concretize pc, bp, sp, and first (argument)
        regs = [self.program_counter_register, self.stack_pointer_register, self.base_pointer_register, self._get_argument_register(0)]
        for reg in regs:
            if reg.getId() in thread.sregs:
                del thread.sregs[reg.getId()]

        thread.cregs[self.program_counter_register.getId()] = new_pc  # set new pc
        thread.cregs[self._get_argument_register(0).getId()] = args   # set args pointer
        thread.cregs[self.base_pointer_register.getId()] = (self.BASE_STACK - ((1 << 28) * tid))
        thread.cregs[self.stack_pointer_register.getId()] = (self.BASE_STACK - ((1 << 28) * tid))

        # Add the thread in the pool of threads
        self._threads[tid] = thread
        return thread

    def set_triton_mode(self, mode: MODE, value: int = True) -> None:
        """
        Set the given mode in the TritonContext.

        :param mode: mode to set in triton context
        :param value: value to set (default True)
        :return: None
        """
        self.tt_ctx.setMode(mode, value)

    def set_thumb(self, enable: bool) -> None:
        """
        Set thumb mode activated in the TritonContext. The mode will automatically
        be switched during execution, but at initialization this method enable
        activating it / disabling it. (Disabled be default)

        :param enable: bool: Wether or not to active thumb
        :return:
        """
        self.tt_ctx.setThumb(enable)

    def set_solver_timeout(self, timeout: int) -> None:
        """
        Set the timeout for all subsequent queries.

        :param timeout: timeout in milliseconds
        :return: None
        """
        self.tt_ctx.setSolverTimeout(timeout)

    def set_solver(self, solver: Union[str, SOLVER]) -> None:
        """
        Set the SMT solver to use in the background.

        :param solver: Solver to use
        :return: None
        """
        if isinstance(solver, str):
            solver = getattr(SOLVER, solver.upper(), SOLVER.Z3)
        self.tt_ctx.setSolver(solver)

    def _get_unique_thread_id(self) -> int:
        """ Return a new unique thread id. Used by thread related functions
        when spawning a new thread.

        :returns: new thread identifier
        """
        self._utid += 1
        return self._utid


    def get_unique_file_id(self) -> int:
        """ Return a new unique file descriptor. Used by routines
        yielding new file descriptors.

        :returns: new file descriptor identifier
        """
        self.fd_id += 1
        return self.fd_id


    @property
    def architecture(self) -> Architecture:
        """ Architecture of the current process state

        :rtype: Architecture
        """
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
        """ Size of a pointer in bytes

        :rtype: :py:obj:`tritondse.types.ByteSize`
        """
        return self.tt_ctx.getGprSize()


    @property
    def ptr_bit_size(self) -> BitSize:
        """ Size of a pointer in bits

        :rtype: :py:obj:`tritondse.types.BitSize`
        """
        return self.tt_ctx.getGprBitSize()


    @property
    def minus_one(self) -> int:
        """ Value -1 according to the architecture size (32 or 64 bits)

        :returns: -1 as an unsigned Python integer
        :rtype: int
        """
        return (1 << self.ptr_bit_size) - 1


    @property
    def registers(self) -> Registers:
        """ All registers according to the current architecture defined.
        The object returned is the TritonContext.register object.

        :rtype: :py:obj:`tritondse.types.Registers`
        """
        return self.tt_ctx.registers


    @property
    def return_register(self) -> Register:
        """ Return the appropriate return register according to the arch

        :rtype: :py:obj:`tritondse.types.Register`
        """
        return getattr(self.registers, self._archinfo.ret_reg)

    @property
    def program_counter_register(self) -> Register:
        """ Return the appropriate pc register according to the arch.

        :rtype: :py:obj:`tritondse.types.Register`
        """
        return getattr(self.registers, self._archinfo.pc_reg)

    @property
    def base_pointer_register(self) -> Register:
        """ Return the appropriate base pointer register according to the arch.

        :rtype: :py:obj:`tritondse.types.Register`
        """
        return getattr(self.registers, self._archinfo.bp_reg)

    @property
    def stack_pointer_register(self) -> Register:
        """ Return the appropriate stack pointer register according to the arch.

        :rtype: :py:obj:`tritondse.types.Register`
        """
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
        Initialize the context with the given architecture

        .. todo:: Protecting that function

        :param arch: The architecture to initialize
        :type arch: Architecture
        :return: None
        """
        self.architecture = arch
        self._archinfo = ARCHS[self.architecture]
        self.cpu = CpuState(self.tt_ctx, self._archinfo)
        self.write_register(self.base_pointer_register, self.BASE_STACK)
        self.write_register(self.stack_pointer_register, self.BASE_STACK)


    def load(self, l: Loader, base_addr: Addr = 0) -> None:
        """
        Load the given program in the process state memory. It sets
        the program counter to the entry point and load all segments
        in the triton context.

        :param l: Loader to use for mapping binary (can by Program, Monolithic etc..)
        :type l: Loader
        :param base_addr: Base address where to load the program (if PIE)
        :type base_addr: :py:obj:`tritondse.types.Addr`
        """
        # Set the program counter to points to entrypoint
        self.cpu.program_counter = l.entry_point

        # Set loading address
        self.load_addr = base_addr
        # TODO: If PIE use this address, if not set absolute address (from binary)

        # Map the stack
        self.memory.map(self.END_STACK, self.BASE_STACK-self.END_STACK+1, Perm.R | Perm.W, "stack")

        self.memory.disable_segmentation()  # Disable segment to be able initializing Read-only segments

        # Load memory areas in memory
        for i, seg in enumerate(l.memory_segments()):
            logging.debug(f"Loading 0x{seg.address:#08x} - {seg.address+len(seg.content):#08x}")
            self.memory.map(seg.address, len(seg.content), seg.perms, f"seg{i}")
            self.memory.write(seg.address, seg.content)

        self.memory.enable_segmentation()

    def read_register(self, register: Union[str, Register]) -> int:
        """
        Read the current concrete value of the given register.

        :param register: string of the register or Register object
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :return: Integer value
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        return self.tt_ctx.getConcreteRegisterValue(reg)


    def write_register(self, register: Union[str, Register], value: int) -> None:
        """
        Read the current concrete value of the given register.

        :param register: string of the register or Register object
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :param value: integer value to assign in the register
        :type value: int
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        return self.tt_ctx.setConcreteRegisterValue(reg, value)


    def register_triton_callback(self, cb_type: CALLBACK, callback: Callable) -> None:
        """
        Register the given ``callback`` as triton callback to hook memory/registers
        read/writes.

        :param cb_type: Callback enum type as defined by Triton
        :type cb_type: `CALLBACK <https://triton.quarkslab.com/documentation/doxygen/py_CALLBACK_page.html>`_
        :param callback: routines to call on the given event
        """
        self.tt_ctx.addCallback(cb_type, callback)


    def clear_triton_callbacks(self) -> None:
        """
        Remove all registered callbacks in triton.
        """
        self.tt_ctx.clearCallbacks()


    def is_heap_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is pointing in the heap area.

        :param ptr: Address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :return: True if pointer points to the heap area *(allocated or not)*.
        """
        if self.BASE_HEAP <= ptr < self.END_HEAP:
            return True
        return False


    def is_stack_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is pointing in stack area.

        :param ptr: Address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :return: True if pointer points to the stack area (allocated or not).
        """
        return self.END_STACK <= ptr <= self.BASE_STACK

    def fetch_instruction(self, address: Addr = None) -> Instruction:
        """
        Fetch the instruction at the given address. If no address
        is specified the current program counter one is used.

        :raise MemoryAccessViolation: If the instruction cannot be fetched in the memory.

        :param address: address where to get the instruction from
        :return: instruction disassembled
        """
        if address is None:
            address = self.cpu.program_counter
        self.memory.disable_segmentation()
        data = self.memory.read(address, 16)
        self.memory.enable_segmentation()
        i = Instruction(address, data)
        i.setThreadId(self.current_thread.tid)
        self.tt_ctx.disassembly(i)  # This needs to be done before using i.getSize()
                                    # otherwise, i.getSize() will always be 16
        map = self.memory.get_map(address, i.getSize())
        if map is None:
            raise MemoryAccessViolation(address, Perm.X, memory_not_mapped=True)
        if Perm.X not in map.perm:  # Note: in this model we can execute code in non-readable pages
            raise MemoryAccessViolation(address, Perm.X, map_perm=map.perm, perm_error=True)
        return i

    def process_instruction(self, instruction: Instruction) -> bool:
        """
        Process the given triton instruction on this process state.

        :param instruction: Triton Instruction object
        :type instruction: `Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_
        :return: True if the processing of the instruction succeeded (False otherwise)
        """
        self.__pcs_updated = False
        __len_pcs = self.tt_ctx.getPathPredicateSize()

        if not instruction.getDisassembly():  # If the insrtuction has not been disassembled
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

        return ret == EXCEPTION.NO_FAULT

    @property
    def path_predicate_size(self) -> int:
        """
        Get the size of the path predicate (conjonction
        of all branches and additionnals constraints added)

        :return: size of the predicate
        """
        return self.tt_ctx.getPathPredicateSize()

    def is_path_predicate_updated(self) -> bool:
        """ Return whether or not the path predicate has been updated """
        return self.__pcs_updated


    @property
    def last_branch_constraint(self) -> PathConstraint:
        """
        Return the last PathConstraint object added in the path predicate.
        Should be called after :py:meth:`is_path_predicate_updated`.

        :raise IndexError: if the path predicate is empty
        :return: the path constraint object as returned by Triton
        :rtype: `PathConstraint <https://triton.quarkslab.com/documentation/doxygen/py_PathConstraint_page.html>`_
        """
        return self.tt_ctx.getPathConstraints()[-1]


    @property
    def current_instruction(self) -> Optional[Instruction]:
        """
        The current instruction being executed. *(None if not set yet)*

        :rtype: Optional[`Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_]
        """
        return self.__current_inst


    def is_register_symbolic(self, register: Union[str, Register]) -> bool:
        """
        Check whether or not the register is symbolic.

        :param register: register string, or Register object
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :return: True if the register is symbolic
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        return self.tt_ctx.getRegisterAst(reg).isSymbolized()

    def read_symbolic_register(self, register: Union[str, Register]) -> Expression:
        """
        Get the symbolic expression associated with the given register.

        :param register: register string, or Register object
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :return: SymbolicExpression of the register as returned by Triton
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str transform to reg
        sym_reg = self.tt_ctx.getSymbolicRegister(reg)

        if sym_reg is None or reg.getBitSize() != sym_reg.getAst().getBitvectorSize():
            return self.tt_ctx.newSymbolicExpression(self.tt_ctx.getRegisterAst(reg))
        else:
            return sym_reg


    def write_symbolic_register(self, register: Union[str, Register], expr: Union[AstNode, Expression], comment: str = "") -> None:
        """
        Assign the given symbolic expression to the register. The given expression can either be a SMT AST node
        or directly an Expression (SymbolicExpression).

        :param register: register identifier (str or Register)
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :param expr: expression to assign (`AstNode <https://triton.quarkslab.com/documentation/doxygen/py_AstNode_page.html>`_
               or `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_)
        :param comment: Comment to add on the symbolic expression created
        :type comment: str
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: memory size in bytes
        :type size: :py:obj:`tritondse.types.ByteSize`
        :raise RuntimeError: If the size if not aligned
        :return: Symbolic Expression associated with the memory
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :return: Symbolic Expression associated with the memory
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: memory size in bytes
        :type size: :py:obj:`tritondse.types.ByteSize`
        :return: Symbolic Expression associated with the memory
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: memory size in bytes
        :type size: :py:obj:`tritondse.types.ByteSize`
        :param expr: expression to assign (`AstNode <https://triton.quarkslab.com/documentation/doxygen/py_AstNode_page.html>`_
                     or `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_)
        :raise RuntimeError: if the size if not aligned
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :param expr: byte expression to assign (`AstNode <https://triton.quarkslab.com/documentation/doxygen/py_AstNode_page.html>`_
                     or `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_)
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
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: size of the memory range to check
        :type size: :py:obj:`tritondse.types.ByteSize`
        :return: True if at least one byte of the memory is symbolic, false otherwise
        """
        for i in range(addr, addr+size):
            if self.tt_ctx.isMemorySymbolized(MemoryAccess(i, 1)):
                return True
        return False


    def push_constraint(self, constraint: AstNode, comment: str = "") -> None:
        """
        Thin wrapper on the triton context underneath to add a path constraint.

        :param constraint: Constraint expression to add
        :type constraint: `AstNode <https://triton.quarkslab.com/documentation/doxygen/py_AstNode_page.html>`_
        :param comment: String comment to attach to the constraint
        :type comment: str
        """
        self.tt_ctx.pushPathConstraint(constraint, comment)

    def get_path_constraints(self) -> List[PathConstraint]:
        """
        Get the list of all path constraints set in the Triton context.

        :return: list of constraints
        """
        return self.tt_ctx.getPathConstraints()

    def concretize_register(self, register: Union[str, Register]) -> None:
        """
        Concretize the given register with its current concrete value.
        **This operation is sound** as it will also add a path constraint
        to enforce that the symbolic register value is equal to its concrete
        value.

        :param register: Register identifier (str or Register)
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        if self.tt_ctx.isRegisterSymbolized(reg):
            value = self.read_register(reg)
            self.push_constraint(self.read_symbolic_register(reg).getAst() == value)
        # Else do not even push the constraint


    def concretize_memory_bytes(self, addr: Addr, size: ByteSize) -> None:
        """
        Concretize the given memory with its current concrete value.
        **This operation is sound** and allows restraining the memory
        value to its constant value.

        :param addr: Address to concretize
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: Size of the integer to concretize
        :type size: :py:obj:`tritondse.types.ByteSize`
        """
        data = self.memory.read(addr, size)
        if self.is_memory_symbolic(addr, size):
            self.push_constraint(self.read_symbolic_memory_bytes(addr, size).getAst() == data)
        # else do not even push the constraint


    def concretize_memory_int(self, addr: Addr, size: ByteSize) -> None:
        """
        Concretize the given memory with its current concrete value.
        **This operation is sound** and allows restraining the memory
        value to its constant value.

        :param addr: Address to concretize
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: Size of the integer to concretize
        :type size: :py:obj:`tritondse.types.ByteSize`
        """
        value = self.memory.read_uint(addr, size)
        if self.tt_ctx.isMemorySymbolized(MemoryAccess(addr, size)):
            self.push_constraint(self.read_symbolic_memory_int(addr, size).getAst() == value)
        # else do not even push the constraint


    def concretize_argument(self, index: int) -> None:
        """
        Concretize the given function parameter following the calling convention
        of the architecture.

        :param index: Argument index
        :type index: int
        """
        try:
            self.concretize_register(self._get_argument_register(index))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            addr = self.cpu.stack_pointer + self.ptr_size + ((index-len_args) * self.ptr_size)  # Retrieve stack address
            self.concretize_memory_int(addr, self.ptr_size)                     # Concretize the value at this addr


    def write_argument_value(self, i: int, val: int) -> None:
        """
        Write the parameter index with the given value. It will take in account
        whether the argument is in a register or the stack.

        :param i: Ith argument of the function
        :param val: integer value of the parameter
        :return: None
        """
        try:
            return self.write_register(self._get_argument_register(i), val)
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            return self.write_stack_value(i-len_args, val, offset=1)


    def get_argument_value(self, i: int) -> int:
        """
        Get the integer value of parameters following the call convention.
        The value originate either from a register or the stack depending
        on the ith argument requested.

        :param i: Ith argument of the function
        :type i: int
        :return: integer value of the parameter
        :rtype: int
        """
        try:
            return self.read_register(self._get_argument_register(i))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            return self.get_stack_value(i-len_args, offset=1)

    def get_argument_symbolic(self, i: int) -> Expression:
        """
        Return the symbolic expression associated with the given ith parameter.

        :param i: Ith function parameter
        :return: Symbolic expression associated
        :rtype: `SymbolicExpression <https://triton.quarkslab.com/documentation/doxygen/py_SymbolicExpression_page.html>`_
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


    def get_string_argument(self, idx: int) -> str:
        """Read a string for which address is a function parameter.
        The function first get the argument value, and then dereference
        the string located at that address.

        :param idx: argument index
        :type idx: int
        :returns: memory string
        :rtype: str
        """
        return self.memory.read_string(self.get_argument_value(idx))


    def get_format_string(self, addr: Addr) -> str:
        """
        Returns a formatted string in Python format from a format string
        located in memory at ``addr``.

        :param addr: Address to concretize
        :type addr: :py:obj:`tritondse.types.Addr`
        :rtype: str
        """
        return self.memory.read_string(addr)                                             \
               .replace("%s", "{}").replace("%d", "{}").replace("%#02x", "{:#02x}")     \
               .replace("%#x", "{:#x}").replace("%x", "{:x}").replace("%02X", "{:02X}") \
               .replace("%c", "{:c}").replace("%02x", "{:02x}").replace("%ld", "{}")    \
               .replace("%*s", "").replace("%lX", "{:X}").replace("%08x", "{:08x}")     \
               .replace("%u", "{}").replace("%lu", "{}").replace("%zu", "{}")           \
               .replace("%02u", "{:02d}").replace("%03u", "{:03d}")                     \
               .replace("%03d", "{:03d}").replace("%p", "{:#x}").replace("%i", "{}")


    def get_format_arguments(self, fmt_addr: Addr, args: List[int]) -> List[Union[int, str]]:
        """
        Read the format string at ``fmt_addr``. For each format item
        which are strings, dereference that associated string and replaces it
        in ``args``.

        :param fmt_addr: Address to concretize
        :type fmt_addr: :py:obj:`tritondse.types.Addr`
        :param args: Parameters associated with the format string
        :type args: List[int]
        :rtype: List[Union[int, str]]
        """
        # FIXME: Modifies inplace args (which is not very nice)
        s_str = self.memory.read_string(fmt_addr)
        postString = [i for i, x in enumerate([i for i, c in enumerate(s_str) if c == '%']) if s_str[x+1] == "s"]
        for p in postString:
            args[p] = self.memory.read_string(args[p])
        return args


    def get_stack_value(self, index: int, offset: int = 0) -> int:
        """
        Returns the value at the ith position further in the stack

        :param index: The index position from the top of the stack
        :type index: int
        :param offset: An integer value offset to apply to stack address
        :type offset: int
        :return: the value got
        :return: the value got
        :rtype: int
        """
        addr = self.cpu.stack_pointer + (offset * self.ptr_size) + (index * self.ptr_size)
        return self.memory.read_uint(addr, self.ptr_size)


    def write_stack_value(self, index: int, value: int, offset: int = 0) -> None:
        """
        Write the given value on the stack at the given index relative to the current
        stack pointer. The index value can be positive to write further down the stack
        or negative to write upward.

        :param index: The index position from the top of the stack
        :type index: int
        :param value: Integer value to write on the stack
        :type value: int
        :param offset: Add an optional ith item offset to add to stack value (not a size)
        :type offset: int
        :return: the value got
        :rtype: int
        """
        addr = self.cpu.stack_pointer + (offset * self.ptr_size) + (index * self.ptr_size)
        self.memory.write_int(addr, value, self.ptr_size)


    def pop_stack_value(self) -> int:
        """
        Pop a stack value, and the re-increment the stack pointer value.
        This operation is fully concrete.

        :return: int
        """
        val = self.memory.read_ptr(self.cpu.stack_pointer)
        self.cpu.stack_pointer += self.ptr_size
        return val

    def push_stack_value(self, value: int) -> None:
        """
        Push a stack value. It then decreement the stack pointer value.

        :param value: The value to push
        """
        self.memory.write_ptr(self.cpu.stack_pointer-self.ptr_size, value)
        self.cpu.stack_pointer -= self.ptr_size

    def is_halt_instruction(self) -> bool:
        """
        Check if the the current instruction is corresponding to an 'halt' instruction
        in the target architecture.

        :returns: Return true if on halt instruction architecture independent
        """
        halt_opc = self._archinfo.halt_inst
        return self.__current_inst.getType() == halt_opc

    def solve(self, constraint: Union[AstNode, List[AstNode]], with_pp: bool = True) -> Tuple[SolverStatus, Model]:
        """
        Solve the given constraint one the current symbolic state and returns both
        a Solver status and a model. If not SAT the model returned is empty. Argument
        ``with_pp`` enables checking the constraint taking in account the path predicate.

        :param constraint: AstNode or list of AstNodes constraints to solve
        :param with_pp: whether to take in account path predicate
        :return: tuple of status and model
        """
        if with_pp:
            cst = constraint if isinstance(constraint, list) else [constraint]
            final_cst = self.actx.land([self.tt_ctx.getPathPredicate()]+cst)
        else:
            final_cst = self.actx.land(constraint) if isinstance(constraint, list) else constraint

        model, status, _ = self.tt_ctx.getModel(final_cst, status=True)
        return SolverStatus(status), model

    def solve_no_pp(self, constraint: Union[AstNode, List[AstNode]]) -> Tuple[SolverStatus, Model]:
        """
        Helper function that solve a constraint forcing not to use
        the path predicate.

        .. warning:: Solving a query without the path predicate gives theoretically
                     unsound results.

        :param constraint: AstNode constraint to solve
        :return: tuple of status and model
        """
        return self.solve(constraint, with_pp=False)

    def symbolize_register(self, register: Union[str, Register], alias: str = None) -> SymbolicVariable:
        """
        Symbolize the given register. This a proxy for the symbolizeRegister
        Triton function.

        :param register: string of the register or Register object
        :type register: Union[str, :py:obj:`tritondse.types.Register`]
        :param alias: alias name to give to the symbolic variable
        :type alias: str
        :return: Triton Symbolic variable created
        """
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  # if str get reg
        if alias:
            var = self.tt_ctx.symbolizeRegister(reg, alias)
        else:
            var = self.tt_ctx.symbolizeRegister(reg)
        return var

    def symbolize_memory_byte(self, addr: Addr, alias: str = None) -> SymbolicVariable:
        """
        Symbolize the given memory cell. Returns the associated
        SymbolicVariable

        :param addr: Address to symbolize
        :type addr: :py:obj:`tritondse.types.Addr`
        :param alias: alias to give the variable
        :return: newly created symbolic variable
        :rtype: :py:obj:`tritondse.types.SymbolicVariable`
        """
        if alias:
            return self.tt_ctx.symbolizeMemory(MemoryAccess(addr, CPUSIZE.BYTE), alias)
        else:
            return self.tt_ctx.symbolizeMemory(MemoryAccess(addr, CPUSIZE.BYTE))

    def symbolize_memory_bytes(self, addr: Addr, size: ByteSize, alias_prefix: str = None, offset: int = 0) -> List[SymbolicVariable]:
        """
        Symbolize a range of memory addresses. Can optionally provide an alias
        prefix.

        :param addr: Address at which to read data
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: Number of bytes to symbolize
        :type size: :py:obj:`tritondse.types.ByteSize`
        :param alias_prefix: prefix name to give the variable
        :type alias_prefix: str
        :return: list of Symbolic variables created
        :rtype: List[:py:obj:`tritondse.types.SymbolicVariable`]
        """
        if alias_prefix:
            return [self.symbolize_memory_byte(addr+i, alias_prefix+f"[{i+offset}]") for i in range(size)]
        else:
            return [self.symbolize_memory_byte(addr+i) for i in range(size)]

    def get_expression_variable_values_model(self, exp: Union[AstNode, Expression], model: Model) -> Dict[SymbolicVariable: int]:
        """
        Given a symbolic expression and a model, returns the valuation
        of all variables involved in the expression.

        :param exp: Symbolic Expression to look into
        :param model: Model generated by the solver
        :return: dictionnary of symbolic variables and their associated value (as int)
        """
        ast = exp.getAst() if hasattr(exp, "getAst") else exp
        ast_vars = self.actx.search(ast, AST_NODE.VARIABLE)
        sym_vars = [x.getSymbolicVariable() for x in ast_vars]
        final_dict = {}
        for avar, svar in zip(ast_vars, sym_vars):
            if svar.getId() in model:
                final_dict[svar] = model[svar.getId()].getValue()
            else:
                final_dict[svar] = avar.evaluate()
        return final_dict

    def evaluate_expression_model(self, exp: Union[AstNode, Expression], model: Model) -> int:
        """
        Evaluate the given expression on the given model. The value returned is the
        integer value corresponding to the bitvector evaluation of the expression.

        :param exp: Symbolic Expression to evaluate
        :param model: Model generated by the solver
        :return: result of the evaluation
        """
        ast = exp.getAst() if hasattr(exp, "getAst") else exp

        vars = self.get_expression_variable_values_model(ast, model)

        backup = {}
        for var, value in vars.items():
            backup[var] = self.tt_ctx.getConcreteVariableValue(var)
            self.tt_ctx.setConcreteVariableValue(var, value)
        final_value = ast.evaluate()
        for var in vars.keys():
            self.tt_ctx.setConcreteVariableValue(var, backup[var])
        return final_value

    # def enumerate_expression_value(self, exp: Union[AstNode, Expression], constraints: List[AstNode], values_blacklist: List[int], limit: int):
    #     # Written for when it will work
    #     solver = z3.SolverFor("QF_BV")
    #     ast = exp.getAst() if hasattr(exp, "getAst") else exp
    #     z3ast = self.actx.tritonToZ3(ast)
    #
    #     solver.add([self.actx.tritonToZ3(x) for x in constraints])
    #     solver.add([z3ast != x for x in values_blacklist])
    #
    #     values = []  # retrieved values
    #
    #     while limit:
    #         res = solver.check()
    #         if res == z3.sat:
    #             model = solver.model()
    #             new_val = model.eval(z3ast)
    #             values.append(new_val)
    #             solver.add(z3ast != new_val)
    #         else:
    #             return values
    #         limit -= 1
    #     return values

    def solve_enumerate_expression(self, exp: Union[AstNode, Expression], constraints: List[AstNode], values_blacklist: List[int], limit: int) -> List[Tuple[Model, int]]:
        # Written for when it will work
        ast = exp.getAst() if hasattr(exp, "getAst") else exp

        constraint = self.actx.land(constraints + [ast != x for x in values_blacklist])

        result = []
        while limit:
            status, model = self.solve(constraint, with_pp=False)
            if status == SolverStatus.SAT:
                new_val = self.evaluate_expression_model(ast, model)
                result.append((model, new_val))
                constraint = self.actx.land([constraint, ast != new_val])
            else:
                return result
            limit -= 1
        return result

    @staticmethod
    def from_loader(loader: Loader) -> 'ProcessState':
        pstate = ProcessState()

        # Initialize the architecture of the processstate
        pstate.initialize_context(loader.architecture)

        # Load segments of the loader
        pstate.load(loader)

        # Apply dynamic relocations
        cur_linkage_address = pstate.EXTERN_FUNC_BASE

        # Disable segmentation
        pstate.memory.disable_segmentation()

        # Link imported functions in EXTERN_FUNC_BASE
        for fname, rel_addr in loader.imported_functions_relocations():
            logging.debug(f"Hooking {fname} at {rel_addr:#x}")

            # Add symbol in dynamic_symbol_table
            pstate.dynamic_symbol_table[fname] = (cur_linkage_address, True)

            # Apply relocation to our custom address in process memory
            pstate.memory.write_ptr(rel_addr, cur_linkage_address)

            # Increment linkage address number
            cur_linkage_address += pstate.ptr_size

        # Reenable segmentation after having performed relocations
        pstate.memory.enable_segmentation()

        # Allocate data for foreign symbols region
        pstate.memory.map(pstate.EXTERN_SYM_BASE, pstate.EXTERN_SYM_SIZE, Perm.R | Perm.W, "extern_syms")
        symb_base = pstate.EXTERN_SYM_BASE

        # Link imported symbols
        for sname, rel_addr in loader.imported_variable_symbols_relocations():
            logging.debug(f"Hooking {sname} at {rel_addr:#x}")

            if pstate.architecture == Architecture.X86_64:  # HACK: Keep rel_addr to directly write symbol on it
                # Add symbol in dynamic_symbol_table
                pstate.dynamic_symbol_table[sname] = (rel_addr, False)
                #pstate.memory.write_ptr(rel_addr, cur_linkage_address)  # Do not write anything as symbolic executor will do it
            else:
                # Add symbol in dynamic_symbol_table
                pstate.dynamic_symbol_table[sname] = (symb_base, False)
                pstate.memory.write_ptr(rel_addr, symb_base)

            symb_base += pstate.ptr_size


        for reg_name in pstate.cpu:
            if reg_name in loader.cpustate:
                setattr(pstate.cpu, reg_name, loader.cpustate[reg_name])

        if loader.arch_mode: # If the processor's mode is provided
            if loader.arch_mode == ArchMode.THUMB:
                pstate.set_thumb(True)
        return pstate
