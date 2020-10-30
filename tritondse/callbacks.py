# built-in imports
from enum   import Enum, auto
from typing import Callable, Tuple, List, Optional
from copy   import deepcopy

# third-party imports
from triton import CALLBACK, Instruction, MemoryAccess

# local imports
from tritondse.process_state  import ProcessState
from tritondse.program        import Program
from tritondse.types          import Addr, Input, Register
from tritondse.thread_context import ThreadContext


class CbPos(Enum):
    BEFORE = auto()
    AFTER = auto()


class CbType(Enum):
    CTX_SWITCH = auto()
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    PORT_RTN = auto()
    POST_ADDR = auto()
    POST_EXEC = auto()
    POST_INST = auto()
    PRE_ADDR = auto()
    PRE_EXEC = auto()
    PRE_INST = auto()
    PRE_RTN = auto()
    REG_READ = auto()
    REG_WRITE = auto()
    SMT_MODEL = auto()


AddrCallback     = Callable[['SymbolicExecutor', ProcessState, Addr], None]
InstrCallback    = Callable[['SymbolicExecutor', ProcessState, Instruction], None]
MemReadCallback  = Callable[['SymbolicExecutor', ProcessState, MemoryAccess], None]
MemWriteCallback = Callable[['SymbolicExecutor', ProcessState, MemoryAccess, int], None]
NewInputCallback = Callable[['SymbolicExecutor', ProcessState, Input], Optional[Input]]
RegReadCallback  = Callable[['SymbolicExecutor', ProcessState, Register], None]
RegWriteCallback = Callable[['SymbolicExecutor', ProcessState, Register, int], None]
RtnCallback      = Callable[['SymbolicExecutor', ProcessState, str, Addr], None]
SymExCallback    = Callable[['SymbolicExecutor', ProcessState], None]
ThreadCallback   = Callable[['SymbolicExecutor', ProcessState, ThreadContext], None]


class ProbeInterface(object):
    """ The Probe interface """
    def __init__(self):
        self.cbs = [] # [(CbType, arg, callback)]


class CallbackManager(object):
    """
    Class used to aggregate all callbacks that can be plugged
    inside a SymbolicExecutor running session. The internal
    structure ensure that check the presence of callback can
    be made in Log(N). All callbacks are designed to be read-only
    """

    def __init__(self, p: Program):
        self.p = p
        self._se = None

        # SymbolicExecutor callbacks
        self._pc_addr_cbs   = {}  # addresses reached
        self._instr_cbs     = {CbPos.BEFORE: [], CbPos.AFTER: []}  # all instructions
        self._pre_exec      = []  # before execution
        self._post_exec     = []  # after execution
        self._ctx_switch    = []  # on each thread context switch (implementing pre/post?)
        self._new_input_cbs = []  # each time an SMT model is get
        self._pre_rtn_cbs   = {}  # before imported routine calls ({str: [RtnCallback]})
        self._post_rtn_cbs  = {}  # after imported routine calls ({str: [RtnCallback]})

        # Triton callbacks
        self._mem_read_cbs  = []  # memory reads
        self._mem_write_cbs = []  # memory writes
        self._reg_read_cbs  = []  # register reads
        self._reg_write_cbs = []  # register writes
        self._empty         = True


    def is_empty(self) -> bool:
        """
        Check whether a callback as alreday been registered or not

        :return: True if no callback were registered
        """
        return self._empty


    def is_binded(self) -> bool:
        """
        Check if the callback manager has already been binded on a given process state.
        :return: True if callbacks are binded on a process state
        """
        return bool(self._se) # and self._se.uid == se.uid)


    def _trampoline_mem_read_cb(self, ctx, mem):
        """
        This function is the trampoline callback on memory read from triton to tritondse
        :param: ctx: TritonContext
        :param: mem: MemoryAccess
        :return: None
        """
        for cb in self._mem_read_cbs:
            cb(self._se, self._se.pstate, mem)


    def _trampoline_mem_write_cb(self, ctx, mem, value):
        """
        This function is the trampoline callback on memory write from triton to tritondse
        :param: ctx: TritonContext
        :param: mem: MemoryAccess
        :param: value: int
        :return: None
        """
        for cb in self._mem_write_cbs:
            cb(self._se, self._se.pstate, mem, value)


    def _trampoline_reg_read_cb(self, ctx, reg):
        """
        This function is the trampoline callback on register read from triton to tritondse
        :param: ctx: TritonContext
        :param: reg: Register
        :return: None
        """
        for cb in self._reg_read_cbs:
            cb(self._se, self._se.pstate, reg)


    def _trampoline_reg_write_cb(self, ctx, reg, value):
        """
        This function is the trampoline callback on register write from triton to tritondse
        :param: ctx: TritonContext
        :param: reg: Register
        :param: value: int
        :return: None
        """
        for cb in self._reg_write_cbs:
            cb(self._se, self._se.pstate, reg, value)


    def bind_to(self, se: 'SymbolicExecutor') -> None:
        """
        Bind callbacks on the given process state. That step is required
        to register callbacks on the Triton Context object. This is also
        used to keep a reference on the SymbolicExecutor object;

        :param se: SymbolicExecutor on which to bind callbacks
        :return: None
        """
        assert not self.is_binded()

        self._se = se

        # Register only one trampoline by kind of callback. It will be the role
        # of the trampoline to call every registered tritondse callbacks.

        if self._mem_read_cbs:
            se.pstate.register_triton_callback(CALLBACK.GET_CONCRETE_MEMORY_VALUE, self._trampoline_mem_read_cb)

        if self._mem_write_cbs:
            se.pstate.register_triton_callback(CALLBACK.SET_CONCRETE_MEMORY_VALUE, self._trampoline_mem_write_cb)

        if self._reg_read_cbs:
            se.pstate.register_triton_callback(CALLBACK.GET_CONCRETE_REGISTER_VALUE, self._trampoline_reg_read_cb)

        if self._reg_write_cbs:
            se.pstate.register_triton_callback(CALLBACK.SET_CONCRETE_REGISTER_VALUE, self._trampoline_reg_write_cb)

        self.rebase_callbacks(self._se.pstate.load_addr)


    def rebase_callbacks(self, addr: Addr) -> None:
        """
        All addresses registered are relative addresse (when PIE). Thus this function will rebase
        all address the currrent address which is meant to represent the loading address

        :param addr:
        :return:
        """
        assert addr == 0
        # TODO: To implement (if PiE rebase otherwise do nothing)


    def register_addr_callback(self, pos: CbPos, addr: Addr, callback: AddrCallback) -> None:
        """
        Register a callback function on a given address before or after the execution
        of the associated instruction.
        :param pos: When to trigger the callback (before or after) execution of the instruction
        :param addr: Address where to trigger the callback
        :param callback: callback function
        :return: None
        """
        if addr not in self._pc_addr_cbs:
            self._pc_addr_cbs[addr] = {CbPos.BEFORE: [], CbPos.AFTER:[]}

        self._pc_addr_cbs[addr][pos].append(callback)
        self._empty = False


    def register_pre_addr_callback(self, addr: Addr, callback: AddrCallback) -> None:
        """
        Register pre address callback
        :param addr: Address where to trigger
        :param callback: Callback function to call
        :return: None
        """
        self.register_addr_callback(CbPos.BEFORE, addr, callback)


    def register_post_addr_callback(self, addr: Addr, callback: AddrCallback) -> None:
        """
        Register post-address callback
        :param addr: Address where to trigger callback
        :param callback: Callback function
        :return: None
        """
        self.register_addr_callback(CbPos.AFTER, addr, callback)


    def get_address_callbacks(self, addr: Addr) -> Tuple[List[AddrCallback], List[AddrCallback]]:
        """
        Get all the pre/post callbacks for a given address.
        :param addr: Address for which to retrieve callbacks
        :return: tuple of lists containing callback functions for pre/post respectively
        """
        cbs = self._pc_addr_cbs.get(addr, None)
        if cbs:
            return cbs[CbPos.BEFORE], cbs[CbPos.AFTER]
        else:
            return [], []


    def register_function_callback(self, func_name: str, callback: AddrCallback) -> bool:
        """
        Register a callback on the address of the given function name.
        The address of the function is resolved through lief. Thus finding
        the function is conditioned by LIEF.
        :param func_name: Function
        :param callback: Callback to be called
        :return: True if registeration succeeded, False otherwise
        """
        f = self.p.find_function(func_name)
        if f:
            self.register_pre_addr_callback(f.address, callback)
            return True
        else:
            return False


    def register_instruction_callback(self, pos: CbPos, callback: InstrCallback) -> None:
        """
        Register a callback triggered on each instruction executed, before or after its
        side effects have been applied to ProcessState.
        :param pos: Before/After execution of the instruction
        :param callback: callback function to trigger
        :return: None
        """
        self._instr_cbs[pos].append(callback)
        self._empty = False


    def register_pre_instruction_callback(self, callback: InstrCallback) -> None:
        """
        Register a pre-execution callback on all instruction executed by the engine.
        :param callback: callback function to trigger
        :return: None
        """
        self.register_instruction_callback(CbPos.BEFORE, callback)


    def register_post_instuction_callback(self, callback: InstrCallback) -> None:
        """
        Register a post-execution callback on all instruction executed by the engine.
        :param callback: callback function to trigger
        :return: None
        """
        self.register_instruction_callback(CbPos.AFTER, callback)


    def get_instruction_callbacks(self) -> Tuple[List[InstrCallback], List[InstrCallback]]:
        """
        Get all the pre/post callbacks for insrtuctions.
        :return: tuple of lists containing callback functions for pre/post respectively
        """
        return self._instr_cbs[CbPos.BEFORE], self._instr_cbs[CbPos.AFTER]


    def register_pre_execution_callback(self, callback: SymExCallback) -> None:
        """
        Register a callback executed after program loading, registers and memory
        initialization. Thus this callback is called just before executing the
        first instruction
        :param callback: Callback function to trigger
        :return: None
        """
        self._pre_exec.append(callback)
        self._empty = False


    def register_post_execution_callback(self, callback: SymExCallback) -> None:
        """
        Register a callback executed after program loading, registers and memory
        initialization. Thus this callback is called after executing upon program
        exit (or crash)
        :param callback: Callback function to trigger
        :return: None
        """
        self._post_exec.append(callback)
        self._empty = False


    def get_execution_callbacks(self) -> Tuple[List[SymExCallback], List[SymExCallback]]:
        """
        Get all the pre/post callbacks for the current symbolic execution
        :return: tuple of lists containing callback functions for pre/post respectively
        """
        return self._pre_exec, self._post_exec


    def register_memory_read_callback(self, callback: MemReadCallback) -> None:
        """
        Register a callback that will be triggered by any read in the concrete
        memory of the process state.

        :param callback: Callback function to be called
        :return: None
        """
        self._mem_read_cbs.append(callback)
        self._empty = False


    def register_memory_write_callback(self, callback: MemWriteCallback) -> None:
        """
        Register a callback called on each write in the concrete memory state
        of the process.

        :param callback: Callback function to be called
        :return: None
        """
        self._mem_write_cbs.append(callback)
        self._empty = False


    def register_register_read_callback(self, callback: RegReadCallback) -> None:
        """
        Register a callback on each register read during the symbolic execution.

        :param callback: Callback function to be called
        :return: None
        """
        self._reg_read_cbs.append(callback)
        self._empty = False


    def register_register_write_callback(self, callback: RegWriteCallback) -> None:
        """
        Register a callback on each register write during the symbolic execution.

        :param callback: Callback function to be called
        :return: None
        """
        self._reg_write_cbs.append(callback)
        self._empty = False


    def register_thread_context_switch_callback(self, callback: ThreadContext) -> None:
        """
        Register a callback triggered upon each thread context switch during the execution.

        :param callback: Callback to be called
        :return: None
        """
        self._ctx_switch.append(callback)
        self._empty = False


    def get_context_switch_callback(self) -> List[ThreadCallback]:
        """
        Get the list of all function callback to call when thread is being scheduled.
        :return: List of callbacks defined when thread is being scheduled
        """
        return self._ctx_switch


    def register_new_input_callback(self, callback: NewInputCallback) -> None:
        """
        Register a callback function called when the SMT solver find a new model namely
        a new input. This callback is called before any treatment on the input (worklist, etc.).
        It thus allow to post-process the input before it getting put in the queue.
        :param callback: callback function
        :return: None
        """
        self._new_input_cbs.append(callback)
        self._empty = False


    def get_new_input_callback(self) -> List[NewInputCallback]:
        """
        Get the list of all function callback to call when an a new
        input is generated by SMT.
        :return: List of callbacks to call on input generation
        """
        return self._new_input_cbs


    def register_pre_imported_routine_callback(self, routine_name: str, callback: RtnCallback) -> None:
        """
        Register a callback before call to imported routines
        :param routine_name: the routine name
        :param callback: callback function
        :return: None
        """
        if routine_name in self._pre_rtn_cbs:
            self._pre_rtn_cbs[routine_name].append(callback)
        else:
            self._pre_rtn_cbs[routine_name] = [callback]
        self._empty = False


    def register_post_imported_routine_callback(self, routine_name: str, callback: RtnCallback) -> None:
        """
        Register a callback after the call to imported routines
        :param routine_name: the routine name
        :param callback: callback function
        :return: None
        """
        if routine_name in self._post_rtn_cbs:
            self._post_rtn_cbs[routine_name].append(callback)
        else:
            self._post_rtn_cbs[routine_name] = [callback]
        self._empty = False


    def get_imported_routine_callbacks(self, routine_name) -> Tuple[List[RtnCallback], List[RtnCallback]]:
        """
        Get the list of all callback for an imported routine
        :param routine_name: the routine name
        :return: List of callbacks
        """
        pre_ret = (self._pre_rtn_cbs[routine_name] if routine_name in self._pre_rtn_cbs else [])
        post_ret = (self._post_rtn_cbs[routine_name] if routine_name in self._post_rtn_cbs else [])
        return pre_ret, post_ret


    def register_probe_callback(self, probe: ProbeInterface) -> None:
        """
        Register a probe callback.
        :param probe: a probe interface
        :return: None
        """
        for (kind, arg, cb) in probe.cbs:
            if kind == CbType.MEMORY_READ:
                self.register_memory_read_callback(cb)

            elif kind == CbType.MEMORY_WRITE:
                self.register_memory_write_callback(cb)

            elif kind == CbType.PRE_RTN:
                self.register_pre_imported_routine_callback(arg, cb)

    def fork(self) -> 'CallbackManager':
        """
        Fork the current CallbackManager in a new object instance
        (that will be unbinded). That method is used by the SymbolicExplorator
        to ensure each SymbolicExecutor running concurrently will have
        their own instance off the CallbackManager.

        :return: Fresh instance of CallbackManager
        """
        cbs = CallbackManager(self.p)

        # SymbolicExecutor callbacks
        cbs._pc_addr_cbs   = self._pc_addr_cbs
        cbs._instr_cbs     = self._instr_cbs
        cbs._pre_exec      = self._pre_exec
        cbs._post_exec     = self._post_exec
        cbs._ctx_switch    = self._ctx_switch
        cbs._new_input_cbs = self._new_input_cbs
        cbs._pre_rtn_cbs   = self._pre_rtn_cbs
        cbs._post_rtn_cbs  = self._post_rtn_cbs
        # Triton callbacks
        cbs._mem_read_cbs  = self._mem_read_cbs
        cbs._mem_write_cbs = self._mem_write_cbs
        cbs._reg_read_cbs  = self._reg_read_cbs
        cbs._reg_write_cbs = self._reg_write_cbs
        cbs._empty         = self._empty

        return cbs
