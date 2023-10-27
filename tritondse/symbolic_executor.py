# built-in imports
import io
import time
import os

if os.name == 'posix':
    import resource

from typing import Optional, Union, List, NoReturn, Dict, Type

# third party imports
from triton import MODE, Instruction, CPUSIZE, MemoryAccess, CALLBACK

# local imports
from tritondse.config import Config
from tritondse.coverage import CoverageSingleRun, BranchSolvingStrategy
from tritondse.process_state import ProcessState
from tritondse.loaders import Loader
from tritondse.seed import Seed, SeedStatus, SeedFormat, CompositeData
from tritondse.types import Expression, Architecture, Addr, Model, SymbolicVariable, Register
from tritondse.routines import SUPPORTED_ROUTINES, SUPORTED_GVARIABLES
from tritondse.callbacks import CallbackManager
from tritondse.workspace import Workspace
from tritondse.heap_allocator import AllocatorException
from tritondse.thread_context import ThreadContext
from tritondse.exception import AbortExecutionException, SkipInstructionException, StopExplorationException, ProbeException
from tritondse.memory import MemoryAccessViolation, Perm
import tritondse.logging

logger = tritondse.logging.get("executor")


class SymbolicExecutor(object):
    """
    Single Program Execution Class.
    That module, is in charge of performing the process loading from the given
    program.
    """

    def __init__(self, config: Config, seed: Seed = Seed(), workspace: Workspace = None, uid=0, callbacks=None):
        """
        :param config: configuration file to use
        :type config: Config
        :param seed: input file to inject either in stdin or argv (optional)
        :type seed: Seed
        :param workspace: Workspace to use. If None it will be instanciated
        :type workspace: Optional[Workspace]
        :param uid: Unique ID. Given by :py:obj:`SymbolicExplorator` to identify uniquely executions
        :type uid: int
        :param callbacks: callbacks to bind on this execution before running *(instanciated if empty !)*
        :type callbacks: CallbackManager
        """
        self.config: Config = config      #: Configuration file used
        self.loader: Type[Loader] = None  #: Loader used to run the code

        self.pstate: ProcessState = None  #: ProcessState

        self.workspace: Workspace = workspace  #: Current workspace
        if self.workspace is None:
            self.workspace = Workspace(config.workspace)

        self.seed: Seed = seed  #: The current seed used for the execution

        # Override config if there is a mismatch between seed format and config file
        if seed.format != self.config.seed_format:
            logger.warning(f"seed format {seed.format} mismatch config {config.seed_format} (override config)")
            self.config.seed_format = seed.format

        self.symbolic_seed = self._init_symbolic_seed(seed) #: symbolic seed (same structure than Seed but with symbols)

        self.coverage: CoverageSingleRun = CoverageSingleRun(self.config.coverage_strategy)  #: Coverage of the execution
        self.rtn_table = dict()   # Addr -> Tuple[fname, routine]
        self.uid: int = uid       #: Unique identifier meant to unique accross Exploration instances
        self.start_time: int = 0  #: start time of the process
        self.end_time: int = 0    #: end time of the process

        # create callback object if not provided as argument, and bind callbacks to the current process state
        self.cbm: CallbackManager = callbacks if callbacks is not None else CallbackManager()
        """callback manager"""

        # List of new seeds filled during the execution and flushed by explorator
        self._pending_seeds = []
        self._run_to_target = None

        self.trace_offset: int = 0  #: counter of instructions executed

        self.previous_pc: int = 0  #: previous program counter executed
        self.current_pc = 0        #: current program counter

        self.debug_pp = False

        self._in_processing = False  # use to know if we are currently processing an instruction

        # TODO: Here we load the binary each time we run an execution (via ELFLoader). We can
        #       avoid this (and so gain in speed) if a TritonContext could be forked from a
        #       state. See: https://github.com/JonathanSalwan/Triton/issues/532

    def _init_symbolic_seed(self, seed: Seed) -> Union[list, CompositeData]:
        if seed.is_raw():
            return [None]*len(seed.content)
        else:  # is composite
            argv = [[None]*len(a) for a in seed.content.argv]
            files = {k: [None]*len(v) for k, v in seed.content.files.items()}
            variables = {k: [None]*(1 if isinstance(v, int) else len(v)) for k, v in seed.content.variables.items()}
            return CompositeData(argv=argv, files=files, variables=variables)

    def load(self, loader: Loader) -> None:
        """
        Use the given loader to initialize the ProcessState.
        It overrides the current ProcessState if any.

        :param loader: Loader describing how to load
        :return: None
        """

        # Initialize the process_state architecture (at this point arch is sure to be supported)
        self.loader = loader
        logger.debug(f"Loading program {self.loader.name} [{self.loader.architecture}]")
        self.pstate = ProcessState.from_loader(loader)
        self._map_dynamic_symbols()
        self._load_seed_process_state(self.pstate, self.seed)

    def load_process(self, pstate: ProcessState) -> None:
        """
        Load the given process state. Do nothing but
        setting the internal ProcessState.

        :param pstate: PrcoessState to set
        """
        self.pstate = pstate
        self._load_seed_process_state(self.pstate, self.seed)

    @staticmethod
    def _load_seed_process_state(pstate: ProcessState, seed: Seed) -> None:
        if seed.is_raw():
            data = seed.content
        else:  # is composite
            if seed.is_file_defined("stdin"):
                data = seed.get_file_input("stdin")
            else:
                return
        filedesc = pstate.get_file_descriptor(0)
        filedesc.fd = io.BytesIO(data)

    @property
    def execution_time(self) -> int:
        """
        Time taken for the execution in seconds

        .. warning:: Only relevant at the end of the execution

        :return: execution time (in s)
        """
        return self.end_time - self.start_time

    @property
    def pending_seeds(self) -> List[Seed]:
        """
        List of pending seeds gathered during execution.

        .. warning:: Only relevant at the end of execution

        :returns: list of new seeds generated
        :rtype: List[Seed]
        """
        return self._pending_seeds

    def enqueue_seed(self, seed: Seed) -> None:
        """
        Add a seed to the queue of seed to be executed in later iterations.
        This function is meant to be used by user callbacks.

        :param seed: Seed to be added
        :type seed: Seed
        """
        self._pending_seeds.append(seed)

    @property
    def callback_manager(self) -> CallbackManager:
        """
        Get the callback manager associated with the execution.

        :rtype: CallbackManager"""
        return self.cbm

    def is_seed_injected(self) -> bool:
        """
        Get whether or not the seed has been injected.

        :return: True if the seed has already been inserted
        """
        if self.config.is_format_raw():
            return bool(self.symbolic_seed)
        elif self.config.is_format_composite():
            # Namely has one of the various input been injected or not
            return bool(self.symbolic_seed.content.files) or bool(self.symbolic_seed.content.variables)
        else:
            assert False

    def _configure_pstate(self) -> None:
        #for mode in [MODE.ALIGNED_MEMORY, MODE.AST_OPTIMIZATIONS, MODE.CONSTANT_FOLDING, MODE.ONLY_ON_SYMBOLIZED]:
        for mode in [MODE.ONLY_ON_SYMBOLIZED]:
            self.pstate.set_triton_mode(mode, True)
        logger.info(f"configure pstate: time_inc:{self.config.time_inc_coefficient}  solver:{self.config.smt_solver.name}  timeout:{self.config.smt_timeout}")
        self.pstate.time_inc_coefficient = self.config.time_inc_coefficient
        self.pstate.set_solver_timeout(self.config.smt_timeout)
        self.pstate.set_solver(self.config.smt_solver)

    def _fetch_next_thread(self, threads: List[ThreadContext]) -> Optional[ThreadContext]:
        """
        Given a list of threads, returns the next to execute. Iterating
        threads in a round-robin style picking the next item in the list.

        :param threads: list of threads
        :return: thread context
        """
        cur_idx = threads.index(self.pstate.current_thread)

        tmp_list = threads[cur_idx+1:]+threads[:cur_idx]  # rotate list (and exclude current_item)
        for th in tmp_list:
            if th.is_running():
                return th  # Return the first thread that is properly running
        return None

    def __schedule_thread(self) -> None:
        threads_list = self.pstate.threads

        if len(threads_list) == 1:  # If there is only one thread no need to schedule another thread
            return

        if self.pstate.current_thread.count > self.config.thread_scheduling:
            # Select the next thread to execute
            next_th = self._fetch_next_thread(threads_list)

            if next_th:  # We found another thread to schedule

                # Call all callbacks related to threads
                for cb in self.cbm.get_context_switch_callback():
                    cb(self, self.pstate, self.pstate.current_thread)

                # Save current context and restore new thread context (+kill current if dead)
                self.pstate.switch_thread(next_th)

            else:  # There are other thread but not other one is available (thus keep current one)
                self.pstate.current_thread.count = 0  # Reset its counter

        else:
            # Increment the instruction counter of the thread (a bit in advance but it does not matter)
            self.pstate.current_thread.count += 1

    def _symbolic_mem_callback(self, se: 'SymbolicExecutor', ps: ProcessState, mem: MemoryAccess, *args):
        tgt_addr = mem.getAddress()
        lea_ast = mem.getLeaAst()
        if lea_ast is None:
            return
        if lea_ast.isSymbolized():
            s = "write" if bool(args) else "read"
            pc = self.pstate.cpu.program_counter
            logger.debug(f"symbolic {s} at 0x{pc:x}: target: 0x{tgt_addr:x} [{lea_ast}]")
            self.pstate.push_constraint(lea_ast == tgt_addr, f"sym-{s}:{self.trace_offset}:{pc}")

    def emulate(self):
        while not self.pstate.stop and self.pstate.threads:
            if not self.step():
                break
        if not self.seed.is_status_set():  # Set a status if it has not already been done
            self.seed.status = SeedStatus.OK_DONE
        return


    def step(self) -> bool:
        """
        Perform a single instruction step. Returns whether the emulation can
        continue or we have to stop.
        """
        try:
            # Schedule thread if it's time
            self.__schedule_thread()

            if not self.pstate.current_thread.is_running():
                logger.warning(f"After scheduling current thread is not running (probably in a deadlock state)")
                return False  # Were not able to find a suitable thread thus exit emulation

            # Fetch program counter (of the thread selected), at this point the current thread should be running!
            self.current_pc = self.pstate.cpu.program_counter  # should normally be already set but still.

            if self.current_pc == self._run_to_target:  # Hit the location we wanted to reach
                return False

            if self.current_pc == 0:
                logger.error(f"PC=0, is it normal ? (stop)")
                return False

            if self.pstate.memory.segmentation_enabled:
                if not self.pstate.memory.has_ever_been_written(self.current_pc, CPUSIZE.BYTE):
                    logger.error(f"Instruction not mapped: 0x{self.current_pc:x}")
                    return False

            instruction = self.pstate.fetch_instruction()
            opcode = instruction.getOpcode()
            mnemonic = instruction.getType()

            try:
                # Trigger pre-address callback
                pre_cbs, post_cbs = self.cbm.get_address_callbacks(self.current_pc)
                for cb in pre_cbs:
                    cb(self, self.pstate, self.current_pc)

                # Trigger pre-opcode callback
                pre_opcode, post_opcode = self.cbm.get_opcode_callbacks(opcode)
                for cb in pre_opcode:
                    cb(self, self.pstate, opcode)

                # Trigger pre-mnemonic callback
                pre_mnemonic, post_mnemonic = self.cbm.get_mnemonic_callbacks(mnemonic)
                for cb in pre_mnemonic:
                    cb(self, self.pstate, mnemonic)

                # Trigger pre-instruction callback
                pre_insts, post_insts = self.cbm.get_instruction_callbacks()
                for cb in pre_insts:
                    cb(self, self.pstate, instruction)
            except SkipInstructionException as _:
                return True

            if self.pstate.is_syscall():
                logger.warning(f"execute syscall instruction {self.pstate.read_register(self.pstate._syscall_register)}")

            # Process
            prev_pc = self.current_pc
            self._in_processing = True
            if not self.pstate.process_instruction(instruction):
                if self.pstate.is_halt_instruction():
                    logger.info(f"hit {str(instruction)} instruction stop.")
                    return False
                else:
                    logger.error('Instruction not supported: %s' % (str(instruction)))
                    if self.config.skip_unsupported_instruction:
                        self.pstate.cpu.program_counter += instruction.getSize() # try to jump over the instruction
                    else:
                        return False  # stop emulation

            self._in_processing = False
            # increment trace offset
            self.trace_offset += 1

            # update previous program counters
            self.previous_pc = prev_pc
            self.current_pc = self.pstate.cpu.program_counter  # current_pc becomes new instruction pointer

            # Update the coverage of the execution
            self.coverage.add_covered_address(self.previous_pc)

            # Update coverage send it the last PathConstraint object if one was added
            if self.pstate.is_path_predicate_updated():
                path_constraint = self.pstate.last_branch_constraint

                if path_constraint.isMultipleBranches():
                    branches = path_constraint.getBranchConstraints()
                    if len(branches) != 2:
                        logger.error("Branching condition has more than two branches")
                    taken, not_taken = branches if branches[0]['isTaken'] else branches[::-1]
                    taken_addr, not_taken_addr = taken['dstAddr'], not_taken['dstAddr']

                    for cb in self.cbm.get_on_branch_covered_callback():
                        cb(self, self.pstate, (self.previous_pc, taken_addr))

                    self.coverage.add_covered_branch(self.previous_pc, taken_addr, not_taken_addr)

                else:  # It is normally a dynamic jump or symbolic memory read/write
                    cmt = path_constraint.getComment()
                    if cmt.startswith("sym-read") or cmt.startswith("sym-write"):
                        pass
                        # NOTE: At the moment it does not seems suitable to count r/w pointers
                        # as part of the coverage. So does not have an influence on covered/not_covered.
                    else:
                        logger.warning(f"New dynamic jump covered at: {self.previous_pc:08x}")
                        path_constraint.setComment(f"dyn-jmp:{self.trace_offset}:{self.previous_pc}")
                        self.coverage.add_covered_dynamic_branch(self.previous_pc, self.current_pc)

            # Trigger post-opcode callback
            for cb in post_opcode:
                cb(self, self.pstate, opcode)

            # Trigger post-mnemonic callback
            for cb in post_mnemonic:
                cb(self, self.pstate, mnemonic)

            # Trigger post-instruction callback
            for cb in post_insts:
                cb(self, self.pstate, instruction)

            # Trigger post-address callbacks
            for cb in post_cbs:
                cb(self, self.pstate, self.previous_pc)

            # Simulate routines
            try:
                self._routines_handler(instruction)
            except AllocatorException as e:
                logger.info(f'An exception has been raised: {e}')
                self.seed.status = SeedStatus.CRASH
                return False

            # Check timeout of the execution
            if self.config.execution_timeout and (time.time() - self.start_time) >= self.config.execution_timeout:
                logger.info('Timeout of an execution reached')
                self.seed.status = SeedStatus.HANG
                return False
            return True

            # Call all the callbacks on the memory violations
            for cb in self.callback_manager.get_memory_violation_callbacks():
                cb(self, self.pstate, e)
        except AbortExecutionException as e:
            return False
        except MemoryAccessViolation as e:
            logger.warning(f"Memory violation: {str(e)}")
        except ProbeException:
            return False
        except Exception as e:
            logger.warning(f"Execution interrupted: {e}")
            self.seed.status = SeedStatus.FAIL
            return False

            # Assign the seed the status of crash
            if not self.seed.is_status_set():
                self.seed.status = SeedStatus.CRASH
            return False


    def __handle_external_return(self, routine_name: str, ret_val: Optional[Union[int, Expression]]) -> None:
        """ Symbolize or concretize return values of external functions """
        if ret_val is not None:
            reg = self.pstate.return_register
            if isinstance(ret_val, int): # Write its concrete value
                self.pstate.write_register(reg, ret_val)
            else:  # It should be a logic expression
                self.pstate.write_symbolic_register(reg, ret_val, f"(routine {routine_name}")

    def _routines_handler(self, instruction: Instruction):
        """
        This function handle external routines calls. When the .plt jmp on an external
        address, we call the appropriate Python routine and setup the returned value
        which may be concrete or symbolic.

        :param instruction: The current instruction executed
        :return: None
        """
        pc = self.pstate.cpu.program_counter
        if pc in self.rtn_table:
            routine_name, routine = self.rtn_table[pc]
            logger.debug(f"Enter external routine: {routine_name}")

            # Trigger pre-address callback
            pre_cbs, post_cbs = self.cbm.get_imported_routine_callbacks(routine_name)

            ret_val = None
            for cb in pre_cbs:
                ret = cb(self, self.pstate, routine_name, pc)
                if ret is not None:  # if the callback return a value the function behavior will be skipped
                    ret_val = ret
                    break  # Set the ret val and break

            if ret_val is None:  # If no ret_val has been set by any callback function call the supported routine
                # Emulate the routine and the return value
                ret_val = routine(self, self.pstate)

            self.__handle_external_return(routine_name, ret_val)

            # Trigger post-address callbacks
            for cb in post_cbs:
                cb(self, self.pstate, routine_name, pc)

            # Do not continue the execution if we are in a locked mutex
            if self.pstate.mutex_locked:
                self.pstate.mutex_locked = False
                self.pstate.cpu.program_counter = instruction.getAddress()
                # It's locked, so switch to another thread
                self.pstate.current_thread.count = self.config.thread_scheduling+1
                return

            # Do not continue the execution if we are in a locked semaphore
            if self.pstate.semaphore_locked:
                self.pstate.semaphore_locked = False
                self.pstate.cpu.program_counter = instruction.getAddress()
                # It's locked, so switch to another thread
                self.pstate.current_thread.count = self.config.thread_scheduling+1
                return

            if self.pstate.architecture == Architecture.AARCH64:
                # Get the return address
                ret_addr = self.pstate.read_register('x30')
            elif self.pstate.architecture in [Architecture.X86, Architecture.X86_64]:
                # Get the return address and restore RSP (simulate RET)
                ret_addr = self.pstate.pop_stack_value()
            else:
                raise Exception("Architecture not supported")

            # Hijack RIP to skip the call
            self.pstate.cpu.program_counter = ret_addr

    def _map_dynamic_symbols(self) -> None:
        """
        Apply dynamic relocations of imported functions and imported symbols
        regardless of the architecture or executable format
        .. FIXME: This function does not apply all possible relocations
        :return: None
        """
        for symbol, (addr, is_func) in self.pstate.dynamic_symbol_table.items():

            if symbol in SUPPORTED_ROUTINES:  # if the routine name is supported
                # Add link to the routine and got tables
                self.rtn_table[addr] = (symbol, SUPPORTED_ROUTINES[symbol])

            elif symbol in SUPORTED_GVARIABLES:
                # if self.pstate.architecture == Architecture.X86_64:
                self.pstate.memory.write_ptr(addr, SUPORTED_GVARIABLES[symbol])  # write directly at addr
                # elif self.pstate.architecture == Architecture.AARCH64:
                #     val = self.pstate.memory.read_ptr(addr)
                #     self.pstate.memory.write_ptr(val, SUPORTED_GVARIABLES[symbol])

            else:  # the symbol is not supported
                if self.uid == 0:  # print warning if first uid (so that it get printed once)
                    logger.warning(f"symbol {symbol} imported but unsupported")
                if is_func:
                    # Add link to a default stub function
                    self.rtn_table[addr] = (symbol, self.__default_stub)
                else:
                    pass # do nothing on unsupported symbols

    def __default_stub(self, se: 'SymbolicExecutor', pstate: ProcessState):
        rtn_name, _ = self.rtn_table[pstate.cpu.program_counter]
        logger.warning(f"calling {rtn_name} which is unsupported")
        if self.config.skip_unsupported_import:
            return None  # Like if function did nothing
        else:
            self.abort()

    def abort(self) -> NoReturn:
        """
        Abort the current execution. It works by raising
        an exception which is caught by the emulation function
        that takes care of returning appropriately afterward.

        :raise AbortExecutionException: to abort execution from anywhere
        """
        raise AbortExecutionException('Execution aborted')

    def skip_instruction(self) -> NoReturn:
        """
        Skip the current instruction before it gets executed. It is only
        relevant to call it from pre-inst or pre-addr callbacks.

        :raise SkipInstructionException: to skip the current instruction
        """
        raise SkipInstructionException("Skip instruction")

    def stop_exploration(self) -> NoReturn:
        """
        Function to call to stop the whole exploration of
        the program. It raises an exception which is caught by SymbolicExplorator.

        :raise StopExplorationException: to stop the exploration
        """
        raise StopExplorationException("Stop exploration")

    def emulation_init(self) -> bool:
        if self.pstate is None:
            logger.error(f"ProcessState is None (have you called \"load\"?")
            return False

        self.start_time = time.time()

        # Configure memory segmentation using configuration
        self.pstate.memory.set_segmentation(self.config.memory_segmentation)
        if self.config.memory_segmentation:
            self.cbm.register_memory_read_callback(self._mem_accesses_callback)
            self.cbm.register_memory_write_callback(self._mem_accesses_callback)

        # Register memory callbacks in case we activated covering mem access
        if BranchSolvingStrategy.COVER_SYM_READ in self.config.branch_solving_strategy:
            self.cbm.register_memory_read_callback(self._symbolic_mem_callback)
        if BranchSolvingStrategy.COVER_SYM_WRITE in self.config.branch_solving_strategy:
            self.cbm.register_memory_write_callback(self._symbolic_mem_callback)

        # bind dbm callbacks on the process state (newly initialized)
        self.cbm.bind_to(self)  # bind call

        # Let's emulate the binary from the entry point
        logger.info('Starting emulation')

        # Get pre/post callbacks on execution
        pre_cb, post_cb = self.cbm.get_execution_callbacks()
        # Iterate through all pre exec callbacks
        for cb in pre_cb:
            cb(self, self.pstate)

        # Call it here to make sure in case of "load_process" the use has properly instanciated the architecture
        self._configure_pstate()
        return True

    def run(self, stop_at: Addr = None) -> None:
        """
        Execute the program.

        If the :py:attr:`tritondse.Config.execution_timeout` is not set
        the execution might hang forever if the program does.

        :param stop_at: Address where to stop (if necessary)
        :return: None
        """
        if stop_at:
            self._run_to_target = stop_at

        # Call init steps
        if not self.emulation_init():
            return

        # Run until reaching a stopping condition
        self.emulate()

        # Call termination steps
        self.emulation_deinit()

    def emulation_deinit(self):
        _, post_cb = self.cbm.get_execution_callbacks()
        # Iterate through post exec callbacks
        for cb in post_cb:
            cb(self, self.pstate)

        self.end_time = time.time()

        # IMPORTANT The next call is necessary otherwise there is a memory
        #           leak issues.
        # Unbind callbacks from the current symbolic executor instance.
        self.cbm.unbind()

        # NOTE Unregister callbacks registered at the beginning of the function.
        #      This is necessary because we currently have a circular dependency
        #      between this class and the callback manager. Note that we create
        #      that circular dependency indirectly when we set the callback to
        #      a method of this class (_mem_accesses_callback and
        #      _symbolic_mem_callback).
        if self.config.memory_segmentation:
            self.cbm.unregister_callback(self._mem_accesses_callback)

        if BranchSolvingStrategy.COVER_SYM_READ in self.config.branch_solving_strategy:
            self.cbm.unregister_callback(self._symbolic_mem_callback)
        if BranchSolvingStrategy.COVER_SYM_WRITE in self.config.branch_solving_strategy:
            self.cbm.unregister_callback(self._symbolic_mem_callback)

        logger.info(f"Emulation done [ret:{self.pstate.read_register(self.pstate.return_register):x}]  (time:{self.execution_time:.02f}s)")
        logger.info(f"Instructions executed: {self.coverage.total_instruction_executed}  symbolic branches: {self.pstate.path_predicate_size}")
        logger.info(f"Memory usage: {self.mem_usage_str()}")

    def _mem_accesses_callback(self, se: 'SymbolicExecutor', ps: ProcessState, mem: MemoryAccess, *args):
        """
        This callback is used to ensure memory accesses performed by side-effect of instructions
        semantic correctly checks memory segmentation. Thus we only do the check during the processing
        of an instruction.
        """
        if ps.memory.segmentation_enabled and self._in_processing:
            perm = Perm.W if bool(args) else Perm.R
            addr = mem.getAddress()
            size = mem.getSize()
            map = ps.memory.get_map(addr, size)  # It raises
            if map is None:
                raise MemoryAccessViolation(addr, perm, memory_not_mapped=True)
            else:
                if perm not in map.perm:
                    raise MemoryAccessViolation(addr, perm, map_perm=map.perm, perm_error=True)

    @property
    def exitcode(self) -> int:
        """ Exit code value of the process. The value
        is simply the concrete value of the register
        marked as return_register (rax, on x86, r0 on ARM..)
        """
        return self.pstate.read_register(self.pstate.return_register) & 0xFF

    @staticmethod
    def mem_usage_str() -> str:
        """
        Debug function to track memory consumption of an execution (not
        implemented on Windows).
        """
        if os.name == "posix":
            size, resident, shared, _, _, _, _ = (int(x) for x in open(f"/proc/{os.getpid()}/statm").read().split(" "))
            resident = resident * resource.getpagesize()
            units = [(float(1024), "Kb"), (float(1024 **2), "Mb"), (float(1024 **3), "Gb")]
            for unit, s in units[::-1]:
                if resident / unit < 1:
                    continue
                else:  # We are on the right unit
                    return "%.2f%s" % (resident/unit, s)
            return "%dB" % resident
        else:
          return "N/A"

    def mk_new_seed_from_model(self, model: Model) -> Seed:
        """
        Creates a new seed from the given SMT model.

        :param model: SMT model
        :return: new seed object
        """
        def repl_bytearray(concrete, symbolic):
            for i, sv in enumerate(symbolic):  # Enumerate symvars associated with each bytes
                if sv is not None:
                    if sv.getId() in model:  # If solver provided a new value for the symvar
                        value = model[sv.getId()].getValue()
                        concrete[i] = value # Replace it in the bytearray
            return concrete

        if self.config.is_format_raw(): # RAW seed. => symbolize_stdin
            content = bytes(repl_bytearray(bytearray(self.seed.content), self.symbolic_seed))

        elif self.config.is_format_composite():
            # NOTE will have to update this if more things are added to CompositeData
            new_files, new_vars = {}, {}

            # Handle argv (its meant to be here)
            args = [bytearray(x) for x in self.seed.content.argv]
            new_argv = [bytes(repl_bytearray(c, s)) for c, s in zip(args, self.symbolic_seed.argv)]

            # Handle stdin and files
            # If the seed provides the content of files (#NOTE stdin is treated as a file)
            new_files = {}
            for k, c in self.seed.content.files.items():
                if k in self.symbolic_seed.files:
                    new_files[k] = bytes(repl_bytearray(bytearray(c), self.symbolic_seed.files[k]))
                else:
                    new_files[k] = c  # keep the current value in the seed

            # Handle variables, if the seed provides some
            new_variables = {}
            for k, c in self.seed.content.variables.items():
                if k in self.symbolic_seed.variables:
                    conc = bytearray(c) if isinstance(c, bytes) else [c]
                    new_vals = repl_bytearray(conc, self.symbolic_seed.variables[k])
                    new_variables[k] = bytes(new_vals) if isinstance(c, bytes) else new_vals[0]  # new variables are either bytes or int
                else:
                    new_variables[k] = c  # If it has not been injected keep the current concrete value

            content = CompositeData(new_argv, new_files, new_variables)
        else:
            assert False

        # Calling callback if user defined one
        new_seed = Seed(content)
        for cb in self.cbm.get_new_input_callback():
            cont = cb(self, self.pstate, new_seed)
            if cont:
                # if the callback return a new input continue with that one
                new_seed = cont
        # Return the
        return new_seed

    def inject_symbolic_argv_memory(self, addr: Addr, index: int, value: bytes) -> None:
        """
        Inject the ith item of argv in memory.
        To be used only with composite seeds and only if seed have a symbolic argv

        :param addr: address where to inject the argv[ith]
        :param index: ith argv item
        :param value: value of the item
        """
        self.pstate.memory.write(addr, value)  # Write concrete bytes in memory
        sym_vars = self.pstate.symbolize_memory_bytes(addr, len(value), f"argv[{index}]") # Symbolize bytes
        self.symbolic_seed.argv[index] = sym_vars # Add symbolic variables to symbolic seed

    def inject_symbolic_file_memory(self, addr: Addr, name: str, value: bytes, offset: int = 0) -> None:
        """
        Inject a symbolic file (or part of it) in memory.

        :param addr: address where to inject the file bytes
        :param name: name of the file in the composite seed
        :param value: bytes content of the file
        :param offset: offset within the file (for partial file injection)
        """
        self.pstate.memory.write(addr, value)  # Write concrete bytes in memory
        sym_vars = self.pstate.symbolize_memory_bytes(addr, len(value), name, offset) # Symbolize bytes
        sym_seed = self.symbolic_seed.files[name] if self.seed.is_composite() else self.symbolic_seed
        sym_seed[offset:offset+len(value)] = sym_vars # Add symbolic variables to symbolic seed
        # FIXME: Handle if reading twice same input bytes !

    def inject_symbolic_variable_memory(self, addr: Addr, name: str, value: bytes, offset: int = 0) -> None:
        """
        Inject a symbolic variable in memory.

        :param addr: address where to inject the variable
        :param name: name of the variable in the composite seed
        :param value: value of the variable
        :param offset: offset within the variable (for partial variable injection)
        :return:
        """
        self.pstate.memory.write(addr, value)  # Write concrete bytes in memory
        sym_vars = self.pstate.symbolize_memory_bytes(addr, len(value), name, offset)  # Symbolize bytes
        self.symbolic_seed.variables[name][offset:offset+len(value)-1] = sym_vars # Add symbolic variables to symbolic seed
        # FIXME: Handle if reading twice same input bytes !

    def inject_symbolic_file_register(self, reg: Union[str, Register], name: str, value: int, offset: int = 0) -> None:
        """
        Inject a symbolic file (or part of it) into a register.
        The value has to be an integer.

        :param reg: register identifier
        :param name: name of the file in the composite seed
        :param value: integer value
        :param offset: offset within the file
        """
        if reg.getSize != 1:
            logger.error("can't call inject_symbolic_file_register with regsiter larger than 1!")
            return
        self.pstate.write_register(reg, value)  # Write concrete value in register
        sym_vars = self.pstate.symbolize_register(reg, f"{name}[{offset}]")  # Symbolize bytes
        sym_seed = self.symbolic_seed.files[name] if self.seed.is_composite() else self.symbolic_seed
        sym_seed[offset] = sym_vars  # Add symbolic variables to symbolic seed

    def inject_symbolic_variable_register(self, reg: Union[str, Register], name: str, value: int) -> None:
        """
        Inject a symbolic variable (or part of it) in a register.
        The value has to be an integer.

        :param reg: register identifier
        :param name: name of the variable
        :param value: integer value
        """
        if not self.seed.is_composite():
            logger.warning("cannot use inject_symbolic_variable_register on raw seeds!")
            return

        if isinstance(value, int):
            self.pstate.write_register(reg, value)                         # write concrete value in register
            sym_var = self.pstate.symbolize_register(reg, f"{name}[{0}]")  # symbolize value
            self.symbolic_seed.variables[name][0] = sym_var               # add the symbolic variables to symbolic seed
        else:  # meant to be bytes
            logger.warning("variable injected in registers have to be integer values")

    def inject_symbolic_raw_input(self, addr: Addr, data: bytes, offset: int = 0) -> None:
        """
        Inject the input in memory. This injection method should
        be used for RAW seed type.

        :param addr: address where to inject input.
        :param data: content of the seed
        :param offset: offset within the content of the seed.
        """
        if self.seed.is_composite():
            logger.warning("inject_symbolic_memory must not be used with composite seeds !")
        else:
            self.inject_symbolic_file_memory(addr, "input", data, offset)
