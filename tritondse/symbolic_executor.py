# built-in imports
import logging
import time
import random
import os
import resource
from typing import Optional, Union, List, NoReturn

# third party imports
from triton import MODE, Instruction, CPUSIZE, ARCH, MemoryAccess

# local imports
from tritondse.config         import Config
from tritondse.coverage       import CoverageSingleRun
from tritondse.process_state  import ProcessState
from tritondse.program        import Program
from tritondse.seed           import Seed, SeedStatus
from tritondse.types          import Expression, Architecture, Addr, Model
from tritondse.routines       import SUPPORTED_ROUTINES, SUPORTED_GVARIABLES
from tritondse.callbacks      import CallbackManager
from tritondse.workspace      import Workspace
from tritondse.heap_allocator import AllocatorException
from tritondse.thread_context import ThreadContext


class SymbolicExecutor(object):
    """
    Single Program Execution Class.
    That module, is in charge of performing the process loading from the given
    program.
    """

    def __init__(self, config: Config, pstate: ProcessState = None, seed: Seed = Seed(), workspace: Workspace = None, uid=0, callbacks=None):
        """
        :param config: configuration file to use
        :type config: Config
        :param pstate: ProcessState instanciated (but not loaded)
        :type pstate: ProcessState
        :param seed: input file to inject either in stdin or argv (optional)
        :type seed: Seed
        :param workspace: Workspace to use. If None it will be instanciated
        :type workspace: Optional[Workspace]
        :param uid: Unique ID. Given by :py:obj:`SymbolicExplorator` to identify uniquely executions
        :type uid: int
        :param callbacks: callbacks to bind on this execution before running *(instanciated if empty !)*
        :type callbacks: CallbackManager
        """
        # FIXME: Change interface, remove ProcessState
        self.config = config    # The config
        self.program = None     # The program to execute

        if pstate:              # If received a ProcessState take it 'as-is'
            self.pstate = pstate
            self._configure_pstate()
        else:
            self.pstate = None  # else should be loaded through load_program

        self.workspace  = workspace                             # The current workspace
        if self.workspace is None:
            self.workspace = Workspace(config.workspace)
        self.seed       = seed              # The current seed used to the execution
        self.symbolic_seed = []           # Will hold SymVars of every bytes of the seed
        self.coverage: CoverageSingleRun = CoverageSingleRun(self.config.coverage_strategy) #: Coverage of the execution
        self.rtn_table  = dict()            # Addr -> Tuple[fname, routine]
        self.uid        = uid               # Unique identifier meant to unique accross Exploration instances
        self.start_time = 0
        self.end_time  = 0
        # NOTE: Temporary datastructure to set hooks on addresses (might be replace later on by a nice visitor)

        # create callback object if not provided as argument, and bind callbacks to the current process state
        self.cbm = callbacks if callbacks is not None else CallbackManager()

        # List of new seeds filled during the execution and flushed by explorator
        self._pending_seeds = []
        self._run_to_target = None

        # TODO: Here we load the binary each time we run an execution (via ELFLoader). We can
        #       avoid this (and so gain in speed) if a TritonContext could be forked from a
        #       state. See: https://github.com/JonathanSalwan/Triton/issues/532

    def load_program(self, program: Program) -> None:
        """
        Load the given program in the symbolic executor's ProcessState.
        It override the current ProcessState if any.

        :param program: Program to load
        :return: None
        """
        self.program = program
        self.pstate = ProcessState.from_program(program)
        self._configure_pstate()
        self._map_dynamic_symbols()

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
        return bool(self.symbolic_seed)

    def _configure_pstate(self) -> None:
        for mode in [MODE.ALIGNED_MEMORY, MODE.AST_OPTIMIZATIONS, MODE.CONSTANT_FOLDING, MODE.ONLY_ON_SYMBOLIZED]:
            self.pstate.set_triton_mode(mode, True)
        self.pstate.time_inc_coefficient = self.config.time_inc_coefficient
        self.pstate.set_solver_timeout(self.config.smt_timeout)


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


    def __emulate(self):
        while not self.pstate.stop and self.pstate.threads:
            # Schedule thread if it's time
            self.__schedule_thread()

            if not self.pstate.current_thread.is_running():
                logging.warning(f"After scheduling current thread is not running (probably in a deadlock state)")
                break  # Were not able to find a suitable thread thus exit emulation

            # Fetch program counter (of the thread selected), at this point the current thread should be running!
            pc = self.pstate.cpu.program_counter

            if pc == self._run_to_target:  # Hit the location we wanted to reach
                break

            if pc == 0:
                logging.error(f"PC=0, is it normal ? (stop)")
                break

            if not self.pstate.is_memory_defined(pc, CPUSIZE.BYTE):
                logging.error(f"Instruction not mapped: 0x{pc:x}")
                break

            instruction = self.pstate.fetch_instruction()

            # Trigger pre-address callback
            pre_cbs, post_cbs = self.cbm.get_address_callbacks(pc)
            for cb in pre_cbs:
                cb(self, self.pstate, pc)

            # Trigger pre-instruction callback
            pre_insts, post_insts = self.cbm.get_instruction_callbacks()
            for cb in pre_insts:
                cb(self, self.pstate, instruction)

            # Process
            if not self.pstate.process_instruction(instruction):
                if self.pstate.is_halt_instruction():
                    logging.info(f"hit {str(instruction)} instruction stop.")
                else:
                    logging.error('Instruction not supported: %s' % (str(instruction)))
                break

            # Trigger post-instruction callback
            for cb in post_insts:
                cb(self, self.pstate, instruction)

            # Update the coverage of the execution
            self.coverage.add_covered_address(pc)

            # Update coverage send it the last PathConstraint object if one was added
            if self.pstate.is_path_predicate_updated():
                branch = self.pstate.last_branch_constraint
                self.coverage.add_covered_branch(pc, branch)

            # Trigger post-address callbacks
            for cb in post_cbs:
                cb(self, self.pstate, pc)

            # Simulate routines
            try:
                self._routines_handler(instruction)
            except AllocatorException as e:
                logging.info(f'An exception has been raised: {e}')
                self.seed.status = SeedStatus.CRASH
                return

            # Check timeout of the execution
            if self.config.execution_timeout and (time.time() - self.start_time) >= self.config.execution_timeout:
                logging.info('Timeout of an execution reached')
                self.seed.status = SeedStatus.HANG
                return

        if not self.seed.is_status_set():  # Set a status if it has not already been done
            self.seed.status = SeedStatus.OK_DONE
        return


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
            logging.debug(f"Enter external routine: {routine_name}")

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

            elif self.pstate.architecture == Architecture.X86_64:
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
                if self.pstate.architecture == Architecture.X86_64:
                    self.pstate.write_memory_ptr(addr, SUPORTED_GVARIABLES[symbol])  # write directly at addr
                elif self.pstate.architecture == Architecture.AARCH64:
                    val = self.pstate.read_memory_ptr(addr)
                    self.pstate.write_memory_ptr(val, SUPORTED_GVARIABLES[symbol])

            else:  # the symbol is not supported
                if self.uid == 0:  # print warning if first uid (so that it get printed once)
                    logging.warning(f"symbol {symbol} imported but unsupported")
                if is_func:
                    # Add link to a default stub function
                    self.rtn_table[addr] = (symbol, self.__default_stub)
                else:
                    pass # do nothing on unsupported symbols


    def __default_stub(self, se: 'SymbolicExecutor', pstate: ProcessState):
        rtn_name, _ = self.rtn_table[pstate.cpu.program_counter]
        logging.warning(f"calling {rtn_name} which is unsupported")
        if self.config.skip_unsupported_import:
            return None  # Like if function did nothing
        else:
            self.abort()


    def abort(self) -> NoReturn:
        """
        Abort the current execution. It works by raising
        an exception which is caught by the emulation function
        that takes care of returning appropriately afterward.

        :raise RuntimeError: to abort execution from anywhere
        """
        raise RuntimeError('Execution aborted')


    def run_to(self, addr: Addr) -> None:
        """
        Execute the programm up until hit the given the address.
        Mostly used to obtain the :py:obj:`ProcessState` at the
        state of the location to perform manual checks.

        :param addr: Address where to stop
        :return: None
        """
        self._run_to_target = addr
        self.run()


    def run(self) -> None:
        """
        Execute the program.

        If the :py:attr:`tritondse.Config.execution_timeout` is not set
        the execution might hang forever if the program does.
        """
        if self.pstate is None:
            logging.error(f"ProcessState is None (have you called load_program ?")
            return

        if not self._check_input_injection_loc():
            return

        self.start_time = time.time()
        # Initialize the process_state architecture (at this point arch is sure to be supported)
        logging.debug(f"Loading program {self.program.path.name} [{self.program.architecture}]")

        # bind dbm callbacks on the process state (newly initialized)
        self.cbm.bind_to(self)  # bind call

        # Let's emulate the binary from the entry point
        logging.info('Starting emulation')

        # Get pre/post callbacks on execution
        pre_cb, post_cb = self.cbm.get_execution_callbacks()
        # Iterate through all pre exec callbacks
        for cb in pre_cb:
            cb(self, self.pstate)

        try:
            self.__emulate()
        except RuntimeError as e:
            pass

        # Iterate through post exec callbacks
        for cb in post_cb:
            cb(self, self.pstate)

        self.end_time = time.time()
        logging.info(f"Emulation done [ret:{self.pstate.read_register(self.pstate.return_register):x}]  (time:{self.execution_time:.02f}s)")
        logging.info(f"Instructions executed: {self.coverage.total_instruction_executed}  symbolic branches: {self.pstate.path_predicate_size}")
        logging.info(f"Memory usage: {self.mem_usage_str()}")

    def _check_input_injection_loc(self) -> bool:
        """ Make sure only stdin or argv are symbolized """
        sum = self.config.symbolize_stdin + self.config.symbolize_argv
        if sum == 0:
            logging.warning("No input injection location selected (neither stdin nor argv) thus user-defined")
            return True  # We allow not defining seed injection point. If so the user has to do it manually
        elif sum == 2:
            logging.error("Cannot inject input on both stdin and argv in the same time")
            return False
        else:
            return True

    @property
    def exitcode(self) -> int:
        """ Exit code value of the process. The value
        is simply the concrete value of the register
        marked as return_register (rax, on x86, r0 on ARM..)
        """
        return self.pstate.read_register(self.pstate.return_register) & 0xFF

    @staticmethod
    def mem_usage_str() -> str:
        """ debug function to track memory consumption of an execution """
        size, resident, shared, _, _, _, _ = (int(x) for x in open(f"/proc/{os.getpid()}/statm").read().split(" "))
        resident = resident * resource.getpagesize()
        units = [(float(1024), "Kb"), (float(1024 **2), "Mb"), (float(1024 **3), "Gb")]
        for unit, s in units[::-1]:
            if resident / unit < 1:
                continue
            else:  # We are on the right unit
                return "%.2f%s" % (resident/unit, s)
        return "%dB" % resident

    def mk_new_seed_from_model(self, model: Model) -> Seed:
        """
        Given a SMT Model creates a new Seed.

        :param model: SMT model
        :return: new seed object
        """
        if self.config.symbolize_stdin:
            content = bytearray(self.seed.content)  # Create the new seed buffer
            for i, sv in enumerate(self.symbolic_seed):  # Enumerate symvars associated with each bytes
                if sv.getId() in model:  # If solver provided a new value for the symvar
                    content[i] = model[sv.getId()].getValue()  # Replace it in the bytearray

        elif self.config.symbolize_argv:
            args = [bytearray(x) for x in self.seed.content.split()]
            for c_arg, sym_arg in zip(args, self.symbolic_seed):
                for i, sv in enumerate(sym_arg):
                    if sv.getId() in model:
                        c_arg[i] = model[sv.getId()].getValue()
            content = b" ".join(args)  # Recreate a full argv string

        else:
            logging.error("In _mk_new_seed() without neither stdin nor argv seed injection loc")
            return Seed()  # Return dummy seed

        # Calling callback if user defined one
        for cb in self.cbm.get_new_input_callback():
            cont = cb(self, self.pstate, content)
            # if the callback return a new input continue with that one
            content = cont if cont is not None else content

        # Create the Seed object and assign the new model
        return Seed(bytes(content))

    def inject_symbolic_input(self, addr: Addr, seed: Seed, var_prefix: str = "input") -> None:
        """
        Inject the given seed at the given address in memory. Then
        all memory bytes are symbolized.

        :param addr: address at which to inject input
        :param seed: Seed to inject in memory
        :param var_prefix: prefix name to give the symbolic variables
        :return: None
        """
        # Write concrete bytes in memory
        self.pstate.write_memory_bytes(addr, seed.content)

        # Symbolize bytes
        sym_vars = self.pstate.symbolize_memory_bytes(addr, seed.size, var_prefix)
        self.symbolic_seed = sym_vars  # Set symbolic_seed to be able to retrieve them in generated models
