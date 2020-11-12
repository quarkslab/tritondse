# built-in imports
import logging
import time
import random
import os
from typing import Optional, Union

# third party imports
from triton import MODE, Instruction, CPUSIZE, ARCH, MemoryAccess

# local imports
from tritondse.config         import Config
from tritondse.coverage       import CoverageSingleRun
from tritondse.process_state  import ProcessState
from tritondse.program        import Program
from tritondse.seed           import Seed, SeedStatus
from tritondse.types          import Expression
from tritondse.routines       import SUPPORTED_ROUTINES, SUPORTED_GVARIABLES
from tritondse.callbacks      import CallbackManager
from tritondse.workspace      import Workspace
from tritondse.heap_allocator import AllocatorException


class SymbolicExecutor(object):
    """
    This class is used to represent the symbolic execution.
    """
    def __init__(self, config: Config, pstate: ProcessState, program: Program, seed: Seed = Seed(), workspace: Workspace = None, uid=0, callbacks=None):
        self.program    = program           # The program to execute
        self.pstate     = pstate            # The process state
        self.config     = config            # The config
        self.workspace  = workspace         # The current workspace
        if self.workspace is None:
            self.workspace = Workspace(config.workspace)
        self.seed       = seed              # The current seed used to the execution
        self.coverage   = CoverageSingleRun(self.config.coverage_strategy) # The coverage state
        self.rtn_table  = dict()            # Addr -> Tuple[fname, routine]
        self.uid        = uid               # Unique identifier meant to unique accross Exploration instances
        # NOTE: Temporary datastructure to set hooks on addresses (might be replace later on by a nice visitor)

        # create callback object if not provided as argument, and bind callbacks to the current process state
        self.cbm = callbacks if callbacks is not None else CallbackManager(self.program)

        # TODO: Here we load the binary each time we run an execution (via ELFLoader). We can
        #       avoid this (and so gain in speed) if a TritonContext could be forked from a
        #       state. See: https://github.com/JonathanSalwan/Triton/issues/532


    @property
    def callback_manager(self) -> CallbackManager:
        return self.cbm


    def __init_optimization(self):
        self.pstate.tt_ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self.pstate.tt_ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self.pstate.tt_ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self.pstate.tt_ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, True)
        self.pstate.tt_ctx.setSolverTimeout(self.config.smt_timeout)


    def __schedule_thread(self):
        if self.pstate.threads[self.pstate.tid].count <= 0:

            # Call all callbacks related to threads
            for cb in self.cbm.get_context_switch_callback():
                cb(self, self.pstate, self.pstate.threads[self.pstate.tid])

            # Reset the counter and save its context
            self.pstate.threads[self.pstate.tid].count = self.config.thread_scheduling
            self.pstate.threads[self.pstate.tid].save(self.pstate.tt_ctx)
            # Schedule to the next thread
            while True:
                self.pstate.tid = (self.pstate.tid + 1) % len(self.pstate.threads.keys())
                try:
                    self.pstate.threads[self.pstate.tid].count = self.config.thread_scheduling
                    self.pstate.threads[self.pstate.tid].restore(self.pstate.tt_ctx)
                    break
                except Exception:
                    continue
        else:
            self.pstate.threads[self.pstate.tid].count -= 1


    def __emulate(self):
        while not self.pstate.stop and self.pstate.threads:
            # Schedule thread if it's time
            self.__schedule_thread()

            # Fetch opcodes
            pc = self.pstate.cpu.program_counter
            opcodes = self.pstate.tt_ctx.getConcreteMemoryAreaValue(pc, 16)

            if (self.pstate.tid and pc == 0) or self.pstate.threads[self.pstate.tid].killed:
                logging.info('End of thread: %d' % self.pstate.tid)
                if pc == 0 and self.pstate.threads[self.pstate.tid].killed is False:
                    logging.warning('PC=0, is it normal?')
                    # TODO: Exit for debug
                    os._exit(-1)
                del self.pstate.threads[self.pstate.tid]
                self.pstate.tid = random.choice(list(self.pstate.threads.keys()))
                self.pstate.threads[self.pstate.tid].count = self.config.thread_scheduling
                self.pstate.threads[self.pstate.tid].restore(self.pstate.tt_ctx)
                continue

            joined = self.pstate.threads[self.pstate.tid].joined
            if joined and joined in self.pstate.threads:
                logging.debug('Thread id %d is joined on thread id %d' % (self.pstate.tid, joined))
                continue

            if not self.pstate.tt_ctx.isConcreteMemoryValueDefined(pc, CPUSIZE.BYTE):
                logging.error('Instruction not mapped: 0x%x' % pc)
                break

            # Create the Triton instruction
            instruction = Instruction(pc, opcodes)
            instruction.setThreadId(self.pstate.tid)

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
                self.routines_handler(instruction)
            except AllocatorException as e:
                logging.info(f'An exception has been raised: {e}')
                self.seed.status = SeedStatus.CRASH
                return

            # Check timeout of the execution
            if self.config.execution_timeout and (time.time() - self.startTime) >= self.config.execution_timeout:
                logging.info('Timeout of an execution reached')
                self.seed.status = SeedStatus.HANG
                return

        self.seed.status = SeedStatus.OK_DONE
        return


    def __handle_external_return(self, ret_val: Optional[Union[int, Expression]]) -> None:
        """ Symbolize or concretize return values of external functions """
        if ret_val is not None:
            reg = self.pstate.return_register
            if isinstance(ret_val, int): # Write its concrete value
                self.pstate.write_register(reg, ret_val)
            else:  # It should be a logic expression
                self.pstate.write_register(reg, ret_val.getAst().evaluate())
                self.pstate.write_symbolic_register(reg, ret_val)


    def routines_handler(self, instruction):
        pc = self.pstate.cpu.program_counter
        if pc in self.rtn_table:
            routine_name, routine = self.rtn_table[pc]

            # Trigger pre-address callback
            pre_cbs, post_cbs = self.cbm.get_imported_routine_callbacks(routine_name)
            for cb in pre_cbs:
                cb(self, self.pstate, routine_name, pc)

            # Emulate the routine and the return value
            ret_val = routine(self)
            self.__handle_external_return(ret_val)

            # Trigger post-address callbacks
            for cb in post_cbs:
                cb(self, self.pstate, routine_name, pc)

            # Do not continue the execution if we are in a locked mutex
            if self.pstate.mutex_locked:
                self.pstate.mutex_locked = False
                self.pstate.cpu.program_counter = instruction.getAddress()
                # It's locked, so switch to another thread
                self.pstate.threads[self.pstate.tid].count = 0
                return

            # Do not continue the execution if we are in a locked semaphore
            if self.pstate.semaphore_locked:
                self.pstate.semaphore_locked = False
                self.pstate.cpu.program_counter = instruction.getAddress()
                # It's locked, so switch to another thread
                self.pstate.threads[self.pstate.tid].count = 0
                return

            # FIXME: What the fuck is that ?
            if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
                # Get the return address
                if routine_name == "__libc_start_main":
                    ret_addr = self.pstate.cpu.program_counter
                else:
                    ret_addr = self.pstate.tt_ctx.getConcreteRegisterValue(self.pstate.tt_ctx.registers.x30)

            elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
                # Get the return address
                ret_addr = self.pstate.tt_ctx.getConcreteMemoryValue(MemoryAccess(self.pstate.cpu.stack_pointer, CPUSIZE.QWORD))
                # Restore RSP (simulate the ret)
                self.pstate.cpu.stack_pointer += CPUSIZE.QWORD

            else:
                raise Exception("Architecture not supported")

            # Hijack RIP to skip the call
            self.pstate.cpu.program_counter = ret_addr


    def _apply_dynamic_relocations(self) -> None:
        """
        Apply dynamic relocations of imported functions and imported symbols
        regardless of the architecture or executable format
        .. FIXME: This function does not apply all possible relocations
        :return: None
        """
        cur_linkage_address = self.pstate.BASE_PLT

        # Link imported functions
        for fname, rel_addr in self.program.imported_functions_relocations():
            if fname in SUPPORTED_ROUTINES:  # if the routine name is supported
                logging.debug(f"Hooking {fname} at {rel_addr:#x}")

                # Add link to the routine and got tables
                self.rtn_table[cur_linkage_address] = (fname, SUPPORTED_ROUTINES[fname])

                # Apply relocation to our custom address in process memory
                self.pstate.write_memory(rel_addr, self.pstate.ptr_size, cur_linkage_address)

                # Increment linkage address number
                cur_linkage_address += self.pstate.ptr_size
            else:
                logging.warning(f"function {fname} imported but unsupported")

        # Link imported symbols
        for sname, rel_addr in self.program.imported_variable_symbols_relocations():
                logging.debug(f"Hooking {sname} at {rel_addr:#x}")
                if sname in SUPORTED_GVARIABLES:  # if the routine name is supported
                    if self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
                        self.pstate.write_memory(rel_addr, self.pstate.ptr_size, SUPORTED_GVARIABLES[sname])

                    elif self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
                        self.pstate.write_memory(rel_addr, self.pstate.ptr_size, cur_linkage_address)
                        self.pstate.write_memory(cur_linkage_address, self.pstate.ptr_size, SUPORTED_GVARIABLES[sname])
                        cur_linkage_address += self.pstate.ptr_size
                else:
                    logging.warning(f"symbol {sname} imported but unsupported")

    def abort(self):
        raise RuntimeError('Execution aborted')


    def run(self):
        # Initialize the process_state architecture (at this point arch is sure to be supported)
        logging.debug(f"Loading program {self.program.path.name} [{self.program.architecture}]")

        # Initialize ProcessState context & optimizations
        self.pstate.initialize_context(self.program.architecture)
        self.__init_optimization()

        # bind dbm callbacks on the process state (newly initialized)
        self.cbm.bind_to(self)  # bind call

        # Load the program in process memory and apply dynamic relocations
        self.pstate.load_program(self.program)
        self._apply_dynamic_relocations()

        # Let's emulate the binary from the entry point
        logging.info('Starting emulation')
        self.startTime = time.time()

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

        self.endTime = time.time()
        logging.info("Emulation done")
        logging.info(f"Return value: 0x{self.pstate.read_register(self.pstate.return_register):x}")
        logging.info('Instructions executed: %d' % self.coverage.total_instruction_executed)
        logging.info('Symbolic branch constraints: %d' % (len(self.pstate.tt_ctx.getPathConstraints())))
        logging.info('Total execution time: %d seconds' % (self.endTime - self.startTime))
