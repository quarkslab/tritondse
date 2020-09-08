# built-in imports
import logging
import time
import random
import os

# third party imports
from triton import MODE, Instruction, CPUSIZE, ARCH, MemoryAccess

# local imports
from tritondse.abi           import ABI
from tritondse.config        import Config
from tritondse.coverage      import Coverage
from tritondse.process_state import ProcessState
from tritondse.program       import Program
from tritondse.seed          import Seed
from tritondse.enums         import Enums
from tritondse.routines      import SUPPORTED_ROUTINES, SUPORTED_GVARIABLES
from tritondse.callbacks     import CallbackManager


class SymbolicExecutor(object):
    """
    This class is used to represent the symbolic execution.
    """
    def __init__(self, config: Config, pstate: ProcessState, program: Program, seed: Seed = None, uid=0, callbacks=None):
        self.program    = program
        self.pstate     = pstate
        self.config     = config
        self.seed       = seed
        self.abi        = ABI(self.pstate)
        self.coverage   = Coverage()
        self.rtn_table  = dict() # Addr -> Tuple[fname, routine]
        self._uid       = uid # Unique identifier meant to unique accross Exploration instances
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


    def __init_registers(self):
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_bp_register(), self.pstate.BASE_STACK)
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_sp_register(), self.pstate.BASE_STACK)
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_pc_register(), self.program.entry_point)


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
            pc = self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_pc_register())
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
            if not self.pstate.tt_ctx.processing(instruction):
                logging.error('Instruction not supported: %s' % (str(instruction)))
                break

            # Trigger post-instruction callback
            for cb in post_insts:
                cb(self, self.pstate, instruction)

            # Simulate that the time of an executed instruction is time_inc_coefficient.
            # For example, if time_inc_coefficient is 0.0001, it means that an instruction
            # takes 100us to be executed. Used to provide a deterministic behavior when
            # calling time functions (e.g gettimeofday(), clock_gettime(), ...).
            self.pstate.time += self.config.time_inc_coefficient

            # Update the coverage of the execution
            self.coverage.add_instruction(pc)

            #print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))

            # Simulate routines
            self.routines_handler(instruction)

            # Trigger post-address callbacks
            for cb in post_cbs:
                cb(self, self.pstate, pc)


            # Check timeout of the execution
            if self.config.execution_timeout and (time.time() - self.startTime) >= self.config.execution_timeout:
                logging.info('Timeout of an execution reached')
                break

        return


    def __handle_external_return(self, ret):
        """ Symbolize or concretize return values of external functions """
        if ret is not None:
            if ret[0] == Enums.CONCRETIZE:
                self.pstate.tt_ctx.concretizeRegister(self.abi.get_ret_register())
                self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_ret_register(), ret[1])
            elif ret[0] == Enums.SYMBOLIZE:
                self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_ret_register(), ret[1].getAst().evaluate())
                self.pstate.tt_ctx.assignSymbolicExpressionToRegister(ret[1], self.abi.get_ret_register())
        return


    def routines_handler(self, instruction):
        pc = self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_pc_register())
        if pc in self.rtn_table:
            routine_name, routine = self.rtn_table[pc]

            # Emulate the routine and the return value
            ret = routine(self)
            self.__handle_external_return(ret)

            # Do not continue the execution if we are in a locked mutex
            if self.pstate.mutex_locked:
                self.pstate.mutex_locked = False
                self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_pc_register(), instruction.getAddress())
                # It's locked, so switch to another thread
                self.pstate.threads[self.pstate.tid].count = 0
                return

            # Do not continue the execution if we are in a locked semaphore
            if self.pstate.semaphore_locked:
                self.pstate.semaphore_locked = False
                self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_pc_register(), instruction.getAddress())
                # It's locked, so switch to another thread
                self.pstate.threads[self.pstate.tid].count = 0
                return

            if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
                # Get the return address
                if routine_name == "__libc_start_main":
                    ret_addr = self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_pc_register())
                else:
                    ret_addr = self.pstate.tt_ctx.getConcreteRegisterValue(self.pstate.tt_ctx.registers.x30)

            elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
                # Get the return address
                ret_addr = self.pstate.tt_ctx.getConcreteMemoryValue(MemoryAccess(self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_sp_register()), CPUSIZE.QWORD))
                # Restore RSP (simulate the ret)
                self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_sp_register(), self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_sp_register()) + CPUSIZE.QWORD)

            else:
                raise Exception("Architecture not supported")

            # Hijack RIP to skip the call
            self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_pc_register(), ret_addr)


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

                # Add link to routine in table
                self.rtn_table[cur_linkage_address] = (fname, SUPPORTED_ROUTINES[fname])

                # Apply relocation to our custom address in process memory
                self.pstate.write_memory(rel_addr, self.pstate.ptr_size, cur_linkage_address)

                # Increment linkage address number
                cur_linkage_address += 1
            else:
                logging.debug(f"function {fname} imported but unsupported")  # should be warning

        # Link imported symbols
        for sname, rel_addr in self.program.imported_variable_symbols_relocations():
            if sname in SUPORTED_GVARIABLES:  # if the routine name is supported
                logging.debug(f"Hooking {sname} at {rel_addr:#x}")
                # Apply relocation to our custom address in process memory
                self.pstate.write_memory(rel_addr, self.pstate.ptr_size, SUPORTED_GVARIABLES[sname])
            else:
                logging.debug(f"symbol {sname} imported but unsupported")  # should be warning


    def run(self):
        # Initialize the process_state architecture (at this point arch is sure to be supported)
        logging.debug(f"Loading an {self.program.architecture.name} architecture")
        self.pstate.architecture = self.program.architecture

        # bind dbm callbacks on the process state
        self.cbm.bind_to(self)  # bind call

        self.__init_optimization()
        self.__init_registers()

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

        self.__emulate()

        # Iterate through post exec callbacks
        for cb in post_cb:
            cb(self, self.pstate)

        self.endTime = time.time()
        logging.info('Emulation done')
        logging.info('Return value: %#x' % (self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_ret_register())))
        logging.info('Instructions executed: %d' % (self.coverage.number_of_instructions_executed()))
        logging.info('Symbolic condition: %d' % (len(self.pstate.tt_ctx.getPathConstraints())))
        logging.info('Time of the execution: %f seconds' % (self.endTime - self.startTime))
