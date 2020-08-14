#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lief
import logging
import time
import random

from triton                 import *
from tritondse.abi          import ABI
from tritondse.config       import Config
from tritondse.coverage     import Coverage
from tritondse.loaders      import ELFLoader
from tritondse.processState import ProcessState
from tritondse.program      import Program
from tritondse.seed         import Seed
from tritondse.enums        import Enums
from tritondse.routines     import *


class SymbolicExecutor(object):
    """
    This class is used to represent the symbolic execution.
    """
    def __init__(self, config : Config, pstate : ProcessState, program : Program, seed : Seed = None):
        self.program    = program
        self.pstate     = pstate
        self.config     = config
        self.seed       = seed
        self.loader     = ELFLoader(config, program, pstate)
        self.abi        = ABI(self.pstate)
        self.coverage   = Coverage()


    def __init_arch__(self):
        if self.program.binary.header.machine_type == lief.ELF.ARCH.AARCH64:
            logging.debug('Loading an AArch64 binary')
            self.pstate.tt_ctx.setArchitecture(ARCH.AARCH64)

        elif self.program.binary.header.machine_type == lief.ELF.ARCH.x86_64:
            logging.debug('Loading an x86_64 binary')
            self.pstate.tt_ctx.setArchitecture(ARCH.X86_64)

        else:
            raise(Exception('Architecture not supported'))


    def __init_optimization__(self):
        self.pstate.tt_ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self.pstate.tt_ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, True)
        self.pstate.tt_ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self.pstate.tt_ctx.setSolverTimeout(self.config.smt_timeout)


    def __init_stack__(self):
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_bp_register(), self.pstate.BASE_STACK)
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_sp_register(), self.pstate.BASE_STACK)
        self.pstate.tt_ctx.setConcreteRegisterValue(self.abi.get_pc_register(), self.program.binary.entrypoint)


    def __schedule_thread__(self):
        if self.pstate.threads[self.pstate.tid].count <= 0:
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
                except:
                    continue
        else:
            self.pstate.threads[self.pstate.tid].count -= 1


    def __emulate__(self):
        while not self.pstate.stop and self.pstate.threads:
            # Schedule thread if it's time
            self.__schedule_thread__()

            # Fetch opcodes
            pc = self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_pc_register())
            opcodes = self.pstate.tt_ctx.getConcreteMemoryAreaValue(pc, 16)

            if (self.pstate.tid and pc == 0) or self.pstate.threads[self.pstate.tid].killed:
                logging.info('End of thread: %d' % self.pstate.tid)
                if pc == 0 and self.pstate.threads[self.pstate.tid].killed == False:
                    logging.warning('PC=0, is it normal?')
                    # TODO: Exit for debug
                    sys.exit(-1)
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

            # Process
            if not self.pstate.tt_ctx.processing(instruction):
                logging.error('Instruction not supported: %s' % (str(instruction)))
                break

            self.coverage.add_instruction(pc)

            print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))

            # Simulate routines
            self.routines_handler(instruction)

            # Check timeout of the execution
            if self.config.execution_timeout and (time.time() - self.startTime) >= self.config.execution_timeout:
                logging.info('Timeout of an execution reached')
                break

        return


    def __handle_external_return__(self, ret):
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
        if pc in self.loader.routines_table:
            # Emulate the routine and the return value
            ret = self.loader.routines_table[pc](self)
            self.__handle_external_return__(ret)

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
                if self.loader.routines_table[pc] == rtn_libc_start_main:
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


    def run(self):
        self.__init_arch__()
        self.__init_optimization__()
        self.__init_stack__()
        self.loader.ld()

        # Let's emulate the binary from the entry point
        logging.info('Starting emulation')
        self.startTime = time.time()
        self.__emulate__()
        self.endTime = time.time()
        logging.info('Emulation done')
        logging.info('Return value: %#x' % (self.pstate.tt_ctx.getConcreteRegisterValue(self.abi.get_ret_register())))
        logging.info('Instructions covered: %d' % (self.coverage.number_of_instructions_covered()))
        logging.info('Instructions executed: %d' % (self.coverage.number_of_instructions_executed()))
        logging.info('Symbolic condition: %d' % (len(self.pstate.tt_ctx.getPathConstraints())))
        logging.info('Time of execution: %f seconds' % (self.endTime - self.startTime))
