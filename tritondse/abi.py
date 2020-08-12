#!/usr/bin/env python
# -*- coding: utf-8 -*-

from triton                 import *
from tritondse.processState import ProcessState


class ABI(object):
    """
    This class is used to represent the ABI.
    """
    def __init__(self, pstate : ProcessState):
        self.pstate = pstate


    def get_ret_register(self):
        """ Return the appropriate return register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            return self.pstate.tt_ctx.registers.x0

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            return self.pstate.tt_ctx.registers.rax

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86:
            return self.pstate.tt_ctx.registers.eax

        raise Exception('Architecture not supported')


    def get_pc_register(self):
        """ Return the appropriate pc register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            return self.pstate.tt_ctx.registers.pc

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            return self.pstate.tt_ctx.registers.rip

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86:
            return self.pstate.tt_ctx.registers.eip

        raise Exception('Architecture not supported')


    def get_bp_register(self):
        """ Return the appropriate base pointer register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            return self.pstate.tt_ctx.registers.sp

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            return self.pstate.tt_ctx.registers.rbp

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86:
            return self.pstate.tt_ctx.registers.ebp

        raise Exception('Architecture not supported')


    def get_sp_register(self):
        """ Return the appropriate stack pointer register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            return self.pstate.tt_ctx.registers.sp

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            return self.pstate.tt_ctx.registers.rsp

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86:
            return self.pstate.tt_ctx.registers.esp

        raise Exception('Architecture not supported')


    def get_sys_register(self):
        """ Return the appropriate syscall id register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            return self.pstate.tt_ctx.registers.x8

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            return self.pstate.tt_ctx.registers.rax

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86:
            return self.pstate.tt_ctx.registers.eax

        raise Exception('Architecture not supported')


    def get_arg_register(self, i):
        """ Return the appropriate register according to the arch """
        if self.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
            aarch64 = {
                0: self.pstate.tt_ctx.registers.x0,
                1: self.pstate.tt_ctx.registers.x1,
                2: self.pstate.tt_ctx.registers.x2,
                3: self.pstate.tt_ctx.registers.x3,
                4: self.pstate.tt_ctx.registers.x4,
                5: self.pstate.tt_ctx.registers.x5,
                6: self.pstate.tt_ctx.registers.x6,
                7: self.pstate.tt_ctx.registers.x7,
            }
            return aarch64[i]

        elif self.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
            x86_64 = {
                0: self.pstate.tt_ctx.registers.rdi,
                1: self.pstate.tt_ctx.registers.rsi,
                2: self.pstate.tt_ctx.registers.rdx,
                3: self.pstate.tt_ctx.registers.rcx,
                4: self.pstate.tt_ctx.registers.r8,
                5: self.pstate.tt_ctx.registers.r9,
            }
            return x86_64[i]

        raise Exception('Architecture or id not supported')
