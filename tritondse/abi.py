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


    def get_memory_string(self, addr):
        """ Returns a string from a memory address """
        s = str()
        index = 0

        while self.pstate.tt_ctx.getConcreteMemoryValue(addr+index):
            c = chr(self.pstate.tt_ctx.getConcreteMemoryValue(addr+index))
            s += c
            index += 1

        return s


    def get_string_argument(self, i):
        return self.get_memory_string(self.pstate.tt_ctx.getConcreteRegisterValue(self.get_arg_register(0)))


    def get_format_string(self, addr):
        """ Returns a formatted string from a memory address """
        return self.get_memory_string(addr)                                             \
               .replace("%s", "{}").replace("%d", "{}").replace("%#02x", "{:#02x}")     \
               .replace("%#x", "{:#x}").replace("%x", "{:x}").replace("%02X", "{:02x}") \
               .replace("%c", "{:c}").replace("%02x", "{:02x}").replace("%ld", "{}")    \
               .replace("%*s", "").replace("%lX", "{:x}").replace("%08x", "{:08x}")     \
               .replace("%u", "{}").replace("%lu", "{}").replace("%zu", "{}")           \


    def find_string_format(self, s):
        pos = 0
        postString = list()
        frmtString = [i for i, letter in enumerate(s) if letter == '%']

        for i in frmtString:
            if s[i+1] == 's':
                postString.append(pos)
            pos += 1

        return postString


    def get_format_arguments(self, s, args):
        postString = self.find_string_format(self.get_memory_string(s))
        for p in postString:
            args[p] = self.get_memory_string(args[p])
        return args
