#!/usr/bin/env python
# -*- coding: utf-8 -*-

from triton             import TritonContext
from tritondse.config   import Config


class ThreadContext(object):

    def __init__(self, config : Config, tid : int):
        self.cregs  = dict()                    # context of concrete registers
        self.sregs  = dict()                    # context of symbolic registers
        self.joined = None                      # joined thread id
        self.tid    = tid                       # the thread id
        self.killed = False                     # is the thread killed
        self.count  = config.thread_scheduling  # Number of instructions executed until scheduling


    def save(self, tt_ctx : TritonContext):
        # Save symbolic registers
        self.sregs = tt_ctx.getSymbolicRegisters()
        # Save concrete registers
        for r in tt_ctx.getParentRegisters():
            self.cregs.update({r.getId(): tt_ctx.getConcreteRegisterValue(r)})


    def restore(self, tt_ctx : TritonContext):
        # Restore concrete registers
        for rid, v in self.cregs.items():
            tt_ctx.setConcreteRegisterValue(tt_ctx.getRegister(rid), v)
        # Restore symbolic registers
        for rid, e in self.sregs.items():
            tt_ctx.assignSymbolicExpressionToRegister(e, tt_ctx.getRegister(rid))
