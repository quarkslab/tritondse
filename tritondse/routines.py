#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from triton import *


def rtn_exit(se):
    raise(Exception('todo'))


def rtn_puts(se):
    raise(Exception('todo'))


def rtn_libc_start_main(se):
    logging.debug('__libc_start_main hooked')

    # Get arguments
    main = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))

    if se.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
        se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_pc_register(), main)

    elif se.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
        # Push the return value to jump into the main() function
        se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_sp_register(), se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_sp_register())-CPUSIZE.QWORD)

        ret2main = MemoryAccess(se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_sp_register()), CPUSIZE.QWORD)
        se.pstate.tt_ctx.concretizeMemory(ret2main)
        se.pstate.tt_ctx.setConcreteMemoryValue(ret2main, main)

    # Define concrete value
    se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_arg_register(0), se.program.get_argc())

    # Define argc / argv
    base = se.pstate.BASE_ARGV
    addrs = list()

    index = 0
    for argv in se.program.argv:
        addrs.append(base)
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(base, argv+b'\x00')
        for indexCell in range(len(argv)):
            if se.config.symbolize_argv:
                var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(base+indexCell, CPUSIZE.BYTE))
                var.setComment('argv[%d][%d]' %(index, indexCell))
        logging.debug('argv[%d] = %s' %(index, repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(base, len(argv)))))
        base += len(argv)+1
        index += 1

    argv = base
    for addr in addrs:
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(base, CPUSIZE.QWORD), addr)
        base += CPUSIZE.QWORD

    # Concrete value
    se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_arg_register(1), argv)

    return None
