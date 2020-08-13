#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os

from triton             import *
from tritondse.enums    import Enums


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

    # Define concrete value of argc
    argc = len(se.config.program_argv)
    se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_arg_register(0), argc)
    logging.debug('argc = %d' %(argc))

    # Define argv
    base = se.pstate.BASE_ARGV
    addrs = list()

    index = 0
    for argv in se.config.program_argv:
        addrs.append(base)
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(base, argv+b'\x00')
        for indexCell in range(len(argv)):
            if se.config.symbolize_argv:
                var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(base+indexCell, CPUSIZE.BYTE))
                var.setAlias('argv[%d][%d]' %(index, indexCell))
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


def rtn_exit(se):
    logging.debug('exit hooked')
    arg = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    se.pstate.stop = True
    return Enums.CONCRETIZE, arg


def rtn_puts(se):
    logging.debug('puts hooked')

    # Get arguments
    arg0 = se.abi.get_string_argument(0)
    sys.stdout.write(arg0 + '\n')
    sys.stdout.flush()

    # Return value
    return Enums.CONCRETIZE, len(arg0) + 1


def rtn_read(se):
    logging.debug('read hooked')

    # Get arguments
    fd   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    buff = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    if fd == 0 and se.config.symbolize_stdin:
        for index in range(size):
            var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)
            if se.seed:
                try:
                    se.pstate.tt_ctx.setConcreteVariableValue(var, se.seed.content[index])
                except:
                    pass
        logging.debug('stdin = %s' % (repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(buff, size))))
        return Enums.CONCRETIZE, size

    if fd in se.pstate.fd_table:
        if fd == 0:
            data = os.read(0, size)
        else:
            data = os.read(se.pstate.fd_table[fd], size)

        se.pstate.tt_ctx.setConcreteMemoryAreaValue(buff, data)

    else:
        return Enums.CONCRETIZE, 0

    # Return value
    return Enums.CONCRETIZE, len(data)
