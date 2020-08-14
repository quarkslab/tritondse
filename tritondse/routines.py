#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os

from triton             import *
from tritondse.enums    import Enums


def rtn_ctype_b_loc(se):
    logging.debug('__ctype_b_loc hooked')

    ctype  = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"  # must point here
    ctype += b"\x02\x00\x03\x20\x02\x20\x02\x20\x02\x20\x02\x20\x02\x00\x02\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"
    ctype += b"\x01\x60\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8"
    ctype += b"\x08\xd8\x08\xd8\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x02\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00"

    size   = se.pstate.tt_ctx.getGprSize()
    ptable = se.pstate.BASE_CTYPE
    table  = (size * 2) + (se.pstate.BASE_CTYPE)
    otable = table + 256

    se.pstate.ctx_tt.setConcreteMemoryValue(MemoryAccess(ptable + 0x00, size), otable)
    se.pstate.ctx_tt.setConcreteMemoryValue(MemoryAccess(ptable + size, size), 0)

    se.pstate.ctx_tt.setConcreteMemoryAreaValue(table, ctype)

    return Enums.CONCRETIZE, ptable


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
        # TODO
        #for indexCell in range(len(argv)):
        #    if se.config.symbolize_argv:
        #        var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(base+indexCell, CPUSIZE.BYTE))
        #        var.setAlias('argv[%d][%d]' %(index, indexCell))
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


def rtn_fwrite(se):
    logging.debug('fwrite hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3))
    size = arg1 * arg2
    data = se.pstate.tt_ctx.getConcreteMemoryAreaValue(arg0, size)

    if arg3 in se.pstate.fd_table:
        if arg3 == 0:
            return Enums.CONCRETIZE, 0
        elif arg3 == 1:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
        elif arg3 == 2:
            sys.stderr.buffer.write(data)
            sys.stderr.flush()
        else:
            fd = open(se.pstate.fd_table[arg3], 'wb+')
            fd.write(data)
    else:
        return Enums.CONCRETIZE, 0

    # Return value
    return Enums.CONCRETIZE, size


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
    minsize = min(len(se.seed.content), size)

    if fd == 0 and se.config.symbolize_stdin:
        for index in range(minsize):
            var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)
            if se.seed:
                try:
                    se.pstate.tt_ctx.setConcreteVariableValue(var, se.seed.content[index])
                except:
                    pass
        logging.debug('stdin = %s' % (repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(buff, minsize))))
        # TODO: Could return the read value as a symbolic one
        return Enums.CONCRETIZE, minsize

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


def rtn_sem_destroy(se):
    logging.debug('sem_destroy hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_ret_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())

    # Destroy the semaphore with the value
    se.pstate.tt_ctx.setConcreteMemoryValue(mem, 0)

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_getvalue(se):
    logging.debug('sem_getvalue hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # int *sval
    memIn = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())
    memOut = MemoryAccess(arg1, se.pstate.tt_ctx.getGprSize())
    value = se.pstate.tt_ctx.getConcreteMemoryValue(memIn)

    # Set the semaphore's value into the output
    se.pstate.tt_ctx.setConcreteMemoryValue(memOut, value)

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_init(se):
    logging.debug('sem_init hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # int pshared
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))  # unsigned int value
    mem = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())

    # Init the semaphore with the value
    se.pstate.tt_ctx.setConcreteMemoryValue(mem, arg2)

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_post(se):
    logging.debug('sem_post hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem  = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())

    # increments (unlocks) the semaphore pointed to by sem
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)

    se.pstate.tt_ctx.setConcreteMemoryValue(mem, value + 1)

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_timedwait(se):
    logging.debug('sem_timedwait hooked')

    arg0  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg0m = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())
    arg1  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # const struct timespec *abs_timeout
    arg1m = MemoryAccess(arg1, se.pstate.tt_ctx.getGprSize())

    # sem_timedwait() is the same as sem_wait(), except that abs_timeout specifies a limit
    # on the amount of time that the call should block if the decrement cannot be immediately
    # performed. The abs_timeout argument points to a structure that specifies an absolute
    # timeout in seconds and nanoseconds since the Epoch, 1970-01-01 00:00:00 +0000 (UTC).
    # This structure is defined as follows:
    #
    #     struct timespec {
    #         time_t tv_sec;      /* Seconds */
    #         long   tv_nsec;     /* Nanoseconds [0 .. 999999999] */
    #     };
    #
    # If the timeout has already expired by the time of the call, and the semaphore could not be
    # locked immediately, then sem_timedwait() fails with a timeout error (errno set to ETIMEDOUT).
    #
    # If  the operation can be performed immediately, then sem_timedwait() never fails with a
    # timeout error, regardless of the value of abs_timeout.  Furthermore, the validity of
    # abs_timeout is not checked in this case.

    # TODO: Take into account the abs_timeout argument
    value = se.pstate.tt_ctx.getConcreteMemoryValue(arg0m)
    if value > 0:
        logging.debug('semaphore still not locked')
        se.pstate.tt_ctx.setConcreteMemoryValue(arg0m, value - 1)
    else:
        logging.debug('semaphore locked')
        se.pstate.semaphore_locked = True

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_trywait(se):
    logging.debug('sem_trywait hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())

    # sem_trywait()  is  the  same as sem_wait(), except that if the decrement
    # cannot be immediately performed, then call returns an error (errno set to
    # EAGAIN) instead of blocking.
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)
    if value > 0:
        logging.debug('semaphore still not locked')
        se.pstate.tt_ctx.setConcreteMemoryValue(mem, value - 1)
    else:
        logging.debug('semaphore locked')
        return Enums.CONCRETIZE, ((1 << se.pstate.tt_ctx.getGprBitSize()) - 1)

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_sem_wait(se):
    logging.debug('sem_wait hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.tt_ctx.getGprSize())

    # decrements (locks) the semaphore pointed to by sem. If the semaphore's value
    # is greater than zero, then the decrement proceeds, and the function returns,
    # immediately. If the semaphore currently has the value zero, then the call blocks
    # until either it becomes possible to perform the decrement (i.e., the semaphore
    # value rises above zero).
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)
    if value > 0:
        logging.debug('semaphore still not locked')
        se.pstate.tt_ctx.setConcreteMemoryValue(mem, value - 1)
    else:
        logging.debug('semaphore locked')
        se.pstate.semaphore_locked = True

    # Return success
    return Enums.CONCRETIZE, 0


def rtn_strncpy(se):
    logging.debug('strncpy hooked')

    dst = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    cnt = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    # TODO: What if the cnt is symbolic ?
    for index in range(cnt):
        dmem = MemoryAccess(dst + index, 1)
        smem = MemoryAccess(src + index, 1)
        cell = se.pstate.tt_ctx.getMemoryAst(smem)
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "strncpy byte")
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)
        if cell.evaluate() == 0:
            break

    return Enums.CONCRETIZE, dst
