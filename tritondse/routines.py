import logging
import os
import random
import re
import sys
import time

from triton                   import *
from tritondse.types          import ConcSymAction as CS
from tritondse.thread_context import ThreadContext


def rtn_ctype_b_loc(se):
    """
    Pure emulation.
    """
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

    size   = se.pstate.ptr_size
    ptable = se.pstate.BASE_CTYPE
    table  = (size * 2) + (se.pstate.BASE_CTYPE)
    otable = table + 256

    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(ptable + 0x00, size), otable)
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(ptable + size, size), 0)

    # FIXME: On pourrait la renvoyer qu'une seule fois ou la charger au demarage direct dans pstate
    se.pstate.tt_ctx.setConcreteMemoryAreaValue(table, ctype)

    return CS.CONCRETIZE, ptable


def rtn_errno_location(se):
    """
    Pure emulation.
    """
    logging.debug('__errno_location hooked')

    # Errno is a int* ptr
    # Initialize it to zero
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(se.pstate.ERRNO_PTR, CPUSIZE.DWORD), 0)

    return CS.CONCRETIZE, se.pstate.ERRNO_PTR


def rtn_libc_start_main(se):
    logging.debug('__libc_start_main hooked')

    # Get arguments
    main = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))

    if se.pstate.tt_ctx.getArchitecture() == ARCH.AARCH64:
        se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_pc_register(), main)

    elif se.pstate.tt_ctx.getArchitecture() == ARCH.X86_64:
        # Push the return value to jump into the main() function
        se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_sp_register(), se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_sp_register()) - CPUSIZE.QWORD)

        ret2main = MemoryAccess(se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_sp_register()), CPUSIZE.QWORD)
        se.pstate.tt_ctx.concretizeMemory(ret2main)
        se.pstate.tt_ctx.setConcreteMemoryValue(ret2main, main)

    # Define concrete value of argc
    argc = len(se.config.program_argv)
    se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_arg_register(0), argc)
    logging.debug('argc = %d' % (argc))

    # Define argv
    base = se.pstate.BASE_ARGV
    addrs = list()

    index = 0
    for argv in se.config.program_argv:
        b_argv = argv.encode("latin-1")
        addrs.append(base)
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(base, b_argv + b'\x00')
        # TODO: Si symbolize_args is True
        #for indexCell in range(len(argv)):
        #    if se.config.symbolize_argv:
        #        var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(base+indexCell, CPUSIZE.BYTE))
        #        var.setAlias('argv[%d][%d]' %(index, indexCell))
        logging.debug('argv[%d] = %s' % (index, repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(base, len(b_argv)))))
        base += len(b_argv) + 1
        index += 1

    b_argv = base
    for addr in addrs:
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(base, CPUSIZE.QWORD), addr)
        base += CPUSIZE.QWORD

    # Concrete value
    se.pstate.tt_ctx.setConcreteRegisterValue(se.abi.get_arg_register(1), b_argv)

    return None


def rtn_stack_chk_fail(se):
    """
    Pure emulation.
    """
    logging.debug('__stack_chk_fail hooked')
    se.pstate.stop = True
    return None


# int __xstat(int ver, const char* path, struct stat* stat_buf);
def rtn_xstat(se):
    """
    Pure emulation.
    """
    logging.debug('__xstat hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # int ver
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # const char* path
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))  # struct stat* stat_buf

    if os.path.isfile(se.abi.get_memory_string(arg1)):
        stat = os.stat(se.abi.get_memory_string(arg1))
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x00, CPUSIZE.QWORD), stat.st_dev)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x08, CPUSIZE.QWORD), stat.st_ino)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x10, CPUSIZE.QWORD), stat.st_nlink)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x18, CPUSIZE.DWORD), stat.st_mode)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x1c, CPUSIZE.DWORD), stat.st_uid)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x20, CPUSIZE.DWORD), stat.st_gid)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x24, CPUSIZE.DWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x28, CPUSIZE.QWORD), stat.st_rdev)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x30, CPUSIZE.QWORD), stat.st_size)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x38, CPUSIZE.QWORD), stat.st_blksize)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x40, CPUSIZE.QWORD), stat.st_blocks)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x48, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x50, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x58, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x60, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x68, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x70, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x78, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x80, CPUSIZE.QWORD), 0)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg2 + 0x88, CPUSIZE.QWORD), 0)
        return CS.CONCRETIZE, 0

    return CS.CONCRETIZE, se.pstate.minus_one


# int atoi(const char *nptr);
def rtn_atoi(se):
    """ Simulate the atoi() function """
    logging.debug('atoi hooked')

    ast = se.pstate.tt_ctx.getAstContext()
    arg = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))

    cells = {
        0: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 0, 1)),
        1: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 1, 1)),
        2: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 2, 1)),
        3: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 3, 1)),
        4: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 4, 1)),
        5: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 5, 1)),
        6: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 6, 1)),
        7: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 7, 1)),
        8: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 8, 1)),
        9: se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg + 9, 1))
    }

    # FIXME: Does not support negative value and all other corner cases.
    # "000000000012372183762173"
    # "         98273483274"
    # "-123123"
    # "18273213"

    def multiply(ast, cells, index):
        n = ast.bv(0, 32)
        for i in range(index):
            n = n * 10 + (ast.zx(24, cells[i]) - 0x30)
        return n

    res = ast.ite(
              ast.lnot(ast.land([cells[0] >= 0x30, cells[0] <= 0x39])),
              multiply(ast, cells, 0),
              ast.ite(
                  ast.lnot(ast.land([cells[1] >= 0x30, cells[1] <= 0x39])),
                  multiply(ast, cells, 1),
                  ast.ite(
                      ast.lnot(ast.land([cells[2] >= 0x30, cells[2] <= 0x39])),
                      multiply(ast, cells, 2),
                      ast.ite(
                          ast.lnot(ast.land([cells[3] >= 0x30, cells[3] <= 0x39])),
                          multiply(ast, cells, 3),
                          ast.ite(
                              ast.lnot(ast.land([cells[4] >= 0x30, cells[4] <= 0x39])),
                              multiply(ast, cells, 4),
                              ast.ite(
                                  ast.lnot(ast.land([cells[5] >= 0x30, cells[5] <= 0x39])),
                                  multiply(ast, cells, 5),
                                  ast.ite(
                                      ast.lnot(ast.land([cells[6] >= 0x30, cells[6] <= 0x39])),
                                      multiply(ast, cells, 6),
                                      ast.ite(
                                          ast.lnot(ast.land([cells[7] >= 0x30, cells[7] <= 0x39])),
                                          multiply(ast, cells, 7),
                                          ast.ite(
                                              ast.lnot(ast.land([cells[8] >= 0x30, cells[8] <= 0x39])),
                                              multiply(ast, cells, 8),
                                              ast.ite(
                                                  ast.lnot(ast.land([cells[9] >= 0x30, cells[9] <= 0x39])),
                                                  multiply(ast, cells, 9),
                                                  multiply(ast, cells, 9)
                                              )
                                          )
                                      )
                                  )
                              )
                          )
                      )
                  )
              )
          )
    res = ast.sx(32, res)

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "atoi summary")

    return CS.SYMBOLIZE, expr


# void *calloc(size_t nmemb, size_t size);
def rtn_calloc(se):
    logging.debug('calloc hooked')

    # Get arguments
    nmemb = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    size  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    # We use nmemb and size as concret values
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)) == nmemb)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == size)

    if nmemb == 0 or size == 0:
        ptr = 0
    else:
        ptr = se.pstate.heap_allocator.alloc(nmemb * size)

    # Return value
    return CS.CONCRETIZE, ptr


# int clock_gettime(clockid_t clockid, struct timespec *tp);
def rtn_clock_gettime(se):
    logging.debug('clock_gettime hooked')

    # Get arguments
    clockid = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    tp      = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    # We use tp as concret value
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == tp)

    # FIXME: We can return something logic
    if tp == 0:
        return CS.CONCRETIZE, se.pstate.minus_one

    if se.config.time_inc_coefficient:
        t = se.pstate.time
    else:
        t = time.time()

    s = se.pstate.ptr_size
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(tp, s), int(t))
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(tp + s, s), int(t * 1000000))

    # Return value
    return CS.CONCRETIZE, 0


def rtn_exit(se):
    logging.debug('exit hooked')
    arg = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    se.pstate.stop = True
    return CS.CONCRETIZE, arg


def rtn_fclose(se):
    logging.debug('fclose hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0)) # fd

    # We use fd as concret value
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)) == arg0)

    if arg0 in se.pstate.fd_table:
        se.pstate.fd_table[arg0].close()
        del se.pstate.fd_table[arg0]
    else:
        return CS.CONCRETIZE, se.pstate.minus_one

    # Return value
    return CS.CONCRETIZE, 0


# char *fgets(char *s, int size, FILE *stream);
def rtn_fgets(se):
    logging.debug('fgets hooked')

    # Get arguments
    buff     = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    buff_ast = se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0))
    size     = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    fd       = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    minsize  = (min(len(se.seed.content), size) if se.seed else size)

    # We use fd as concret value
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == fd)

    if fd == 0 and se.config.symbolize_stdin:
        # We use fd as concret value
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == minsize)

        if se.seed:
            se.pstate.tt_ctx.setConcreteMemoryAreaValue(buff, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug('stdin = %s' % (repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(buff, minsize))))
        return CS.SYMBOLIZE, se.pstate.tt_ctx.newSymbolicExpression(buff_ast)

    if fd in se.pstate.fd_table:
        # We use fd as concret value
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == size)
        data = (os.read(0, size) if fd == 0 else os.read(se.pstate.fd_table[fd], size))
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(buff, data)
        return CS.SYMBOLIZE, se.pstate.tt_ctx.newSymbolicExpression(buff_ast)
    else:
        logging.warning(f'File descriptor ({fd}) not found')

    return CS.CONCRETIZE, 0


# fopen(const char *pathname, const char *mode);
def rtn_fopen(se):
    logging.debug('fopen hooked')

    # Get arguments
    arg0  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # const char *pathname
    arg1  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # const char *mode
    arg0s = se.abi.get_memory_string(arg0)
    arg1s = se.abi.get_memory_string(arg1)

    # We use pathname as concrete value
    for i, c in enumerate(arg0s):
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg0 + i, CPUSIZE.BYTE)) == ord(c))

    # Ensure terminating with \0
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(arg0 + len(arg0s), CPUSIZE.BYTE)) == 0x00)

    # We use mode as concrete value
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == arg1)

    fd = open(arg0s, arg1s)
    fd_id = se.pstate.get_unique_file_id()
    se.pstate.fd_table.update({fd_id: fd})

    # Return value
    return CS.CONCRETIZE, fd_id


def rtn_fprintf(se):
    logging.debug('fprintf hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3))
    arg4 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(4))
    arg5 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(5))
    arg6 = se.abi.get_stack_value(0)
    arg7 = se.abi.get_stack_value(1)
    arg8 = se.abi.get_stack_value(2)

    # FIXME: ARM64
    # FIXME: pushPathConstraint

    arg1f = se.abi.get_format_string(arg1)
    nbArgs = arg1f.count("{")
    args = se.abi.get_format_arguments(arg1, [arg2, arg3, arg4, arg5, arg6, arg7, arg8][:nbArgs])
    s = arg1f.format(*args)

    if arg0 in se.pstate.fd_table:
        se.pstate.fd_table[arg0].write(s)
        se.pstate.fd_table[arg0].flush()
    else:
        return CS.CONCRETIZE, 0

    # Return value
    return CS.CONCRETIZE, len(s)


# fputc(int c, FILE *stream);
def rtn_fputc(se):
    logging.debug('fputc hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)) == arg0)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == arg1)

    if arg1 in se.pstate.fd_table:
        if arg1 == 0:
            return CS.CONCRETIZE, 0
        elif arg1 == 1:
            sys.stdout.write(chr(arg0))
            sys.stdout.flush()
        elif arg1 == 2:
            sys.stderr.write(chr(arg0))
            sys.stderr.flush()
        else:
            fd = open(se.pstate.fd_table[arg1], 'wb+')
            fd.write(chr(arg0))
    else:
        return CS.CONCRETIZE, 0

    # FIXME: We can iterate over all fd_tables and do the disjunction of all available fd

    # Return value
    return CS.CONCRETIZE, 1


def rtn_fputs(se):
    logging.debug('fputs hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)) == arg0)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)) == arg1)

    # FIXME: What if the fd is coming from the memory (fmemopen) ?

    if arg1 in se.pstate.fd_table:
        if arg1 == 0:
            return CS.CONCRETIZE, 0
        elif arg1 == 1:
            sys.stdout.write(se.abi.get_memory_string(arg0))
            sys.stdout.flush()
        elif arg1 == 2:
            sys.stderr.write(se.abi.get_memory_string(arg0))
            sys.stderr.flush()
        else:
            fd = open(se.pstate.fd_table[arg1], 'wb+')
            fd.write(se.abi.get_memory_string(arg0))
    else:
        return CS.CONCRETIZE, 0

    # Return value
    return CS.CONCRETIZE, len(se.abi.get_memory_string(arg0))


def rtn_fread(se):
    logging.debug('fread hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0)) # ptr
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1)) # size
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2)) # nmemb
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3)) # stream
    size = arg1 * arg2

    minsize = (min(len(se.seed.content), size) if se.seed else size)

    # FIXME: pushPathConstraint

    if arg3 == 0 and se.config.symbolize_stdin:
        if se.seed:
            se.pstate.tt_ctx.setConcreteMemoryAreaValue(arg0, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(arg0 + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug('stdin = %s' % (repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(arg0, minsize))))
        # TODO: Could return the read value as a symbolic one
        return CS.CONCRETIZE, minsize

    elif arg3 in se.pstate.fd_table:
        data = se.pstate.fd_table[arg3].read(arg1 * arg2)
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(arg0, data)

    else:
        return CS.CONCRETIZE, 0

    # Return value
    return CS.CONCRETIZE, len(data)


def rtn_free(se):
    logging.debug('free hooked')

    # Get arguments
    ptr = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    se.pstate.heap_allocator.free(ptr)

    return None


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
            return CS.CONCRETIZE, 0
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
        return CS.CONCRETIZE, 0

    # Return value
    return CS.CONCRETIZE, size


def rtn_gettimeofday(se):
    logging.debug('gettimeofday hooked')

    # Get arguments
    tv = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    tz = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    if tv == 0:
        return CS.CONCRETIZE, ((1 << se.pstate.ptr_bit_size) - 1)

    if se.config.time_inc_coefficient:
        t = se.pstate.time
    else:
        t = time.time()

    s = se.pstate.ptr_size
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(tv, s), int(t))
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(tv + s, s), int(t * 1000000))

    # Return value
    return CS.CONCRETIZE, 0


def rtn_malloc(se):
    logging.debug('malloc hooked')

    # Get arguments
    size = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    ptr  = se.pstate.heap_allocator.alloc(size)

    # Return value
    return CS.CONCRETIZE, ptr


def rtn_memcmp(se):
    logging.debug('memcmp hooked')

    s1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    s2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    ast = se.pstate.tt_ctx.getAstContext()
    res = ast.bv(0, 64)

    # We constrain the logical value of size
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == size)

    for index in range(size):
        cells1 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + index, 1))
        cells2 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + index, 1))
        res = res + ast.ite(
                        cells1 == cells2,
                        ast.bv(0, 64),
                        ast.bv(1, 64)
                    )

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "memcmp summary")

    return CS.SYMBOLIZE, expr


def rtn_memcpy(se):
    logging.debug('memcpy hooked')

    dst = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    cnt = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    # We constrain the logical value of size
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == cnt)

    for index in range(cnt):
        dmem  = MemoryAccess(dst + index, CPUSIZE.BYTE)
        smem  = MemoryAccess(src + index, CPUSIZE.BYTE)
        cell = se.pstate.tt_ctx.getMemoryAst(smem)
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "memcpy byte")
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)

    return CS.SYMBOLIZE, se.pstate.tt_ctx.newSymbolicExpression(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)))


def rtn_memmem(se):
    logging.debug('memmem hooked')

    haystack    = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # const void*
    haystacklen = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # size_t
    needle      = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))  # const void *
    needlelen   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3))  # size_t

    s1 = se.pstate.tt_ctx.getConcreteMemoryAreaValue(haystack, haystacklen)  # haystack
    s2 = se.pstate.tt_ctx.getConcreteMemoryAreaValue(needle, needlelen)      # needle

    offset = s1.find(s2)
    if offset == -1:
        #FIXME: faut s'assurer que le marquer dans le string
        return CS.CONCRETIZE, 0

    for i, c in enumerate(s2):
        se.pstate.tt_ctx.pushPathConstraint(
            se.pstate.tt_ctx.getMemoryAst(MemoryAccess(haystack + offset + i, CPUSIZE.BYTE))
            ==
            se.pstate.tt_ctx.getMemoryAst(MemoryAccess(needle + i, CPUSIZE.BYTE))
        )

    # FIXME: à reflechir si on doit contraindre offset ou pas

    # faut s'assurer que le marquer est bien présent à l'offset trouvé
    return CS.CONCRETIZE, haystack + offset


def rtn_memmove(se):
    logging.debug('memmove hooked')

    dst = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    cnt = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    # We constrain the logical value of cnt
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == cnt)

    src_cells = []
    # TODO: What if cnt is symbolic ?
    for index in range(cnt):
        smem  = MemoryAccess(src + index, CPUSIZE.BYTE)
        cell = se.pstate.tt_ctx.getMemoryAst(smem)
        src_cells.append(cell)

    for index in range(cnt):
        dmem  = MemoryAccess(dst + index, CPUSIZE.BYTE)
        cell = src_cells.pop(0)
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "memmove byte")
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)

    return CS.SYMBOLIZE, se.pstate.tt_ctx.newSymbolicExpression(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)))


def rtn_memset(se):
    logging.debug('memset hooked')

    dst = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    # We constrain the logical value of size
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == size)

    # TODO: What if size is symbolic ?
    for index in range(size):
        dmem = MemoryAccess(dst + index, CPUSIZE.BYTE)
        cell = se.pstate.tt_ctx.getAstContext().extract(7, 0, se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(1)))
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "memset byte")
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)

    return CS.SYMBOLIZE, se.pstate.tt_ctx.newSymbolicExpression(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)))


def rtn_printf(se):
    logging.debug('printf hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3))
    arg4 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(4))
    arg5 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(5))
    arg6 = se.abi.get_stack_value(0)
    arg7 = se.abi.get_stack_value(1)
    arg8 = se.abi.get_stack_value(2)

    arg0f = se.abi.get_format_string(arg0)
    nbArgs = arg0f.count("{")
    args = se.abi.get_format_arguments(arg0, [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8][:nbArgs])
    s = arg0f.format(*args)

    se.pstate.fd_table[1].write(s)
    se.pstate.fd_table[1].flush()

    # Return value
    return CS.CONCRETIZE, len(s)


def rtn_pthread_create(se):
    logging.debug('pthread_create hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0)) # pthread_t *thread
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1)) # const pthread_attr_t *attr
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2)) # void *(*start_routine) (void *)
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3)) # void *arg

    tid = se.pstate.get_unique_thread_id()
    thread = ThreadContext(se.config, tid)
    thread.save(se.pstate.tt_ctx)

    # Concretize pc
    if se.abi.get_pc_register().getId() in thread.sregs:
        del thread.sregs[se.abi.get_pc_register().getId()]

    # Concretize bp
    if se.abi.get_bp_register().getId() in thread.sregs:
        del thread.sregs[se.abi.get_bp_register().getId()]

    # Concretize sp
    if se.abi.get_sp_register().getId() in thread.sregs:
        del thread.sregs[se.abi.get_sp_register().getId()]

    # Concretize arg0
    if se.abi.get_arg_register(0).getId() in thread.sregs:
        del thread.sregs[se.abi.get_arg_register(0).getId()]

    thread.cregs[se.abi.get_pc_register().getId()] = arg2
    thread.cregs[se.abi.get_arg_register(0).getId()] = arg3
    thread.cregs[se.abi.get_bp_register().getId()] = (se.pstate.BASE_STACK - ((1 << 28) * tid))
    thread.cregs[se.abi.get_sp_register().getId()] = (se.pstate.BASE_STACK - ((1 << 28) * tid))

    se.pstate.threads.update({tid: thread})

    # Save out the thread id
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg0, se.pstate.ptr_size), tid)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_pthread_exit(se):
    logging.debug('pthread_exit hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))

    # Kill the thread
    se.pstate.threads[se.pstate.tid].killed = True

    # Return value
    return None


def rtn_pthread_join(se):
    logging.debug('pthread_join hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))

    if arg0 in se.pstate.threads:
        se.pstate.threads[se.pstate.tid].joined = arg0
        logging.info('Thread id %d joined thread id %d' % (se.pstate.tid, arg0))
    else:
        se.pstate.threads[se.pstate.tid].joined = None
        logging.debug('Thread id %d already destroyed' % arg0)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_pthread_mutex_destroy(se):
    logging.debug('pthread_mutex_destroy hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # pthread_mutex_t *restrict mutex
    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg0, se.pstate.ptr_size), se.pstate.PTHREAD_MUTEX_INIT_MAGIC)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_pthread_mutex_init(se):
    logging.debug('pthread_mutex_init hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # pthread_mutex_t *restrict mutex
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # const pthread_mutexattr_t *restrict attr)

    se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(arg0, se.pstate.ptr_size), se.pstate.PTHREAD_MUTEX_INIT_MAGIC)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_pthread_mutex_lock(se):
    logging.debug('pthread_mutex_lock hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # pthread_mutex_t *mutex
    mem = MemoryAccess(arg0, se.pstate.ptr_size)
    mutex = se.pstate.tt_ctx.getConcreteMemoryValue(mem)

    # If the thread has been initialized and unused, define the tid has lock
    if mutex == se.pstate.PTHREAD_MUTEX_INIT_MAGIC:
        logging.debug('mutex unlocked')
        se.pstate.tt_ctx.setConcreteMemoryValue(mem, se.pstate.tid)

    # The mutex is locked and we are not allowed to continue the execution
    elif mutex != se.pstate.tid:
        logging.debug('mutex locked')
        se.pstate.mutex_locked = True

    # Return value
    return CS.CONCRETIZE, 0


def rtn_pthread_mutex_unlock(se):
    logging.debug('pthread_mutex_unlock hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # pthread_mutex_t *mutex
    mem = MemoryAccess(arg0, se.pstate.ptr_size)

    se.pstate.tt_ctx.setConcreteMemoryValue(mem, se.pstate.PTHREAD_MUTEX_INIT_MAGIC)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_puts(se):
    logging.debug('puts hooked')

    # Get arguments
    arg0 = se.abi.get_string_argument(0)
    sys.stdout.write(arg0 + '\n')
    sys.stdout.flush()

    # Return value
    return CS.CONCRETIZE, len(arg0) + 1


def rtn_rand(se):
    logging.debug('rand hooked')
    return Enums.CONCRETIZE, random.randrange(0, 0xffffffff)


def rtn_read(se):
    logging.debug('read hooked')

    # Get arguments
    fd   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    buff = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    minsize = (min(len(se.seed.content), size) if se.seed else size)

    if se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)).isSymbolized():
        logging.warning(f'Reading from the file descriptor ({fd}) with a symbolic size')

    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(0)) == fd)

    if fd == 0 and se.config.symbolize_stdin:
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == minsize)
        if se.seed:
            se.pstate.tt_ctx.setConcreteMemoryAreaValue(buff, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = se.pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug('stdin = %s' % (repr(se.pstate.tt_ctx.getConcreteMemoryAreaValue(buff, minsize))))
        # TODO: Could return the read value as a symbolic one
        return CS.CONCRETIZE, minsize

    if fd in se.pstate.fd_table:
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getRegisterAst(se.abi.get_arg_register(2)) == size)
        data = (os.read(0, size) if fd == 0 else os.read(se.pstate.fd_table[fd], size))
        se.pstate.tt_ctx.setConcreteMemoryAreaValue(buff, data)
        return CS.CONCRETIZE, len(data)

    return CS.CONCRETIZE, 0


def rtn_sem_destroy(se):
    logging.debug('sem_destroy hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_ret_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.ptr_size)

    # Destroy the semaphore with the value
    se.pstate.tt_ctx.setConcreteMemoryValue(mem, 0)

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_getvalue(se):
    logging.debug('sem_getvalue hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # int *sval
    memIn = MemoryAccess(arg0, se.pstate.ptr_size)
    memOut = MemoryAccess(arg1, CPUSIZE.DWORD)
    value = se.pstate.tt_ctx.getConcreteMemoryValue(memIn)

    # Set the semaphore's value into the output
    se.pstate.tt_ctx.setConcreteMemoryValue(memOut, value)

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_init(se):
    logging.debug('sem_init hooked')

    # Get arguments
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # int pshared
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))  # unsigned int value
    mem = MemoryAccess(arg0, se.pstate.ptr_size)

    # Init the semaphore with the value
    se.pstate.tt_ctx.setConcreteMemoryValue(mem, arg2)

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_post(se):
    logging.debug('sem_post hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem  = MemoryAccess(arg0, se.pstate.ptr_size)

    # increments (unlocks) the semaphore pointed to by sem
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)

    se.pstate.tt_ctx.setConcreteMemoryValue(mem, value + 1)

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_timedwait(se):
    logging.debug('sem_timedwait hooked')

    arg0  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    arg0m = MemoryAccess(arg0, se.pstate.ptr_size)
    arg1  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))  # const struct timespec *abs_timeout
    arg1m = MemoryAccess(arg1, se.pstate.ptr_size)

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
        se.pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked')
        se.pstate.semaphore_locked = True

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_trywait(se):
    logging.debug('sem_trywait hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.ptr_size)

    # sem_trywait()  is  the  same as sem_wait(), except that if the decrement
    # cannot be immediately performed, then call returns an error (errno set to
    # EAGAIN) instead of blocking.
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)
    if value > 0:
        logging.debug('semaphore still not locked')
        se.pstate.tt_ctx.setConcreteMemoryValue(mem, value - 1)
        se.pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked but continue')
        se.pstate.semaphore_locked = False
        # Setting errno to EAGAIN (3406)
        se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(se.pstate.ERRNO_PTR, CPUSIZE.DWORD), 3406)
        # Return -1
        return CS.CONCRETIZE, ((1 << se.pstate.ptr_bit_size) - 1)

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sem_wait(se):
    logging.debug('sem_wait hooked')

    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))  # sem_t *sem
    mem = MemoryAccess(arg0, se.pstate.ptr_size)

    # decrements (locks) the semaphore pointed to by sem. If the semaphore's value
    # is greater than zero, then the decrement proceeds, and the function returns,
    # immediately. If the semaphore currently has the value zero, then the call blocks
    # until either it becomes possible to perform the decrement (i.e., the semaphore
    # value rises above zero).
    value = se.pstate.tt_ctx.getConcreteMemoryValue(mem)
    if value > 0:
        logging.debug('semaphore still not locked')
        se.pstate.tt_ctx.setConcreteMemoryValue(mem, value - 1)
        se.pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked')
        se.pstate.semaphore_locked = True

    # Return success
    return CS.CONCRETIZE, 0


def rtn_sleep(se):
    logging.debug('sleep hooked')

    # Get arguments
    t = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    #time.sleep(t)

    # Return value
    return CS.CONCRETIZE, 0


def rtn_sprintf(se):
    logging.debug('sprintf hooked')

    # Get arguments
    buff = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    arg0 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    arg1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    arg2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(3))
    arg3 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(4))
    arg4 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(5))
    arg5 = se.abi.get_stack_value(0)
    arg6 = se.abi.get_stack_value(1)
    arg7 = se.abi.get_stack_value(2)

    arg0f = se.abi.get_format_string(arg0)
    nbArgs = arg0f.count("{")
    args = se.abi.get_format_arguments(arg0, [arg1, arg2, arg3, arg4, arg5, arg6, arg7][:nbArgs])
    s = arg0f.format(*args)

    # FIXME: todo

    index = 0
    for c in s:
        se.pstate.tt_ctx.concretizeMemory(buff + index)
        se.pstate.tt_ctx.setConcreteMemoryValue(buff + index, ord(c))
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + index, 1)) == ord(c))
        index += 1

    # including the terminating null byte ('\0')
    se.pstate.tt_ctx.setConcreteMemoryValue(buff + len(s), 0x00)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + len(s), 1)) == 0x00)

    return CS.CONCRETIZE, len(s)


def rtn_strcasecmp(se):
    logging.debug('strcasecmp hooked')

    s1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    s2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = min(len(se.abi.get_memory_string(s1)), len(se.abi.get_memory_string(s2)))

    s = s1 if len(se.abi.get_memory_string(s1)) < len(se.abi.get_memory_string(s2)) else s2
    for i in range(size):
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)

    ast = se.pstate.tt_ctx.getAstContext()
    res = ast.bv(0, se.pstate.ptr_bit_size)
    for index in range(size):
        cells1 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + index, 1))
        cells2 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + index, 1))
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strcasecmp summary")

    return CS.SYMBOLIZE, expr


def rtn_strchr(se):
    logging.debug('strchr hooked')

    string = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    char   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    ast    = se.pstate.tt_ctx.getAstContext()

    def rec(res, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(string + deep, 1))
        res  = ast.ite(cell == (char & 0xff), ast.bv(string + deep, 64), rec(res, deep + 1, maxdeep))
        return res

    sze = len(se.abi.get_memory_string(string))
    res = rec(ast.bv(0, 64), 0, sze)

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strchr summary")

    return CS.SYMBOLIZE, expr


def rtn_strcmp(se):
    logging.debug('strcmp hooked')

    s1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    s2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = min(len(se.abi.get_memory_string(s1)), len(se.abi.get_memory_string(s2)))

    s = s1 if len(se.abi.get_memory_string(s1)) < len(se.abi.get_memory_string(s2)) else s2
    for i in range(size):
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
        se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)

    ast = se.pstate.tt_ctx.getAstContext()
    res = ast.bv(0, 64)
    for index in range(size):
        cells1 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + index, 1))
        cells2 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + index, 1))
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strcmp summary")

    return CS.SYMBOLIZE, expr


def rtn_strcpy(se):
    logging.debug('strcpy hooked')

    dst  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    size = len(se.abi.get_memory_string(src))

    for index in range(size):
        dmem = MemoryAccess(dst + index, CPUSIZE.BYTE)
        smem = MemoryAccess(src + index, CPUSIZE.BYTE)
        cell = se.pstate.tt_ctx.getMemoryAst(smem)
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "strcpy byte")
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)

    # including the terminating null byte ('\0')
    se.pstate.tt_ctx.setConcreteMemoryValue(dst + size, 0x00)
    se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(MemoryAccess(dst + size, 1)) == 0)

    return CS.CONCRETIZE, dst


def rtn_strlen(se):
    logging.debug('strlen hooked')

    # Get arguments
    s = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    ast = se.pstate.tt_ctx.getAstContext()

    def rec(res, s, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s + deep, 1))
        res  = ast.ite(cell == 0x00, ast.bv(deep, 64), rec(res, s, deep + 1, maxdeep))
        return res

    sze = len(se.abi.get_memory_string(s))
    res = ast.bv(sze, 64)
    res = rec(res, s, 0, sze)

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strlen summary")

    return CS.SYMBOLIZE, expr


def rtn_strncasecmp(se):
    logging.debug('strncasecmp hooked')

    s1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    s2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    sz = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    maxlen = min(sz, min(len(se.abi.get_memory_string(s1)), len(se.abi.get_memory_string(s2))) + 1)

    ast = se.pstate.tt_ctx.getAstContext()
    res = ast.bv(0, se.pstate.ptr_bit_size)
    for index in range(maxlen):
        cells1 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + index, 1))
        cells2 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + index, 1))
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strncasecmp summary")

    return CS.SYMBOLIZE, expr


def rtn_strncmp(se):
    logging.debug('strncmp hooked')

    s1 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    s2 = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    sz = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    maxlen = min(sz, min(len(se.abi.get_memory_string(s1)), len(se.abi.get_memory_string(s2))) + 1)

    ast = se.pstate.tt_ctx.getAstContext()
    res = ast.bv(0, 64)
    for index in range(maxlen):
        cells1 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + index, 1))
        cells2 = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + index, 1))
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    # create a new symbolic expression for this summary
    expr = se.pstate.tt_ctx.newSymbolicExpression(res, "strncmp summary")

    return CS.SYMBOLIZE, expr


def rtn_strncpy(se):
    logging.debug('strncpy hooked')

    dst = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    src = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    cnt = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    # TODO: What if the cnt is symbolic ?
    for index in range(cnt):
        dmem = MemoryAccess(dst + index, CPUSIZE.BYTE)
        smem = MemoryAccess(src + index, CPUSIZE.BYTE)
        cell = se.pstate.tt_ctx.getMemoryAst(smem)
        expr = se.pstate.tt_ctx.newSymbolicExpression(cell, "strncpy byte")
        se.pstate.tt_ctx.setConcreteMemoryValue(dmem, cell.evaluate())
        se.pstate.tt_ctx.assignSymbolicExpressionToMemory(expr, dmem)
        #se.pstate.tt_ctx.pushPathConstraint(se.pstate.tt_ctx.getMemoryAst(dmem) == cell)
        if cell.evaluate() == 0:
            break

    return CS.CONCRETIZE, dst


def rtn_strtok_r(se):
    logging.debug('strtok_r hooked')

    string  = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    delim   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    saveptr = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))
    saveMem = se.pstate.tt_ctx.getConcreteMemoryValue(MemoryAccess(saveptr, se.pstate.ptr_size))
    actx    = se.pstate.tt_ctx.getAstContext()

    if string == 0:
        string = saveMem

    d = se.abi.get_memory_string(delim)
    s = se.abi.get_memory_string(string)

    tokens = re.split('[' + re.escape(d) + ']', s)

    # TODO: Make it symbolic
    for token in tokens:
        if token:
            offset = s.find(token)
            # Init the \0 at the delimiter position
            node = se.pstate.tt_ctx.getMemoryAst(MemoryAccess(string + offset + len(token), 1))
            try:
                se.pstate.tt_ctx.pushPathConstraint(actx.lor([node == ord(c) for c in d]))
            except:
                se.pstate.tt_ctx.pushPathConstraint(node == ord(d))
            se.pstate.tt_ctx.setConcreteMemoryValue(string + offset + len(token), 0)
            # Save the pointer
            se.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(saveptr, se.pstate.ptr_size), string + offset + len(token) + 1)
            # Return the token
            return CS.CONCRETIZE, string + offset

    return CS.CONCRETIZE, 0


def rtn_strtoul(se):
    logging.debug('strtoul hooked')

    nptr   = se.abi.get_string_argument(0)
    endptr = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    base   = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(2))

    # TODO: Make it symbolic

    try:
        return CS.CONCRETIZE, int(nptr, base)
    except:
        return CS.CONCRETIZE, 0xffffffff



SUPPORTED_ROUTINES = {
    # TODO:
    #   - tolower
    #   - toupper
    '__ctype_b_loc':           rtn_ctype_b_loc,
    '__errno_location':        rtn_errno_location,
    '__libc_start_main':       rtn_libc_start_main,
    '__stack_chk_fail':        rtn_stack_chk_fail,
    '__xstat':                 rtn_xstat,
    'atoi':                    rtn_atoi,
    'calloc':                  rtn_calloc,
    'clock_gettime':           rtn_clock_gettime,
    'exit':                    rtn_exit,
    'fclose':                  rtn_fclose,
    'fgets':                   rtn_fgets,
    'fopen':                   rtn_fopen,
    'fprintf':                 rtn_fprintf,
    'fputc':                   rtn_fputc,
    'fputs':                   rtn_fputs,
    'fread':                   rtn_fread,
    'free':                    rtn_free,
    'fwrite':                  rtn_fwrite,
    'gettimeofday':            rtn_gettimeofday,
    'malloc':                  rtn_malloc,
    'memcmp':                  rtn_memcmp,
    'memcpy':                  rtn_memcpy,
    'memmem':                  rtn_memmem,
    'memmove':                 rtn_memmove,
    'memset':                  rtn_memset,
    'printf':                  rtn_printf,
    'pthread_create':          rtn_pthread_create,
    'pthread_exit':            rtn_pthread_exit,
    'pthread_join':            rtn_pthread_join,
    'pthread_mutex_destroy':   rtn_pthread_mutex_destroy,
    'pthread_mutex_init':      rtn_pthread_mutex_init,
    'pthread_mutex_lock':      rtn_pthread_mutex_lock,
    'pthread_mutex_unlock':    rtn_pthread_mutex_unlock,
    'puts':                    rtn_puts,
    'rand':                    rtn_rand,
    'read':                    rtn_read,
    'sem_destroy':             rtn_sem_destroy,
    'sem_getvalue':            rtn_sem_getvalue,
    'sem_init':                rtn_sem_init,
    'sem_post':                rtn_sem_post,
    'sem_timedwait':           rtn_sem_timedwait,
    'sem_trywait':             rtn_sem_trywait,
    'sem_wait':                rtn_sem_wait,
    'sleep':                   rtn_sleep,
    'sprintf':                 rtn_sprintf,
    'strcasecmp':              rtn_strcasecmp,
    'strchr':                  rtn_strchr,
    'strcmp':                  rtn_strcmp,
    'strcpy':                  rtn_strcpy,
    'strlen':                  rtn_strlen,
    'strncasecmp':             rtn_strncasecmp,
    'strncmp':                 rtn_strncmp,
    'strncpy':                 rtn_strncpy,
    'strtok_r':                rtn_strtok_r,
    'strtoul':                 rtn_strtoul,
}


SUPORTED_GVARIABLES = {
    '__stack_chk_guard':    0xdead,
    'stderr':               0x0002,
    'stdin':                0x0000,
    'stdout':               0x0001,
}
