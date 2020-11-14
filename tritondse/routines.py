import logging
import os
import random
import re
import sys
import time

from triton                   import CPUSIZE, MemoryAccess
from tritondse.thread_context import ThreadContext
from tritondse.types          import Architecture
from tritondse.seed           import SeedStatus


def rtn_ctype_b_loc(se, pstate):
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

    size   = pstate.ptr_size
    ptable = pstate.BASE_CTYPE
    table  = (size * 2) + pstate.BASE_CTYPE
    otable = table + 256

    pstate.write_memory_ptr(ptable + 0x00, otable)
    pstate.write_memory_ptr(ptable+size, 0)

    # FIXME: On pourrait la renvoyer qu'une seule fois ou la charger au demarage direct dans pstate
    pstate.write_memory_bytes(table, ctype)

    return ptable


def rtn_errno_location(se, pstate):
    """
    Pure emulation.
    """
    logging.debug('__errno_location hooked')

    # Errno is a int* ptr
    # Initialize it to zero
    pstate.write_memory_int(pstate.ERRNO_PTR, CPUSIZE.DWORD, 0)

    return pstate.ERRNO_PTR


def rtn_libc_start_main(se, pstate):
    logging.debug('__libc_start_main hooked')

    # Get arguments
    main = pstate.get_argument_value(0)

    if pstate.architecture == Architecture.AARCH64:
        pstate.cpu.program_counter = main

    elif pstate.architecture == Architecture.X86_64:
        # Push the return value to jump into the main() function
        pstate.push_stack_value(main)

    # Define concrete value of argc
    argc = len(se.config.program_argv)
    pstate.write_register(pstate._get_argument_register(0), argc)
    logging.debug(f"argc = {argc}")

    # Define argv
    base = pstate.BASE_ARGV
    addrs = list()

    index = 0
    for argv in se.config.program_argv:
        b_argv = argv.encode("latin-1")
        addrs.append(base)
        pstate.write_memory_bytes(base, b_argv + b'\x00')
        # TODO: Si symbolize_args is True
        #for indexCell in range(len(argv)):
        #    if se.config.symbolize_argv:
        #        var = pstate.tt_ctx.symbolizeMemory(MemoryAccess(base+indexCell, CPUSIZE.BYTE))
        #        var.setAlias('argv[%d][%d]' %(index, indexCell))
        logging.debug(f"argv[{index}] = {repr(pstate.read_memory_bytes(base, len(b_argv)))}")
        base += len(b_argv) + 1
        index += 1

    b_argv = base
    for addr in addrs:
        pstate.write_memory_ptr(base, addr)
        base += CPUSIZE.QWORD

    # Concrete value
    pstate.write_register(pstate._get_argument_register(1), b_argv)

    return None


def rtn_stack_chk_fail(se, pstate):
    """
    Pure emulation.
    """
    logging.debug('__stack_chk_fail hooked')
    logging.critical('*** stack smashing detected ***: terminated')
    se.seed.status = SeedStatus.CRASH
    se.abort()


# int __xstat(int ver, const char* path, struct stat* stat_buf);
def rtn_xstat(se, pstate):
    """
    Pure emulation.
    """
    logging.debug('__xstat hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # int ver
    arg1 = pstate.get_argument_value(1)  # const char* path
    arg2 = pstate.get_argument_value(2)  # struct stat* stat_buf

    if os.path.isfile(pstate.get_memory_string(arg1)):
        stat = os.stat(pstate.get_memory_string(arg1))
        pstate.write_memory_int(arg2 + 0x00, CPUSIZE.QWORD, stat.st_dev)
        pstate.write_memory_int(arg2 + 0x08, CPUSIZE.QWORD, stat.st_ino)
        pstate.write_memory_int(arg2 + 0x10, CPUSIZE.QWORD, stat.st_nlink)
        pstate.write_memory_int(arg2 + 0x18, CPUSIZE.DWORD, stat.st_mode)
        pstate.write_memory_int(arg2 + 0x1c, CPUSIZE.DWORD, stat.st_uid)
        pstate.write_memory_int(arg2 + 0x20, CPUSIZE.DWORD, stat.st_gid)
        pstate.write_memory_int(arg2 + 0x24, CPUSIZE.DWORD, 0)
        pstate.write_memory_int(arg2 + 0x28, CPUSIZE.QWORD, stat.st_rdev)
        pstate.write_memory_int(arg2 + 0x30, CPUSIZE.QWORD, stat.st_size)
        pstate.write_memory_int(arg2 + 0x38, CPUSIZE.QWORD, stat.st_blksize)
        pstate.write_memory_int(arg2 + 0x40, CPUSIZE.QWORD, stat.st_blocks)
        pstate.write_memory_int(arg2 + 0x48, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x50, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x58, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x60, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x68, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x70, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x78, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x80, CPUSIZE.QWORD, 0)
        pstate.write_memory_int(arg2 + 0x88, CPUSIZE.QWORD, 0)
        return 0

    return pstate.minus_one


# int atoi(const char *nptr);
def rtn_atoi(se, pstate):
    """ Simulate the atoi() function """
    logging.debug('atoi hooked')

    ast = pstate.actx
    arg = pstate.get_argument_value(0)

    cells = {i: pstate.read_symbolic_memory_byte(arg+i).getAst() for i in range(10)}

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

    return res


# void *calloc(size_t nmemb, size_t size);
def rtn_calloc(se, pstate):
    logging.debug('calloc hooked')

    # Get arguments
    nmemb = pstate.get_argument_value(0)
    size  = pstate.get_argument_value(1)

    # We use nmemb and size as concret values
    pstate.concretize_argument(0)  # will be concretized with nmemb value
    pstate.concretize_argument(1)  # will be concretized with size value

    if nmemb == 0 or size == 0:
        ptr = 0
    else:
        ptr = pstate.heap_allocator.alloc(nmemb * size)

    # Return value
    return ptr


# int clock_gettime(clockid_t clockid, struct timespec *tp);
def rtn_clock_gettime(se, pstate):
    logging.debug('clock_gettime hooked')

    # Get arguments
    clockid = pstate.get_argument_value(0)
    tp      = pstate.get_argument_value(1)

    # We use tp as concret value
    pstate.concretize_argument(1)

    # FIXME: We can return something logic
    if tp == 0:
        return pstate.minus_one

    if pstate.time_inc_coefficient:
        t = pstate.time
    else:
        t = time.time()

    pstate.write_memory_int(tp, pstate.ptr_size, int(t))
    pstate.write_memory_int(tp+pstate.ptr_size, pstate.ptr_size, int(t * 1000000))

    # Return value
    return 0


def rtn_exit(se, pstate):
    logging.debug('exit hooked')
    arg = pstate.get_argument_value(0)
    pstate.stop = True
    return arg


def rtn_fclose(se, pstate):
    logging.debug('fclose hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0) # fd

    # We use fd as concret value
    pstate.concretize_argument(0)

    if arg0 in pstate.fd_table:
        pstate.fd_table[arg0].close()
        del pstate.fd_table[arg0]
    else:
        return pstate.minus_one

    # Return value
    return 0


# char *fgets(char *s, int size, FILE *stream);
def rtn_fgets(se, pstate):
    logging.debug('fgets hooked')

    # Get arguments
    buff, buff_ast = pstate.get_full_argument(0)
    size, size_ast = pstate.get_full_argument(1)
    fd       = pstate.get_argument_value(2)
    minsize  = (min(len(se.seed.content), size) if se.seed else size)

    # We use fd as concret value
    pstate.concretize_argument(2)

    if fd == 0 and se.config.symbolize_stdin:
        # We use fd as concret value
        pstate.push_constraint(size_ast.getAst() == minsize)

        if se.seed:
            pstate.write_memory_bytes(buff, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug(f"stdin = {repr(pstate.read_memory_bytes(buff, minsize))}")
        return buff_ast

    if fd in pstate.fd_table:
        # We use fd as concret value
        pstate.concretize_argument(1)
        data = (os.read(0, size) if fd == 0 else os.read(pstate.fd_table[fd], size))
        pstate.write_memory_bytes(buff, data)
        return buff_ast
    else:
        logging.warning(f'File descriptor ({fd}) not found')

    return 0


# fopen(const char *pathname, const char *mode);
def rtn_fopen(se, pstate):
    logging.debug('fopen hooked')

    # Get arguments
    arg0  = pstate.get_argument_value(0)  # const char *pathname
    arg1  = pstate.get_argument_value(1)  # const char *mode
    arg0s = pstate.get_memory_string(arg0)
    arg1s = pstate.get_memory_string(arg1)

    # Concretize the whole path name
    pstate.concretize_memory_bytes(arg0, len(arg0s)+1)  # Concretize the whole string + \0

    # We use mode as concrete value
    pstate.concretize_argument(1)

    try:
        fd = open(arg0s, arg1s)
        fd_id = se.pstate.get_unique_file_id()
        se.pstate.fd_table.update({fd_id: fd})
    except:
        # Return value
        return 0

    # Return value
    return fd_id


def rtn_fprintf(se, pstate):
    logging.debug('fprintf hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)
    arg3 = pstate.get_argument_value(3)
    arg4 = pstate.get_argument_value(4)
    arg5 = pstate.get_argument_value(5)
    arg6 = pstate.get_argument_value(6)
    arg7 = pstate.get_argument_value(7)
    arg8 = pstate.get_argument_value(8)

    # FIXME: ARM64
    # FIXME: pushPathConstraint

    arg1f = pstate.get_format_string(arg1)
    nbArgs = arg1f.count("{")
    args = pstate.get_format_arguments(arg1, [arg2, arg3, arg4, arg5, arg6, arg7, arg8][:nbArgs])
    try:
        s = arg1f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    if arg0 in pstate.fd_table:
        pstate.fd_table[arg0].write(s)
        pstate.fd_table[arg0].flush()
    else:
        return 0

    # Return value
    return len(s)


# fputc(int c, FILE *stream);
def rtn_fputc(se, pstate):
    logging.debug('fputc hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    pstate.concretize_argument(0)
    pstate.concretize_argument(1)

    if arg1 in pstate.fd_table:
        if arg1 == 0:
            return 0
        elif arg1 == 1:
            sys.stdout.write(chr(arg0))
            sys.stdout.flush()
        elif arg1 == 2:
            sys.stderr.write(chr(arg0))
            sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg1], 'wb+')
            fd.write(chr(arg0))
    else:
        return 0

    # FIXME: We can iterate over all fd_tables and do the disjunction of all available fd

    # Return value
    return 1


def rtn_fputs(se, pstate):
    logging.debug('fputs hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    pstate.concretize_argument(0)
    pstate.concretize_argument(1)

    # FIXME: What if the fd is coming from the memory (fmemopen) ?

    if arg1 in pstate.fd_table:
        if arg1 == 0:
            return 0
        elif arg1 == 1:
            sys.stdout.write(pstate.get_memory_string(arg0))
            sys.stdout.flush()
        elif arg1 == 2:
            sys.stderr.write(pstate.get_memory_string(arg0))
            sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg1], 'wb+')
            fd.write(pstate.get_memory_string(arg0))
    else:
        return 0

    # Return value
    return len(pstate.get_memory_string(arg0))


def rtn_fread(se, pstate):
    logging.debug('fread hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0) # ptr
    arg1 = pstate.get_argument_value(1) # size
    arg2 = pstate.get_argument_value(2) # nmemb
    arg3 = pstate.get_argument_value(3) # stream
    size = arg1 * arg2

    minsize = (min(len(se.seed.content), size) if se.seed else size)

    # FIXME: pushPathConstraint

    if arg3 == 0 and se.config.symbolize_stdin:
        if se.seed:
            pstate.write_memory_bytes(arg0, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = pstate.tt_ctx.symbolizeMemory(MemoryAccess(arg0 + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug(f"stdin = {repr(pstate.read_memory_bytes(arg0, minsize))}")
        # TODO: Could return the read value as a symbolic one
        return minsize

    elif arg3 in pstate.fd_table:
        data = pstate.fd_table[arg3].read(arg1 * arg2)
        pstate.write_memory_bytes(arg0, data)

    else:
        return 0

    # Return value
    return len(data)


def rtn_free(se, pstate):
    logging.debug('free hooked')

    # Get arguments
    ptr = pstate.get_argument_value(0)
    pstate.heap_allocator.free(ptr)

    return None


def rtn_fwrite(se, pstate):
    logging.debug('fwrite hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)
    arg3 = pstate.get_argument_value(3)
    size = arg1 * arg2
    data = pstate.read_memory_bytes(arg0, size)

    if arg3 in pstate.fd_table:
        if arg3 == 0:
            return 0
        elif arg3 == 1:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
        elif arg3 == 2:
            sys.stderr.buffer.write(data)
            sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg3], 'wb+')
            fd.write(data)
    else:
        return 0

    # Return value
    return size


def rtn_gettimeofday(se, pstate):
    logging.debug('gettimeofday hooked')

    # Get arguments
    tv = pstate.get_argument_value(0)
    tz = pstate.get_argument_value(1)

    if tv == 0:
        return pstate.minus_one

    if pstate.time_inc_coefficient:
        t = pstate.time
    else:
        t = time.time()

    s = pstate.ptr_size
    pstate.write_memory_int(tv, s, int(t))
    pstate.write_memory_int(tv+s, s, int(t * 1000000))

    # Return value
    return 0


def rtn_malloc(se, pstate):
    logging.debug('malloc hooked')

    # Get arguments
    size = pstate.get_argument_value(0)
    ptr  = pstate.heap_allocator.alloc(size)

    # Return value
    return ptr


def rtn_memcmp(se, pstate):
    logging.debug('memcmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = pstate.get_argument_value(2)

    ast = pstate.actx
    res = ast.bv(0, 64)

    # We constrain the logical value of size
    pstate.concretize_argument(2)

    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1+index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2+index).getAst()
        res = res + ast.ite(
                        cells1 == cells2,
                        ast.bv(0, 64),
                        ast.bv(1, 64)
                    )

    return res


def rtn_memcpy(se, pstate):
    logging.debug('memcpy hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    # We constrain the logical value of size
    pstate.concretize_argument(2)

    for index in range(cnt):
        # Read symbolic src value and copy symbolically in dst
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst_ast


def rtn_memmem(se, pstate):
    logging.debug('memmem hooked')

    haystack    = pstate.get_argument_value(0)  # const void*
    haystacklen = pstate.get_argument_value(1)  # size_t
    needle      = pstate.get_argument_value(2)  # const void *
    needlelen   = pstate.get_argument_value(3)  # size_t

    s1 = pstate.read_memory_bytes(haystack, haystacklen)  # haystack
    s2 = pstate.read_memory_bytes(needle, needlelen)      # needle

    offset = s1.find(s2)
    if offset == -1:
        #FIXME: faut s'assurer que le marquer dans le string
        return 0

    for i, c in enumerate(s2):
        c1 = pstate.read_symbolic_memory_byte(haystack+offset+i)
        c2 = pstate.read_symbolic_memory_byte(needle+i)
        pstate.push_constraint(c1.getAst() == c2.getAst())

    # FIXME: à reflechir si on doit contraindre offset ou pas

    # faut s'assurer que le marquer est bien présent à l'offset trouvé
    return haystack + offset


def rtn_memmove(se, pstate):
    logging.debug('memmove hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    # We constrain the logical value of cnt
    pstate.concretize_argument(2)

    # TODO: What if cnt is symbolic ?
    for index in range(cnt):
        # Read symbolic src value and copy symbolically in dst
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst_ast


def rtn_memset(se, pstate):
    logging.debug('memset hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src, src_ast = pstate.get_full_argument(1)
    size = pstate.get_argument_value(2)

    # We constrain the logical value of size
    pstate.concretize_argument(2)

    sym_cell = pstate.actx.extract(7, 0, src_ast.getAst())

    # TODO: What if size is symbolic ?
    for index in range(size):
        pstate.write_symbolic_memory_byte(dst+index, sym_cell)

    return dst_ast


def rtn_printf(se, pstate):
    logging.debug('printf hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)
    arg3 = pstate.get_argument_value(3)
    arg4 = pstate.get_argument_value(4)
    arg5 = pstate.get_argument_value(5)
    arg6 = pstate.get_argument_value(6)
    arg7 = pstate.get_argument_value(7)
    arg8 = pstate.get_argument_value(8)

    arg0f = pstate.get_format_string(arg0)
    nbArgs = arg0f.count("{")
    args = pstate.get_format_arguments(arg0, [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8][:nbArgs])
    try:
        s = arg0f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    pstate.fd_table[1].write(s)
    pstate.fd_table[1].flush()

    # Return value
    return len(s)


def rtn_pthread_create(se, pstate):
    logging.debug('pthread_create hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0) # pthread_t *thread
    arg1 = pstate.get_argument_value(1) # const pthread_attr_t *attr
    arg2 = pstate.get_argument_value(2) # void *(*start_routine) (void *)
    arg3 = pstate.get_argument_value(3) # void *arg

    tid = pstate.get_unique_thread_id()
    thread = ThreadContext(tid, pstate.thread_scheduling_count)
    thread.save(pstate.tt_ctx)

    # Concretize pc
    if pstate.program_counter_register.getId() in thread.sregs:
        del thread.sregs[pstate.program_counter_register.getId()]

    # Concretize bp
    if pstate.base_pointer_register.getId() in thread.sregs:
        del thread.sregs[pstate.base_pointer_register.getId()]

    # Concretize sp
    if pstate.stack_pointer_register.getId() in thread.sregs:
        del thread.sregs[pstate.stack_pointer_register.getId()]

    # Concretize arg0
    if pstate._get_argument_register(0).getId() in thread.sregs:
        del thread.sregs[pstate._get_argument_register(0).getId()]

    thread.cregs[pstate.program_counter_register.getId()] = arg2
    thread.cregs[pstate._get_argument_register(0).getId()] = arg3
    thread.cregs[pstate.base_pointer_register.getId()] = (pstate.BASE_STACK - ((1 << 28) * tid))
    thread.cregs[pstate.stack_pointer_register.getId()] = (pstate.BASE_STACK - ((1 << 28) * tid))

    pstate.threads.update({tid: thread})

    # Save out the thread id
    pstate.write_memory_ptr(arg0, tid)

    # Return value
    return 0


def rtn_pthread_exit(se, pstate):
    logging.debug('pthread_exit hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)

    # Kill the thread
    pstate.threads[pstate.tid].killed = True

    # Return value
    return None


def rtn_pthread_join(se, pstate):
    logging.debug('pthread_join hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    if arg0 in pstate.threads:
        pstate.threads[pstate.tid].joined = arg0
        logging.info('Thread id %d joined thread id %d' % (pstate.tid, arg0))
    else:
        pstate.threads[pstate.tid].joined = None
        logging.debug('Thread id %d already destroyed' % arg0)

    # Return value
    return 0


def rtn_pthread_mutex_destroy(se, pstate):
    logging.debug('pthread_mutex_destroy hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *restrict mutex
    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_init(se, pstate):
    logging.debug('pthread_mutex_init hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *restrict mutex
    arg1 = pstate.get_argument_value(1)  # const pthread_mutexattr_t *restrict attr)

    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_lock(se, pstate):
    logging.debug('pthread_mutex_lock hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *mutex
    mutex = pstate.read_memory_ptr(arg0)  # deref pointer and read a uint64 int

    # If the thread has been initialized and unused, define the tid has lock
    if mutex == pstate.PTHREAD_MUTEX_INIT_MAGIC:
        logging.debug('mutex unlocked')
        pstate.write_memory_ptr(arg0, pstate.tid)

    # The mutex is locked and we are not allowed to continue the execution
    elif mutex != pstate.tid:
        logging.debug('mutex locked')
        pstate.mutex_locked = True

    # Return value
    return 0


def rtn_pthread_mutex_unlock(se, pstate):
    logging.debug('pthread_mutex_unlock hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *mutex

    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    # Return value
    return 0


def rtn_puts(se, pstate):
    logging.debug('puts hooked')

    # Get arguments
    arg0 = pstate.get_string_argument(0)
    sys.stdout.write(arg0 + '\n')
    sys.stdout.flush()

    # Return value
    return len(arg0) + 1


def rtn_rand(se, pstate):
    logging.debug('rand hooked')
    return random.randrange(0, 0xffffffff)


def rtn_read(se, pstate):
    logging.debug('read hooked')

    # Get arguments
    fd   = pstate.get_argument_value(0)
    buff = pstate.get_argument_value(1)
    size, size_ast = pstate.get_full_argument(2)
    minsize = (min(len(se.seed.content), size) if se.seed else size)

    if size_ast.isSymbolized():
        logging.warning(f'Reading from the file descriptor ({fd}) with a symbolic size')

    pstate.concretize_argument(0)

    if fd == 0 and se.config.symbolize_stdin:
        pstate.push_constraint(size_ast.getAst() == minsize)
        if se.seed:
            pstate.write_memory_bytes(buff, se.seed.content[:minsize])
        else:
            se.seed.content = b'\x00' * minsize

        for index in range(minsize):
            var = pstate.tt_ctx.symbolizeMemory(MemoryAccess(buff + index, CPUSIZE.BYTE))
            var.setComment('stdin[%d]' % index)

        logging.debug(f"stdin = {repr(pstate.read_memory_bytes(buff, minsize))}")
        # TODO: Could return the read value as a symbolic one
        return minsize

    if fd in pstate.fd_table:
        pstate.concretize_argument(2)
        data = (os.read(0, size) if fd == 0 else os.read(pstate.fd_table[fd], size))
        pstate.write_memory_bytes(buff, data)
        return len(data)

    return 0


def rtn_sem_destroy(se, pstate):
    logging.debug('sem_destroy hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # Destroy the semaphore with the value
    pstate.write_memory_ptr(arg0, 0)

    # Return success
    return 0


def rtn_sem_getvalue(se, pstate):
    logging.debug('sem_getvalue hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem
    arg1 = pstate.get_argument_value(1)  # int *sval

    value = pstate.read_memory_ptr(arg0)  # deref pointer

    # Set the semaphore's value into the output
    pstate.write_memory_int(arg1, CPUSIZE.DWORD, value)  # WARNING: read uint64 to uint32

    # Return success
    return 0


def rtn_sem_init(se, pstate):
    logging.debug('sem_init hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem
    arg1 = pstate.get_argument_value(1)  # int pshared
    arg2 = pstate.get_argument_value(2)  # unsigned int value

    # Init the semaphore with the value
    pstate.write_memory_ptr(arg0, arg2)

    # Return success
    return 0


def rtn_sem_post(se, pstate):
    logging.debug('sem_post hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # increments (unlocks) the semaphore pointed to by sem
    value = pstate.read_memory_ptr(arg0)
    pstate.write_memory_ptr(arg0, value + 1)

    # Return success
    return 0


def rtn_sem_timedwait(se, pstate):
    logging.debug('sem_timedwait hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem
    arg1 = pstate.get_argument_value(1)  # const struct timespec *abs_timeout

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
    value = pstate.read_memory_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.write_memory_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked')
        pstate.semaphore_locked = True

    # Return success
    return 0


def rtn_sem_trywait(se, pstate):
    logging.debug('sem_trywait hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # sem_trywait()  is  the  same as sem_wait(), except that if the decrement
    # cannot be immediately performed, then call returns an error (errno set to
    # EAGAIN) instead of blocking.
    value = pstate.read_memory_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.write_memory_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked but continue')
        pstate.semaphore_locked = False
        # Setting errno to EAGAIN (3406)
        pstate.write_memory_int(pstate.ERRNO_PTR, CPUSIZE.DWORD, 3406)
        # Return -1
        return pstate.minus_one

    # Return success
    return 0


def rtn_sem_wait(se, pstate):
    logging.debug('sem_wait hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # decrements (locks) the semaphore pointed to by sem. If the semaphore's value
    # is greater than zero, then the decrement proceeds, and the function returns,
    # immediately. If the semaphore currently has the value zero, then the call blocks
    # until either it becomes possible to perform the decrement (i.e., the semaphore
    # value rises above zero).
    value = pstate.read_memory_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.write_memory_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked')
        pstate.semaphore_locked = True

    # Return success
    return 0


def rtn_sleep(se, pstate):
    logging.debug('sleep hooked')

    # Get arguments
    t = pstate.get_argument_value(0)
    #time.sleep(t)

    # Return value
    return 0


def rtn_sprintf(se, pstate):
    logging.debug('sprintf hooked')

    # Get arguments
    buff = pstate.get_argument_value(0)
    arg0 = pstate.get_argument_value(1)
    arg1 = pstate.get_argument_value(2)
    arg2 = pstate.get_argument_value(3)
    arg3 = pstate.get_argument_value(4)
    arg4 = pstate.get_argument_value(5)
    arg5 = pstate.get_argument_value(6)
    arg6 = pstate.get_argument_value(7)
    arg7 = pstate.get_argument_value(8)

    arg0f = pstate.get_format_string(arg0)
    nbArgs = arg0f.count("{")
    args = pstate.get_format_arguments(arg0, [arg1, arg2, arg3, arg4, arg5, arg6, arg7][:nbArgs])
    try:
        s = arg0f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    # FIXME: todo

    # FIXME: THIS SEEMS NOT OK
    for index, c in enumerate(s):
        pstate.tt_ctx.concretizeMemory(buff + index)
        pstate.tt_ctx.setConcreteMemoryValue(buff + index, ord(c))
        pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + index, 1)) == ord(c))

    # including the terminating null byte ('\0')
    pstate.tt_ctx.setConcreteMemoryValue(buff + len(s), 0x00)
    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + len(s), 1)) == 0x00)

    return len(s)


def rtn_strcasecmp(se, pstate):
    logging.debug('strcasecmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = min(len(pstate.get_memory_string(s1)), len(pstate.get_memory_string(s2)) + 1)

    #s = s1 if len(pstate.get_memory_string(s1)) < len(pstate.get_memory_string(s2)) else s2
    #for i in range(size):
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + len(pstate.get_memory_string(s1)), CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + len(pstate.get_memory_string(s2)), CPUSIZE.BYTE)) == 0x00)

    # FIXME: Il y a des truc chelou avec le +1 et le logic ci-dessous

    ast = pstate.actx
    res = ast.bv(0, pstate.ptr_bit_size)
    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    return res


def rtn_strchr(se, pstate):
    logging.debug('strchr hooked')

    string = pstate.get_argument_value(0)
    char   = pstate.get_argument_value(1)
    ast    = pstate.actx

    def rec(res, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(string + deep).getAst()
        res  = ast.ite(cell == (char & 0xff), ast.bv(string + deep, 64), rec(res, deep + 1, maxdeep))
        return res

    sze = len(pstate.get_memory_string(string))
    res = rec(ast.bv(0, 64), 0, sze)

    for i, c in enumerate(pstate.get_memory_string(string)):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(string+i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(string+sze).getAst() == 0x00)

    return res


def rtn_strcmp(se, pstate):
    logging.debug('strcmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = min(len(pstate.get_memory_string(s1)), len(pstate.get_memory_string(s2))) + 1

    #s = s1 if len(pstate.get_memory_string(s1)) <= len(pstate.get_memory_string(s2)) else s2
    #for i in range(size):
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + len(pstate.get_memory_string(s1)), CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + len(pstate.get_memory_string(s2)), CPUSIZE.BYTE)) == 0x00)

    # FIXME: Il y a des truc chelou avec le +1 et le logic ci-dessous

    ast = pstate.actx
    res = ast.bv(0, 64)
    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    return res


def rtn_strcpy(se, pstate):
    logging.debug('strcpy hooked')

    dst  = pstate.get_argument_value(0)
    src  = pstate.get_argument_value(1)
    src_str = pstate.get_memory_string(src)
    size = len(src_str)

    # constraint src buff to be != \00 and last one to be \00 (indirectly concretize length)
    for i, c in enumerate(src_str):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(src + i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(src + size).getAst() == 0x00)

    # Copy symbolically bytes (including \00)
    for index in range(size+1):
        sym_c = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_c)

    return dst


def rtn_strlen(se, pstate):
    logging.debug('strlen hooked')

    # Get arguments
    s = pstate.get_argument_value(0)
    ast = pstate.actx

    # FIXME: Not so sure its is really the strlen semantic
    def rec(res, s, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(s+deep).getAst()
        res  = ast.ite(cell == 0x00, ast.bv(deep, 64), rec(res, s, deep + 1, maxdeep))
        return res

    sze = len(pstate.get_memory_string(s))
    res = ast.bv(sze, 64)
    res = rec(res, s, 0, sze)

    # FIXME: That routine should do something like below to be SOUND !
    # for i, c in enumerate(pstate.get_memory_string(src)):
    #     pstate.push_constraint(pstate.read_symbolic_memory_byte(src + i) != 0x00)
    # pstate.push_constraint(pstate.read_symbolic_memory_byte(src + size) == 0x00)

    pstate.push_constraint(pstate.read_symbolic_memory_byte(s+sze).getAst() == 0x00)

    return res


def rtn_strncasecmp(se, pstate):
    logging.debug('strncasecmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    sz = pstate.get_argument_value(2)
    maxlen = min(sz, min(len(pstate.get_memory_string(s1)), len(pstate.get_memory_string(s2))) + 1)

    ast = pstate.actx
    res = ast.bv(0, pstate.ptr_bit_size)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    return res


def rtn_strncmp(se, pstate):
    logging.debug('strncmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    sz = pstate.get_argument_value(2)
    maxlen = min(sz, min(len(pstate.get_memory_string(s1)), len(pstate.get_memory_string(s2))) + 1)

    ast = pstate.actx
    res = ast.bv(0, 64)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = res + ast.ite(cells1 == cells2, ast.bv(0, 64), ast.bv(1, 64))

    return res


def rtn_strncpy(se, pstate):
    logging.debug('strncpy hooked')

    dst = pstate.get_argument_value(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    pstate.concretize_argument(2)

    for index in range(cnt):
        src_sym = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, src_sym)

        if src_sym.getAst().evaluate() == 0:
            pstate.push_constraint(src_sym.getAst() == 0x00)
            break
        else:
            pstate.push_constraint(src_sym.getAst() != 0x00)

    return dst


def rtn_strtok_r(se, pstate):
    logging.debug('strtok_r hooked')

    string  = pstate.get_argument_value(0)
    delim   = pstate.get_argument_value(1)
    saveptr = pstate.get_argument_value(2)
    saveMem = pstate.read_memory_ptr(saveptr)

    if string == 0:
        string = saveMem

    d = pstate.get_memory_string(delim)
    s = pstate.get_memory_string(string)

    tokens = re.split('[' + re.escape(d) + ']', s)

    # TODO: Make it symbolic
    for token in tokens:
        if token:
            offset = s.find(token)
            # Init the \0 at the delimiter position
            node = pstate.read_symbolic_memory_byte(string + offset + len(token)).getAst()
            try:
                pstate.push_constraint(pstate.actx.lor([node == ord(c) for c in d]))
            except:  # dafuck is that?
                pstate.push_constraint(node == ord(d))
            pstate.write_memory_byte(string + offset + len(token), 0)
            # Save the pointer
            pstate.write_memory_ptr(saveptr, string + offset + len(token) + 1)
            # Return the token
            return string + offset

    return 0


def rtn_strtoul(se, pstate):
    logging.debug('strtoul hooked')

    nptr   = pstate.get_argument_value(0)
    nptrs  = pstate.get_string_argument(0)
    endptr = pstate.get_argument_value(1)
    base   = pstate.get_argument_value(2)

    for i, c in enumerate(nptrs):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(nptr+i).getAst() == ord(c))

    pstate.concretize_argument(2)  # Concretize base

    try:
        return int(nptrs, base)
    except:
        return 0xffffffff



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
