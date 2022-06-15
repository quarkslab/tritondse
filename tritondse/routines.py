import logging
import os
import random
import re
import sys
import time

from typing import Union

from triton                   import CPUSIZE, MemoryAccess
from tritondse.thread_context import ThreadContext
from tritondse.types          import Architecture
from tritondse.seed           import SeedType, SeedStatus, Seed


def rtn_ctype_b_loc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __ctype_b_loc behavior.
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


def rtn_errno_location(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __errno_location behavior.
    """
    logging.debug('__errno_location hooked')

    # Errno is a int* ptr
    # Initialize it to zero
    pstate.write_memory_int(pstate.ERRNO_PTR, CPUSIZE.DWORD, 0)

    return pstate.ERRNO_PTR


def rtn_libc_start_main(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __libc_start_main behavior.
    """
    logging.debug('__libc_start_main hooked')

    # Get arguments
    main = pstate.get_argument_value(0)

    # WARNING: Dirty trick to make sure to jump on main after
    # the emulation of that stub
    if pstate.architecture == Architecture.AARCH64:
        pstate.cpu.x30 = main
    elif pstate.architecture in [Architecture.X86_64, Architecture.X86]:
        # Push the return value to jump into the main() function
        pstate.push_stack_value(main)
    else:
        assert False

    # Define concrete value of argc (from either the seed or the program_argv)
    if se.config.seed_type == SeedType.RAW:
        argc = len(se.seed.content.split(b"\x00")) if se.config.symbolize_argv else len(se.config.program_argv)
    else: # SeedType.COMPOSITE
        if se.config.symbolize_argv and "argv" not in se.seed.content:
            logging.error("symbolized_argv specified but seed does not contain \"argv\"")
            assert False
        argc = len(se.seed.content["argv"]) if se.config.symbolize_argv else len(se.config.program_argv)
    pstate.write_argument_value(0, argc)
    logging.debug(f"argc = {argc}")

    # Define argv
    base = pstate.BASE_ARGV
    addrs = list()

    index = 0

    if se.config.symbolize_argv:  # Use the seed provided (and ignore config.program_argv !!)
        if se.config.seed_type == SeedType.RAW:
            argvs = [x for x in se.seed.content.split(b"\x00")]
        else: # SeedType.COMPOSITE
            argvs = se.seed.content["argv"] 
    else:  # use the config argv
        argvs = [x.encode("latin-1") for x in se.config.program_argv]  # Convert it from str to bytes

    for i, arg in enumerate(argvs):
        addrs.append(base)
        pstate.write_memory_bytes(base, arg + b'\x00')

        if se.config.symbolize_argv: # If the symbolic input injection point is a argv

            # Symbolize the argv string
            sym_vars = pstate.symbolize_memory_bytes(base, len(arg), f"argv[{i}]")
            if se.config.seed_type == SeedType.RAW:
                se.symbolic_seed.append(sym_vars)  # Set symbolic_seed to be able to retrieve them in generated models
            else: # SeedType.COMPOSITE
                if "argv" not in se.symbolic_seed:
                    se.symbolic_seed["argv"] = []
                se.symbolic_seed["argv"].append(sym_vars)
            # FIXME: Shall add a constraint on every char to be != \x00

        logging.debug(f"argv[{index}] = {repr(pstate.read_memory_bytes(base, len(arg)))}")
        base += len(arg) + 1
        index += 1

    b_argv = base
    for addr in addrs:
        pstate.write_memory_ptr(base, addr)
        base += CPUSIZE.QWORD

    # Concrete value
    pstate.write_argument_value(1, b_argv)
    return None


def rtn_stack_chk_fail(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __stack_chk_fail behavior.
    """
    logging.debug('__stack_chk_fail hooked')
    logging.critical('*** stack smashing detected ***: terminated')
    se.seed.status = SeedStatus.CRASH
    pstate.stop = True


# int __xstat(int ver, const char* path, struct stat* stat_buf);
def rtn_xstat(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __xstat behavior.
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


def rtn_abort(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """::

        void abort(void);

    Mark the input seed as OK and stop execution.

    [`Man Page <https://man7.org/linux/man-pages/man3/abort.3.html>`_]
    """
    logging.debug('abort hooked')
    se.seed.status = SeedStatus.OK_DONE
    pstate.stop = True


# int atoi(const char *nptr);
def rtn_atoi(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """::

        int atoi(const char *nptr);

    **Description**: The atoi() function converts the initial portion of the string
    pointed to by nptr to int.  The behavior is the same as

    Concrete: /

    Symbolic: Represent the return value symbolically with 10 nested if
    to represent the value.

    .. warning:: The function does not support all possibles representation
                 of an integer. It does not support negative integer nor values
                 prefixed by spaces.

    [`Man Page <https://man7.org/linux/man-pages/man3/abort.3.html>`_]

    :return: Symbolic value of the integer base on the symbolic string ``nptr``
    """
    logging.debug('atoi hooked')

    ast = pstate.actx
    arg = pstate.get_argument_value(0)

    # FIXME: On ne concretize pas correctement la taille de la chaine

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
def rtn_calloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The calloc behavior.
    """
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
        # Once the ptr allocated, the memory area must be filled with zero
        for index in range(nmemb * size):
            pstate.write_symbolic_memory_byte(ptr+index, pstate.actx.bv(0, 8))

    # Return value
    return ptr


# int clock_gettime(clockid_t clockid, struct timespec *tp);
def rtn_clock_gettime(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The clock_gettime behavior.
    """
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


def rtn_exit(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The exit behavior.
    """
    logging.debug('exit hooked')
    arg = pstate.get_argument_value(0)
    pstate.stop = True
    return arg


def rtn_fclose(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fclose behavior.
    """
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
def rtn_fgets(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fgets behavior.
    """
    logging.debug('fgets hooked')

    # Get arguments
    buff, buff_ast = pstate.get_full_argument(0)
    size, size_ast = pstate.get_full_argument(1)
    fd       = pstate.get_argument_value(2)
    if se.config.seed_type == SeedType.RAW:
        minsize  = (min(len(se.seed.content), size) if se.seed else size)
    else: # SeedType.COMPOSITE 
        minsize  = (min(len(se.seed.content["stdin"]), size) if se.seed else size)
    # We use fd as concret value
    pstate.concretize_argument(2)

    if fd == 0 and se.config.symbolize_stdin:
        if se.is_seed_injected():
            logging.warning("fgets reading stdin, while seed already injected (return EOF)")
            #return 0
        #else:
        # We use fd as concret value
        pstate.push_constraint(size_ast.getAst() == minsize)

        if se.config.seed_type == SeedType.RAW:
            content = se.seed.content[:minsize] if se.seed else b'\x00' * minsize
            se.inject_symbolic_input(buff, Seed(content), "stdin")
        else: # SeedType.COMPOSITE 
            content = se.seed.content["stdin"][:minsize] if se.seed else b'\x00' * minsize
            se.inject_symbolic_input(buff, Seed(content), "stdin")

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
def rtn_fopen(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fopen behavior.
    """
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


def rtn_fprintf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fprintf behavior.
    """
    logging.debug('fprintf hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    # FIXME: ARM64
    # FIXME: pushPathConstraint

    arg1f = pstate.get_format_string(arg1)
    nbArgs = arg1f.count("{")
    args = pstate.get_format_arguments(arg1, [pstate.get_argument_value(x) for x in range(2, nbArgs+2)])
    try:
        s = arg1f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    if arg0 in pstate.fd_table:
        if arg0 not in [1, 2] or (arg0 == 1 and se.config.pipe_stdout) or (arg0 == 2 and se.config.pipe_stderr):
            pstate.fd_table[arg0].write(s)
            pstate.fd_table[arg0].flush()
    else:
        return 0

    # Return value
    return len(s)


# fputc(int c, FILE *stream);
def rtn_fputc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fputc behavior.
    """
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
            if se.config.pipe_stdout:
                sys.stdout.write(chr(arg0))
                sys.stdout.flush()
        elif arg1 == 2:
            if se.config.pipe_stderr:
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


def rtn_fputs(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fputs behavior.
    """
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
            if se.config.pipe_stdout:
                sys.stdout.write(pstate.get_memory_string(arg0))
                sys.stdout.flush()
        elif arg1 == 2:
            if se.config.pipe_stderr:
                sys.stderr.write(pstate.get_memory_string(arg0))
                sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg1], 'wb+')
            fd.write(pstate.get_memory_string(arg0))
    else:
        return 0

    # Return value
    return len(pstate.get_memory_string(arg0))


def rtn_fread(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fread behavior.
    """
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
        if se.is_seed_injected():
            logging.warning("fread reading stdin, while seed already injected (return EOF)")
            return 0
        else:

            content = se.seed.content[:minsize] if se.seed else b'\x00' * minsize

            se.inject_symbolic_input(arg0, Seed(content), "stdin")

            logging.debug(f"stdin = {repr(pstate.read_memory_bytes(arg0, minsize))}")
            # TODO: Could return the read value as a symbolic one
            return minsize

    elif arg3 in pstate.fd_table:
        data = pstate.fd_table[arg3].read(arg1 * arg2)
        if isinstance(data, str): data = data.encode()
        pstate.write_memory_bytes(arg0, data)

    else:
        return 0

    # Return value
    return len(data)


def rtn_free(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The free behavior.
    """
    logging.debug('free hooked')

    # Get arguments
    ptr = pstate.get_argument_value(0)
    pstate.heap_allocator.free(ptr)

    return None


def rtn_fwrite(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fwrite behavior.
    """
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
            if se.config.pipe_stdout:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
        elif arg3 == 2:
            if se.config.pipe_stderr:
                sys.stderr.buffer.write(data)
                sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg3], 'wb+')
            fd.write(data)
    else:
        return 0

    # Return value
    return size


def rtn_gettimeofday(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The gettimeofday behavior.
    """
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


def rtn_malloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The malloc behavior.
    """
    logging.debug('malloc hooked')

    # Get arguments
    size = pstate.get_argument_value(0)
    ptr  = pstate.heap_allocator.alloc(size)

    # Return value
    return ptr


def rtn_memcmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memcmp behavior.
    """
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


def rtn_memcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memcpy behavior.
    """
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


def rtn_memmem(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memmem behavior.
    """
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


def rtn_memmove(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memmove behavior.
    """
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


def rtn_memset(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memset behavior.
    """
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


def rtn_printf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The printf behavior.
    """
    logging.debug('printf hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)

    arg0f = pstate.get_format_string(arg0)
    nbArgs = arg0f.count("{")
    args = pstate.get_format_arguments(arg0, [pstate.get_argument_value(x) for x in range(1, nbArgs+1)])
    try:
        s = arg0f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    if se.config.pipe_stdout:
        pstate.fd_table[1].write(s)
        pstate.fd_table[1].flush()

    # Return value
    return len(s)


def rtn_pthread_create(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_create behavior.
    """
    logging.debug('pthread_create hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0) # pthread_t *thread
    arg1 = pstate.get_argument_value(1) # const pthread_attr_t *attr
    arg2 = pstate.get_argument_value(2) # void *(*start_routine) (void *)
    arg3 = pstate.get_argument_value(3) # void *arg

    th = pstate.spawn_new_thread(arg2, arg3)

    # Save out the thread id
    pstate.write_memory_ptr(arg0, th.tid)

    # Return value
    return 0


def rtn_pthread_exit(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_exit behavior.
    """
    logging.debug('pthread_exit hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)

    # Kill the thread
    pstate.current_thread.kill()

    # FIXME: I guess the thread exiting never return, so should not continue
    # FIXME: iterating instructions

    # Return value
    return None


def rtn_pthread_join(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_join behavior.
    """
    logging.debug('pthread_join hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    if arg0 in pstate.threads:
        pstate.current_thread.join_thread(arg0)
        logging.info(f"Thread id {pstate.current_thread.tid} joined thread id {arg0}")
    else:
        pstate.current_thread.cancel_join()
        logging.debug(f"Thread id {arg0} already destroyed")

    # Return value
    return 0


def rtn_pthread_mutex_destroy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_destroy behavior.
    """
    logging.debug('pthread_mutex_destroy hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *restrict mutex
    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_init(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_init behavior.
    """
    logging.debug('pthread_mutex_init hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *restrict mutex
    arg1 = pstate.get_argument_value(1)  # const pthread_mutexattr_t *restrict attr)

    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_lock(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_lock behavior.
    """
    logging.debug('pthread_mutex_lock hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *mutex
    mutex = pstate.read_memory_ptr(arg0)  # deref pointer and read a uint64 int

    # If the thread has been initialized and unused, define the tid has lock
    if mutex == pstate.PTHREAD_MUTEX_INIT_MAGIC:
        logging.debug('mutex unlocked')
        pstate.write_memory_ptr(arg0, pstate.current_thread.tid)

    # The mutex is locked and we are not allowed to continue the execution
    elif mutex != pstate.current_thread.tid:
        logging.debug('mutex locked')
        pstate.mutex_locked = True

    # Return value
    return 0


def rtn_pthread_mutex_unlock(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_unlock behavior.
    """
    logging.debug('pthread_mutex_unlock hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *mutex

    pstate.write_memory_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    # Return value
    return 0


def rtn_puts(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The puts behavior.
    """
    logging.debug('puts hooked')

    arg0 = pstate.get_string_argument(0)

    # Get arguments
    if se.config.pipe_stdout:  # Only perform printing if pipe_stdout activated
        sys.stdout.write(arg0 + '\n')
        sys.stdout.flush()

    # Return value
    return len(arg0) + 1


def rtn_rand(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The rand behavior.
    """
    logging.debug('rand hooked')
    return random.randrange(0, 0xffffffff)


def rtn_read(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The read behavior.
    """
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
        if se.is_seed_injected():
            logging.warning("reading stdin, while seed already injected (return EOF)")
            return 0
        else:
            pstate.push_constraint(size_ast.getAst() == minsize)

            content = se.seed.content[:minsize] if se.seed else b'\x00' * minsize

            se.inject_symbolic_input(buff, Seed(content), "stdin")

            logging.debug(f"stdin = {repr(pstate.read_memory_bytes(buff, minsize))}")
            # TODO: Could return the read value as a symbolic one
            return minsize


    if fd in pstate.fd_table:
        pstate.concretize_argument(2)
        data = (os.read(0, size) if fd == 0 else os.read(pstate.fd_table[fd], size))
        pstate.write_memory_bytes(buff, data)
        return len(data)

    return 0


def rtn_sem_destroy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_destroy behavior.
    """
    logging.debug('sem_destroy hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # Destroy the semaphore with the value
    pstate.write_memory_ptr(arg0, 0)

    # Return success
    return 0


def rtn_sem_getvalue(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_getvalue behavior.
    """
    logging.debug('sem_getvalue hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem
    arg1 = pstate.get_argument_value(1)  # int *sval

    value = pstate.read_memory_ptr(arg0)  # deref pointer

    # Set the semaphore's value into the output
    pstate.write_memory_int(arg1, CPUSIZE.DWORD, value)  # WARNING: read uint64 to uint32

    # Return success
    return 0


def rtn_sem_init(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_init behavior.
    """
    logging.debug('sem_init hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem
    arg1 = pstate.get_argument_value(1)  # int pshared
    arg2 = pstate.get_argument_value(2)  # unsigned int value

    # Init the semaphore with the value
    pstate.write_memory_ptr(arg0, arg2)

    # Return success
    return 0


def rtn_sem_post(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_post behavior.
    """
    logging.debug('sem_post hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # increments (unlocks) the semaphore pointed to by sem
    value = pstate.read_memory_ptr(arg0)
    pstate.write_memory_ptr(arg0, value + 1)

    # Return success
    return 0


def rtn_sem_timedwait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_timedwait behavior.
    """
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


def rtn_sem_trywait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_trywait behavior.
    """
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


def rtn_sem_wait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_wait behavior.
    """
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


def rtn_sleep(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sleep behavior.
    """
    logging.debug('sleep hooked')

    # Get arguments
    if not se.config.skip_sleep_routine:
        t = pstate.get_argument_value(0)
        time.sleep(t)

    # Return value
    return 0


def rtn_sprintf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sprintf behavior.
    """
    logging.debug('sprintf hooked')

    # Get arguments
    buff = pstate.get_argument_value(0)
    arg0 = pstate.get_argument_value(1)

    try:
        arg0f = pstate.get_format_string(arg0)
        nbArgs = arg0f.count("{")
        args = pstate.get_format_arguments(arg0, [pstate.get_argument_value(x) for x in range(2, nbArgs+2)])
        s = arg0f.format(*args)
    except:
        # FIXME: Les chars UTF8 peuvent foutre le bordel. Voir avec ground-truth/07.input
        logging.warning('Something wrong, probably UTF-8 string')
        s = ""

    # FIXME: todo

    # FIXME: THIS SEEMS NOT OK
    # for index, c in enumerate(s):
    #     pstate.tt_ctx.concretizeMemory(buff + index)
    #     pstate.tt_ctx.setConcreteMemoryValue(buff + index, ord(c))
    #     pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + index, 1)) == ord(c))
    #
    # # including the terminating null byte ('\0')
    # pstate.tt_ctx.setConcreteMemoryValue(buff + len(s), 0x00)
    # pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(buff + len(s), 1)) == 0x00)

    return len(s)


def rtn_strcasecmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strcasecmp behavior.
    """
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


def rtn_strchr(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strchr behavior.
    """
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


def rtn_strcmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strcmp behavior.
    """
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


def rtn_strcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strcpy behavior.
    """
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


def rtn_strerror(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strerror behavior.

    :param se: The current symbolic execution instance
    :param pstate: The current process state
    :return: a concrete value
    """
    logging.debug('strerror hooked')

    sys_errlist = [
        b"Success",
        b"Operation not permitted",
        b"No such file or directory",
        b"No such process",
        b"Interrupted system call",
        b"Input/output error",
        b"No such device or address",
        b"Argument list too long",
        b"Exec format error",
        b"Bad file descriptor",
        b"No child processes",
        b"Resource temporarily unavailable",
        b"Cannot allocate memory",
        b"Permission denied",
        b"Bad address",
        b"Block device required",
        b"Device or resource busy",
        b"File exists",
        b"Invalid cross-device link",
        b"No such device",
        b"Not a directory",
        b"Is a directory",
        b"Invalid argument",
        b"Too many open files in system",
        b"Too many open files",
        b"Inappropriate ioctl for device",
        b"Text file busy",
        b"File too large",
        b"No space left on device",
        b"Illegal seek",
        b"Read-only file system",
        b"Too many links",
        b"Broken pipe",
        b"Numerical argument out of domain",
        b"Numerical result out of range",
        b"Resource deadlock avoided",
        b"File name too long",
        b"No locks available",
        b"Function not implemented",
        b"Directory not empty",
        b"Too many levels of symbolic links",
        None,
        b"No message of desired type",
        b"Identifier removed",
        b"Channel number out of range",
        b"Level 2 not synchronized",
        b"Level 3 halted",
        b"Level 3 reset",
        b"Link number out of range",
        b"Protocol driver not attached",
        b"No CSI structure available",
        b"Level 2 halted",
        b"Invalid exchange",
        b"Invalid request descriptor",
        b"Exchange full",
        b"No anode",
        b"Invalid request code",
        b"Invalid slot",
        None,
        b"Bad font file format",
        b"Device not a stream",
        b"No data available",
        b"Timer expired",
        b"Out of streams resources",
        b"Machine is not on the network",
        b"Package not installed",
        b"Object is remote",
        b"Link has been severed",
        b"Advertise error",
        b"Srmount error",
        b"Communication error on send",
        b"Protocol error",
        b"Multihop attempted",
        b"RFS specific error",
        b"Bad message",
        b"Value too large for defined data type",
        b"Name not unique on network",
        b"File descriptor in bad state",
        b"Remote address changed",
        b"Can not access a needed shared library",
        b"Accessing a corrupted shared library",
        b".lib section in a.out corrupted",
        b"Attempting to link in too many shared libraries",
        b"Cannot exec a shared library directly",
        b"Invalid or incomplete multibyte or wide character",
        b"Interrupted system call should be restarted",
        b"Streams pipe error",
        b"Too many users",
        b"Socket operation on non-socket",
        b"Destination address required",
        b"Message too long",
        b"Protocol wrong type for socket",
        b"Protocol not available",
        b"Protocol not supported",
        b"Socket type not supported",
        b"Operation not supported",
        b"Protocol family not supported",
        b"Address family not supported by protocol",
        b"Address already in use",
        b"Cannot assign requested address",
        b"Network is down",
        b"Network is unreachable",
        b"Network dropped connection on reset",
        b"Software caused connection abort",
        b"Connection reset by peer",
        b"No buffer space available",
        b"Transport endpoint is already connected",
        b"Transport endpoint is not connected",
        b"Cannot send after transport endpoint shutdown",
        b"Too many references: cannot splice",
        b"Connection timed out",
        b"Connection refused",
        b"Host is down",
        b"No route to host",
        b"Operation already in progress",
        b"Operation now in progress",
        b"Stale NFS file handle",
        b"Structure needs cleaning",
        b"Not a XENIX named type file",
        b"No XENIX semaphores available",
        b"Is a named type file",
        b"Remote I/O error",
        b"Disk quota exceeded",
        b"No medium found",
        b"Wrong medium type",
        b"Operation canceled",
        b"Required key not available",
        b"Key has expired",
        b"Key has been revoked",
        b"Key was rejected by service",
        b"Owner died",
        b"State not recoverable"
    ]

    # Get arguments
    errnum = pstate.get_argument_value(0)
    try:
        str = sys_errlist[errnum]
    except:
        # invalid errnum
        str = b'Error'

    # TODO: We allocate the string at every hit of this function with a
    # potential memory leak. We should allocate the sys_errlist only once
    # and then refer to this table instead of allocate string.

    ptr = pstate.heap_allocator.alloc(len(str) + 1)
    pstate.write_memory_bytes(ptr, str + b'\0')

    return ptr


def rtn_strlen(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strlen behavior.
    """
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


def rtn_strncasecmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strncasecmp behavior.
    """
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


def rtn_strncmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strncmp behavior.
    """
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


def rtn_strncpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strncpy behavior.
    """
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


def rtn_strtok_r(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strtok_r behavior.
    """
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

            # Token must not contain delimiters
            for index, char in enumerate(token):
                node = pstate.read_symbolic_memory_byte(string + offset + index).getAst()
                for delim in d:
                    pstate.push_constraint(node != ord(delim))

            pstate.write_memory_byte(string + offset + len(token), 0)
            # Save the pointer
            pstate.write_memory_ptr(saveptr, string + offset + len(token) + 1)
            # Return the token
            return string + offset

    return 0


def rtn_strtoul(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strtoul behavior.
    """
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
    'abort':                   rtn_abort,
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
    'strerror':                rtn_strerror,
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
