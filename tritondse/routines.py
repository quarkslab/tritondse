import logging
import os
import random
import re
import sys
import time

from typing import Union

from triton                   import CPUSIZE, MemoryAccess
from tritondse.thread_context import ThreadContext
from tritondse.types          import Architecture, FileDesc
from tritondse.seed           import SeedFormat, SeedStatus, Seed, CompositeField


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

    # Allocate on heap enough to make the table to fit
    alloc_size = 2*pstate.ptr_size + len(ctype)
    base_ctype = pstate.heap_allocator.alloc(alloc_size)

    ctype_table_offset = base_ctype + (pstate.ptr_size * 2)
    otable_offset = ctype_table_offset + 256

    pstate.memory.write_ptr(base_ctype, otable_offset)
    pstate.memory.write_ptr(base_ctype+pstate.ptr_size, 0)

    # FIXME: On pourrait la renvoyer qu'une seule fois ou la charger au demarage direct dans pstate
    pstate.memory.write(ctype_table_offset, ctype)

    return base_ctype

def rtn_ctype_toupper_loc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    # FIXME: Not sure about the array and where to place the pointer 
    # https://codebrowser.dev/glibc/glibc/locale/C-ctype.c.html
    """
    The __ctype_toupper_loc behavior.
    """
    logging.debug('__ctype_toupper_loc hooked')

    ctype  = b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    ctype += b"\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
    ctype += b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
    ctype += b"\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
    ctype += b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
    ctype += b"\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
    ctype += b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
    ctype += b"\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff\xff\xff\xff"
    ctype += b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    ctype += b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    ctype += b"\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f"
    ctype += b"\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f"
    ctype += b"\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f"
    ctype += b"\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f"
    ctype += b"\x60\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f"
    ctype += b"\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x7b\x7c\x7d\x7e\x7f"
    ctype += b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    ctype += b"\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
    ctype += b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
    ctype += b"\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
    ctype += b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
    ctype += b"\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
    ctype += b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
    ctype += b"\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff"

    # Allocate on heap enough to make the table to fit
    alloc_size = 2*pstate.ptr_size + len(ctype)
    base_ctype = pstate.heap_allocator.alloc(alloc_size)

    ctype_table_offset = base_ctype + (pstate.ptr_size * 2)
    otable_offset = ctype_table_offset + 256

    pstate.memory.write_ptr(base_ctype, otable_offset)
    pstate.memory.write_ptr(base_ctype+pstate.ptr_size, 0)

    # FIXME: On pourrait la renvoyer qu'une seule fois ou la charger au demarage direct dans pstate
    pstate.memory.write(ctype_table_offset, ctype)

    return base_ctype


def rtn_errno_location(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __errno_location behavior.
    """
    logging.debug('__errno_location hooked')

    # Errno is a int* ptr, initialize it to zero
    # We consider it is located in the [extern] segment
    # Thus the process must have one of this map
    segs = pstate.memory.find_map(pstate.EXTERN_SEG)
    if segs:
        map = segs[0]
        ERRNO = map.start + map.size - 4  # Point is last int of the mapping
    else:
        assert False
    pstate.memory.write_dword(ERRNO, 0)

    return ERRNO


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
    if se.config.seed_format == SeedFormat.RAW:
        # Cannot provide argv in RAW seeds
        argc = len(se.config.program_argv)
    else: # SeedFormat.COMPOSITE
        argc = len(se.seed.content.argv) if se.seed.content.argv else len(se.config.program_argv)

    if pstate.architecture == Architecture.X86:
        # Because of the "Dirty trick" described above, we RET to main instead of CALLing it.
        # Because of that, the arguments end up 1 slot off on the stack
        pstate.write_argument_value(0 + 1, argc)
    else:
        pstate.write_argument_value(0, argc)
    logging.debug(f"argc = {argc}")

    # Define argv
    addrs = list()

    if se.config.seed_format == SeedFormat.COMPOSITE and se.seed.content.argv: # Use the seed provided (and ignore config.program_argv !!)
        argvs = se.seed.content.argv
    else:  # use the config argv
        argvs = [x.encode("latin-1") for x in se.config.program_argv]  # Convert it from str to bytes

    # Compute the allocation size: size of strings, + all \x00 + all pointers
    size = sum(len(x) for x in argvs)+len(argvs)+len(argvs)*pstate.ptr_size
    if size == 0:  # Falback on a single pointer that will hold not even be initialized
        size = pstate.ptr_size

    # We put the ARGV stuff on the heap even though its normally on stack
    base = pstate.heap_allocator.alloc(size)

    for i, arg in enumerate(argvs):
        addrs.append(base)
        pstate.memory.write(base, arg + b'\x00')

        if se.config.seed_format == SeedFormat.COMPOSITE and se.seed.content.argv: # Use the seed provided (and ignore config.program_argv !!)
            # Symbolize the argv string
            se.inject_symbolic_input(base, arg, f"argv[{i}]", CompositeField.ARGV)
            # FIXME: Shall add a constraint on every char to be != \x00

        logging.debug(f"argv[{i}] = {repr(pstate.memory.read(base, len(arg)))}")
        base += len(arg) + 1


    # NOTE: the array of pointers will be after the string themselves
    b_argv = base
    for addr in addrs:
        pstate.memory.write_ptr(base, addr)
        base += pstate.ptr_size

    # Concrete value
    if pstate.architecture == Architecture.X86:
        # Because of the "Dirty trick" described above, we RET to main instead of CALLing it.
        # Because of that, the arguments end up 1 slot off on the stack
        pstate.write_argument_value(1 + 1, b_argv)
    else:
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

    if os.path.isfile(pstate.memory.read_string(arg1)):
        stat = os.stat(pstate.memory.read_string(arg1))
        pstate.memory.write_qword(arg2 + 0x00, stat.st_dev)
        pstate.memory.write_qword(arg2 + 0x08, stat.st_ino)
        pstate.memory.write_qword(arg2 + 0x10, stat.st_nlink)
        pstate.memory.write_dword(arg2 + 0x18, stat.st_mode)
        pstate.memory.write_dword(arg2 + 0x1c, stat.st_uid)
        pstate.memory.write_dword(arg2 + 0x20, stat.st_gid)
        pstate.memory.write_dword(arg2 + 0x24, 0)
        pstate.memory.write_qword(arg2 + 0x28, stat.st_rdev)
        pstate.memory.write_qword(arg2 + 0x30, stat.st_size)
        pstate.memory.write_qword(arg2 + 0x38, stat.st_blksize)
        pstate.memory.write_qword(arg2 + 0x40, stat.st_blocks)
        pstate.memory.write_qword(arg2 + 0x48, 0)
        pstate.memory.write_qword(arg2 + 0x50, 0)
        pstate.memory.write_qword(arg2 + 0x58, 0)
        pstate.memory.write_qword(arg2 + 0x60, 0)
        pstate.memory.write_qword(arg2 + 0x68, 0)
        pstate.memory.write_qword(arg2 + 0x70, 0)
        pstate.memory.write_qword(arg2 + 0x78, 0)
        pstate.memory.write_qword(arg2 + 0x80, 0)
        pstate.memory.write_qword(arg2 + 0x88, 0)
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

    pstate.memory.write_ptr(tp, int(t))
    pstate.memory.write_ptr(tp+pstate.ptr_size, int(t * 1000000))

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
        fd = pstate.fd_table[arg0].fd
        if fd: fd.close()
        del pstate.fd_table[arg0]
    else:
        return pstate.minus_one

    # Return value
    return 0


def rtn_fseek(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The fseek behavior.
    """

    #define SEEK_SET    0   /* set file offset to offset */
    #define SEEK_CUR    1   /* set file offset to current plus offset */
    #define SEEK_END    2   /* set file offset to EOF plus offset */
    logging.debug('fseek hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)

    if arg2 != 0 and arg2 != 1 and arg2 != 2:
        return -22 # EINVAL

    if arg0 in pstate.fd_table:
        fs = pstate.fd_table[arg0]
        if arg2 == 0: # SEEK_SET
            fs.offset = arg1
        if arg2 == 1: # SEEK_CUR
            fs.offset += arg1
        if arg2 == 2: # SEEK_END
            # Get the file length. There are two options:
            # either the file is in the composite seed or it's not
            if se.config.seed_format == SeedFormat.RAW \
                    or not se.seed.content.files \
                    or pstate.fd_table[arg0].name not in se.seed.content.files:
                # Get the real file's length
                fd = pstate.fd_table[arg0].fd
                filelength = os.fstat(fd.fileno()).st_size
            else: # file content is provided in the seed
                filelength = len(se.seed.content.files[fs.name])

            fs.offset = filelength + arg1
    else: return -22

    return 0


def rtn_ftell(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The ftell behavior.
    """

    logging.debug('ftell hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)

    if arg0 not in pstate.fd_table: 
        return -22 # EINVAL
    else:
        fs = pstate.fd_table[arg0]
        return fs.offset



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
    # We use fd as concret value
    pstate.concretize_argument(2)

    if fd == 0 and se.config.seed_format == SeedFormat.RAW: # symbolize_stdin
        minsize  = (min(len(se.seed.content), size) if se.seed else size)

        if se.is_seed_injected():
            logging.warning("fgets reading stdin, while seed already injected (return EOF)")
            #return 0
        #else:
        # We use fd as concret value
        pstate.push_constraint(size_ast.getAst() == minsize)

        content = se.seed.content[:minsize] if se.seed else b'\x00' * minsize
        se.inject_symbolic_input(buff, content, "stdin")

        logging.debug(f"stdin = {repr(pstate.memory.read(buff, minsize))}")
        return buff_ast

    if fd in pstate.fd_table:
        if se.config.seed_format == SeedFormat.RAW \
                or not se.seed.content.files \
                or pstate.fd_table[fd].name not in se.seed.content.files:
            # We use fd as concret value
            pstate.concretize_argument(1)
            data = (os.read(0, size) if fd == 0 else os.read(pstate.fd_table[fd].fd, size))
            pstate.memory.write(buff, data)
            return buff_ast
        else: # SeedFormat.COMPOSITE
            # File state (name and offset)
            fs = pstate.fd_table[fd]
            filecontent = se.seed.content.files[fs.name][fs.offset:]
            minsize  = min(len(filecontent), size)
            data = filecontent[:minsize]
            se.inject_symbolic_input(buff, data, fs.name, CompositeField.FILE, fs.offset)
            fs.offset += minsize
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
    arg0s = pstate.memory.read_string(arg0)
    arg1s = pstate.memory.read_string(arg1)

    # Concretize the whole path name
    pstate.concretize_memory_bytes(arg0, len(arg0s)+1)  # Concretize the whole string + \0

    # We use mode as concrete value
    pstate.concretize_argument(1)

    try:
        fd = open(arg0s, arg1s)
    except Exception as e:
        logging.debug(f"Failed to open {arg0s} {e}")
        if se.config.seed_format == SeedFormat.RAW \
                or not se.seed.content.files \
                or arg0s not in se.seed.content.files:
            return 0

        # If the file doesn't exist in the real filesystem but is present in the seed,
        # we simulate a success.
        fd = None

    fd_id = se.pstate.get_unique_file_id()
    se.pstate.fd_table.update({fd_id: FileDesc(arg0s, 0, fd, fd_id)})

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
            pstate.fd_table[arg0].fd.write(s)
            pstate.fd_table[arg0].fd.flush()
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
            fd = open(pstate.fd_table[arg1].fd, 'wb+')
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
                sys.stdout.write(pstate.memory.read_string(arg0))
                sys.stdout.flush()
        elif arg1 == 2:
            if se.config.pipe_stderr:
                sys.stderr.write(pstate.memory.read_string(arg0))
                sys.stderr.flush()
        else:
            fd = open(pstate.fd_table[arg1].fd, 'wb+')
            fd.write(pstate.memory.read_string(arg0))
    else:
        return 0

    # Return value
    return len(pstate.memory.read_string(arg0))


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

    # FIXME: pushPathConstraint

    if arg3 == 0 and se.config.seed_format == SeedFormat.RAW: # symbolize_stdin
        minsize  = (min(len(se.seed.content), size) if se.seed else size)

        if se.is_seed_injected():
            logging.warning("fread reading stdin, while seed already injected (return EOF)")
            return 0
        else:
            data = se.seed.content[:minsize] if se.seed else b'\x00' * minsize
            se.inject_symbolic_input(arg0, data, "stdin")
            logging.debug(f"stdin = {repr(pstate.memory.read(arg0, minsize))}")
            # TODO: Could return the read value as a symbolic one
            return minsize

    elif arg3 in pstate.fd_table:
        if se.config.seed_format == SeedFormat.RAW \
                or not se.seed.content.files \
                or pstate.fd_table[arg3].name not in se.seed.content.files:
            data = pstate.fd_table[arg3].fd.read(arg1 * arg2)
            if isinstance(data, str): data = data.encode()
            pstate.memory.write(arg0, data)
        else: # SeedFormat.COMPOSITE and it contains filename
            # File state (name and offset)
            fs = pstate.fd_table[arg3]
            filecontent = se.seed.content.files[fs.name][fs.offset:]
            minsize  = min(len(filecontent), size)
            data = filecontent[:minsize]
            se.inject_symbolic_input(arg0, data, fs.name, CompositeField.FILE, fs.offset)
            fs.offset += minsize

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
    if ptr == 0: # free(NULL) is a nop
        return None
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
    data = pstate.memory.read(arg0, size)

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
            fd = open(pstate.fd_table[arg3].fd, 'wb+')
            fd.write(data)
    else:
        return 0

    # Return value
    return size


def rtn_write(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The write behavior.
    """
    logging.debug('write hooked')

    # Get arguments
    fd = pstate.get_argument_value(0)
    buf = pstate.get_argument_value(1)
    size = pstate.get_argument_value(2)
    data = pstate.memory.read(buf, size)

    if fd in pstate.fd_table:
        if fd == 0:
            return 0
        elif fd == 1:
            if se.config.pipe_stdout:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
        elif fd == 2:
            if se.config.pipe_stderr:
                sys.stderr.buffer.write(data)
                sys.stderr.flush()
        else:
            file_desc = open(pstate.fd_table[fd].fd, 'wb+')
            file_desc.write(data)
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
    pstate.memory.write_ptr(tv, int(t))
    pstate.memory.write_ptr(tv+pstate.ptr_size, int(t * 1000000))

    # Return value
    return 0


def rtn_malloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The malloc behavior.
    """
    logging.debug('malloc hooked')

    # Get arguments
    size = pstate.get_argument_value(0)
    ptr = pstate.heap_allocator.alloc(size)

    # Return value
    return ptr


def rtn_realloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The realloc behavior.
    """
    logging.debug('realloc hooked')

    # Get arguments
    oldptr = pstate.get_argument_value(0)
    size = pstate.get_argument_value(1)

    if oldptr == 0:
        # malloc behaviour
        ptr = pstate.heap_allocator.alloc(size)
        return ptr

    ptr = pstate.heap_allocator.alloc(size)
    if ptr == 0: return ptr

    if ptr not in pstate.heap_allocator.alloc_pool:
        logging.warning("Invalid ptr passed to realloc")
        pstate.heap_allocator.free(ptr) # This will raise an error

    old_memmap = pstate.heap_allocator.alloc_pool[oldptr]
    old_size = old_memmap.size
    size_to_copy = min(size, old_size)

    #data = pstate.memory.read(oldptr, size_to_copy)
    #pstate.memory.write(ptr, data)

    # Copy bytes symbolically
    for index in range(size_to_copy):
        sym_c = pstate.read_symbolic_memory_byte(oldptr+index)
        pstate.write_symbolic_memory_byte(ptr+index, sym_c)

    pstate.heap_allocator.free(oldptr)

    return ptr


def rtn_memcmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memcmp behavior.
    """
    logging.debug('memcmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = pstate.get_argument_value(2)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)

    # We constrain the logical value of size
    pstate.concretize_argument(2)

    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1+index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2+index).getAst()
        res = res + ast.ite(
                        cells1 == cells2,
                        ast.bv(0, ptr_bit_size),
                        ast.bv(1, ptr_bit_size)
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


def rtn_mempcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The mempcpy behavior.
    """
    logging.debug('mempcpy hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    # We constrain the logical value of size
    pstate.concretize_argument(2)

    for index in range(cnt):
        # Read symbolic src value and copy symbolically in dst
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst + cnt


def rtn_memmem(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The memmem behavior.
    """
    logging.debug('memmem hooked')

    haystack    = pstate.get_argument_value(0)  # const void*
    haystacklen = pstate.get_argument_value(1)  # size_t
    needle      = pstate.get_argument_value(2)  # const void *
    needlelen   = pstate.get_argument_value(3)  # size_t

    s1 = pstate.memory.read(haystack, haystacklen)  # haystack
    s2 = pstate.memory.read(needle, needlelen)      # needle

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
        pstate.fd_table[1].fd.write(s)
        pstate.fd_table[1].fd.flush()

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
    pstate.memory.write_ptr(arg0, th.tid)

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
    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_init(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_init behavior.
    """
    logging.debug('pthread_mutex_init hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *restrict mutex
    arg1 = pstate.get_argument_value(1)  # const pthread_mutexattr_t *restrict attr)

    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_lock(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The pthread_mutex_lock behavior.
    """
    logging.debug('pthread_mutex_lock hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # pthread_mutex_t *mutex
    mutex = pstate.memory.read_ptr(arg0)  # deref pointer and read a uint64 int

    # If the thread has been initialized and unused, define the tid has lock
    if mutex == pstate.PTHREAD_MUTEX_INIT_MAGIC:
        logging.debug('mutex unlocked')
        pstate.memory.write_ptr(arg0, pstate.current_thread.tid)

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

    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

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

    if size_ast.isSymbolized():
        logging.warning(f'Reading from the file descriptor ({fd}) with a symbolic size')

    pstate.concretize_argument(0)

    if fd == 0 and se.config.seed_format == SeedFormat.RAW: # symbolize_stdin
        minsize  = (min(len(se.seed.content), size) if se.seed else size)
        if se.is_seed_injected():
            logging.warning("reading stdin, while seed already injected (return EOF)")
        pstate.push_constraint(size_ast.getAst() == minsize)
        content = se.seed.content[:minsize] if se.seed else b'\x00' * minsize
        se.inject_symbolic_input(buff, content, "stdin")

        logging.debug(f"stdin = {repr(pstate.memory.read(buff, minsize))}")
        # TODO: Could return the read value as a symbolic one
        return minsize

    if fd in pstate.fd_table:
        if se.config.seed_format == SeedFormat.RAW\
                or pstate.fd_table[fd] not in se.seed.content:
            pstate.concretize_argument(2)
            data = (os.read(0, size) if fd == 0 else os.read(pstate.filname_table[fd].fd, size))
            pstate.memory.write(buff, data)
            return len(data)
        else: # SeedFormat.COMPOSITE
            filename = pstate.fd_table[fd]
            minsize  = (min(len(se.seed.content.files[filename]), size) if se.seed else size)
            data = se.seed.content.size[filename][:minsize] if se.seed else b'\x00' * minsize
            se.inject_symbolic_input(buff, data, filename, CompositeField.FILE)
            return len(data)

    return 0


def rtn_getchar(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The getchar behavior.
    """
    logging.debug('getchar hooked')

    # Get arguments
    fd   = 0
    fs = pstate.fd_table[fd]

    if se.config.seed_format == SeedFormat.RAW: # symbolize_stdin
        content = int(se.seed.content[fs.offset]) if se.seed and len(se.seed.content) > fs.offset else 0
        se.inject_symbolic_register(pstate.return_register, content, "stdin", None, fs.offset)
    else: # COMPOSITE
        if "stdin" in se.seed.content.files:
            data = se.seed.content.files["stdin"]
            content = int(data[fs.offset]) if len(data) > fs.offset else 0
            if len(data) <= fs.offset:
                logging.warning("getchar reading past the end of the provided 'stdin', returning 0")
            se.inject_symbolic_register(pstate.return_register, content, "stdin", CompositeField.FILE, fs.offset)
        else:
            return 0

    fs.offset += 1


def rtn_sem_destroy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_destroy behavior.
    """
    logging.debug('sem_destroy hooked')

    # Get arguments
    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # Destroy the semaphore with the value
    pstate.memory.write_ptr(arg0, 0)

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

    value = pstate.memory.read_ptr(arg0)  # deref pointer

    # Set the semaphore's value into the output
    pstate.memory.write_dword(arg1, value)  # WARNING: read uint64 to uint32

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
    pstate.memory.write_ptr(arg0, arg2)

    # Return success
    return 0


def rtn_sem_post(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The sem_post behavior.
    """
    logging.debug('sem_post hooked')

    arg0 = pstate.get_argument_value(0)  # sem_t *sem

    # increments (unlocks) the semaphore pointed to by sem
    value = pstate.memory.read_ptr(arg0)
    pstate.memory.write_ptr(arg0, value + 1)

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
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
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
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logging.debug('semaphore locked but continue')
        pstate.semaphore_locked = False

        # Setting errno to EAGAIN (3406)
        segs = pstate.memory.find_map(pstate.EXTERN_SEG)
        if segs:
            map = segs[0]
            ERRNO = map.start + map.size - 4  # Point is last int of the mapping
            pstate.memory.write_dword(ERRNO, 3406)
        else:
            assert False

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
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logging.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
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
    size = min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2)) + 1)

    #s = s1 if len(pstate.memory.read_string(s1)) < len(pstate.memory.read_string(s2)) else s2
    #for i in range(size):
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + len(pstate.memory.read_string(s1)), CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + len(pstate.memory.read_string(s2)), CPUSIZE.BYTE)) == 0x00)

    # FIXME: Il y a des truc chelou avec le +1 et le logic ci-dessous

    ptr_bit_size = pstate.ptr_bit_size
    ast = pstate.actx
    res = ast.bv(0, pstate.ptr_bit_size)
    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strchr(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strchr behavior.
    """
    logging.debug('strchr hooked')

    string = pstate.get_argument_value(0)
    char   = pstate.get_argument_value(1)
    ast    = pstate.actx
    ptr_bit_size = pstate.ptr_bit_size

    def rec(res, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(string + deep).getAst()
        res  = ast.ite(cell == (char & 0xff), ast.bv(string + deep, ptr_bit_size), rec(res, deep + 1, maxdeep))
        return res

    sze = len(pstate.memory.read_string(string))
    res = rec(ast.bv(0, ptr_bit_size), 0, sze)

    for i, c in enumerate(pstate.memory.read_string(string)):
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
    size = min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1
    
    #s = s1 if len(pstate.memory.read_string(s1)) <= len(pstate.memory.read_string(s2)) else s2
    #for i in range(size):
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + i, CPUSIZE.BYTE)) != 0x00)
    #    pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + i, CPUSIZE.BYTE)) != 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s + size, CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s1 + len(pstate.memory.read_string(s1)), CPUSIZE.BYTE)) == 0x00)
    #pstate.tt_ctx.pushPathConstraint(pstate.tt_ctx.getMemoryAst(MemoryAccess(s2 + len(pstate.memory.read_string(s2)), CPUSIZE.BYTE)) == 0x00)

    # FIXME: Il y a des truc chelou avec le +1 et le logic ci-dessous

    ptr_bit_size = pstate.ptr_bit_size
    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strcpy behavior.
    """
    logging.debug('strcpy hooked')

    dst  = pstate.get_argument_value(0)
    src  = pstate.get_argument_value(1)
    src_str = pstate.memory.read_string(src)
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


def rtn_strdup(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strdup behavior.
    """
    logging.debug('strdup hooked')

    s  = pstate.get_argument_value(0)
    s_str = pstate.memory.read_string(s)
    size = len(s_str)

    #print(f"strdup s={s:#x} s_str={s_str} size={size}")

    # constrain src buff to be != \00 and last one to be \00 (indirectly concretize length)
    for i, c in enumerate(s_str):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(s + i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(s + size).getAst() == 0x00)


    # Malloc a chunk
    ptr = pstate.heap_allocator.alloc(size + 1)

    # Copy symbolically bytes (including \00)
    for index in range(size+1):
        sym_c = pstate.read_symbolic_memory_byte(s+index)
        pstate.write_symbolic_memory_byte(ptr+index, sym_c)

    return ptr


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
    pstate.memory.write(ptr, str + b'\0')

    return ptr


def rtn_strlen(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strlen behavior.
    """
    logging.debug('strlen hooked')

    ptr_bit_size = pstate.ptr_bit_size

    # Get arguments
    s = pstate.get_argument_value(0)
    ast = pstate.actx

    # FIXME: Not so sure its is really the strlen semantic
    def rec(res, s, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(s+deep).getAst()
        res  = ast.ite(cell == 0x00, ast.bv(deep, ptr_bit_size), rec(res, s, deep + 1, maxdeep))
        return res

    sze = len(pstate.memory.read_string(s))
    res = ast.bv(sze, ptr_bit_size)
    res = rec(res, s, 0, sze)

    # FIXME: That routine should do something like below to be SOUND !
    # for i, c in enumerate(pstate.memory.read_string(src)):
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
    maxlen = min(sz, min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1) # upper case
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2) # upper case
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strncmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The strncmp behavior.
    """
    logging.debug('strncmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    sz = pstate.get_argument_value(2)
    maxlen = min(sz, min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

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
    saveMem = pstate.memory.read_ptr(saveptr)

    if string == 0:
        string = saveMem

    d = pstate.memory.read_string(delim)
    s = pstate.memory.read_string(string)

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

            pstate.memory.write_char(string + offset + len(token), 0)
            # Save the pointer
            pstate.memory.write_ptr(saveptr, string + offset + len(token) + 1)
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


def rtn_getenv(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The getenv behavior.
    """
    # TODO
    name = pstate.get_argument_value(0)

    if name == 0:
        return 0

    environ_name = pstate.memory.read_string(name)
    logging.warning(f"Target called getenv({environ_name})")
    host_env_val = os.getenv(environ_name)
    return host_env_val if host_env_val is not None else 0


def rtn___assert_fail(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The __assert_fail behavior.
    """
    msg = pstate.get_argument_value(0)

    msg = pstate.memory.read_string(msg)
    logging.warning(f"__assert_fail called : {msg}")
    
def rtn_setlocale(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The setlocale behavior.
    """
    logging.debug('setlocale hooked')

    category = pstate.get_argument_value(0)
    locale   = pstate.get_argument_value(1)

    if locale != 0:
        logging.warning(f"Attempt to modify Locale. Currently not supported.")
        return 0

    # This is a bit hacky but we just store the LOCALEs in the [extern] segment
    segs = pstate.memory.find_map(pstate.EXTERN_SEG)
    if segs:
        map = segs[0]
        LC_ALL = map.start + map.size - 0x20 # Point to the end of seg. But keep in mind LC_ALL is at end - 4.
    else:
        assert False
    print(f"selocale writing at {LC_ALL:#x}")

    if category == 0:
        pstate.memory.write(LC_ALL, b"en_US.UTF-8\x00")
    else:
        logging.warning(f"setlocale called with unsupported category={category}.")

    return LC_ALL


def rtn__setjmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The _setjmp behavior. 
    """
    # TODO
    logging.warning("hooked _setjmp")
    return 0


def rtn_longjmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    """
    The longjmp behavior.
    """
    # NOTE All the programs tested so far used `longjmp` as an error handling mechanism, right
    # before exiting. This is why, `longjmp` is currently considered an exit condition. 
    # TODO Real implementation
    logging.debug('longjmp hooked')
    pstate.stop = True



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

    'write':                   rtn_write,
    'getenv':                  rtn_getenv,
    'fseek':                   rtn_fseek,
    'ftell':                   rtn_ftell,
    '__assert_fail':           rtn___assert_fail,
    '_setjmp':                 rtn__setjmp,
    'longjmp':                 rtn_longjmp,
    'realloc':                 rtn_realloc,
    #'setlocale':               rtn_setlocale,
    'strdup':                  rtn_strdup,
    'mempcpy':                 rtn_mempcpy,
    'getchar':                 rtn_getchar,

    '__ctype_toupper_loc':           rtn_ctype_toupper_loc,
}


SUPORTED_GVARIABLES = {
    '__stack_chk_guard':    0xdead,
    'stderr':               0x0002,
    'stdin':                0x0000,
    'stdout':               0x0001,
}

