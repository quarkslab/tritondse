#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import lief

from triton                 import *
from tritondse.config       import Config
from tritondse.processState import ProcessState
from tritondse.program      import Program
from tritondse.routines     import *


class ELFLoader(object):
    """
    This class is used to represent the ELF loader mechanism.
    """
    def __init__(self, config : Config, program : Program, pstate : ProcessState):
        self.program = program
        self.pstate  = pstate
        self.config  = config

        # Mapping of the .plt and .got to the Python routines
        self.routines_table = dict()
        self.plt = [
            ['__ctype_b_loc',           rtn_ctype_b_loc,            None],
            ['__errno_location',        rtn_errno_location,         None],
            ['__libc_start_main',       rtn_libc_start_main,        None],
            ['__stack_chk_fail',        rtn_stack_chk_fail,         None],
            ['__xstat',                 rtn_xstat,                  None],
            ['clock_gettime',           rtn_clock_gettime,          None],
            ['exit',                    rtn_exit,                   None],
            ['fclose',                  rtn_fclose,                 None],
            ['fopen',                   rtn_fopen,                  None],
            ['fprintf',                 rtn_fprintf,                None],
            ['fputc',                   rtn_fputc,                  None],
            ['fputs',                   rtn_fputs,                  None],
            ['fread',                   rtn_fread,                  None],
            ['free',                    rtn_free,                   None],
            ['fwrite',                  rtn_fwrite,                 None],
            ['gettimeofday',            rtn_gettimeofday,           None],
            ['malloc',                  rtn_malloc,                 None],
            ['memcmp',                  rtn_memcmp,                 None],
            ['memcpy',                  rtn_memcpy,                 None],
            ['memmove',                 rtn_memmove,                None],
            ['memset',                  rtn_memset,                 None],
            ['pthread_create',          rtn_pthread_create,         None],
            ['pthread_exit',            rtn_pthread_exit,           None],
            ['pthread_join',            rtn_pthread_join,           None],
            ['pthread_mutex_destroy',   rtn_pthread_mutex_destroy,  None],
            ['pthread_mutex_init',      rtn_pthread_mutex_init,     None],
            ['pthread_mutex_lock',      rtn_pthread_mutex_lock,     None],
            ['pthread_mutex_unlock',    rtn_pthread_mutex_unlock,   None],
            ['puts',                    rtn_puts,                   None],
            ['read',                    rtn_read,                   None],
            ['sem_destroy',             rtn_sem_destroy,            None],
            ['sem_getvalue',            rtn_sem_getvalue,           None],
            ['sem_init',                rtn_sem_init,               None],
            ['sem_post',                rtn_sem_post,               None],
            ['sem_timedwait',           rtn_sem_timedwait,          None],
            ['sem_trywait',             rtn_sem_trywait,            None],
            ['sem_wait',                rtn_sem_wait,               None],
            ['sleep',                   rtn_sleep,                  None],
            ['sprintf',                 rtn_sprintf,                None],
            ['strcasecmp',              rtn_strcasecmp,             None],
            ['strchr',                  rtn_strchr,                 None],
            ['strcmp',                  rtn_strcmp,                 None],
            ['strcpy',                  rtn_strcpy,                 None],
            ['strlen',                  rtn_strlen,                 None],
            ['strncpy',                 rtn_strncpy,                None],
            ['strtok_r',                rtn_strtok_r,               None],
        ]
        self.gvariables = {
            'stderr': 2,
        }


    def __loading__(self):
        phdrs = self.program.binary.segments
        for phdr in phdrs:
            size  = phdr.physical_size
            vaddr = phdr.virtual_address
            if size:
                logging.debug('Loading 0x%08x - 0x%08x' %(vaddr, vaddr+size))
                self.pstate.tt_ctx.setConcreteMemoryAreaValue(vaddr, phdr.content)


    def __dynamic_relocation__(self, vaddr : int = 0):
        # Initialize our routines table
        for index in range(len(self.plt)):
            self.plt[index][2] = self.pstate.BASE_PLT + index
            self.routines_table.update({self.pstate.BASE_PLT + index: self.plt[index][1]})

        # Initialize the pltgot
        try:
            for rel in self.program.binary.pltgot_relocations:
                symbolName = rel.symbol.name
                symbolRelo = vaddr + rel.address
                for crel in self.plt:
                    if symbolName == crel[0]:
                        logging.debug('Hooking %s at %#x' %(symbolName, symbolRelo))
                        self.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(symbolRelo, CPUSIZE.QWORD), crel[2])
        except:
            logging.error('Something wrong with the pltgot relocations')

        try:
            for rel in self.program.binary.dynamic_relocations:
                symbolName = rel.symbol.name
                symbolRelo = vaddr + rel.address
                for crel in self.plt:
                    if symbolName == crel[0]:
                        logging.debug('Hooking %s at %#x' %(symbolName, symbolRelo))
                        self.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(symbolRelo, CPUSIZE.QWORD), crel[2])
        except:
            logging.error('Something wrong with the dynamic relocations')

        for k, v in self.gvariables.items():
            try:
                vaddr = self.program.binary.get_symbol(k).value
                logging.debug('Hooking %s at %#x' % (k, vaddr))
                self.pstate.tt_ctx.setConcreteMemoryValue(MemoryAccess(vaddr, self.pstate.tt_ctx.getGprSize()), 2)
            except:
                logging.debug('Cannot find the symbol %s' %(k))

        return


    def ld(self):
        self.__loading__()
        self.__dynamic_relocation__()
