#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tritondse.callbacks import CbType, ProbeInterface


class UAFSanitizer(ProbeInterface):
    """ The UAF sanitizer """
    def __init__(self):
        super(UAFSanitizer, self).__init__()
        self.cbs[CbType.MEMORY_READ] = self.memory_read
        self.cbs[CbType.MEMORY_WRITE] = self.memory_write


    @staticmethod
    def memory_read(se, pstate, mem):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            print(f'UAF detected at {mem}')
            se.abort()


    @staticmethod
    def memory_write(se, pstate, mem, value):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            print(f'UAF detected at {mem}')
            se.abort()



class MemoryCorruptionSanitizer(ProbeInterface):
    """ The memory corruption sanitizer """
    def __init__(self):
        super(MemoryCorruptionSanitizer, self).__init__()
        self.cbs[CbType.MEMORY_READ] = self.memory_read
        self.cbs[CbType.MEMORY_WRITE] = self.memory_write


    @staticmethod
    def memory_read(se, pstate, mem):
        # TODO: Récuperer l'AST de MemoryAcces, puis faire une requete SMT
        # pour savoir si l'adresse peut pointer en dehors d'une zone mappée.
        #
        # FIXME: Faire en sorte que Triton fournit l'AST dans l'objet MemoryAccess
        # envoyé dans les callbacks.
        pass


    @staticmethod
    def memory_write(se, pstate, mem, value):
        # TODO: Récuperer l'AST de MemoryAcces, puis faire une requete SMT
        # pour savoir si l'adresse peut pointer en dehors d'une zone mappée.
        #
        # FIXME: Faire en sorte que Triton fournit l'AST dans l'objet MemoryAccess
        # envoyé dans les callbacks.
        pass
