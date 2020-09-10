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
        #print('Memory read', mem)
        #se.abort()
        pass


    @staticmethod
    def memory_write(se, pstate, mem, value):
        #print('Memory write', mem)
        pass
