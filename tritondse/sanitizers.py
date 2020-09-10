#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tritondse.callbacks import CbType


class Sanitizer(object):
    """ The Sanitizer interface """
    def __init__(self):
        self.cbs = dict()


class UAFSanitizer(Sanitizer):
    """ The UAF sanitizer """
    def __init__(self):
        Sanitizer.__init__(self)
        self.cbs[CbType.MEMORY_READ] = self.memory_read
        self.cbs[CbType.MEMORY_WRITE] = self.memory_write


    @staticmethod
    def memory_read(se, pstate, mem):
        #print('Memory read', mem)
        pass


    @staticmethod
    def memory_write(se, pstate, mem, value):
        #print('Memory write', mem)
        pass
