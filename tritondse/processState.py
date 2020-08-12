#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from triton                  import TritonContext
from tritondse.threadContext import ThreadContext
from tritondse.config        import Config


class ProcessState(object):
    """
    This class is used to represent the state of a process.
    """
    def __init__(self, config : Config):
        # Memory mapping
        self.BASE_PLT   = 0x01000000
        self.BASE_ARGV  = 0x02000000
        self.BASE_ALLOC = 0x03000000
        self.BASE_STACK = 0xefffffff
        self.BASE_LIBC  = 0x04000000
        self.BASE_CTYPE = 0x05000000
        self.START_MAP  = 0x01000000
        self.END_MAP    = 0xf0000000

        # The Triton's context
        self.tt_ctx = TritonContext()

        # Used to define that the process must exist
        self.stop = False

        # Signals table used by raise(), signal(), etc.
        self.signals_table = dict()

        # File descriptors table used by fopen(), fprintf(), etc.
        self.fd_table = {
            0: sys.stdin,
            1: sys.stdout,
            2: sys.stderr,
        }

        # Allocation information used by malloc()
        self.mallocMaxAllocation = 0x03ffffff
        self.mallocBase = self.BASE_ALLOC

        # Unique thread id incrementation
        self.utid = 0

        # Current thread id
        self.tid = self.utid

        # Threads contexts
        self.threads = {
            self.utid: ThreadContext(config, self.tid)
        }

        # Thread mutext init magic number
        self.PTHREAD_MUTEX_INIT_MAGIC = 0xdead

        # Mutex and semaphore
        self.mutex_locked = False
        self.semaphore_locked = False
