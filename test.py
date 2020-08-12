#!/usr/bin/env python
## -*- coding: utf-8 -*-

from tritondse import *

config  = Config()
pstate  = ProcessState(config)
seed    = Seed(b'salut')
program = Program('./samples/crackme_xor', [b'./samples/crackme_xor', b'salut'])
se      = SymbolicExecutor(config, program, pstate, seed)

se.run()
