#!/usr/bin/env python
## -*- coding: utf-8 -*-

from tritondse import *

config = Config()
config.symbolize_argv = True

pstate  = ProcessState(config)
program = Program('./samples/crackme_xor', [b'./samples/crackme_xor', b'salut'])
se      = SymbolicExecutor(config, pstate, program)

se.run()
