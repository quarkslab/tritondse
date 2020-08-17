#!/usr/bin/env python
## -*- coding: utf-8 -*-

from tritondse import *

config = Config(debug=False)
config.symbolize_stdin = True
config.execution_timeout = 20
config.program_argv = [
    b'../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln',
    b'wlp0s20f3',
    b'5c:80:b6:96:d7:3c',
    b'192.168.1.45',
    b'255.255.255.0',
    b'192.168.1.255'
]

program = Program('../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln')
seed    = SeedFile('../programme_etalon_final/micro_http_server/private/frame.seed')
dse     = SymbolicExplorator(config, program, seed)

dse.explore()
