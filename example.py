#!/usr/bin/env python3

import sys

from tritondse import *


def trace_debug(se: SymbolicExecutor, state: ProcessState, instruction: 'Instruction'):
    print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))


config = Config(debug=False)
config.execution_timeout    = 120   # 2 minutes
config.smt_timeout          = 5000  # 5 seconds
config.smt_queries_limit    = 0
config.thread_scheduling    = 4
config.time_inc_coefficient = 0.00001
config.coverage_strategy    = CoverageStrategy.EDGE_COVERAGE
config.symbolize_stdin      = True
config.program_argv         = [
    b'../aarch64/micro_http_server_tt_fuzz_single_without_vuln',
    #b'../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_with_vuln',
    b'wlp0s20f3',
    b'48:e2:44:f5:9b:01',
    b'10.0.13.86',
    b'255.255.255.0',
    b'10.0.13.254'
]

try:
    program = Program('../aarch64/micro_http_server_tt_fuzz_single_without_vuln')
    #program = Program('../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_with_vuln')
except FileNotFoundError as e:
    print(e)
    sys.exit(-1)

seed = SeedFile('../programme_etalon_final/micro_http_server/misc/frame.seed')

#dse = SymbolicExplorator(config, program, seed)
#dse.explore()

ps = ProcessState(config)
execution = SymbolicExecutor(config, ps, program, seed)
#execution.callback_manager.register_post_instuction_callback(trace_debug)
execution.run()
