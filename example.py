#!/usr/bin/env python3

import sys

from tritondse import *


def trace_debug(se: SymbolicExecutor, state: ProcessState, instruction: 'Instruction'):
    print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))


config = Config(debug=False)
config.execution_timeout    = 120   # 2 minutes
config.smt_timeout          = 5000  # 5 seconds
config.smt_queries_limit    = 0
config.thread_scheduling    = 400
config.time_inc_coefficient = 0.00001
config.coverage_strategy    = CoverageStrategy.EDGE_COVERAGE
config.symbolize_stdin      = True
config.program_argv         = [
    '../cyclonetcp_harness/harness/harness_triton_vuln_ON'
]

try:
    program = Program('../cyclonetcp_harness/harness/harness_triton_vuln_ON')
except FileNotFoundError as e:
    print(e)
    sys.exit(-1)

seed = Seed.from_file('../cyclonetcp_harness/harness/misc/new_frame.seed')

dse = SymbolicExplorator(config, program)
dse.add_input_seed(seed)
dse.explore()

#ps = ProcessState(config.thread_scheduling, config.time_inc_coefficient)
#execution = SymbolicExecutor(config, ps, program, seed)
##execution.callback_manager.register_post_instuction_callback(trace_debug)
#execution.run()
