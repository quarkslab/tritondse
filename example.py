#!/usr/bin/env python3

import sys

from tritondse import *


def trace_debug(se: SymbolicExecutor, state: ProcessState, instruction: 'Instruction'):
    print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))


config = Config(debug=False)
config.execution_timeout    = 120   # 2 minutes
config.smt_timeout          = 5000  # 5 seconds
config.smt_queries_limit    = 0
config.thread_scheduling    = 123
config.time_inc_coefficient = 0.00001
config.coverage_strategy    = CoverageStrategy.EDGE_COVERAGE
config.symbolize_stdin      = True
config.program_argv         = [
    #b'../aarch64/micro_http_server_tt_fuzz_single_without_vuln',
    '../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_with_vuln',
    'wlp0s20f3',
    '48:e2:44:f5:9b:01',
    '10.0.13.86',
    '255.255.255.0',
    '10.0.13.254'
    #b'/home/jonathan/Works/QuarksLab/Missions/pastis/programme_etalon_etatique/bin/target'
]

try:
    #program = Program('../aarch64/micro_http_server_tt_fuzz_single_without_vuln')
    program = Program('../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_with_vuln')
    #program = Program('/home/jonathan/Works/QuarksLab/Missions/pastis/programme_etalon_etatique/bin/target')
except FileNotFoundError as e:
    print(e)
    sys.exit(-1)

seed = Seed.from_file('../programme_etalon_final/micro_http_server/misc/frame.seed')
#seed = SeedFile('/home/jonathan/Works/QuarksLab/Missions/pastis/programme_etalon_etatique/fuzzing/in/tcp_echo_1')
#seed = SeedFile('/home/jonathan/Works/QuarksLab/Missions/pastis/programme_etalon_etatique/fuzzing/in/frame_ip4_tcp_syn')

#dse = SymbolicExplorator(config, program, seed)
#dse.explore()

ps = ProcessState(config.thread_scheduling, config.time_inc_coefficient)
execution = SymbolicExecutor(config, ps, program, seed)
#execution.callback_manager.register_post_instuction_callback(trace_debug)
execution.run()
