#!/usr/bin/env python3

import sys

from tritondse import *

config = Config(debug=True)
config.execution_timeout    = 120   # 2 minutes
config.smt_timeout          = 5000  # 5 seconds
config.smt_queries_limit    = 0
config.thread_scheduling    = 300
config.time_inc_coefficient = 0.00001
config.coverage_strategy    = CoverageStrategy.EDGE_COVERAGE
config.symbolize_stdin      = True

program = Program(sys.argv[1])
seed = Seed(b'this is my initial input')
dse = SymbolicExplorator(config, program, seed)
dse.explore()
