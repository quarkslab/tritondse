#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from tritondse.enums import CoverageStrategy



class Config(object):
    """
    Data class holding tritondse configurations
    """
    def __init__(self, debug=True):
        self.symbolize_argv         = False                           # Not symbolized by default
        self.symbolize_stdin        = False                           # Not symbolized by default
        self.smt_timeout            = 10000                           # 10 seconds by default (milliseconds)
        self.execution_timeout      = 0                               # Unlimited by default (seconds)
        self.exploration_timeout    = 0                               # Unlimited by default (seconds)
        self.exploration_limit      = 0                               # Unlimited by default (number of traces)
        self.thread_scheduling      = 200                             # Number of instructions executed by thread before scheduling
        self.smt_queries_limit      = 2000                            # Limit of SMT queries by execution
        self.coverage_strategy      = CoverageStrategy.CODE_COVERAGE  # Coverage strategy
        self.debug                  = debug                           # Enable debug info by default
        self.corpus_dir             = './corpus'                      # The corpus directory
        self.crash_dir              = './crash'                       # The crash directory
        self.worklist_dir           = './worklist'                    # The worklist directory
        self.metadata_dir           = './metadata'                    # The metadata directory. Contains some data like code already covered, constrains already asked, etc.
        self.program_argv           = list()                          # The program arguments (ex. argv[0], argv[1], etc.). List of Bytes.
        self.time_inc_coefficient   = 0                               # Time increment coefficient at each instruction to provide a deterministic
                                                                      # behavior when calling time functions (e.g gettimeofday(), clock_gettime(), ...).
                                                                      # For example, if 0.0001 is defined, each instruction will increment the time representation
                                                                      # of the execution by 100us.

        logging.basicConfig(format="%(threadName)s\033[0m [%(levelname)s] %(message)s", level=logging.DEBUG if self.debug else logging.INFO)


    def __str__(self):
        s  = f'symbolize_argv       = {self.symbolize_argv}\n'
        s += f'symbolize_stdin      = {self.symbolize_stdin}\n'
        s += f'smt_timeout          = {self.smt_timeout}\n'
        s += f'execution_timeout    = {self.execution_timeout}\n'
        s += f'exploration_timeout  = {self.exploration_timeout}\n'
        s += f'exploration_limit    = {self.exploration_limit}\n'
        s += f'thread_scheduling    = {self.thread_scheduling}\n'
        s += f'smt_queries_limit    = {self.smt_queries_limit}\n'
        s += f'coverage_strategy    = {self.coverage_strategy}\n'
        s += f'debug                = {self.debug}\n'
        s += f'corpus_dir           = {self.corpus_dir}\n'
        s += f'crash_dir            = {self.crash_dir}\n'
        s += f'worklist_dir         = {self.worklist_dir}\n'
        s += f'metadata_dir         = {self.metadata_dir}\n'
        s += f'program_argv         = {self.program_argv}\n'
        s += f'time_inc_coefficient = {self.time_inc_coefficient}'
        return s
