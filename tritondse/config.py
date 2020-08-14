#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


class Config(object):
    """
    Data class holding tritondse configurations
    """
    def __init__(self, **args):
        self.symbolize_argv         = False         # Not symbolized by default
        self.symbolize_stdin        = False         # Not symbolized by default
        self.smt_timeout            = 10000         # 10 seconds by default
        self.execution_timeout      = 0             # Unlimited by default
        self.exploration_timeout    = 0             # Unlimited by default
        self.thread_scheduling      = 200           # Number of instructions executed by thread before scheduling
        self.debug                  = True          # Enable debug info by default
        self.corpus_dir             = './corpus'    # The corpus directory
        self.crash_dir              = './crash'     # The crash directory
        self.worklist_dir           = './worklist'  # The worklist directory
        self.metadata_dir           = './metadata'  # The metadata directory. Contains some data like code already covered, constrains already asked, etc.
        self.program_argv           = list()        # The program arguments (ex. argv[0], argv[1], etc.). List of Bytes.

        if 'debug' in args:
            self.debug = args['debug']

        logging.basicConfig(format="%(threadName)s\033[0m [%(levelname)s] %(message)s", level=logging.DEBUG if self.debug else logging.INFO)


    def __str__(self):
        s  = f'symbolize_argv       = {self.symbolize_argv}\n'
        s += f'symbolize_stdin      = {self.symbolize_stdin}\n'
        s += f'smt_timeout          = {self.smt_timeout}\n'
        s += f'execution_timeout    = {self.execution_timeout}\n'
        s += f'exploration_timeout  = {self.exploration_timeout}\n'
        s += f'thread_scheduling    = {self.thread_scheduling}\n'
        s += f'debug                = {self.debug}\n'
        s += f'corpus_dir           = {self.corpus_dir}\n'
        s += f'crash_dir            = {self.crash_dir}\n'
        s += f'worklist_dir         = {self.worklist_dir}\n'
        s += f'data_dir             = {self.data_dir}\n'
        s += f'program_argv         = {self.program_argv}'
        return s
