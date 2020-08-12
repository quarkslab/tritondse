#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


class Config(object):
    """
    Data class holding tritondse configurations
    """
    def __init__(self):
        self.symbolize_argv         = False     # not symbolized by default
        self.symbolize_stdin        = False     # Not symbolized by default
        self.smt_timeout            = 10        # 10 seconds by default
        self.execution_timeout      = 0         # unlimited by default
        self.exploration_timeout    = 0         # unlimited by default
        self.thread_scheduling      = 200       # Number of instructions executed by thread before scheduling
        self.debug_info             = True

        logging.basicConfig(format="%(threadName)s\033[0m [%(levelname)s] %(message)s", level=logging.DEBUG if self.debug_info else logging.INFO)


    def __str__(self):
        s  = f'symbolize_argv       = {self.symbolize_argv}\n'
        s += f'symbolize_stdin      = {self.symbolize_stdin}\n'
        s += f'smt_timeout          = {self.smt_timeout}\n'
        s += f'execution_timeout    = {self.execution_timeout}\n'
        s += f'exploration_timeout  = {self.exploration_timeout}\n'
        s += f'thread_scheduling    = {self.thread_scheduling}\n'
        s += f'debug_info           = {self.debug_info}'
        return s
