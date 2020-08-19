#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import threading

from triton                      import *
from tritondse.config            import Config
from tritondse.process_state     import ProcessState
from tritondse.program           import Program
from tritondse.seed              import Seed, SeedFile
from tritondse.seeds_manager     import SeedsManager
from tritondse.enums             import Enums
from tritondse.symbolic_executor import SymbolicExecutor


class SymbolicExplorator(object):
    """
    This class is used to represent the symbolic exploration.
    """
    def __init__(self, config : Config, program : Program, seed : Seed = Seed()):
        self.program       = program
        self.config        = config
        self.seeds_manager = SeedsManager(self.config, seed)
        self.stop          = False
        self.total_exec    = 0
        self.ts            = time.time()


    def __time_delta(self):
        return time.time() - self.ts


    def worker(self, seed):
        """ Worker thread """
        # Inc number of trace execution
        self.total_exec += 1

        if self.config.exploration_timeout and self.__time_delta() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self.stop = True
            return

        # Execute the binary with seeds
        execution = SymbolicExecutor(self.config, ProcessState(self.config), self.program, seed)
        execution.run()

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed)

        logging.info('Total time of the exploration: %f seconds' % (self.__time_delta()))


    def explore(self):
        # TODO: We could run several threads. Number of threads run may be defined in Config.
        while self.seeds_manager.worklist and not self.stop:
            # Take an input
            seed = self.seeds_manager.pick_seed()

            # Execution into a thread
            t = threading.Thread(
                    name='\033[0;%dm[exec:%08d]' % ((31 + (self.total_exec % 4)), self.total_exec),
                    target=self.worker,
                    args=[seed],
                    daemon=True
                )
            t.start()

            try:
                t.join()
            except KeyboardInterrupt:
                logging.warning("keyboard interrupt, stop symbolic exploration")
                self.stop = True
