#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import threading
import gc

from triton                      import *
from tritondse.config            import Config
from tritondse.process_state     import ProcessState
from tritondse.program           import Program
from tritondse.seed              import Seed, SeedFile
from tritondse.seeds_manager     import SeedsManager
from tritondse.enums             import Enums
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.callbacks         import CallbackManager


class SymbolicExplorator(object):
    """
    This class is used to represent the symbolic exploration.
    """
    def __init__(self, config: Config, program: Program, seed: Seed = Seed()):
        self.program       = program
        self.config        = config
        self.cbm           = CallbackManager(program)
        self.seeds_manager = SeedsManager(self.config, self.cbm, seed)
        self.stop          = False
        self.total_exec    = 0
        self.ts            = time.time()
        self.uid_counter   = 0


    def __time_delta(self):
        return time.time() - self.ts


    @property
    def callback_manager(self) -> CallbackManager:
        return self.cbm


    def worker(self, seed, uid):
        """ Worker thread """
        # Inc number of trace execution
        self.total_exec += 1

        if self.config.exploration_timeout and self.__time_delta() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self.stop = True
            return

        # Execute the binary with seeds
        cbs = None if self.cbm.is_empty() else self.cbm.fork()
        execution = SymbolicExecutor(self.config, ProcessState(self.config), self.program, seed=seed, uid=uid, callbacks=cbs)
        execution.run()

        if self.config.exploration_limit and self.total_exec >= self.config.exploration_limit:
            logging.info('Exploration limit reached')
            self.stop = True
            return

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed)

        logging.info('Total time of the exploration: %f seconds' % (self.__time_delta()))


    def explore(self):
        # TODO: We could run several threads. Number of threads run may be defined in Config.
        # Note: J'ai test, c'est pas forcement plus rapide... En Python ce n'est pas des vrais threads.
        while self.seeds_manager.worklist and not self.stop:
            # Take an input
            seed = self.seeds_manager.pick_seed()

            # Execution into a thread

            t = threading.Thread(
                    name='\033[0;%dm[exec:%08d]' % ((31 + (self.total_exec % 4)), self.total_exec),
                    target=self.worker,
                    args=[seed, self.uid_counter],
                    daemon=True
                )
            t.start()
            self.uid_counter += 1

            try:
                t.join()
                gc.collect()  # FIXME: Why we have to force the collect to avoid memory leak?
            except KeyboardInterrupt:
                logging.warning("keyboard interrupt, stop symbolic exploration")
                self.stop = True
