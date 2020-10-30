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
        self.ts            = time.time()
        self.uid_counter   = 0


    @property
    def callback_manager(self) -> CallbackManager:
        return self.cbm


    def __time_delta(self):
        return time.time() - self.ts


    def worker(self, seed, uid):
        """ Worker thread """
        logging.info(f'Pickuping {self.config.worklist_dir}/{seed.get_file_name()}')

        if self.config.exploration_timeout and self.__time_delta() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self.stop = True
            return

        # Execute the binary with seeds
        execution = SymbolicExecutor(self.config, ProcessState(self.config), self.program, seed=seed, uid=uid, callbacks=self.cbm)
        execution.run()

        if self.config.exploration_limit and uid >= self.config.exploration_limit:
            logging.info('Exploration limit reached')
            self.stop = True
            return

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed)

        logging.info('Total time of the exploration: %f seconds' % (self.__time_delta()))


    def explore(self):
        while self.seeds_manager.worklist and not self.stop:

            # Take an input
            seed = self.seeds_manager.pick_seed()

            # Execution into a thread
            t = threading.Thread(
                    name='\033[0;%dm[exec:%08d]' % ((31 + (self.uid_counter % 4)), self.uid_counter),
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
