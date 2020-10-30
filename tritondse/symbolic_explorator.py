import logging
import time
import threading
import gc
from enum import Enum

from tritondse.config            import Config
from tritondse.process_state     import ProcessState
from tritondse.program           import Program
from tritondse.seed              import Seed, SeedFile
from tritondse.seeds_manager     import SeedsManager
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.callbacks         import CallbackManager


class ExplorationStatus(Enum):
    """ Enum representing the current state of the exploration """
    NOT_RUNNING = 0
    RUNNING     = 1
    IDLE        = 2
    STOPPED     = 3


class SymbolicExplorator(object):
    """
    This class is used to represent the symbolic exploration.
    """
    def __init__(self, config: Config, program: Program):
        self.program       = program
        self.config        = config
        self.cbm           = CallbackManager(program)
        self.seeds_manager = SeedsManager(self.config, self.cbm)
        self._stop          = False
        self.ts            = time.time()
        self.uid_counter   = 0
        self.status = ExplorationStatus.NOT_RUNNING


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
            self._stop = True
            return

        # Execute the binary with seeds
        cbs = None if self.cbm.is_empty() else self.cbm.fork()
        execution = SymbolicExecutor(self.config, ProcessState(self.config), self.program, seed=seed, uid=uid, callbacks=cbs)
        execution.run()

        if self.config.exploration_limit and (uid+1) >= self.config.exploration_limit:
            logging.info('Exploration limit reached')
            self._stop = True
            return

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed)

        logging.info('Total time of the exploration: %f seconds' % (self.__time_delta()))


    def explore(self) -> ExplorationStatus:
        self.status = ExplorationStatus.RUNNING

        try:
            while self.seeds_manager.worklist and not self._stop:
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

                while True:
                    t.join(0.001)
                    if not t.is_alive():
                        break
                gc.collect()  # FIXME: Why we have to force the collect to avoid memory leak?

            # Exited loop
            self.status = ExplorationStatus.STOPPED if self._stop else ExplorationStatus.IDLE

        except KeyboardInterrupt:
            logging.warning("keyboard interrupt, stop symbolic exploration")
            self.status = ExplorationStatus.STOPPED

        return self.status

    def add_input_seed(self, data: bytes) -> None:
        """ Add the given bytes as input for the exploration """
        self.seeds_manager.add_seed(Seed(data))

    def stop_exploration(self) -> None:
        """ Interrupt exploration """
        self._stop = True
