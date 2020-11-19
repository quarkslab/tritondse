import logging
import time
import threading
import gc
from enum   import Enum
from typing import Union

from tritondse.config            import Config
from tritondse.process_state     import ProcessState
from tritondse.program           import Program
from tritondse.seed              import Seed
from tritondse.seeds_manager     import SeedManager
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.callbacks         import CallbackManager
from tritondse.workspace         import Workspace
from tritondse.coverage          import GlobalCoverage


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
        self._stop          = False
        self.ts            = time.time()
        self.uid_counter   = 0
        self.status = ExplorationStatus.NOT_RUNNING

        # Initialize the workspace
        self.workspace = Workspace(self.config.workspace)
        self.workspace.initialize(flush=False)

        # Save both the program and configuration in the workspace (for later resume if needed)
        self.workspace.save_file("config.json", self.config.to_json())
        self.workspace.save_file(self.program.path.name, self.program.path.read_bytes())

        # Initialize coverage
        self.coverage = GlobalCoverage(self.config.coverage_strategy, self.workspace)

        # Initialize the seed manager
        self.seeds_manager = SeedManager(self.config, self.cbm, self.coverage, self.workspace)

        # running executors (for debugging purposes)
        self.last_executors = None

    @property
    def callback_manager(self) -> CallbackManager:
        return self.cbm


    def __time_delta(self):
        return time.time() - self.ts


    def worker(self, seed, uid):
        """ Worker thread """
        logging.info(f'Pick-up seed: {seed.filename} (fresh: {seed.is_fresh()})')

        if self.config.exploration_timeout and self.__time_delta() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self._stop = True
            return

        # Execute the binary with seeds
        cbs = None if self.cbm.is_empty() else self.cbm.fork()
        logging.info(f"Initialize ProcessState with thread scheduling: {self.config.thread_scheduling}")
        pstate = ProcessState(self.config.thread_scheduling, self.config.time_inc_coefficient)
        execution = SymbolicExecutor(self.config, pstate, self.program, seed=seed, workspace=self.workspace, uid=uid, callbacks=cbs)
        self.last_executors = execution
        execution.run()

        if self.config.exploration_limit and (uid+1) >= self.config.exploration_limit:
            logging.info('Exploration limit reached')
            self._stop = True
            return

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed)

        logging.info(f"Elapsed time: {self._fmt_elpased(self.__time_delta())}\n")


    def explore(self) -> ExplorationStatus:
        self.status = ExplorationStatus.RUNNING

        try:
            while self.seeds_manager.seeds_available() and not self._stop:
                # Take an input
                seed = self.seeds_manager.pick_seed()

                # Execution into a thread
                t = threading.Thread(
                        name='\033[0;%dm[exec:%08d]\033[0m' % ((31 + (self.uid_counter % 4)), self.uid_counter),
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

        # Call all termination functions
        self.seeds_manager.post_exploration()
        self.coverage.post_exploration()
        logging.info(f"Total time of the exploration: {self._fmt_elpased(self.__time_delta())}")

        return self.status

    def add_input_seed(self, seed: Union[bytes, Seed]) -> None:
        """ Add the given bytes as input for the exploration """
        seed = seed if isinstance(seed, Seed) else Seed(seed)
        self.seeds_manager.add_new_seed(seed)

    def stop_exploration(self) -> None:
        """ Interrupt exploration """
        self._stop = True

    def _fmt_elpased(self, seconds) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return (f"{h}s" if h else '')+f"{int(m)}m{int(s)}s"
