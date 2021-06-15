import logging
import time
import threading
import gc
from enum   import Enum
from typing import Union
from pathlib import Path
import stat

from tritondse.config            import Config
from tritondse.process_state     import ProcessState
from tritondse.program           import Program
from tritondse.seed              import Seed
from tritondse.seeds_manager     import SeedManager
from tritondse.worklist          import SeedScheduler
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.callbacks         import CallbackManager
from tritondse.workspace         import Workspace
from tritondse.coverage          import GlobalCoverage
from tritondse.types             import Addr


class ExplorationStatus(Enum):
    """ Enum representing the current state of the exploration """
    NOT_RUNNING = 0
    RUNNING     = 1
    IDLE        = 2
    STOPPED     = 3


class SymbolicExplorator(object):
    """
    Symbolic Exploration. This class is in charge of iterating
    executions with the different seeds available in the workspace
    and generated along the way.
    """
    def __init__(self, config: Config, program: Program = None, seed_scheduler: SeedScheduler = None, executor_stop_at: Addr = None):
        self.program: Program     = program  #: Program being analyzed
        self.config: Config       = config   #: Configuration file
        self.cbm: CallbackManager = CallbackManager()
        self._stop          = False
        self.ts            = time.time()
        self.uid_counter   = 0
        self.status: ExplorationStatus = ExplorationStatus.NOT_RUNNING  #: status of the execution
        self._executor_stop_at = executor_stop_at

        # Initialize the workspace
        self.workspace: Workspace = Workspace(self.config.workspace)  #: workspace object
        self.workspace.initialize(flush=False)

        # Save both the program and configuration in the workspace (for later resume if needed)
        self.workspace.save_file("config.json", self.config.to_json())
        if self.program:  # If the program is not None
            self.workspace.save_file(self.program.path.name, self.program.path.read_bytes())
            self.workspace.get_metadata_file_path(self.program.path.name).chmod(stat.S_IRWXU)  # Set the binary to be executable

        # Configure logfile
        self._configure_file_logger()

        # Initialize coverage
        self.coverage: GlobalCoverage = GlobalCoverage(self.config.coverage_strategy, self.workspace, self.config.branch_solving_strategy)
        """ GlobalCoverage object holding information about the global coverage.
        *(not really meant to be manipulated by the user)*
        """

        # Initialize the seed manager
        self.seeds_manager: SeedManager = SeedManager(self.coverage, self.workspace, self.config.smt_queries_limit, seed_scheduler)
        """ Manager of seed, holding all seeds related data and various statistics """

        # running executors (for debugging purposes)
        self.current_executor: SymbolicExecutor = None  #: last symbolic executor executed

        # General purpose attributes
        self._exec_count = 0

    @property
    def callback_manager(self) -> CallbackManager:
        """
        CallbackManager global instance that will be transmitted to
        all :py:obj:`SymbolicExecutor`.

        :rtype: CallbackManager
        """
        return self.cbm

    @property
    def execution_count(self) -> int:
        """
        Get the number of execution performed.

        :return: number of execution performed
        :rtype: int
        """
        return self._exec_count

    def __time_delta(self):
        return time.time() - self.ts


    def _worker(self, seed, uid):
        """ Worker thread """
        logging.info(f'Pick-up seed: {seed.filename} (fresh: {seed.is_fresh()})')

        if self.config.exploration_timeout and self.__time_delta() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self._stop = True
            return

        # Execute the binary with seeds
        cbs = None if self.cbm.is_empty() else self.cbm.fork()
        logging.info(f"Initialize ProcessState with thread scheduling: {self.config.thread_scheduling}")
        execution = SymbolicExecutor(self.config, seed=seed, workspace=self.workspace, uid=uid, callbacks=cbs)
        if self.program:  # If doing the exploration from a program
            execution.load_program(self.program)
        self.current_executor = execution
        execution.run(self._executor_stop_at)

        # increment exec_count
        self._exec_count += 1

        if self.config.exploration_limit and (uid+1) >= self.config.exploration_limit:
            logging.info('Exploration limit reached')
            self._stop = True

        # Some analysis in post execution
        self.seeds_manager.post_execution(execution, seed, not self._stop)

        logging.info(f"Elapsed time: {self._fmt_elpased(self.__time_delta())}\n")


    def step(self) -> None:
        """
        Perform a single exploration step. That means it execute
        a single :py:obj:`SymbolicExecutor`. Then it gives the hand
        back to the user.
        """
        # Take an input
        seed = self.seeds_manager.pick_seed()

        # Iterate the callback to be called at each steps
        for cb in self.cbm.get_exploration_step_callbacks():
            cb(self)

        # Execution into a thread
        t = threading.Thread(
            name='\033[0;%dm[exec:%08d]\033[0m' % ((31 + (self.uid_counter % 4)), self.uid_counter),
            target=self._worker,
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


    def explore(self) -> ExplorationStatus:
        """
        Start the symbolic exploration. That function
        holds until the exploration is interrupted or finished.

        :returns: the status of the exploration
        :rtype: ExplorationStatus
        """
        self.status = ExplorationStatus.RUNNING

        try:
            while self.seeds_manager.seeds_available() and not self._stop:
                self.step()

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
        """
        Add the given bytes or Seed object as input for the exploration.

        :param seed: input seed to add in the pending inputs to process
        :type seed: Union[bytes, Seed]
        """
        seed = seed if isinstance(seed, Seed) else Seed(seed)
        self.seeds_manager.add_new_seed(seed)


    def stop_exploration(self) -> None:
        """ Interrupt the exploration """
        self._stop = True


    def _fmt_elpased(self, seconds) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return (f"{int(h)}h" if h else '')+f"{int(m)}m{int(s)}s"


    def _configure_file_logger(self) -> None:
        """ Configure the filehandler to log to file """
        hldr = logging.FileHandler(self.workspace.logfile_path)
        hldr.setLevel(logging.DEBUG)
        hldr.setFormatter(logging.Formatter("%(asctime)s %(threadName)s [%(levelname)s] %(message)s"))
        logging.root.addHandler(hldr)
