import logging
import json
from enum import Enum
from pathlib import Path

# triton-based libraries
from tritondse.coverage import CoverageStrategy, BranchCheckStrategy



class Config(object):
    """
    Data class holding tritondse configurations
    """
    def __init__(self, debug=True):
        self.symbolize_argv         = False                                       # Not symbolized by default
        self.symbolize_stdin        = False                                       # Not symbolized by default
        self.pipe_stdout            = False                                       # Whether to forward program stdout on current output
        self.pipe_stderr            = False                                       # Whether to forward program stderrr on current stderr
        self.skip_sleep_routine     = False                                       # If True, sleeps are not waiting
        self.smt_timeout            = 5000                                        # 10 seconds by default (milliseconds)
        self.execution_timeout      = 120                                         # Unlimited by default (seconds)
        self.exploration_timeout    = 0                                           # Unlimited by default (seconds)
        self.exploration_limit      = 0                                           # Unlimited by default (number of traces)
        self.thread_scheduling      = 200                                         # Number of instructions executed by thread before scheduling
        self.smt_queries_limit      = 1200                                        # Limit of SMT queries by execution
        self.coverage_strategy      = CoverageStrategy.CODE_COVERAGE              # Coverage strategy
        self.branch_solving_strategy= BranchCheckStrategy.FIRST_LAST_NOT_COVERED  # Only checks the first and last branch
        self.debug                  = debug                                       # Enable debug info by default
        self.workspace              = "workspace"                                 # Workspace directory
        self.program_argv           = list()                                      # The program arguments (ex. argv[0], argv[1], etc.). List of Bytes.
        self.time_inc_coefficient   = 0.00001                                     # Time increment coefficient at each instruction to provide a deterministic
                                                                                  # behavior when calling time functions (e.g gettimeofday(), clock_gettime(), ...).
                                                                                  # For example, if 0.0001 is defined, each instruction will increment the time representation
                                                                                  # of the execution by 100us.


    def __str__(self):
        return "\n".join(f"{k.ljust(21)}= {v}" for k, v in self.__dict__.items())


    def to_file(self, file: str) -> None:
        """
        Save the current configuration to a file

        :param file: The path name
        :return: None
        """
        with open(file, "w") as f:
            json.dump({k: (x.name if isinstance(x, Enum) else x) for k, x in self.__dict__.items()}, f, indent=2)


    @staticmethod
    def from_file(file: str) -> 'Config':
        """
        Load a configuration from a file to a new instance of Config

        :param file: The path name
        :return: A fresh instance of Config
        """
        raw = Path(file).read_text()
        return Config.from_json(raw)


    @staticmethod
    def from_json(s: str) -> 'Config':
        """
        Load a configuration from a json input to a new instance of Config

        :param s: The json text
        :return: A fresh instance of Config
        """
        data = json.loads(s)
        c = Config()
        for k, v in data.items():
            if hasattr(c, k):
                if k == "coverage_strategy":
                    v = CoverageStrategy[v]
                setattr(c, k, v)
            else:
                logging.warning(f"config unknown parameter: {k}")
        return c


    def to_json(self) -> str:
        """
        Convert the current configuration to a json output
        :return: json text
        """
        return json.dumps({k: (x.name if isinstance(x, Enum) else x) for k, x in self.__dict__.items()}, indent=2)
