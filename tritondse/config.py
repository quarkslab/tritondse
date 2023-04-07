import logging
import json
from enum import Enum, IntFlag
from pathlib import Path
from typing import List
from functools import reduce

# triton-based libraries
from tritondse.coverage import CoverageStrategy, BranchSolvingStrategy
from tritondse.types import SmtSolver
from tritondse.seed import SeedFormat


class Config(object):
    """
    Data class holding tritondse configuration
    parameter
    """

    def __init__(self,
                 seed_format: SeedFormat = SeedFormat.RAW,
                 pipe_stdout: bool = False,
                 pipe_stderr: bool = False,
                 skip_sleep_routine: bool = False,
                 smt_solver: SmtSolver = SmtSolver.Z3,
                 smt_timeout: int = 5000,
                 execution_timeout: int = 0,
                 exploration_timeout: int = 0,
                 exploration_limit: int = 0,
                 thread_scheduling: int = 200,
                 smt_queries_limit: int = 1200,
                 smt_enumeration_limit: int = 40,
                 coverage_strategy: CoverageStrategy = CoverageStrategy.BLOCK,
                 branch_solving_strategy: BranchSolvingStrategy = BranchSolvingStrategy.FIRST_LAST_NOT_COVERED,
                 debug: bool = False,
                 workspace: str = "",
                 program_argv: List[str] = None,
                 time_inc_coefficient: float = 0.00001,
                 skip_unsupported_import: bool = False,
                 skip_unsupported_instruction: bool = False,
                 memory_segmentation: bool = True):
        """
        :param debug: Enable debugging logging
        :type debug: bool
        """

        self.seed_format: SeedFormat = seed_format
        """ Seed type is either Raw (raw bytes) or Composite (more expressive).
            See seeds.py for more information on each format.
        """
        
        self.pipe_stdout: bool = pipe_stdout
        """ Pipe the program stdout to Python's stdout. *(default: False)*"""

        self.pipe_stderr: bool = pipe_stderr
        """ Pipe the program stderr to Python's stderr *(default: False)*"""

        self.skip_sleep_routine: bool = skip_sleep_routine
        """ Whether to emulate sleeps routine or not *(default: False)*"""

        self.smt_solver: SmtSolver = smt_solver
        """ SMT solver to perform queries solving """

        self.smt_timeout: int = smt_timeout
        """ Timeout for a single SMT query in milliseconds *(default: 10)*"""

        self.execution_timeout: int = execution_timeout
        """ Timeout of a single execution. If it is triggered the associated
        input file is marked as 'hanging'. In seconds, 0 means unlimited *(default: 0)*"""

        self.exploration_timeout: int = exploration_timeout
        """ Overall timeout of the exploration in seconds. 0 means unlimited *(default: 0)* """

        self.exploration_limit: int = exploration_limit
        """ Number of execution iterations. 0 means unlimited *(default: 0)*"""

        self.thread_scheduling: int = thread_scheduling
        """ Number of instructions to execute before switching to the next thread.
        At the moment all threads are scheduled in a round-robin manner *(default: 200)*"""

        self.smt_queries_limit: int = smt_queries_limit
        """ Limit of SMT queries to perform for a single execution *(default: 1200)*"""

        self.smt_enumeration_limit: int = smt_enumeration_limit
        """ Limit of model values retrieved when enumerating a dynamic jump or symbolic memory accesses"""

        self.coverage_strategy: CoverageStrategy = coverage_strategy
        """ Coverage strategy to apply for the whole exploration, default: :py:obj:`CoverageStrategy.BLOCK`"""

        self.branch_solving_strategy: BranchSolvingStrategy = branch_solving_strategy
        """ Branch solving strategy to apply for a single execution. For a given non-covered
        branch allows changing whether we try to solve it at all occurences or more seldomly.
        default: :py:obj:`BranchSolvingStrategy.FIRST_LAST_NOT_COVERED`
        """

        self.debug: bool = debug
        """ Enable debug logging or not. Value taken from constructor.
        FIXME: What if the value is changed during execution ?
        """

        self.workspace: str = workspace
        """ Workspace directory to use. *(default: 'workspace')* """

        self.program_argv: List[str] = [] if program_argv is None else program_argv
        """ Concrete program argument as given on the command line."""

        self.time_inc_coefficient: float = time_inc_coefficient
        """ Time increment coefficient at each instruction to provide a deterministic
        behavior when calling time functions (e.g gettimeofday(), clock_gettime(), ...).
        For example, if 0.0001 is defined, each instruction will increment the time representation
        of the execution by 100us. *(default: 0.00001)*
        """

        self.skip_unsupported_import: bool = skip_unsupported_import
        """ Whether or not to stop the emulation when hitting a external
        call to a function that is not supported.
        """

        self.skip_unsupported_instruction: bool = skip_unsupported_instruction
        """ Whether or not to stop the emulation when hitting an instruction
        for which the semantic is not defined.
        """

        self.memory_segmentation: bool = memory_segmentation
        """ This option defines whether or not memory segmentation is enforced.
        If activated all memory accesses must belong to a mapped memory area.
        """

        self.custom = {}
        """
        Custom carrier field enabling user to add parameters of their own.
        """

    def __str__(self):
        return "\n".join(f"{k.ljust(23)}= {v}" for k, v in self.__dict__.items())

    def to_file(self, file: str) -> None:
        """
        Save the current configuration to a file

        :param file: The path name
        """
        with open(file, "w") as f:
            f.write(self.to_json())

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

        :param s: The JSON text
        :return: A fresh instance of Config
        """
        data = json.loads(s)
        c = Config()
        for k, v in data.items():
            if hasattr(c, k):
                mapping = {"coverage_strategy": CoverageStrategy, "smt_solver": SmtSolver,
                           "seed_format": SeedFormat}
                if k in mapping:
                    v = mapping[k][v]
                elif k == "branch_solving_strategy":
                    v = reduce(lambda acc, x: BranchSolvingStrategy[x] | acc, v, 0)
                setattr(c, k, v)
            else:
                logging.warning(f"config unknown parameter: {k}")
        return c

    def to_json(self) -> str:
        """
        Convert the current configuration to a json output

        :return: JSON text
        """
        def to_str_list(value):
            return [x.name for x in list(BranchSolvingStrategy) if x in value]
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, IntFlag):
                d[k] = to_str_list(v)
            elif isinstance(v, Enum):
                d[k] = v.name
            else:
                d[k] = v
        return json.dumps(d, indent=2)

    def is_format_composite(self) -> bool:
        """ Return true if the seed format is composite"""
        return self.seed_format == SeedFormat.COMPOSITE

    def is_format_raw(self) -> bool:
        """ Return true if the seed format is raw """
        return self.seed_format == SeedFormat.RAW
