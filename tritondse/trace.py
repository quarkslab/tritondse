# built-in imports
import json
import logging
import os
import subprocess
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Optional, Union
from collections import Counter


# local imports
import tritondse # NOTE We need this import so we can use it to determine the path of this file.
from tritondse import Config, Program, SymbolicExecutor, CoverageStrategy, CoverageSingleRun



class TraceException(Exception):
    pass


class Trace:
    def __init__(self):
        pass

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str], output_path: str, stdin_file=None) -> bool:
        """Run the binary passed as argument and return the coverage.

        :param strategy: Coverage strategy.
        :type strategy: :py:obj:`CoverageStrategy`.
        :param binary_path: Path to the binary.
        :type binary_path: :py:obj:`str`.
        :param args: List of arguments to pass to the binary.
        :type args: :py:obj:`List[str]`.
        :type output_path: File where to store trace
        :param stdin_file: Path to the file that will act as stdin.
        :type args: :py:obj:`str`.
        """
        raise NotImplementedError()

    @staticmethod
    def from_file(file: Union[str, Path]) -> 'QBDITrace':
        raise NotImplementedError()

    @property
    def coverage(self) -> CoverageSingleRun:
        """
        Coverage generated by the trace

        :return: CoverageSingleRun object
        """
        raise NotImplementedError()

    def get_coverage(self) -> CoverageSingleRun:
        """Return the execution coverage.

        :return: :py:obj:`CoverageSingleRun`.
        """
        return self.coverage

    @property
    def strategy(self) -> CoverageStrategy:
        """
        Return the coverage strategy with which this trace
        was generated with.

        :return: :py:obj:`CoverageStrategy`
        """
        return self.coverage.strategy


class TritonTrace(Trace):

    def __init__(self):
        super().__init__()

        self._coverage = None

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str], output_path: str, stdin_file=None) -> bool:
        # Override stdin with the input file.
        if stdin_file:
            os.dup2(os.open(stdin_file, os.O_RDONLY), 0)

        config = Config(coverage_strategy=strategy)

        se = SymbolicExecutor(config)

        se.load(Program(binary_path))

        se.run()

        trace = TritonTrace()
        trace._coverage = se.coverage
        # FIXME: Writing the coverage to a file

    @staticmethod
    def from_file(file: Union[str, Path]) -> 'QBDITrace':
        # FIXME: Reading coverage file from a file
        return trace

    @property
    def coverage(self) -> CoverageSingleRun:
        return self._coverage



class QBDITrace(Trace):

    QBDI_SCRIPT_FILEPATH = Path(tritondse.__file__).parent / 'qbdi_trace.py'

    def __init__(self):
        super().__init__()
        self._coverage = None

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str], output_path: str, stdin_file=None, timeout=None, cwd=None) -> bool:
        if not Path(binary_path).exists():
            raise FileNotFoundError()

        if stdin_file and not Path(stdin_file).exists():
            raise FileNotFoundError()

        args = [] if not args else args

        cmdlne = f'python -m pyqbdipreload {QBDITrace.QBDI_SCRIPT_FILEPATH}'.split(' ') + [binary_path] + args

        logging.debug(f'Command line: {" ".join(cmdlne)}')

        # Set environment variables.
        environ = {
            'PYQBDIPRELOAD_COVERAGE_STRATEGY': strategy.name,
            'PYQBDIPRELOAD_OUTPUT_FILEPATH': output_path,
            'PYQBDIPRELOAD_LONGJMP_ADDR': os.getenv("TT_LONGJMP_ADDR", default="0"),
        }
        environ.update(os.environ)

        # Open stdin file if it is present.
        stdin_fp = open(stdin_file, 'rb') if stdin_file else None

        # Run QBDI tool.
        process = subprocess.Popen(cmdlne, stdin=stdin_fp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=environ)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            logging.debug(stdout)
            logging.debug(stderr)
        except subprocess.TimeoutExpired:
            process.wait()
            logging.error('QBDI tracer timeout expired!')
            raise TraceException('QBDI tracer timeout expired')

        if stdin_fp:
            stdin_fp.close()

        return Path(output_path).exists()

    @staticmethod
    def from_file(coverage_path: str) -> 'QBDITrace':
        """Load coverage from a file.

        :param coverage_path: Path to the coverage file.
        :type coverage_path: :py:obj:`str`.
        """
        trace = QBDITrace()

        logging.debug(f'Loading coverage file: {coverage_path}')
        with open(coverage_path, 'rb') as fd:
            data = json.load(fd)

        cov = CoverageSingleRun(CoverageStrategy[data["coverage_strategy"]])
        cov.covered_instructions = Counter(data["covered_instructions"])

        for (src, dst, not_taken) in data["covered_items"]:
            if not_taken is None:
                cov.add_covered_dynamic_branch(src, dst)
            else:
                cov.add_covered_branch(src, dst, not_taken)

        trace._coverage = cov

        return trace

    @property
    def coverage(self) -> CoverageSingleRun:
        if not self._coverage:
            logging.warning("Please .run() the trace before querying coverage")

        return self._coverage


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: trace.py program [args]")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG)

    if QBDITrace.run(CoverageStrategy.EDGE, sys.argv[1], sys.argv[2:], "/tmp/test.cov"):
        coverage = QBDITrace.from_file("/tmp/test.cov")
    else:
        print("Something went wrong during trace generation")
