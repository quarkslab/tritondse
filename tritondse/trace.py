import json
import atexit
import bisect
import logging
import os
import pickle
import subprocess
import tempfile
import pyqbdi

from pathlib import Path
from typing import List, Optional

# NOTE We need this import so we can use it to determine the path of this file.
import tritondse

from tritondse import Config
from tritondse import Program
from tritondse import SymbolicExecutor
from tritondse.coverage import CoverageSingleRun, CoverageStrategy
from tritondse.coverage_utils import *
from tritondse.types import Addr

#logging.basicConfig(level=logging.INFO)



class TraceException(Exception):
    pass


class Trace:

    def __init__(self):
        pass

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str] = None, stdin_file=None) -> 'Trace':
        """Run the binary passed as argument and return the coverage.

        :param strategy: Coverage strategy.
        :type strategy: :py:obj:`CoverageStrategy`.
        :param binary_path: Path to the binary.
        :type binary_path: :py:obj:`str`.
        :param args: List of arguments to pass to the binary.
        :type args: :py:obj:`List[str]`.
        :param stdin_file: Path to the file that will act as stdin.
        :type args: :py:obj:`str`.
        """
        raise NotImplementedError()

    def get_coverage(self) -> CoverageSingleRun:
        """Return the execution coverage.

        :return: :py:obj:`CoverageSingleRun`.
        """
        raise NotImplementedError()


class TritonTrace(Trace):

    def __init__(self):
        super().__init__()

        self._strategy = None

        self._coverage = None

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str] = None, stdin_file=None) -> 'TritonTrace':
        # Override stdin with the input file.
        if stdin_file:
            os.dup2(os.open(stdin_file, os.O_RDONLY), 0)

        config = Config(coverage_strategy=strategy)

        se = SymbolicExecutor(config)

        se.load_program(Program(binary_path))

        se.run()

        trace = TritonTrace()
        trace._coverage = se.coverage

        return trace

    def get_coverage(self) -> CoverageSingleRun:
        return self._coverage


class QBDITrace(Trace):

    QBDI_SCRIPT_FILEPATH = Path(tritondse.__file__).parent / 'qbdi_trace.py'

    def __init__(self):
        super().__init__()
        self._strategy = None
        self._coverage = None
        self._branches = None
        self._instructions = None

    @staticmethod
    def run(strategy: CoverageStrategy, binary_path: str, args: List[str] = None, stdin_file=None, timeout=None, cwd=None) -> 'QBDITrace':
        if not Path(binary_path).exists():
            raise FileNotFoundError()

        if stdin_file and not Path(stdin_file).exists():
            raise FileNotFoundError()

        args = [] if not args else args

        cmdlne = f'python -m pyqbdipreload {QBDITrace.QBDI_SCRIPT_FILEPATH}'.split(' ') + [binary_path] + args

        logging.debug(f'Command line: {" ".join(cmdlne)}')

        # Set output filepath.
        output_path = None
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_path = f.name

        # Set environment variables.

        environ = {
            'PYQBDIPRELOAD_COVERAGE_STRATEGY': strategy.name,
            'PYQBDIPRELOAD_OUTPUT_FILEPATH': output_path,
            'PYQBDIPRELOAD_LONGJMP_ADDR': os.getenv("TT_LONGJMP_ADDR", default=0x0),
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
            logging.error('QBDI tracer timeout expired!')
            raise TraceException('QBDI tracer timeout expired')

        if stdin_fp:
            stdin_fp.close()

        if not Path(output_path).exists():
            raise Exception('Error getting coverage')

        return QBDITrace.load_coverage_from_file(output_path)

    @staticmethod
    def load_coverage_from_file(coverage_path: str) -> 'QBDITrace':
        """Load coverage from a file.

        :param coverage_path: Path to the coverage file.
        :type coverage_path: :py:obj:`str`.
        """
        trace = QBDITrace()

        logging.debug(f'Loading coverage file')
        with open(coverage_path, 'rb') as fd:
            data = json.load(fd)

        trace._coverage = CoverageSingleRun.from_json(data)
        trace._strategy = CoverageStrategy[data["coverage_strategy"]]
        trace._branches = data["covered_branches"]
        trace._instructions = data["covered_instructions"]

        return trace


    def get_coverage(self) -> CoverageSingleRun:
        if not self._coverage:
            logging.warning("Please .run() the trace before querying coverage")

        return self._coverage


# NOTE Below you'll find the code of the QBDI tool to collect the trace
#      information.

def convert_permission(permission):
    p_read = Permission.READ if (permission & pyqbdi.PF_READ) == pyqbdi.PF_READ else Permission.NONE
    p_write = Permission.WRITE if (permission & pyqbdi.PF_WRITE) == pyqbdi.PF_WRITE else Permission.NONE
    p_exec = Permission.EXEC if (permission & pyqbdi.PF_EXEC) == pyqbdi.PF_EXEC else Permission.NONE

    return p_read | p_write | p_exec


def get_modules():
    # Collect modules.
    modules = {}
    for mm in pyqbdi.getCurrentProcessMaps(True):
        logging.debug(f'Processing memory map: {mm.name}')

        # TODO What should we do with special modules (anonymous, heap, stack, etc)?
        if Path(mm.name).exists():
            module = modules.get(mm.name, Module(mm.name))
            segment = Segment(Range(mm.range.start, mm.range.end), convert_permission(mm.permission))

            module.add(segment)

            modules[mm.name] = module

    modules = {m.range.start: m for m in modules.values()}

    for _, m in modules.items():
        logging.debug(f'module.name: {m.name} ({m.range.start:#x} -> {m.range.end:#x})')

    return modules


def write_coverage(coverage_data):
    """Write coverage into a file.
    """
    coverage_strategy = coverage_data['coverage_strategy']
    output_filepath = coverage_data['output_filepath']

    modules = get_modules()
    branches = coverage_data['branches']
    instructions = coverage_data['instructions']

    logging.debug(f'Writing coverage file to {output_filepath}')

    with open(output_filepath, 'ab') as f:
        pickle.dump(coverage_strategy, f)
        pickle.dump(modules, f)
        pickle.dump(branches, f)
        pickle.dump(instructions, f)


def register_instruction_coverage(vm, gpr, fpr, data):
    inst_analysis = vm.getInstAnalysis(type=pyqbdi.AnalysisType.ANALYSIS_INSTRUCTION)

    # Save basic block data.
    data['instructions'].append(inst_analysis.address)

    return pyqbdi.CONTINUE


def register_basic_block_coverage(vm, evt, gpr, fpr, data):
    addr = evt.basicBlockStart
    size = evt.basicBlockEnd - addr

    # Save basic block data.
    data['basic_blocks']['start'].add(addr)
    data['basic_blocks']['size'][addr] = size

    # Process branch data in case there is one pending.
    if data['branch_data']:
        # Unpack branch data.
        branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic = data['branch_data']

        # Check if the branch was taken.
        is_taken = addr == true_branch_addr

        # Save branch data.
        data['branches'].add((branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic))

        # Clear branch data for next occurrence.
        data['branch_data'] = None

    return pyqbdi.CONTINUE


def register_branch_coverage(vm, gpr, fpr, data):
    inst_analysis = vm.getInstAnalysis(type=pyqbdi.AnalysisType.ANALYSIS_INSTRUCTION | pyqbdi.AnalysisType.ANALYSIS_OPERANDS)

    operand = inst_analysis.operands[0]

    branch_addr = inst_analysis.address
    true_branch_addr = None
    false_branch_addr = inst_analysis.address + inst_analysis.instSize
    is_taken = None
    is_dynamic = False

    if operand.type == pyqbdi.OperandType.OPERAND_IMM:
        true_branch_addr = inst_analysis.address + inst_analysis.instSize + operand.value
    else:
        raise Exception('Invalid operand type')

    # Save current branch data.
    data['branch_data'] = (branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic)

    return pyqbdi.CONTINUE

import time
def pyqbdipreload_on_run(vm, start, stop):
    s = time.time()
    # Read parameters.
    #coverage_strategy = os.getenv('PYQBDIPRELOAD_COVERAGE_STRATEGY', 'BLOCK')
    coverage_strategy = os.getenv('PYQBDIPRELOAD_COVERAGE_STRATEGY', 'EDGE')
    output_filepath = os.getenv('PYQBDIPRELOAD_OUTPUT_FILEPATH', 'a.cov')

    # Initialize variables.
    coverage_data = {
        'coverage_strategy': coverage_strategy,
        'output_filepath': output_filepath,
        'instructions': list(),
        'branches': set(),
        'basic_blocks': {
            'start': set(),
            'size': dict(),
        },
        'branch_data': None,    # Current branch data.
    }

    # Remove all instrumented modules except the main one.
    vm.removeAllInstrumentedRanges()
    vm.addInstrumentedModuleFromAddr(start)

    if coverage_strategy == 'BLOCK':
        # Add callback on instruction execution.
        vm.addCodeCB(pyqbdi.PREINST, register_instruction_coverage, coverage_data)

    # Add callback on basic block entry.
    vm.addVMEventCB(pyqbdi.BASIC_BLOCK_ENTRY, register_basic_block_coverage, coverage_data)

    # Add callback on the JCC mnemonic.
    vm.addMnemonicCB('JCC', pyqbdi.InstPosition.POSTINST, register_branch_coverage, coverage_data)

    # Write coverage on exit.
    # TODO This does not work with bins that crash.
    atexit.register(write_coverage, coverage_data)

#    def showInstruction(vm, gpr, fpr, data):
#        instAnalysis = vm.getInstAnalysis()
#        print("0x{:x}: {}".format(instAnalysis.address, instAnalysis.disassembly))
#        return pyqbdi.CONTINUE    
#    vm.addCodeCB(pyqbdi.PREINST, showInstruction, None)

    # Run program.
    print("Run start")
    vm.run(start, stop)
    e = time.time()
    print("Run finished")
    print(e - s)
