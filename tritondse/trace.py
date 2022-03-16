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


logging.basicConfig(level=logging.INFO)


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

    QBDI_SCRIPT_FILEPATH = Path(tritondse.__file__).parent / 'trace.py'

    def __init__(self):
        super().__init__()

        self._strategy = None

        self._coverage = None

        self._modules = None
        self._modules_addrs = None
        self._modules_sizes = None
        self._modules_count = 0

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

        try:
            with open(coverage_path, 'rb') as f:
                trace._strategy = CoverageStrategy[pickle.load(f)]
                trace._modules = pickle.load(f)
                trace._modules_addrs = sorted([m.range.start for m in trace._modules.values()])
                trace._modules_sizes = {m.range.start: m.range.end for m in trace._modules.values()}
                trace._modules_count = len(trace._modules_addrs)
                trace._branches = pickle.load(f)
                trace._instructions = pickle.load(f)
        except (pickle.PickleError, EOFError):
            logging.warning('Error loading QBDI tracer coverage file.')
            raise TraceException('QBDI tracer timeout expired')

        return trace

    def get_module_name(self, address: Addr) -> Optional[str]:
        """Return the name of the module that contains the address passed as
        argument.

        :param address: Address.
        :type address: :py:obj:`Addr`.

        :return: :py:obj:`Optional[str]`.
        """
        # Find insertion index.
        idx = bisect.bisect_left(self._modules_addrs, address)

        # Leftmost case. Only check current index.
        if idx == 0:
            m_addr = self._modules_addrs[idx]
            m_size = self._modules_sizes[m_addr]

            return self._modules[m_addr].name if m_addr <= address < m_addr + m_size else None

        # Rightmost case. Only check previous index.
        if idx == self._modules_count:
            m_addr = self._modules_addrs[idx - 1]
            m_size = self._modules_sizes[m_addr]

            return self._modules[m_addr].name if m_addr <= address < m_addr + m_size else None

        # Middle case. Check both previous and current indexes.
        m_addr = self._modules_addrs[idx - 1]
        m_size = self._modules_sizes[m_addr]

        if m_addr <= address < m_addr + m_size:
            return self._modules[m_addr].name

        if address == self._modules_addrs[idx]:
            return self._modules[address].name

        return None

    def get_module(self, address: Addr) -> Optional[Module]:
        """Return the module that contains the address passed as
        argument.

        :param address: Address.
        :type address: :py:obj:`Addr`.

        :return: :py:obj:`Optional[str]`.
        """
        # Find insertion index.
        idx = bisect.bisect_left(self._modules_addrs, address)

        # Leftmost case. Only check current index.
        if idx == 0:
            m_addr = self._modules_addrs[idx]
            m_size = self._modules_sizes[m_addr]

            return self._modules[m_addr] if m_addr <= address < m_addr + m_size else None

        # Rightmost case. Only check previous index.
        if idx == self._modules_count:
            m_addr = self._modules_addrs[idx - 1]
            m_size = self._modules_sizes[m_addr]

            return self._modules[m_addr] if m_addr <= address < m_addr + m_size else None

        # Middle case. Check both previous and current indexes.
        m_addr = self._modules_addrs[idx - 1]
        m_size = self._modules_sizes[m_addr]

        if m_addr <= address < m_addr + m_size:
            return self._modules[m_addr]

        if address == self._modules_addrs[idx]:
            return self._modules[address]

        return None

    def get_coverage(self) -> CoverageSingleRun:
        if not self._coverage:
            self._load_coverage()

        return self._coverage

    def _load_coverage(self):
        self._coverage = CoverageSingleRun(self._strategy)

        # Add covered instructions.
        if self._strategy == CoverageStrategy.BLOCK and len(self._instructions) > 0:
            # Get main module and transform absolute addresses into relative.
            # As we only trace the main module, we only need to check that one
            # to get the base address.
            main_module = self.get_module(self._instructions[0])

            module_start_addr = main_module.range.start

            for addr in self._instructions:
                self._coverage.add_covered_address(addr - module_start_addr)

        # Add covered branches.
        if len(self._branches) > 0:
            # Get main module and transform absolute addresses into relative.
            # As we only trace the main module, we only need to check that one
            # to get the base address.
            main_module = self.get_module(list(self._branches)[0][0])

            module_start_addr = main_module.range.start

            # TODO Find a better way to do this.
            addr_mask = 0xffffffffffffffff

            for branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic in self._branches:
                taken_addr, not_taken_addr = (true_branch_addr, false_branch_addr) if is_taken else (false_branch_addr, true_branch_addr)

                if is_dynamic:
                    source = (branch_addr - module_start_addr) & addr_mask
                    target = (taken_addr - module_start_addr) & addr_mask

                    self._coverage.add_covered_dynamic_branch(source, target)
                else:
                    program_counter = (branch_addr - module_start_addr) & addr_mask
                    taken_addr = (taken_addr - module_start_addr) & addr_mask
                    not_taken_addr = (not_taken_addr - module_start_addr) & addr_mask

                    self._coverage.add_covered_branch(program_counter, taken_addr, not_taken_addr)


# NOTE Below you'll find the code of the QBDI tool to collect the trace
#      information.

def convert_permission(permission):
    p_read = int(permission & pyqbdi.PF_READ == pyqbdi.PF_READ) << Permission.READ
    p_write = int(permission & pyqbdi.PF_WRITE == pyqbdi.PF_WRITE) << Permission.WRITE
    p_exec = int(permission & pyqbdi.PF_EXEC == pyqbdi.PF_EXEC) << Permission.EXEC

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


def pyqbdipreload_on_run(vm, start, stop):
    # Read parameters.
    coverage_strategy = os.getenv('PYQBDIPRELOAD_COVERAGE_STRATEGY', 'BLOCK')
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

    # Run program.
    vm.run(start, stop)
