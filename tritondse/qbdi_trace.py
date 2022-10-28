# This script is usede with pyqbdipreload to generate a json file that can be parsed with CoverageSingleRun.from_json
# This needs to be fast which is why we cannot import tritondse and generate the CoverageSingleRun directly 
# (`import tritondse` adds ~0.3 s to the execution time of the script in my experience).

import atexit
import bisect
import logging
import os
import pickle
import subprocess
import tempfile
import json
import pyqbdi

from pathlib import Path
from typing import List, Optional
from collections import Counter


#logging.basicConfig(level=logging.INFO)

MODULES = None
MODULES_ADDRS = None
MODULES_SIZES = None
MODULES_COUNT = None

from enum import auto, IntFlag

class Permission(IntFlag):
    NONE = auto()
    READ = auto()
    WRITE = auto()
    EXEC = auto()


class Range:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def overlaps(self, other_range: 'Range') -> bool:
        return self.start <= other_range.start < self.end or \
               self.start < other_range.end <= self.end

class Segment:

    def __init__(self, range_: Range, permissions: Permission):
        self.range = range_
        self.permissions = permissions


class Module:

    def __init__(self, name: str):
        self.name = name
        self.range = None
        self.segments = dict()

    def add(self, segment: Segment) -> None:
        if not self.range:
            self.range = Range(segment.range.start, segment.range.end)

        if segment.range.start < self.range.start:
            self.range.start = segment.range.start

        if self.range.end < segment.range.end:
            self.range.end = segment.range.end

        if not segment.range.start in self.segments:
            self.segments[segment.range.start] = segment


# NOTE Below you'll find the code of the QBDI tool to collect the trace
#      information.

def convert_permission(permission):
    p_read = Permission.READ if (permission & pyqbdi.PF_READ) == pyqbdi.PF_READ else Permission.NONE
    p_write = Permission.WRITE if (permission & pyqbdi.PF_WRITE) == pyqbdi.PF_WRITE else Permission.NONE
    p_exec = Permission.EXEC if (permission & pyqbdi.PF_EXEC) == pyqbdi.PF_EXEC else Permission.NONE

    return p_read | p_write | p_exec


def get_module(address) -> Optional[Module]:
    """Return the module that contains the address passed as
    argument.

    :param address: Address.
    :type address: :py:obj:`Addr`.

    :return: :py:obj:`Optional[str]`.
    """
    # Find insertion index.
    idx = bisect.bisect_left(MODULES_ADDRS, address)

    # Leftmost case. Only check current index.
    if idx == 0:
        m_addr = MODULES_ADDRS[idx]
        m_size = MODULES_SIZES[m_addr]

        return MODULES[m_addr] if m_addr <= address < m_addr + m_size else None

    # Rightmost case. Only check previous index.
    if idx == MODULES_COUNT:
        m_addr = MODULES_ADDRS[idx - 1]
        m_size = MODULES_SIZES[m_addr]

        return MODULES[m_addr] if m_addr <= address < m_addr + m_size else None

    # Middle case. Check both previous and current indexes.
    m_addr = MODULES_ADDRS[idx - 1]
    m_size = MODULES_SIZES[m_addr]

    if m_addr <= address < m_addr + m_size:
        return MODULES[m_addr]

    if address == MODULES_ADDRS[idx]:
        return MODULES[address]

    return None

def get_module_name(self, address) -> Optional[str]:
    """Return the name of the module that contains the address passed as
    argument.

    :param address: Address.
    :type address: :py:obj:`Addr`.

    :return: :py:obj:`Optional[str]`.
    """
    mod = get_module(address)
    if mod: return mod.name
    else: return None

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

def convert_coverage(coverage_data):
    #coverage_single_run = CoverageSingleRun(self._strategy)
    strategy = coverage_data['coverage_strategy']
    instructions = coverage_data['instructions']
    branches = coverage_data['branches']

    # CoverageSingleRun.covered_instructions
    covered_instructions = Counter()
    # CoverageSingleRun.covered_branches
    covered_branches = list()
    # CoverageSingleRun.covered_instructions
    covered_dynamic_branches = list()

    # Add covered instructions.
    if strategy == "BLOCK" and len(instructions) > 0:
        # Get main module and transform absolute addresses into relative.
        # As we only trace the main module, we only need to check that one
        # to get the base address.
        main_module = get_module(list(branches)[0][0])

        module_start_addr = main_module.range.start

        for addr in instructions:
            covered_instructions[addr - module_start_addr] += 1

    # Add covered branches.
    if len(branches) > 0:
        # Get main module and transform absolute addresses into relative.
        # As we only trace the main module, we only need to check that one
        # to get the base address.
        main_module = get_module(list(branches)[0][0])

        module_start_addr = main_module.range.start

        # TODO Find a better way to do this.
        addr_mask = 0xffffffffffffffff

        for branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic in branches:
            taken_addr, not_taken_addr = (true_branch_addr, false_branch_addr) if is_taken else (false_branch_addr, true_branch_addr)

            if is_dynamic:
                source = (branch_addr - module_start_addr) & addr_mask
                target = (taken_addr - module_start_addr) & addr_mask
                covered_dynamic_branches.append((source, target))
            else:
                program_counter = (branch_addr - module_start_addr) & addr_mask
                taken_addr = (taken_addr - module_start_addr) & addr_mask
                not_taken_addr = (not_taken_addr - module_start_addr) & addr_mask
                covered_branches.append((program_counter, taken_addr, not_taken_addr))

        return (covered_instructions, covered_dynamic_branches, covered_branches)


def write_coverage(coverage_data):
    """Write coverage into a file.
    """

    global MODULES, MODULES_ADDRS, MODULES_SIZES, MODULES_COUNT
    MODULES = get_modules()
    MODULES_ADDRS = sorted([m.range.start for m in MODULES.values()])
    MODULES_SIZES = {m.range.start: m.range.end for m in MODULES.values()}
    MODULES_COUNT = len(MODULES_ADDRS)

    covered_instructions, covered_dynamic_branches, covered_branches = convert_coverage(coverage_data)
    coverage_strategy = coverage_data['coverage_strategy']
    output_filepath = coverage_data['output_filepath']

    data = {"coverage_strategy": coverage_strategy,
            "covered_instructions": covered_instructions, 
            "covered_dynamic_branches": covered_dynamic_branches,
            "covered_branches": covered_branches}

    with open(output_filepath+".json", "w") as fd:
        json.dump(data, fd)

    coverage_strategy = coverage_data['coverage_strategy']
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


    # TODO Find a better way
    # There is no generic way to do that with LIEF https://github.com/lief-project/LIEF/issues/762
    longjmp_plt = int(os.getenv("PYQBDIPRELOAD_LONGJMP_ADDR", default=0x0))
    #longjmp_plt = 0x401ce0 # FT
    #longjmp_plt = 0x401530 # libpng
    #longjmp_plt = 0x401800 # libjpeg
    print(f"in qbdi_trace {longjmp_plt}")
    if longjmp_plt != 0:
        def longjmp_callback(vm, gpr, fpr, data):
            print("in longjmp callback")
            return pyqbdi.STOP
        vm.addCodeAddrCB(longjmp_plt, pyqbdi.PREINST, longjmp_callback, None)

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
