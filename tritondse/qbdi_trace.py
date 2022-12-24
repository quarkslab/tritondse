# This script is used by pyqbdipreload to generate a json file that can be parsed with CoverageSingleRun.from_json
# This needs to be fast which is why we cannot import tritondse and generate the CoverageSingleRun directly 
# (`import tritondse` adds ~0.3 s to the execution time of the script in my experience).

# built-in modules
import atexit
import bisect
import os
import json
import time
import ctypes
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import Counter
from dataclasses import dataclass
import lief
import sys

# Third-party modules
import pyqbdi


@dataclass
class CoverageTrace:
    strategy: str  # BLOCK, EDGE, etc..
    covered_instructions: Counter
    covered_items: List[Tuple[int, int, Optional[int]]]
    modules: Dict[str, int]
    trace: List[int]

@dataclass
class CoverageData:
    strategy: str  # BLOCK, EDGE, etc..
    branch_data: Optional[Tuple[int, int, int, bool, bool]]  # (temporary data): branch pc, true-branch, false-branch, is_taken, is_dynamic
    trace: CoverageTrace
    modules_base: List[int]
    pie: bool

    def to_relative(self, addr: int) -> int:
        if self.pie:
            return addr - self.modules_base[bisect.bisect_right(self.modules_base, addr)-1]
        else:
            return addr



def get_modules() -> Dict[str, int]:
    """ Retrieve modules base address to remove ASLR. """
    modules = {}
    for m in pyqbdi.getCurrentProcessMaps(True):
        if m.name in modules:
            modules[m.name] = min(m.range[0], modules[m.name])
        else:
            modules[m.name] = m.range[0]
        print(f"{m.name}: {m.range}, {m.permission}")
    return modules

def get_module_bases() -> List[int]:
    """ Retrieve modules base address to remove ASLR. """
    return sorted(get_modules().values())


def write_coverage(covdata: CoverageData, output_file: str):
    """Write coverage into a file.
    """
    data = {
        "coverage_strategy": covdata.trace.strategy,
        "covered_instructions": covdata.trace.covered_instructions,
        "covered_items": covdata.trace.covered_items,
        "trace": covdata.trace.trace,
        "modules_base": covdata.trace.modules
    }
    with open(output_file, "w") as fd:
        json.dump(data, fd)


def register_instruction_coverage(vm, gpr, fpr, data: CoverageData):
    # inst_analysis = vm.getInstAnalysis(type=pyqbdi.AnalysisType.ANALYSIS_INSTRUCTION)

    # Save instruction covered
    rel_rip = data.to_relative(gpr.rip)
    data.trace.covered_instructions[rel_rip] += 1  # change to be portable

    # Also save the trace
    data.trace.trace.append(rel_rip)

    return pyqbdi.CONTINUE


def register_basic_block_coverage(vm, evt, gpr, fpr, data: CoverageData):
    addr = evt.basicBlockStart

    # Process branch data in case there is one pending.
    if data.branch_data:
        # Unpack branch data.
        branch_addr, true_branch_addr, false_branch_addr, is_taken, is_dynamic = data.branch_data
        br_a, true_a, false_a = data.to_relative(branch_addr), data.to_relative(true_branch_addr), data.to_relative(false_branch_addr)

        # Check if the branch was taken.
        taken_a, not_taken_a = (true_a, false_a) if bool(addr == true_branch_addr) else (false_a, true_a)

        if is_dynamic:
            data.trace.covered_items.append((br_a, taken_a, None))
        else:
            data.trace.covered_items.append((br_a, taken_a, not_taken_a))

        # Clear branch data for next occurrence.
        data.branch_data = None
    else:
        pass
        # FIXME: There is a problem in that script is_dynamic can never be true !
        # FIXME: as it's never set to true. I feel like the else here is the case where its dynamic?

    return pyqbdi.CONTINUE


def register_branch_coverage(vm, gpr, fpr, data):
    inst_analysis = vm.getInstAnalysis(type=pyqbdi.AnalysisType.ANALYSIS_INSTRUCTION | pyqbdi.AnalysisType.ANALYSIS_OPERANDS)

    operand = inst_analysis.operands[0]

    branch_addr = inst_analysis.address
    false_branch_addr = inst_analysis.address + inst_analysis.instSize

    if operand.type == pyqbdi.OperandType.OPERAND_IMM:
        # FIXME: Isn't it assuming the jump is relative ?
        true_branch_addr = inst_analysis.address + inst_analysis.instSize + ctypes.c_longlong(operand.value).value
    else:
        raise Exception('Invalid operand type')

    # Save current branch data
    data.branch_data = (branch_addr, true_branch_addr, false_branch_addr, None, False)

    return pyqbdi.CONTINUE



def pyqbdipreload_on_run(vm, start, stop):
    s = time.time()
    # Read parameters.
    strat = os.getenv('PYQBDIPRELOAD_COVERAGE_STRATEGY', 'BLOCK')
    output = os.getenv('PYQBDIPRELOAD_OUTPUT_FILEPATH', 'a.cov')
    bool_trace = os.getenv('PYQBDIPRELOAD_DUMP_TRACE', 'False')
    bool_trace = True if bool_trace in ['true', 'True'] else False

    mods = get_modules()
    base_addresses = sorted(get_modules().values())
    covtrace = CoverageTrace(strat, Counter(), [], mods, [])

    # Open binary in LIEF to check if PIE or not
    p = lief.parse(sys.argv[0])

    coverage_data = CoverageData(strat, None, covtrace, base_addresses, p.is_pie)

    # Remove all instrumented modules except the main one.
    vm.removeAllInstrumentedRanges()
    vm.addInstrumentedModuleFromAddr(start)

    if coverage_data.strategy == 'BLOCK' or bool_trace:
        # Add callback on instruction execution.
        vm.addCodeCB(pyqbdi.PREINST, register_instruction_coverage, coverage_data)

    # Add callback on basic block entry.
    vm.addVMEventCB(pyqbdi.BASIC_BLOCK_ENTRY, register_basic_block_coverage, coverage_data)

    # Add callback on the JCC mnemonic.
    vm.addMnemonicCB('JCC', pyqbdi.InstPosition.POSTINST, register_branch_coverage, coverage_data)

    # Write coverage on exit.
    # TODO This does not work with bins that crash.
    atexit.register(write_coverage, coverage_data, output)

    # TODO Find a better way
    # There is no generic way to do that with LIEF https://github.com/lief-project/LIEF/issues/762
    longjmp_plt = int(os.getenv("PYQBDIPRELOAD_LONGJMP_ADDR", default="0"))

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
    print(f"Run finished: {time.time() - s:.02f}s")
