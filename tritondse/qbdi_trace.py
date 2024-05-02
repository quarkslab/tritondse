# This script is used by pyqbdipreload to generate a json file that can be parsed with CoverageSingleRun.from_json
# This needs to be fast which is why we cannot import tritondse and generate the CoverageSingleRun directly
# (`import tritondse` adds ~0.3 s to the execution time of the script in my experience).

# built-in imports
import atexit
import bisect
import ctypes
import ctypes.util
import json
import os
import sys
import time

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# Third-party imports
import lief
import pyqbdi


setjmp_data = {}
libdl = None


class Dl_info(ctypes.Structure):
    _fields_ = [
        ('dli_fname', ctypes.c_char_p),
        ('dli_fbase', ctypes.c_void_p),
        ('dli_sname', ctypes.c_char_p),
        ('dli_saddr', ctypes.c_void_p)
    ]


def dladdr(addr):
    res = Dl_info()
    libdl.dladdr(ctypes.cast(addr, ctypes.c_void_p), ctypes.byref(res))

    return res.dli_sname


def is_symbol(name, addr):
    sname = dladdr(addr)
    sname = sname.decode() if sname else ""

    return sname == name


def hook_post_setjmp(vm, state, gpr, fpr, data):
    setjmp_data[data]['rip'] = gpr.rip

    if setjmp_data[data]['setjmp_cbk_id'] is not None:
        vm.deleteInstrumentation(setjmp_data[data]['setjmp_cbk_id'])
        setjmp_data[data]['setjmp_cbk_id'] = None

    return pyqbdi.CONTINUE


def hook_post_longjmp(vm, state, gpr, fpr, data):
    if data in setjmp_data:
        gpr.rip = setjmp_data[data]['rip']
    else:
        print(f'[FATAL] longjmp arg ({data:x}) not found!')
        sys.exit(-1)

    if setjmp_data[data]['longjmp_cbk_id'] is not None:
        vm.deleteInstrumentation(setjmp_data[data]['longjmp_cbk_id'])
        setjmp_data[data]['longjmp_cbk_id'] = None

    return pyqbdi.CONTINUE


def handle_exec_transfer_call(vm, state, gpr, fpr, data):
    if is_symbol("_setjmp", gpr.rip):
        arg1 = gpr.rdi      # get env argument.

        setjmp_data[arg1] = {}
        setjmp_data[arg1]['setjmp_cbk_id'] = vm.addVMEventCB(pyqbdi.EXEC_TRANSFER_RETURN, hook_post_setjmp, arg1)
    elif is_symbol("longjmp", gpr.rip):
        arg1 = gpr.rdi      # get env argument.

        setjmp_data[arg1]['longjmp_cbk_id'] = vm.addVMEventCB(pyqbdi.EXEC_TRANSFER_RETURN, hook_post_longjmp, arg1)

    return pyqbdi.CONTINUE


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
    dump_trace: bool

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
        # print(f"{m.name}: {m.range}, {m.permission}")
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

    if data.dump_trace:
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
    global libdl

    s = time.time()

    # Load dl library.
    libdl_path = ctypes.util.find_library('dl')
    if libdl_path is None:
        raise Exception('Unable to found dl library')
    libdl = ctypes.cdll.LoadLibrary(libdl_path)
    libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))

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

    coverage_data = CoverageData(strat, None, covtrace, base_addresses, p.is_pie, bool_trace)

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

    # Add callback for handling calls to functions setjmp and longjmp.
    vm.addVMEventCB(pyqbdi.EXEC_TRANSFER_CALL, handle_exec_transfer_call, None)

    # Run program.
    print("Run start")
    vm.run(start, stop)
    print(f"Run finished: {time.time() - s:.02f}s")
