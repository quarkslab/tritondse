#!/usr/bin/env python
from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed, SeedFormat, Loader, MonolithicLoader, CompositeData, CompositeField
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction
from pwn import *
import logging
#logging.basicConfig(level=logging.DEBUG)

cnt = 0
n_inst = 0
LIGHTHOUSE = True
TRACE_FOLDER = "./lighthouse/"

def trace_inst(exec: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    if LIGHTHOUSE: 
        with open(TRACE_FOLDER + str(exec.seed.hash) + ".trace", "a") as fd:
            fd.write(f"0x{inst.getAddress():x}\n")
    global n_inst
    if n_inst > 50000:
        print("Executed more than 50000 instructions")
        exec.seed.status = SeedStatus.CRASH
        pstate.stop = True
    n_inst += 1

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    global cnt
    cnt += 1
    print(f"{cnt}. : seed:{se.seed.hash} ({repr(se.seed.content.variables['buffer'])})   [exitcode:{se.exitcode}]")

def pre_exec_hook(se: SymbolicExecutor, state: ProcessState):
    global n_inst 
    n_inst = 0

def fptr_stub(exec: SymbolicExecutor, pstate: ProcessState, addr: int):
    print(f"fptr_stub addr : {addr:#x}")
    if addr == 0x81d1bf0:
        pstate.cpu.r0 = 1
    elif addr == 0x81d1252:
        pstate.cpu.r0 = 0
    pstate.cpu.program_counter += 2
    exec.skip_instruction()

def hook_start(exec: SymbolicExecutor, pstate: ProcessState, addr: int):
    buffer = pstate.get_argument_value(0)
    length = pstate.get_argument_value(1)
    JSON_ctx = pstate.get_argument_value(2)

    exec.inject_symbolic_input(buffer, exec.seed.content.variables,\
            var_prefix="buffer", compfield=CompositeField.VARIABLE)
    exec.inject_symbolic_input(JSON_ctx, exec.seed.content.variables,\
            var_prefix="JSON_ctx", compfield=CompositeField.VARIABLE)

    # We hardcode the length
    pstate.cpu.r1 = 128

conf = Config(skip_unsupported_import=True,\
        seed_format=SeedFormat.COMPOSITE)

ldr = MonolithicLoader(\
        "./bugged_json_parser.bin", \
        Architecture.ARM32, \
        0x8000000,\
        cpustate = {"pc": 0x81dc46e, \
                    "r0": 0x800000, \
                    "r2": 0x700000}, \
        set_thumb=True,\
        vmmap = { 0x800000 : b"\x00"*128,\
                0x700000 : b"\x00"*512,\
                0x600000 : b"\x00"*0x1000})

dse = SymbolicExplorator(conf,\
        ldr, executor_stop_at=0x81dc472)

membuf_addr = 0x600000
composite_data = CompositeData(variables={"buffer" : b"A"*128, "JSON_ctx": p32(membuf_addr)*128})
dse.add_input_seed(Seed(composite_data))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_pre_execution_callback(pre_exec_hook)
dse.callback_manager.register_post_instruction_callback(trace_inst)
dse.callback_manager.register_pre_addr_callback(0x81d140e, hook_start)
dse.callback_manager.register_pre_addr_callback(0x81d1bf0, fptr_stub)
dse.callback_manager.register_pre_addr_callback(0x81d1252, fptr_stub)

dse.explore()
