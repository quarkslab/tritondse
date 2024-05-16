import logging

from pathlib import Path

from triton import Instruction

from tritondse.arch import Architecture
from tritondse.config import Config
from tritondse.loaders.loader import LoadableSegment
from tritondse.loaders.loader import MonolithicLoader
from tritondse.process_state import ProcessState
from tritondse.seed import CompositeData
from tritondse.seed import Seed
from tritondse.seed import SeedFormat
from tritondse.seed import SeedStatus
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.symbolic_explorator import SymbolicExplorator
from tritondse.types import Addr
from tritondse.types import Perm
import tritondse.logging


logging.basicConfig(level=logging.INFO)
# tritondse.logging.enable(level=logging.INFO)

# Memory mapping
STACK_ADDR   = 0x1000000
STACK_SIZE   = 1024*6
STRUC_ADDR   = 0x3000000
BUFFER_ADDR  = 0x2000000
BASE_ADDRESS = 0x8000000
USER_CB      = 0x40000000

# Addresses
ENTRY_POINT = 0x81dc46e
EXIT_POINT  = 0x81dc472
STUB_ADDR1  = 0x81d1bf0
STUB_ADDR2  = 0x81d1252


def pre_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[{se.trace_offset}] {inst.getAddress():#08x}: {inst.getDisassembly()}")
    if se.trace_offset > 40000:
        se.seed.status = SeedStatus.HANG
        se.abort()


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.info(f"[{se.uid}] seed:{se.seed.hash} ({repr(se.seed.content.variables['buffer'][:100])}) => {se.seed.status.name} [exitcode:{se.exitcode}]")


def fptr_stub(se: SymbolicExecutor, pstate: ProcessState, addr: Addr):
    logging.info(f"fptr_stub addr : {addr:#x}")
    if addr == STUB_ADDR1:
        pstate.cpu.r0 = 1
    elif addr == STUB_ADDR2:
        pstate.cpu.r0 = 0
    pstate.cpu.program_counter += 2
    se.skip_instruction()


def hook_start(se: SymbolicExecutor, pstate: ProcessState):
    buffer = pstate.get_argument_value(0)
    _ = pstate.get_argument_value(1)
    json_ctx = pstate.get_argument_value(2)

    se.inject_symbolic_variable_memory(buffer, "buffer", se.seed.content.variables["buffer"])
    se.inject_symbolic_variable_memory(json_ctx, "JSON_ctx", se.seed.content.variables["JSON_ctx"])

    # Take the length of the buffer (which is not meant to change)
    pstate.cpu.r1 = len(se.seed.content.variables['buffer'])


conf = Config(seed_format=SeedFormat.COMPOSITE,
              skip_unsupported_import=True,
              workspace="ws",
              workspace_reset=True)

raw_firmware = Path("./bugged_json_parser.bin").read_bytes()

ldr = MonolithicLoader(Architecture.ARM32,
                       cpustate={"pc": ENTRY_POINT,
                                 "r0": BUFFER_ADDR,
                                 "r2": STRUC_ADDR,
                                 "sp": STACK_ADDR+STACK_SIZE},
                       set_thumb=True,
                       maps=[LoadableSegment(BASE_ADDRESS, len(raw_firmware), Perm.R | Perm.X, content=raw_firmware, name="bugged_json_parser"),
                             LoadableSegment(BUFFER_ADDR, 40, Perm.R | Perm.W, name="input"),
                             LoadableSegment(STRUC_ADDR, 512, Perm.R | Perm.W, name="JSON_ctx"),
                             LoadableSegment(USER_CB, 1000, Perm.R | Perm.X, name="user_cb"),
                             LoadableSegment(STACK_ADDR, STACK_SIZE, Perm.R | Perm.W, name="[stack]")])

dse = SymbolicExplorator(conf, ldr, executor_stop_at=EXIT_POINT)

seed = Seed(CompositeData(variables={"buffer": b"A"*40,
                                     "JSON_ctx": b"\x00"*128}))

dse.add_input_seed(seed)

dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_pre_execution_callback(hook_start)

dse.callback_manager.register_pre_instruction_callback(pre_inst)

dse.callback_manager.register_pre_addr_callback(STUB_ADDR1, fptr_stub)
dse.callback_manager.register_pre_addr_callback(STUB_ADDR2, fptr_stub)

dse.explore()
