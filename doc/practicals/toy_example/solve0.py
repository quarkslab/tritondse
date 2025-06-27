import logging

from triton import Instruction

from tritondse.config import Config
from tritondse.loaders.program import Program
from tritondse.process_state import ProcessState
from tritondse.sanitizers import NullDerefSanitizer
from tritondse.seed import CompositeData
from tritondse.seed import Seed
from tritondse.seed import SeedFormat
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.symbolic_explorator import SymbolicExplorator
import tritondse.logging


logging.basicConfig(level=logging.DEBUG)
tritondse.logging.enable(level=logging.DEBUG)


def trace_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.debug(f"seed:{se.seed.hash} ({repr(se.seed.content)})")


prog = Program("./bin/0")

config = Config(pipe_stdout=True,
                skip_unsupported_import=True,
                seed_format=SeedFormat.COMPOSITE)

dse = SymbolicExplorator(config, prog)

dse.add_input_seed(Seed(CompositeData(argv=[b"./bin/0", b"XXXX"], files={"stdin": b"ZZZZ"})))
dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
# dse.callback_manager.register_post_instruction_callback(trace_inst)

dse.explore()
