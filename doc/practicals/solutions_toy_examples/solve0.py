from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed, SeedFormat, CompositeData
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction

def trace_inst(exec: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    print(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})")

p = Program("./7")
dse = SymbolicExplorator(Config(\
        skip_unsupported_import=True,\
        seed_format=SeedFormat.COMPOSITE), p)

dse.add_input_seed(Seed(CompositeData(argv=[b"./7", b"XXXX"], files={"stdin": b"ZZZZ"})))
dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
#dse.callback_manager.register_post_instruction_callback(trace_inst)

dse.explore()
