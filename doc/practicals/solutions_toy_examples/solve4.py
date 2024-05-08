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
from tritondse.types import Addr
from tritondse.types import SolverStatus
import tritondse.logging


logging.basicConfig(level=logging.DEBUG)
tritondse.logging.enable(level=logging.DEBUG)

once_flag = False


def trace_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.debug(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")


def hook_strlen(se: SymbolicExecutor, pstate: ProcessState, rtn_name: str, address: Addr):
    global once_flag

    if once_flag:
        return

    # Get argument.
    s = pstate.get_argument_value(0)

    ast = pstate.actx

    def rec(res, s, depth, max_depth):
        if depth == max_depth:
            return res
        byte_ast = pstate.read_symbolic_memory_byte(s + depth).getAst()
        res = ast.ite(byte_ast == 0x00, ast.bv(depth, 64), rec(res, s, depth + 1, max_depth))
        return res

    length = len(pstate.memory.read_string(s))
    length_ast = ast.bv(length, 64)
    length_ast = rec(length_ast, s, 0, length)

    pstate.push_constraint(pstate.read_symbolic_memory_byte(s + length).getAst() == 0x00)

    # Generate models to get strings with increasing length up to the length of
    # the current input.
    constraint = length_ast != length
    status, model = pstate.solve(constraint)

    while status == SolverStatus.SAT:
        # Get length of the generated string.
        length = pstate.evaluate_expression_model(length_ast, model)

        # Generate new seed from model.
        new_seed = se.mk_new_seed_from_model(model)

        logging.debug(f'new_seed: {new_seed.content} len = {length:x}')

        # Enqueue seed.
        se.enqueue_seed(new_seed)

        # Add newly generated length to the constraints in order to
        # generate a string with a different length for the next
        # iteration.
        constraint = pstate.actx.land([constraint, length_ast != length])

        # Generate new model.
        status, model = pstate.solve(constraint)

    once_flag = True

    return length_ast


prog = Program("./bin/4")

config = Config(skip_unsupported_import=True,
                seed_format=SeedFormat.COMPOSITE)

dse = SymbolicExplorator(config, prog)

dse.add_input_seed(Seed(CompositeData(argv=[b"./bin/4", b"AAAAAA"])))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_pre_imported_routine_callback("strlen", hook_strlen)
# dse.callback_manager.register_post_instruction_callback(trace_inst)

dse.explore()
