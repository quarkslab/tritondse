import logging
import sys

from triton import Instruction

from tritondse.callbacks import CbType
from tritondse.callbacks import ProbeInterface
from tritondse.config import Config
from tritondse.loaders.program import Program
from tritondse.process_state import ProcessState
from tritondse.sanitizers import NullDerefSanitizer
from tritondse.seed import Seed
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.symbolic_explorator import SymbolicExplorator
from tritondse.types import Addr
from tritondse.types import SolverStatus
import tritondse.logging


logging.basicConfig(level=logging.DEBUG)
tritondse.logging.enable(level=logging.DEBUG)

buffers_len_g = dict()      # buffer_address : buffer_len


class StrncpySanitizer(ProbeInterface):

    def __init__(self):
        super(StrncpySanitizer, self).__init__()

        self._add_callback(CbType.PRE_RTN, self.strncpy_check, 'strncpy')

    def strncpy_check(self, se: SymbolicExecutor, pstate: ProcessState, rtn_name: str, address: Addr):
        # char *strncpy(char *dest, const char *src, size_t n);
        dest = se.pstate.get_argument_value(0)
        src = pstate.get_argument_value(1)
        n = se.pstate.get_argument_value(2)

        if dest not in buffers_len_g:
            return

        buffer_len = buffers_len_g[dest]

        if n > buffer_len:
            logging.critical(f"Found overflowing strncpy buf: {dest:x} bufsize: {buffer_len} copysize: {n}")

            # Generate input to trigger the overflow.
            ast = pstate.actx

            def rec(res, s, depth, max_depth):
                if depth == max_depth:
                    return res
                byte_ast = pstate.read_symbolic_memory_byte(s + depth).getAst()
                res = ast.ite(byte_ast == 0x00, ast.bv(depth, 64), rec(res, s, depth + 1, max_depth))
                return res

            length = len(pstate.memory.read_string(src))
            length_ast = ast.bv(length, 64)
            length_ast = rec(length_ast, src, 0, length)

            # Add the terminating null byte as a constraint.
            pstate.push_constraint(pstate.read_symbolic_memory_byte(src + length).getAst() == 0x00)

            # Generate model to get string with the proper length.
            constraint = length_ast > n
            status, model = pstate.solve(constraint)

            while status == SolverStatus.SAT:
                # Get length of the generated string.
                length = pstate.evaluate_expression_model(length_ast, model)

                # Generate new seed from model.
                new_seed = se.mk_new_seed_from_model(model)

                logging.debug(f"new_seed: {new_seed.content} len = {length:x}")

                # Enqueue seed.
                se.enqueue_seed(new_seed)

                # Add newly generated length to the constraints in order to
                # generate a string with a different length for the next
                # iteration.
                constraint = pstate.actx.land([constraint, length_ast != length])

                # Generate new model.
                status, model = pstate.solve(constraint)


def trace_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.debug(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")


def hook_alert_placeholder(se: SymbolicExecutor, pstate: ProcessState, address: int):
    buffer_len = pstate.get_argument_value(2)
    buffer_addr = pstate.get_argument_value(3)
    buffers_len_g[buffer_addr] = buffer_len


prog = Program("./bin/5")

config = Config(pipe_stdout=True,
                skip_unsupported_import=True)

alert_placeholder_addr = prog.find_function_addr("__alert_placeholder")

if alert_placeholder_addr is None:
    logging.fatal(f'alert_placeholder not present!')
    sys.exit(1)

logging.debug(f'alert_placeholder_addr: {alert_placeholder_addr:#x}')

dse = SymbolicExplorator(config, prog)

# dse.add_input_seed(Seed(b"AZERAZAZERA"))
dse.add_input_seed(Seed(b"AZER"))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_pre_addr_callback(alert_placeholder_addr, hook_alert_placeholder)
dse.callback_manager.register_probe(StrncpySanitizer())

dse.explore()
