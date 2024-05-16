import logging

from triton import Instruction
from triton import MemoryAccess

from tritondse.config import Config
from tritondse.loaders.program import Program
from tritondse.process_state import ProcessState
from tritondse.sanitizers import NullDerefSanitizer
from tritondse.seed import CompositeData
from tritondse.seed import Seed
from tritondse.seed import SeedFormat
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.symbolic_explorator import SymbolicExplorator
from tritondse.types import SolverStatus
import tritondse.logging


logging.basicConfig(level=logging.DEBUG)
tritondse.logging.enable(level=logging.DEBUG)

once_flag = False


def trace_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.debug(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")


def memory_read_callback(se: SymbolicExecutor, pstate: ProcessState, maccess: MemoryAccess):
    global once_flag

    if once_flag:
        return

    inst_address = pstate.read_register(pstate.registers.rip)

    if inst_address == 0x11c6:
        # Check whether the register containing the array index is symbolic.
        if not pstate.is_register_symbolic(pstate.registers.rax):
            logging.debug("rax is not symbolic, returning")
            return

        rax_sym = pstate.read_symbolic_register(pstate.registers.rax)

        # Get address of the array item that will be compared.
        rax = pstate.read_register(pstate.registers.rax)
        rbp = pstate.read_register(pstate.registers.rbp)
        target = rbp + rax * 4 - 0x20

        lea_ast = maccess.getLeaAst()
        if lea_ast is None:
            return

        logging.debug(f"argv[1] = {se.seed.content.argv[1]} Target = {target:x}")

        # Generate model to get an array item different from the current one.
        constraint = lea_ast != target
        status, model = pstate.solve(constraint)

        while status == SolverStatus.SAT:
            # Generate new seed from model.
            new_seed = se.mk_new_seed_from_model(model)

            # Enqueue seed.
            se.enqueue_seed(new_seed)

            # Obtain the new target value from the current model.
            target = pstate.evaluate_expression_model(lea_ast, model)

            # Generate new index value from the current model.
            var_values = pstate.get_expression_variable_values_model(rax_sym, model)

            for var, value in var_values.items():
                logging.debug(f"{var}: {chr(value)} Target = {target:x}")

            # Add newly generated target to the constraints in order to generate
            # a different value for the next iteration.
            constraint = pstate.actx.land([constraint, lea_ast != target])

            # Generate new model.
            status, model = pstate.solve(constraint)

        once_flag = True


prog = Program("./bin/2")

config = Config(skip_unsupported_import=True,
                seed_format=SeedFormat.COMPOSITE)

dse = SymbolicExplorator(config, prog)

dse.add_input_seed(Seed(CompositeData(argv=[b"./bin/2", b"AZ\nERAZER"])))

dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_memory_read_callback(memory_read_callback)
# dse.callback_manager.register_pre_instruction_callback(trace_inst)

dse.explore()
