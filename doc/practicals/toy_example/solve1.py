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
import tritondse.logging


logging.basicConfig(level=logging.DEBUG)
tritondse.logging.enable(level=logging.DEBUG)


def trace_inst(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    logging.debug(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")


def post_exec_hook(se: SymbolicExecutor, pstate: ProcessState):
    logging.debug(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")


def hook_sscanf4(se: SymbolicExecutor, pstate: ProcessState, rtn_name: str, address: Addr):
    # sscanf(buffer, "%d", &j) is treated as j = atoi(buffer)
    ast = pstate.actx

    str_addr = pstate.get_argument_value(0)
    int_addr = pstate.get_argument_value(2)

    fmt_str = pstate.memory.read_string(str_addr)

    cells = {i: pstate.read_symbolic_memory_byte(str_addr+i).getAst() for i in range(10)}

    def multiply(ast, cells, index):
        n = ast.bv(0, 32)
        for i in range(index):
            n = n * 10 + (ast.zx(24, cells[i]) - 0x30)
        return n

    int_ast = ast.ite(
                ast.lnot(ast.land([cells[0] >= 0x30, cells[0] <= 0x39])),
                multiply(ast, cells, 0),
                ast.ite(
                    ast.lnot(ast.land([cells[1] >= 0x30, cells[1] <= 0x39])),
                    multiply(ast, cells, 1),
                    ast.ite(
                        ast.lnot(ast.land([cells[2] >= 0x30, cells[2] <= 0x39])),
                        multiply(ast, cells, 2),
                        ast.ite(
                            ast.lnot(ast.land([cells[3] >= 0x30, cells[3] <= 0x39])),
                            multiply(ast, cells, 3),
                            ast.ite(
                                ast.lnot(ast.land([cells[4] >= 0x30, cells[4] <= 0x39])),
                                multiply(ast, cells, 4),
                                ast.ite(
                                    ast.lnot(ast.land([cells[5] >= 0x30, cells[5] <= 0x39])),
                                    multiply(ast, cells, 5),
                                    ast.ite(
                                        ast.lnot(ast.land([cells[6] >= 0x30, cells[6] <= 0x39])),
                                        multiply(ast, cells, 6),
                                        ast.ite(
                                            ast.lnot(ast.land([cells[7] >= 0x30, cells[7] <= 0x39])),
                                            multiply(ast, cells, 7),
                                            ast.ite(
                                                ast.lnot(ast.land([cells[8] >= 0x30, cells[8] <= 0x39])),
                                                multiply(ast, cells, 8),
                                                ast.ite(
                                                    ast.lnot(ast.land([cells[9] >= 0x30, cells[9] <= 0x39])),
                                                    multiply(ast, cells, 9),
                                                    multiply(ast, cells, 9)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
    int_ast = ast.sx(32, int_ast)

    pstate.write_symbolic_memory_int(int_addr, 8, int_ast)

    try:
        i = int(fmt_str)

        pstate.push_constraint(int_ast == i)

        rv = 1
    except ValueError:
        logging.debug("Failed to convert to int!")
        rv = 0

    return rv


prog = Program("./bin/1")

config = Config(skip_unsupported_import=True,
                seed_format=SeedFormat.COMPOSITE)

dse = SymbolicExplorator(config, prog)

dse.add_input_seed(Seed(CompositeData(files={"tmp.covpro": b"AZERAEZR"})))

dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_pre_imported_routine_callback("__isoc99_sscanf", hook_sscanf4)
# dse.callback_manager.register_pre_instruction_callback(trace_inst)

dse.explore()
