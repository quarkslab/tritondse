from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed, SeedFormat, CompositeData
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer

from tritondse.routines import rtn_atoi

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")

def hook_fread(exec: SymbolicExecutor, pstate: ProcessState, routine: str, addr: int):
    # We hook fread to symbolize what is being read
    arg = pstate.get_argument_value(0)
    sizeof = pstate.get_argument_value(2)
    exec.inject_symbolic_input(arg, exec.seed)
    print("Symbolizing {} bytes at {}".format(hex(sizeof), hex(arg)))
    s = pstate.memory.read_string(arg)
    print(f"fread: {s}")
    return 0

def hook_sscanf4(exec: SymbolicExecutor, pstate: ProcessState, routine: str, addr: int):
    # sscanf(buffer, "%d", &j) is treated as j = atoi(buffer)
    ast = pstate.actx
    addr_j = pstate.get_argument_value(2)
    arg = pstate.get_argument_value(0)
    int_str = pstate.memory.read_string(arg)

    cells = {i: pstate.read_symbolic_memory_byte(arg+i).getAst() for i in range(10)}

    def multiply(ast, cells, index):
        n = ast.bv(0, 32)
        for i in range(index):
            n = n * 10 + (ast.zx(24, cells[i]) - 0x30)
        return n

    res = ast.ite(
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
    res = ast.sx(32, res)

    pstate.write_symbolic_memory_int(addr_j, 8, res)

    try:
        i = int(int_str)
        constraint = res == i
        pstate.push_constraint(constraint)
    except:
        print("Failed to convert to int")

    return res

p = Program("./1")
dse = SymbolicExplorator(Config(\
        skip_unsupported_import=True,\
        seed_format=SeedFormat.COMPOSITE), p)

dse.add_input_seed(Seed(CompositeData(files={"stdin": b"AZERZAER", "tmp.covpro": b"AZERAEZR"})))

dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_probe(NullDerefSanitizer())
#dse.callback_manager.register_post_imported_routine_callback("fread", hook_fread)
dse.callback_manager.register_pre_imported_routine_callback("__isoc99_sscanf", hook_sscanf4)

dse.explore()
