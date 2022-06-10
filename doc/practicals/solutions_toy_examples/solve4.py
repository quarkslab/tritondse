from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction

once_flag = False

def trace_inst(exec: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    print(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")

def hook_strlen(se: SymbolicExecutor, pstate: ProcessState, routine: str, addr: int):
    global once_flag
    if once_flag: return

    # Get arguments
    s = pstate.get_argument_value(0)
    ast = pstate.actx

    def rec(res, s, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(s+deep).getAst()
        res  = ast.ite(cell == 0x00, ast.bv(deep, 64), rec(res, s, deep + 1, maxdeep))
        return res

    sze = 20#len(pstate.get_memory_string(s))
    res = ast.bv(sze, 64)
    res = rec(res, s, 0, sze)

    print('test')
    print(type(pstate.read_symbolic_memory_byte(s+sze).getAst()))
    print(type(pstate.read_symbolic_memory_byte(s+sze)))
    pstate.push_constraint(pstate.read_symbolic_memory_byte(s+sze).getAst() == 0x00)

    # Manual state coverage of strlen(s) 
    exp = res != sze
    status, model = pstate.solve(exp)
    while status == SolverStatus.SAT:
        sze = pstate.evaluate_expression_model(res, model)
        new_seed = se.mk_new_seed_from_model(model)
        #print(f"new_seed: {new_seed.content} len = {hex(sze)}")

        # TODO Voir s'il y a mieux 
        tmp = new_seed.content[::-1]
        last_null = len(tmp) - tmp.index(b"\x00") - 1
        new_seed_truncated = Seed(new_seed.content[:last_null])
        se.enqueue_seed(new_seed_truncated)
        ##########################

        se.enqueue_seed(new_seed)
        var_values = pstate.get_expression_variable_values_model(res, model)
        exp = pstate.actx.land([exp, res != sze])
        status, model = pstate.solve(exp)

    once_flag = True
    return res


p = Program("./4")
dse = SymbolicExplorator(Config(symbolize_argv=True, skip_unsupported_import=True), p)

#dse.add_input_seed(Seed(b"./4\x00AZERAZERAZER"))
dse.add_input_seed(Seed(b"./4\x00AAAA"))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
#dse.callback_manager.register_memory_write_callback(memory_write_callback)
#dse.callback_manager.register_post_instruction_callback(trace_inst)
dse.callback_manager.register_pre_imported_routine_callback("strlen", hook_strlen)

dse.explore()
