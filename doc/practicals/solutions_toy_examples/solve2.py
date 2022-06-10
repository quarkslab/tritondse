from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer

once_flag = False

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")

def memory_read_callback(se: SymbolicExecutor, pstate: ProcessState, addr):
    global once_flag
    if once_flag: return
    read_address = addr.getAddress()
    inst_address = pstate.read_register(pstate.registers.rip)
    if inst_address == 0x11c6:
        rax_sym = pstate.read_symbolic_register(pstate.registers.rax)
        rax = pstate.read_register(pstate.registers.rax)
        rbp = pstate.read_register(pstate.registers.rbp)
        target = rbp + rax * 4 - 0x20

        if not pstate.is_register_symbolic(pstate.registers.rax):
            print("rax not symbolic")
            return

        lea = addr.getLeaAst()
        if lea == None: return
        print(f"argv[1] = {se.seed.content} Target = {hex(target)}")
        exp = lea != target
        status, model = pstate.solve(exp)
        while status == SolverStatus.SAT:
            new_seed = se.mk_new_seed_from_model(model)
            se.enqueue_seed(new_seed)
            target = pstate.evaluate_expression_model(lea, model)
            var_values = pstate.get_expression_variable_values_model(rax_sym, model)
            for var, value in var_values.items():
                print(f"{var}: {chr(value)} Target = {hex(target)}")
            exp = pstate.actx.land([exp, lea != target])
            status, model = pstate.solve(exp)
        once_flag = True

p = Program("./2")
dse = SymbolicExplorator(Config(symbolize_argv=True, skip_unsupported_import=True), p)

dse.add_input_seed(Seed(b"./1\x00AZERAZER"))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_memory_read_callback(memory_read_callback)

dse.explore()
