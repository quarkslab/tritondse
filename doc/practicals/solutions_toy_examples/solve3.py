from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction

once_flag_write = False
once_flag_read = False

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")

def memory_read_callback(se: SymbolicExecutor, pstate: ProcessState, addr):
    global once_flag_read
    if once_flag_read: return
    read_address = addr.getAddress()
    inst_address = pstate.read_register(pstate.registers.rip)
    lea = addr.getLeaAst()
    if lea == None: return
    #print(f"inst: {hex(inst_address)} read: {hex(read_address)}")
    if inst_address == 0x1234:
        print(lea)
        rax_sym = pstate.read_symbolic_register(pstate.registers.rax)
        rax = pstate.read_register(pstate.registers.rax)
        rbp = pstate.read_register(pstate.registers.rbp)
        target = rbp + rax * 4 - 0x80

        if not pstate.is_register_symbolic(pstate.registers.rax):
            print("rax not symbolic")
            return

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
        once_flag_read = True

#   0010120f 89 54 85 80     MOV        dword ptr [RBP + RAX*0x4 + -0x80],EDX
def memory_write_callback(se: SymbolicExecutor, pstate: ProcessState, addr, value):
    global once_flag_write
    if once_flag_write: return
    read_address = addr.getAddress()
    inst_address = pstate.read_register(pstate.registers.rip)
    lea = addr.getLeaAst()
    if lea == None: return
    #print(f"inst: {hex(inst_address)} write {hex(value)} to {hex(read_address)}")
    if inst_address == 0x120f:
        print(lea)
        rax_sym = pstate.read_symbolic_register(pstate.registers.rax)
        rax = pstate.read_register(pstate.registers.rax)
        rbp = pstate.read_register(pstate.registers.rbp)
        target = rbp + rax * 4 - 0x80

        if not pstate.is_register_symbolic(pstate.registers.rax):
            print("rax not symbolic")
            return

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
        once_flag_write = True

p = Program("./3")
dse = SymbolicExplorator(Config(symbolize_stdin=True, skip_unsupported_import=True), p)

dse.add_input_seed(Seed(b"AZERAZER"))

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_memory_read_callback(memory_read_callback)
dse.callback_manager.register_memory_write_callback(memory_write_callback)

dse.explore()
