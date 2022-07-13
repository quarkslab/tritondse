from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed, SeedFormat, Loader, CompositeData
from tritondse.types import Addr, SolverStatus, Architecture, ArchMode
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction

once_flag = False

def trace_inst(exec: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    print(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")

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
conf = Config(\
    skip_unsupported_import=True, \
    seed_format=SeedFormat.COMPOSITE)

dse = SymbolicExplorator(conf, p)

composite_data = CompositeData(argv=[b"./1", b"AZ\nERAZER"])
dse.add_input_seed(composite_data)

dse.callback_manager.register_probe(NullDerefSanitizer())
dse.callback_manager.register_post_execution_callback(post_exec_hook)
dse.callback_manager.register_memory_read_callback(memory_read_callback)
#dse.callback_manager.register_pre_instruction_callback(trace_inst)

dse.explore()
