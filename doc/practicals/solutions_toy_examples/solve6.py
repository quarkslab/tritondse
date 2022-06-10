from tritondse import ProbeInterface, SymbolicExecutor, Config, Program, SymbolicExplorator, ProcessState, CbType, SeedStatus, Seed
from tritondse.types import Addr, SolverStatus, Architecture
from tritondse.sanitizers import NullDerefSanitizer
from triton import Instruction

import logging

buffers_len_g = dict() # buffer_address : buffer_len

class StrncpySanitizer(ProbeInterface):
    # TODO handle strncpy into buff + offset

    def __init__(self):
        super(StrncpySanitizer, self).__init__()
        self._add_callback(CbType.PRE_RTN, self.strncpy_check, 'strncpy')

    def strncpy_check(self, se: SymbolicExecutor, pstate: ProcessState, rtn_name: str, addr: Addr):
        buffer_addr = se.pstate.get_argument_value(0)
        if buffer_addr not in buffers_len_g:
            return 

        buffer_len = buffers_len_g[buffer_addr]
        n = se.pstate.get_argument_value(2)
        n_sym = se.pstate.get_argument_symbolic(2)

        if n > buffer_len:
            logging.critical(f"Found overflowing strncpy buf: {hex(buffer_addr)} bufsize: {buffer_len} copysize: {n}")

            # Generate input to trigger the overflow
            s = pstate.get_argument_value(1)
            ast = pstate.actx

            def rec(res, s, deep, maxdeep):
                if deep == maxdeep:
                    return res
                cell = pstate.read_symbolic_memory_byte(s+deep).getAst()
                res  = ast.ite(cell == 0x00, ast.bv(deep, 64), rec(res, s, deep + 1, maxdeep))
                return res

            sze = len(pstate.get_memory_string(s))
            res = ast.bv(sze, 64)
            res = rec(res, s, 0, sze)

            pstate.push_constraint(pstate.read_symbolic_memory_byte(s+sze).getAst() == 0x00)

            # Manual state coverage of strlen(s) 
            exp = res > n
            status, model = pstate.solve(exp)
            while status == SolverStatus.SAT:
                sze = pstate.evaluate_expression_model(res, model)
                new_seed = se.mk_new_seed_from_model(model)
                #print(f"new_seed: {new_seed.content} len = {hex(sze)}")

                se.enqueue_seed(new_seed)
                var_values = pstate.get_expression_variable_values_model(res, model)
                exp = pstate.actx.land([exp, res != sze])
                status, model = pstate.solve(exp)
            return

        # If n is symbolic, we try to make if bigger than buffer_len
        if n_sym.isSymbolized():
            const = n_sym.getAst() > buffer_len
            st, model = pstate.solve(const)
            if st == SolverStatus.SAT:
                new_seed = se.mk_new_seed_from_model(model)
                #new_seed.status = SeedStatus.CRASH
                se.enqueue_seed(new_seed) 

def trace_inst(exec: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    print(f"[tid:{inst.getThreadId()}] 0x{inst.getAddress():x}: {inst.getDisassembly()}")

def post_exec_hook(se: SymbolicExecutor, state: ProcessState):
    print(f"seed:{se.seed.hash} ({repr(se.seed.content)})   [exitcode:{se.exitcode}]")

def hook_alert_placeholder(exec: SymbolicExecutor, pstate: ProcessState, addr: int):
    buffer_len = pstate.get_argument_value(2)
    buffer_addr = pstate.get_argument_value(3)
    buffers_len_g[buffer_addr] = buffer_len

p = Program("./6")
alert_placeholder_addr = p.find_function_addr("__alert_placeholder")
dse = SymbolicExplorator(Config(symbolize_argv=True, skip_unsupported_import=True), p)

dse.add_input_seed(Seed(b"./6\x00AZERAZER\x00AZERAZER"))

dse.callback_manager.register_post_execution_callback(post_exec_hook)
#dse.callback_manager.register_post_instruction_callback(trace_inst)
dse.callback_manager.register_pre_addr_callback(alert_placeholder_addr, hook_alert_placeholder)
dse.callback_manager.register_probe(StrncpySanitizer())
dse.callback_manager.register_probe(NullDerefSanitizer())

dse.explore()
