#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tritondse.callbacks import CbType, ProbeInterface
from tritondse.program   import Program
from tritondse.seed      import Seed



def save_model(se, model):
    """
    This function is used by every sanitizers to dump the model found in order
    to trigger a bug into the crash directory.
    """
    new_input = bytearray(se.seed.content)
    for k, v in model.items():
        new_input[k] = v.getValue()
    new_seed = Seed(new_input)
    new_seed.save_on_disk(se.config.crash_dir)



class UAFSanitizer(ProbeInterface):
    """
    The UAF sanitizer
        - UFM.DEREF.MIGHT: Use after free
        - UFM.FFM.MUST: Double free
    """
    def __init__(self):
        super(UAFSanitizer, self).__init__()
        self.cbs.append((CbType.MEMORY_READ, None, self.memory_read))
        self.cbs.append((CbType.MEMORY_WRITE, None, self.memory_write))
        self.cbs.append((CbType.PRE_RTN, 'free', self.free_routine)) # FIXME: ATM it's not possible on imported functions


    @staticmethod
    def memory_read(se, pstate, mem):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            print(f'UAF detected at {mem}')
            se.seed.save_on_disk(se.config.crash_dir)
            se.abort()


    @staticmethod
    def memory_write(se, pstate, mem, value):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            print(f'UAF detected at {mem}')
            se.seed.save_on_disk(se.config.crash_dir)
            se.abort()


    @staticmethod
    def free_routine(se, pstate, addr):
        ptr = se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            print(f'Double free detected at {addr:#x}')
            se.seed.save_on_disk(se.config.crash_dir)
            se.abort()



class NullDerefSanitizer(ProbeInterface):
    """
    The null deref sanitizer
        - NPD.FUNC.MUST
    """
    def __init__(self):
        super(NullDerefSanitizer, self).__init__()
        self.cbs.append((CbType.MEMORY_READ, None, self.memory_read))
        self.cbs.append((CbType.MEMORY_WRITE, None, self.memory_write))


    @staticmethod
    def memory_read(se, pstate, mem):
        ast = pstate.tt_ctx.getAstContext()
        access_ast = mem.getLeaAst()
        if access_ast is not None and access_ast.isSymbolized():
            model = pstate.tt_ctx.getModel(access_ast == 0)
            if model:
                print(f'Potential null deref when reading at {mem}')
                save_model(se, model)
                se.abort()


    @staticmethod
    def memory_write(se, pstate, mem, value):
        ast = pstate.tt_ctx.getAstContext()
        access_ast = mem.getLeaAst()
        if access_ast is not None and access_ast.isSymbolized():
            model = pstate.tt_ctx.getModel(access_ast == 0)
            if model:
                print(f'Potential null deref when writing at {mem}')
                save_model(se, model)
                se.abort()



class FormatStringSanitizer(ProbeInterface):
    """
    The format string sanitizer
        - SV.TAINTED.FMTSTR
    """
    def __init__(self):
        super(FormatStringSanitizer, self).__init__()
        self.cbs.append((CbType.PRE_RTN, 'printf',  self.xprintf_arg0_routine))
        self.cbs.append((CbType.PRE_RTN, 'fprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'sprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'dprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'snprintf', self.xprintf_arg1_routine))


    @staticmethod
    def printf_family_routines(se, pstate, addr, string):
        symbolic_cells = 0

        # Count the number of cells which is symbolic
        while se.pstate.tt_ctx.getConcreteMemoryValue(string):
            if se.pstate.tt_ctx.isMemorySymbolized(string):
                symbolic_cells += 1
            string += 1

        if symbolic_cells:
            print(f'Potential format string of {symbolic_cells} symbolic cells at {addr:#x}')
            se.seed.save_on_disk(se.config.crash_dir)
            se.abort()


    @staticmethod
    def xprintf_arg0_routine(se, pstate, addr):
        self.printf_family_routines(se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0)))


    @staticmethod
    def xprintf_arg1_routine(se, pstate, addr):
        self.printf_family_routines(se.pstate.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1)))
