import logging

from tritondse.callbacks import CbType, ProbeInterface
from tritondse.seed      import Seed, SeedStatus
from tritondse.types     import Architecture



def mk_new_crashing_seed(se, model) -> Seed:
    """
    This function is used by every sanitizers to dump the model found in order
    to trigger a bug into the crash directory.
    """
    new_input = bytearray(se.seed.content)
    for k, v in model.items():
        new_input[k] = v.getValue()
    return Seed(new_input, SeedStatus.CRASH)



class UAFSanitizer(ProbeInterface):
    """
    The UAF sanitizer
        - UFM.DEREF.MIGHT: Use after free
        - UFM.FFM.MUST: Double free
        - UFM.FFM.MIGHT: Double free
    """
    def __init__(self):
        super(UAFSanitizer, self).__init__()
        self.cbs.append((CbType.MEMORY_READ, None, self.memory_read))
        self.cbs.append((CbType.MEMORY_WRITE, None, self.memory_write))
        self.cbs.append((CbType.PRE_RTN, 'free', self.free_routine))


    @staticmethod
    def memory_read(se, pstate, mem):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            logging.critical(f'UAF detected at {mem}')
            se.seed.status = SeedStatus.CRASH
            se.abort()


    @staticmethod
    def memory_write(se, pstate, mem, value):
        ptr = mem.getAddress()
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            logging.critical(f'UAF detected at {mem}')
            se.seed.status = SeedStatus.CRASH
            se.abort()


    @staticmethod
    def free_routine(se, pstate, name, addr):
        ptr = se.pstate.get_argument_value(0)
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            logging.critical(f'Double free detected at {addr:#x}')
            se.seed.status = SeedStatus.CRASH
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
        access_ast = mem.getLeaAst()
        if access_ast is not None and access_ast.isSymbolized():
            model = pstate.tt_ctx.getModel(access_ast == 0)
            if model:
                logging.warning(f'Potential null deref when reading at {mem}')
                crash_seed = mk_new_crashing_seed(se, model)
                se.workspace.save_seed(crash_seed)
                se.seed.status = SeedStatus.OK_DONE
                #se.abort()
        elif access_ast is not None and access_ast.evaluate() == 0:
                # FIXME: Mettre toutes les zones valide en read
                logging.critical(f'Invalid memory access when reading at {mem}')
                se.seed.status = SeedStatus.CRASH
                se.abort()


    @staticmethod
    def memory_write(se, pstate, mem, value):
        access_ast = mem.getLeaAst()
        if access_ast is not None and access_ast.isSymbolized():
            model = pstate.tt_ctx.getModel(access_ast == 0)
            if model:
                logging.warning(f'Potential null deref when writing at {mem}')
                crash_seed = mk_new_crashing_seed(se, model)
                se.workspace.save_seed(crash_seed)
                se.seed.status = SeedStatus.OK_DONE
                #se.abort()
        elif access_ast is not None and access_ast.evaluate() == 0:
                # FIXME: Mettre toutes les zones valide en read
                logging.critical(f'Invalid memory access when writting at {mem}')
                se.seed.status = SeedStatus.CRASH
                se.abort()



class FormatStringSanitizer(ProbeInterface):
    """
    The format string sanitizer
        - SV.TAINTED.FMTSTR
        - SV.FMTSTR.GENERIC
    """
    def __init__(self):
        super(FormatStringSanitizer, self).__init__()
        self.cbs.append((CbType.PRE_RTN, 'printf',  self.xprintf_arg0_routine))
        self.cbs.append((CbType.PRE_RTN, 'fprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'sprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'dprintf', self.xprintf_arg1_routine))
        self.cbs.append((CbType.PRE_RTN, 'snprintf', self.xprintf_arg1_routine))


    @staticmethod
    def printf_family_routines(se, pstate, addr, string_ptr):
        symbolic_cells = 0

        # Count the number of cells which is symbolic
        # TODO: What if there no null byte?
        while se.pstate.tt_ctx.getConcreteMemoryValue(string_ptr):
            if se.pstate.tt_ctx.isMemorySymbolized(string_ptr):
                symbolic_cells += 1
            string_ptr += 1

        if symbolic_cells:
            logging.warning(f'Potential format string of {symbolic_cells} symbolic cells at {addr:#x}')
            se.seed.status = SeedStatus.OK_DONE
            #se.abort()


    @staticmethod
    def xprintf_arg0_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(0)
        FormatStringSanitizer.printf_family_routines(se, pstate, addr, string_ptr)


    @staticmethod
    def xprintf_arg1_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(1)
        FormatStringSanitizer.printf_family_routines(se, pstate, addr, string_ptr)



class IntegerOverflowSanitizer(ProbeInterface):
    """
    The integer overflow sanitizer
        - NUM.OVERFLOW
    """
    def __init__(self):
        super(IntegerOverflowSanitizer, self).__init__()
        self.cbs.append((CbType.POST_INST, None, self.post_instruction))


    @staticmethod
    def post_instruction(se, pstate, instruction):
        # This probe is only available for X86_64 and AARCH64
        # TODO: Should be done only once, but where?
        assert(pstate.architecture == Architecture.X86_64 or pstate.architecture == Architecture.AARCH64)

        rf = (pstate.tt_ctx.registers.of if pstate.architecture == Architecture.X86_64 else pstate.tt_ctx.registers.v)
        flag = pstate.tt_ctx.getRegisterAst(pstate.tt_ctx.registers.rf)
        # TODO: What if it's a legit overflow (signed / unsigned).
        # What if it's not symbolic and 1?
        if flag.isSymbolized():
            model = pstate.tt_ctx.getModel(flag == 1)
            if model:
                logging.warning(f'Potential integer overflow at {instruction}')
                crash_seed = mk_new_crashing_seed(se, model)
                se.workspace.save_seed(crash_seed)
                se.seed.status = SeedStatus.OK_DONE
                #se.abort()
