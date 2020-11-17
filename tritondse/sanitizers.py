import logging

from triton              import MemoryAccess, CPUSIZE
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
    # Don't tag the seed as CRASH before executing it.
    # At this stage, we do not know if the seed will really make the
    # program crash or not.
    return Seed(new_input)
    #return Seed(new_input, SeedStatus.CRASH)



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
    def check(se, pstate, ptr, desc):
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            logging.critical(desc)
            se.seed.status = SeedStatus.CRASH
            pstate.stop = True
            return True
        return False


    @staticmethod
    def memory_read(se, pstate, mem):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')


    @staticmethod
    def memory_write(se, pstate, mem, value):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')


    @staticmethod
    def free_routine(se, pstate, name, addr):
        ptr = se.pstate.get_argument_value(0)
        return UAFSanitizer.check(se, pstate, ptr, f'Double free detected at {addr:#x}')


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
    def check(se, pstate, mem, desc):
        ptr = mem.getAddress()
        valid_access = False

        # The execution has not started yet
        if pstate.current_instruction is None:
            return False

        # FIXME: Takes so much time...
        #if access_ast is not None and access_ast.isSymbolized():
        #    model = pstate.tt_ctx.getModel(access_ast == 0)
        #    if model:
        #        logging.warning(f'Potential null deref when reading at {mem}')
        #        crash_seed = mk_new_crashing_seed(se, model)
        #        se.workspace.save_seed(crash_seed)
        #        se.seed.status = SeedStatus.OK_DONE
        #        # Do not abort, just continue the execution
        #if access_ast is not None and access_ast.evaluate() == 0:

        # FIXME: Ici on rajoute 16 car nous avons un problème si une instruction se situe
        # en fin de page mappée. Lors du fetching des opcodes, nous fetchons 16 bytes car
        # nous ne connaissons pas la taille d'une instruction, ici, en fetchant en fin de
        # page on déclenche ce sanitizer...

        if ptr == 0 or pstate.is_valid_memory_mapping(ptr, padding_segment=16) == False:
            logging.critical(desc)
            se.seed.status = SeedStatus.CRASH
            pstate.stop = True
            return True

        return False


    @staticmethod
    def memory_read(se, pstate, mem):
        return NullDerefSanitizer.check(se, pstate, mem, f'Invalid memory access when reading at {mem} from {pstate.current_instruction}')


    @staticmethod
    def memory_write(se, pstate, mem, value):
        return NullDerefSanitizer.check(se, pstate, mem, f'Invalid memory access when writting at {mem} from {pstate.current_instruction}')



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
    def solve_query(se, pstate, query):
        model = pstate.tt_ctx.getModel(query)
        if model:
            crash_seed = mk_new_crashing_seed(se, model)
            se.enqueue_seed(crash_seed)
            logging.warning(f'Model found for a seed which may lead to a crash ({crash_seed.filename})')


    @staticmethod
    def check(se, pstate, addr, string_ptr):
        symbolic_cells = []

        # Count the number of cells which is symbolic
        while se.pstate.tt_ctx.getConcreteMemoryValue(string_ptr):
            if se.pstate.tt_ctx.isMemorySymbolized(string_ptr):
                symbolic_cells.append(string_ptr)
            string_ptr += 1

        if symbolic_cells:
            logging.warning(f'Potential format string of {len(symbolic_cells)} symbolic memory cells at {addr:#x}')
            se.seed.status = SeedStatus.OK_DONE
            actx = pstate.tt_ctx.getAstContext()
            query1 = pstate.tt_ctx.getPathPredicate()
            query2 = (actx.bvtrue() == actx.bvtrue())
            for i in range(int(len(symbolic_cells) / 2)):
                cell1  = pstate.tt_ctx.getMemoryAst(MemoryAccess(symbolic_cells.pop(0), CPUSIZE.BYTE))
                cell2  = pstate.tt_ctx.getMemoryAst(MemoryAccess(symbolic_cells.pop(0), CPUSIZE.BYTE))
                query1 = actx.land([query1, cell1 == ord('%'), cell2 == ord('s')])
                query2 = actx.land([query2, cell1 == ord('%'), cell2 == ord('s')])
                FormatStringSanitizer.solve_query(se, pstate, query1)
                FormatStringSanitizer.solve_query(se, pstate, query2) # This method may be incorrect but help to discover bugs
            # Do not stop the execution, just continue the execution
            pstate.stop = False
            return True
        return False


    @staticmethod
    def xprintf_arg0_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(0)
        FormatStringSanitizer.check(se, pstate, addr, string_ptr)


    @staticmethod
    def xprintf_arg1_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(1)
        FormatStringSanitizer.check(se, pstate, addr, string_ptr)



class IntegerOverflowSanitizer(ProbeInterface):
    """
    The integer overflow sanitizer
        - NUM.OVERFLOW
    """
    def __init__(self):
        super(IntegerOverflowSanitizer, self).__init__()
        self.cbs.append((CbType.POST_INST, None, self.check))


    @staticmethod
    def check(se, pstate, instruction):
        # This probe is only available for X86_64 and AARCH64
        assert(pstate.architecture == Architecture.X86_64 or pstate.architecture == Architecture.AARCH64)

        rf = (pstate.tt_ctx.registers.of if pstate.architecture == Architecture.X86_64 else pstate.tt_ctx.registers.v)
        flag = pstate.tt_ctx.getRegisterAst(pstate.tt_ctx.registers.rf)
        if flag.evaluate():
            logging.warning(f'Integer overflow at {instruction}')
            # FIXME: What if it's normal behavior?
            se.seed.status = SeedStatus.CRASH
            return True

        if flag.isSymbolized():
            model = pstate.tt_ctx.getModel(flag == 1)
            if model:
                logging.warning(f'Potential integer overflow at {instruction}')
                crash_seed = mk_new_crashing_seed(se, model)
                se.enqueue_seed(crash_seed)
                return True

        return False
