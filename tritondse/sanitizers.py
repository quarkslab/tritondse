from __future__ import annotations
import logging

from triton              import MemoryAccess, CPUSIZE, Instruction
from tritondse.callbacks import CbType, ProbeInterface
from tritondse.seed      import Seed, SeedStatus
from tritondse.types     import Architecture, Addr, Tuple, SolverStatus
from tritondse           import SymbolicExecutor, ProcessState


def mk_new_crashing_seed(se, model) -> Seed:
    """
    This function is used by every sanitizers to dump the model found in order
    to trigger a bug into the crash directory.

    :return: A fresh Seed
    """
    new_input = bytearray(se.seed.content)
    for k, v in model.items():
        new_input[k] = v.getValue()
    # Don't tag the seed as CRASH before executing it.
    # At this stage, we do not know if the seed will really make the
    # program crash or not.
    return Seed(new_input)



class UAFSanitizer(ProbeInterface):
    """
    Use-After-Free Sanitizer.
    It is able to detect UaF and double-free. It works by hooking
    all memory read/write if it points to the heap in a freed area
    then the Use-After-Free is detected. It also hooks the free
    routine to detect double-free.
    """
    def __init__(self):
        super(UAFSanitizer, self).__init__()
        self._add_callback(CbType.MEMORY_READ, self._memory_read)
        self._add_callback(CbType.MEMORY_WRITE, self._memory_write)
        self._add_callback(CbType.PRE_RTN, self._free_routine, 'free')


    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, ptr: Addr, description: str = None) -> bool:
        """
        Checks whether the given ``ptr`` is symptomatic of a Use-After-Free by querying
        various methods of :py:obj:`tritondse.heap_allocator.HeapAllocator`.

        :param se: symbolic executor
        :type se: SymbolicExecutor
        :param pstate: process state
        :type pstate: ProcessState
        :param ptr: pointer address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :param description: description string printed in logger if an issue is detected
        :return: True if the bug is present
        """
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            if description:
                logging.critical(description)
            se.seed.status = SeedStatus.CRASH
            pstate.stop = True
            return True
        return False


    @staticmethod
    def _memory_read(se, pstate, mem):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')


    @staticmethod
    def _memory_write(se, pstate, mem, value):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')


    @staticmethod
    def _free_routine(se, pstate, name, addr):
        ptr = se.pstate.get_argument_value(0)
        return UAFSanitizer.check(se, pstate, ptr, f'Double free detected at {addr:#x}')


class NullDerefSanitizer(ProbeInterface):
    """
    Null Dereference Sanitizer.
    Simply checks if any memory read or write is performed at address 0.
    If so an error is raised.
    """
    def __init__(self):
        super(NullDerefSanitizer, self).__init__()
        self._add_callback(CbType.MEMORY_READ, self._memory_read)
        self._add_callback(CbType.MEMORY_WRITE, self._memory_write)


    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, ptr: Addr, description: str = None) -> bool:
        """
        Checks that the ``ptr`` given is basically not 0.

        :param se: symbolic executor
        :type se: SymbolicExecutor
        :param pstate: process state
        :type pstate: ProcessState
        :param ptr: pointer address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :param description: description string printed in logger if an issue is detected
        :return: True if the bug is present
        """

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

        # FIXME: Why do we call is_valid_memory_mapping ? It is not a "Null Deref vulnerability", it is more a segmentation error
        if ptr == 0 or not pstate.is_valid_memory_mapping(ptr, padding_segment=16):
            if description:
                logging.critical(description)
            se.seed.status = SeedStatus.CRASH
            pstate.stop = True
            return True

        return False


    @staticmethod
    def _memory_read(se, pstate, mem):
        return NullDerefSanitizer.check(se, pstate, mem.getAddress(), f'Invalid memory access when reading at {mem} from {pstate.current_instruction}')


    @staticmethod
    def _memory_write(se, pstate, mem, value):
        return NullDerefSanitizer.check(se, pstate, mem.getAddress(), f'Invalid memory access when writting at {mem} from {pstate.current_instruction}')



class FormatStringSanitizer(ProbeInterface):
    """
    Format String Sanitizer.
    This probes hooks standard libc functions like 'printf', 'fprintf', 'sprintf',
    'dprintf', 'snprintf' and if one of them is triggered it checks the format string.
    If the format string is symbolic then it is user controlled. A warning is shown
    but the execution not interrupted. However the sanitizer tries through SMT to
    generate format strings with many '%s'. If satisfiable a new input is generated
    which will then be added to inputs to process. That subsequent input might lead
    to a crash.
    """
    def __init__(self):
        super(FormatStringSanitizer, self).__init__()
        self._add_callback(CbType.PRE_RTN,  self._xprintf_arg0_routine, 'printf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'fprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'sprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'dprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'snprintf')


    @staticmethod
    def check(se, pstate, fmt_ptr, extra_data: Tuple[str, Addr] = None):
        """
        Checks that the format string at ``fmt_ptr`` does not contain
        symbolic bytes. If so shows an alert and tries to generate new
        inputs with as many '%s' as possible.

        :param se: symbolic executor
        :type se: SymbolicExecutor
        :param pstate: process state
        :type pstate: ProcessState
        :param fmt_ptr: pointer address to check
        :type fmt_ptr: :py:obj:`tritondse.types.Addr`
        :param extra_data: additionnal infos given by the callbacks on routines (indicating function address)
        :type extra_data: Tuple[str, :py:obj:`tritondse.types.Addr`]
        :return: True if the bug is present
        """
        symbolic_cells = []

        # Count the number of cells which is symbolic
        cur_ptr = fmt_ptr
        while se.pstate.read_memory_int(cur_ptr, 1):  # while different from 0
            if se.pstate.is_memory_symbolic(cur_ptr, 1):
                symbolic_cells.append(cur_ptr)
            cur_ptr += 1

        if symbolic_cells:
            extra = f"(function {extra_data[0]}@{extra_data[1]:#x})" if extra_data else ""
            logging.warning(f'Potential format string of length {len(symbolic_cells)} on {fmt_ptr:x} {extra}')
            se.seed.status = SeedStatus.OK_DONE
            pp_seeds = []
            nopp_seeds = []

            for i in range(int(len(symbolic_cells) / 2)):
                # FIXME: Does not check that cell1 and cell2 are contiguous
                cell1 = pstate.read_symbolic_memory_byte(symbolic_cells.pop(0)).getAst()
                cell2 = pstate.read_symbolic_memory_byte(symbolic_cells.pop(0)).getAst()

                # Try to solve once with the path predicate
                st, model = pstate.solve([cell1 == ord('%'), cell2 == ord('s')], with_pp=True)
                if st == SolverStatus.SAT and model:
                    pp_seeds.append(mk_new_crashing_seed(se, model))

                # Try once again without the path predicate (may be incorrect but help to discover bug)
                st, model = pstate.solve_no_pp([cell1 == ord('%'), cell2 == ord('s')])
                if st == SolverStatus.SAT and model:
                    pp_seeds.append(mk_new_crashing_seed(se, model))

            # If found some seeds
            if pp_seeds:
                s = pp_seeds[-1]
                se.enqueue_seed(s)  # Only keep last seed
                logging.warning(f'Found model that might lead to a crash: {s.hash} (with path predicate)')
            if nopp_seeds:
                s = nopp_seeds[-1]
                se.enqueue_seed(s)  # Only keep last seed
                logging.warning(f'Found model that might lead to a crash: {s.hash} (without path predicate)')

            # Do not stop the execution, just continue the execution
            pstate.stop = False
            return True
        return False


    @staticmethod
    def _xprintf_arg0_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(0)
        FormatStringSanitizer.check(se, pstate, string_ptr, (name, addr))


    @staticmethod
    def _xprintf_arg1_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(1)
        FormatStringSanitizer.check(se, pstate, string_ptr, (name, addr))



class IntegerOverflowSanitizer(ProbeInterface):
    """
    Integer Overflow Sanitizer.
    This probe checks on every instructions that the overflow
    flag is not set. If so mark the input as a crashing input.
    If not, but the value is symbolic, via SMT solving to make
    it to be set (and thus to overflow). If possible generates
    a new input to be executed.
    """
    def __init__(self):
        super(IntegerOverflowSanitizer, self).__init__()
        self._add_callback(CbType.POST_INST, self.check)


    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, instruction: Instruction) -> bool:
        """
        The entry point of the sanitizer. This function check if a bug is present

        :param se: symbolic executor
        :type se: SymbolicExecutor
        :param pstate: process state
        :type pstate: ProcessState
        :param instruction: Instruction that has just been executed
        :type instruction: `Instruction <https://triton.quarkslab.com/documentation/doxygen/py_Instruction_page.html>`_
        :return: True if the bug is present
        """
        # This probe is only available for X86_64 and AARCH64
        assert(pstate.architecture == Architecture.X86_64 or pstate.architecture == Architecture.AARCH64)

        rf = (pstate.registers.of if pstate.architecture == Architecture.X86_64 else pstate.registers.v)
        flag = pstate.read_symbolic_register(rf)
        if flag.evaluate():
            logging.warning(f'Integer overflow at {instruction}')
            # FIXME: What if it's normal behavior?
            se.seed.status = SeedStatus.CRASH
            return True

        if flag.isSymbolized():
            _, model = pstate.solve_no_pp(flag == 1)
            if model:
                logging.warning(f'Potential integer overflow at {instruction}')
                crash_seed = mk_new_crashing_seed(se, model)
                se.enqueue_seed(crash_seed)
                return True

        return False
