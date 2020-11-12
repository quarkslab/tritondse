# built-ins
import sys
import time
import logging
from typing import Union, Callable, Optional

# third-party
from triton import TritonContext, MemoryAccess, CALLBACK, CPUSIZE, Instruction

# local imports
from tritondse.thread_context import ThreadContext
from tritondse.config         import Config
from tritondse.program        import Program
from tritondse.heap_allocator import HeapAllocator
from tritondse.types          import Architecture, Addr, ByteSize, BitSize, PathConstraint


class ProcessState(object):
    """
    This class is used to represent the state of a process.
    """
    def __init__(self, thread_scheduling: int, time_coefficient: int):
        # Memory mapping
        self.BASE_PLT   = 0x01000000
        self.BASE_ARGV  = 0x02000000
        self.BASE_CTYPE = 0x03000000
        self.ERRNO_PTR  = 0x04000000
        self.BASE_HEAP  = 0x10000000
        self.END_HEAP   = 0x6fffffff
        self.BASE_STACK = 0xefffffff
        self.END_STACK  = 0x70000000
        self.START_MAP  = 0x01000000
        self.END_MAP    = 0xf0000000

        # The Triton's context
        self.tt_ctx = TritonContext()

        # Used to define that the process must exist
        self.stop = False

        # Signals table used by raise(), signal(), etc.
        self.signals_table = dict()

        # File descriptors table used by fopen(), fprintf(), etc.
        self.fd_table = {
            0: sys.stdin,
            1: sys.stdout,
            2: sys.stderr,
        }
        # Unique file id incrementation
        self.fd_id = len(self.fd_table)

        # Allocation information used by malloc()
        self.heap_allocator = HeapAllocator(self.BASE_HEAP, self.END_HEAP)

        # Unique thread id incrementation
        self.utid = 0

        # Current thread id
        self.tid = self.utid

        # Threads contexts
        self.threads = {
            self.utid: ThreadContext(self.tid, thread_scheduling)
        }

        # Thread mutext init magic number
        self.PTHREAD_MUTEX_INIT_MAGIC = 0xdead

        # Mutex and semaphore
        self.mutex_locked = False
        self.semaphore_locked = False

        # The time when the ProcessState is instancied.
        # It's used to provide a deterministic behavior when calling functions
        # like gettimeofday(), clock_gettime(), etc.
        self.time = time.time()

        # Hold the loading address where the main program has been loaded
        self.load_addr = 0

        # Configuration values
        self.thread_scheduling_count = thread_scheduling
        self.time_inc_coefficient = time_coefficient

        # Runtime temporary variables
        self.__pcs_updated = False

        # The current instruction executed
        self.__current_inst = None

        # The memory mapping of the program ({vaddr_s : vaddr_e})
        self.__program_segments_mapping = {}


    def get_unique_thread_id(self):
        self.utid += 1
        return self.utid


    def get_unique_file_id(self):
        self.fd_id += 1
        return self.fd_id


    @property
    def architecture(self) -> Architecture:
        """ Return architecture of the current process state """
        return Architecture(self.tt_ctx.getArchitecture())


    @architecture.setter
    def architecture(self, arch: Architecture) -> None:
        """
        Set the architecture of the process state.
        Internal set it in the TritonContext
        """
        self.tt_ctx.setArchitecture(arch)


    @property
    def ptr_size(self) -> ByteSize:
        """ Size of a pointer in bytes """
        return self.tt_ctx.getGprSize()


    @property
    def ptr_bit_size(self) -> BitSize:
        """ Size of a pointer in bits """
        return self.tt_ctx.getGprBitSize()


    @property
    def minus_one(self) -> int:
        """ -1 according to the architecture size """
        return ((1 << self.ptr_bit_size) - 1)


    def load_program(self, p: Program, base_addr: Addr = 0) -> None:
        """
        Load the given program in the process state memory
        :param p: Program to load in the process memory
        :param base_addr: Base address where to load the program (if PIE)
        :return: True on whether loading succeeded or not
        """
        # Set loading address
        self.load_addr = base_addr
        # TODO: If PIE use this address, if not set absolute address (from binary)

        # Load memory areas in memory
        for vaddr, data in p.memory_segments():
            logging.debug(f"Loading {vaddr:#08x} - {vaddr+len(data):#08x}")
            self.tt_ctx.setConcreteMemoryAreaValue(vaddr, data)
            size = len(data)
            if vaddr in self.__program_segments_mapping and self.__program_segments_mapping[vaddr] > vaddr + size:
                # If we already have a vaddr entry, keep the larger one
                pass
            else:
                self.__program_segments_mapping.update({vaddr : vaddr + size})


    def is_valid_memory_mapping(self, ptr, padding_segment=0) -> bool:
        valid_access = False

        # Check stack area
        if ptr <= self.BASE_STACK and ptr >= self.END_STACK:
            valid_access = True

        # Check heap area
        if ptr >= self.BASE_HEAP and ptr <= self.END_HEAP:
            if self.is_heap_ptr(ptr) and self.heap_allocator.is_ptr_allocated(ptr):
                valid_access = True

        # Check other areas
        if ptr >= self.BASE_PLT and ptr <= self.ERRNO_PTR + CPUSIZE.QWORD:
            valid_access = True

        # Check segments mapping
        for vaddr_s, vaddr_e in self.__program_segments_mapping.items():
            if ptr >= vaddr_s and ptr < vaddr_e + padding_segment:
                valid_access = True

        return valid_access


    def read_memory(self, addr: Addr, size: ByteSize) -> bytes:
        """
        Read in the process memory ``size`` amount of bytes at ``addr``.
        Data read is returned as bytes.

        :param addr: Address at which to read data
        :param size: Number of bytes to read
        :return: Data read
        """
        return self.tt_ctx.getConcreteMemoryAreaValue(addr, size)


    def write_memory(self, addr: Addr, size: ByteSize, data: Union[int, bytes]) -> None:
        """
        Write in the process memory the given data of the given size at
        a specific address.
        :param addr: address where to write data
        :param size: size of data to write in bytes
        :param data: data to write represented as an integer
        :return: None

        .. todo:: Adding a parameter to specify endianess if needed
        """
        assert type(data) is int or type(data) is bytes

        if type(data) is int:
            self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, size), data)
        else:
            self.tt_ctx.setConcreteMemoryAreaValue(addr, data)


    def register_triton_callback(self, cb_type: CALLBACK, callback: Callable):
        """
        Register the given ``callback`` as triton callback to hook memory/registers
        read/writes.

        :param cb_type: CALLBACK type as defined by Triton
        :param callback: routines to call on the given event
        :return: None
        """
        self.tt_ctx.addCallback(cb_type, callback)


    def is_heap_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is coming from the heap area.

        :param ptr: Address to check
        :return: True if pointer points to the heap area (allocated or not).
        """
        if self.BASE_HEAP <= ptr < self.END_HEAP:
            return True
        return False


    def is_stack_ptr(self, ptr: Addr) -> bool:
        """
        Check whether a given address is coming from the stack area.

        :param ptr: Address to check
        :return: True if pointer points to the stack area (allocated or not).
        """
        if self.BASE_STACK <= ptr < self.END_STACK:
            return True
        return False


    def process_instruction(self, instruction: Instruction) -> bool:
        """
        Process the given triton instruction on this process state.
        :param instruction:
        :return:
        """
        self.__pcs_updated = False
        __len_pcs = self.tt_ctx.getPathPredicateSize()

        self.tt_ctx.disassembly(instruction)
        self.__current_inst = instruction
        ret = self.tt_ctx.buildSemantics(instruction)

        # Simulate that the time of an executed instruction is time_inc_coefficient.
        # For example, if time_inc_coefficient is 0.0001, it means that an instruction
        # takes 100us to be executed. Used to provide a deterministic behavior when
        # calling time functions (e.g gettimeofday(), clock_gettime(), ...).
        self.time += self.time_inc_coefficient

        if self.tt_ctx.getPathPredicateSize() > __len_pcs:
            self.__pcs_updated = True

        return ret


    def is_path_predicate_updated(self) -> bool:
        """ Return whether or not the path predicate has been updated """
        return self.__pcs_updated


    @property
    def last_branch_constraint(self) -> PathConstraint:
        """
        Return the last PathConstraint object added in the path predicate.
        Should be called after ``is_path_predicate_updated``.

        :raise: IndexError if the path predicate is empty
        :return:
        """
        return self.tt_ctx.getPathConstraints()[-1]


    @property
    def current_instruction(self) -> Optional[Instruction]:
        """
        Return the current Instruction.
        """
        return self.__current_inst
