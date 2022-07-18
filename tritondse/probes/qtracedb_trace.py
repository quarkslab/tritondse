from datetime import datetime
import logging

from triton import Instruction, MemoryAccess

from qtracedb import DatabaseManager, ArchsManager, MemAccessType
from qtracedb.archs.manager import SupportedArchs


from tritondse import SymbolicExecutor, ProcessState
from tritondse.callbacks import CbType, ProbeInterface
from tritondse.types import Architecture


ARCH_MAPPER = {
    Architecture.X86: SupportedArchs.x86,
    Architecture.X86_64: SupportedArchs.x86_64,
    Architecture.ARM32: SupportedArchs.ARM,
    Architecture.AARCH64: SupportedArchs.ARM64
}


class TraceGenerator(ProbeInterface):
    """
    Generate a Qtrace-DB execution trace from
    the given execution.
    """
    def __init__(self, name_prefix: str = None):
        super(TraceGenerator, self).__init__()
        self._add_callbacks()
        self.dbm = DatabaseManager.from_qtracedb_config()
        self.trace = None
        self.name_prefix = name_prefix if name_prefix else ""
        self.cur_inst = None
        self.arch = None
        self.reentry_lock = False  # lock to avoid re-entrency on mem read

        # Accumulated data
        self.insts = []
        self.mems = []

    def _add_callbacks(self):
        self._add_callback(CbType.PRE_EXEC, self.pre_exec_hook)
        self._add_callback(CbType.POST_EXEC, self.post_exec_hook)
        self._add_callback(CbType.PRE_INST, self.instr_hook)
        self._add_callback(CbType.POST_INST, self.mem_hook)
        # self._add_callback(CbType.MEMORY_READ, self.mem_read_hook)
        # self._add_callback(CbType.MEMORY_WRITE, self.mem_write_hook)

    def pre_exec_hook(self, se: SymbolicExecutor, pstate: ProcessState):
        name = self.name_prefix if self.name_prefix else se.program.path.name
        name = f"{name}-{se.seed.hash}"
        if name in self.dbm.list_traces():
            logging.info(f"[qtracedb] clear trace: {name}")
            self.dbm.remove_trace(name)
        else:
            logging.info(f"[qtracedb] create trace: {name}")

        self.arch = ARCH_MAPPER[pstate.architecture]
        self.trace = self.dbm.create_trace(self.arch.name, name=name)
        self.insts = []  # clear instructions and mems from on exec to the other
        self.mems = []
        self.trace.add_module(pstate.load_addr, se.program._binary.virtual_size, se.program.path.name)

    def instr_hook(self, se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
        regs = {x.name: getattr(pstate.cpu, x.name.lower()) for x in ArchsManager.get_supported_regs(self.arch)}
        self.cur_inst = self.trace.make_instr(opcode=inst.getOpcode(), **regs)
        self.insts.append(self.cur_inst)

    def mem_hook(self, se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
        for load in inst.getLoadAccess():
            self._mem_hook(pstate, load[0], MemAccessType.read)
        for store in inst.getStoreAccess():
            self._mem_hook(pstate, store[0], MemAccessType.write)
    #
    # def mem_read_hook(self, se: SymbolicExecutor, pstate: ProcessState, mem: MemoryAccess):
    #     self._mem_hook(pstate, mem, MemAccessType.read)
    #
    # def mem_write_hook(self, se: SymbolicExecutor, pstate: ProcessState, mem: MemoryAccess):
    #     self._mem_hook(pstate, mem, MemAccessType.write)

    def _mem_hook(self, pstate: ProcessState, mem: MemoryAccess, kind: MemAccessType):
        addr = mem.getAddress()
        size = mem.getSize()
        data = pstate.memory.read(addr, size)
        self.mems.append(self.trace.add_memaccess(kind, addr, data, self.cur_inst))

    def post_exec_hook(self, se: SymbolicExecutor, pstate: ProcessState):
        self.trace.add_all(self.insts)
        self.trace.add_all(self.mems)
