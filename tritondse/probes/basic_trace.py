from tritondse import ProbeInterface, CbType, SymbolicExecutor, ProcessState, SymbolicExplorator
import tritondse.logging

logger = tritondse.logging.get("probe.basictrace")


class BasicDebugTrace(ProbeInterface):
    """
    Basic probe that print instruction trace
    to logging debug.
    """
    NAME = "debugtrace-probe"

    def __init__(self):
        super(BasicDebugTrace, self).__init__()
        self._add_callback(CbType.PRE_INST, self.trace_debug)

    def trace_debug(self, exec: SymbolicExecutor, __: ProcessState, ins: 'Instruction'):
        logger.debug(f"[tid:{ins.getThreadId()}] {exec.trace_offset} [0x{ins.getAddress():x}]: {ins.getDisassembly()}")



class BasicTextTrace(ProbeInterface):
    """
    Basic probe that generate a txt execution trace
    for each run.
    """
    NAME = "txttrace-probe"

    def __init__(self):
        super(BasicTextTrace, self).__init__()
        self._add_callback(CbType.PRE_EXEC, self.pre_execution)
        self._add_callback(CbType.POST_EXEC, self.post_execution)
        self._add_callback(CbType.PRE_INST, self.trace_debug)

        # File in which to write the trace
        self._file = None

    def pre_execution(self, executor: SymbolicExecutor, _: ProcessState):
        # Triggered before each execution
        name = f"{executor.uid}-{executor.seed.hash}.txt"
        file = executor.workspace.get_metadata_file_path(f"{self.NAME}/{name}")
        self._file = open(file, "w")

    def post_execution(self, executor: SymbolicExecutor, _: ProcessState):
        self._file.close()

    def trace_debug(self, exec: SymbolicExecutor, __: ProcessState, ins: 'Instruction'):
        """
        This function is mainly used for debug.

        :param _: The current symbolic executor
        :param __: The current processus state of the execution
        :param ins: The current instruction executed
        :return: None
        """
        self._file.write(f"[tid:{ins.getThreadId()}] {exec.trace_offset} [0x{ins.getAddress():x}]: {ins.getDisassembly()}\n")
