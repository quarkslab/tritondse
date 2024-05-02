class SkipInstructionException(Exception):
    """
    Exception to raise in a PRE callback to skip the evaluation
    of the current instruction. It will thus force a SymbolicExecutor
    to fetch the next instruction. Thus, the user have to update the
    RIP of the ProcessState currently being executed.
    """
    pass


class AbortExecutionException(Exception):
    """
    Exception to rais in a callback to stop current SymbolicExecutor.
    The user should be careful to set the status of the current seed
    being executed.
    """
    pass


class StopExplorationException(Exception):
    """
    Exception to raise in a callback to stop the whole exploration of
    the program. It is caught by SymbolicExplorator.
    """
    pass


class AllocatorException(Exception):
    """
    Class used to represent a heap allocator exception.
    This exception can be raised in the following conditions:

    * trying to allocate data which overflow heap size
    * trying to free a pointer already freed
    * trying to free a non-allocated pointer
    """
    def __init__(self, message):
        super(Exception, self).__init__(message)


class ProbeException(Exception):
    """
    Exception to raise in a probe to stop the current exception.
    It is caught by SymbolicExplorator.
    """
    def __init__(self, message):
        super(Exception, self).__init__(message)
