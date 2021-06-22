class SkipInstructionException(Exception):
    """
    Exception to raise in a PRE callback to skip the evaluation
    of the current instruction. It will thus force a SymbolicExecutor
    to fetch the next instruction. Thus the user have to update the
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
