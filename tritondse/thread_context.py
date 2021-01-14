from triton import TritonContext



class ThreadContext(object):
    """
    Thread data structure holding all information related to it.
    Purposely used to save registers and to restore them in a
    TritonContext.
    """

    def __init__(self, tid: int, thread_scheduling: int):
        """

        :param tid: thread id
        :type tid: int
        :param thread_scheduling: Thread scheduling value, see :py:attr:`tritondse.Config.thread_scheduling`
        :type thread_scheduling: int
        """
        self.cregs  = dict()                    # context of concrete registers
        self.sregs  = dict()                    # context of symbolic registers
        self.joined = None                      # joined thread id
        self.tid    = tid                       # the thread id
        self.killed = False                     # is the thread killed
        self.count  = thread_scheduling         # Number of instructions executed until scheduling
        # FIXME: Keep the thread_scheduling and automated the reset on restore

    def save(self, tt_ctx: TritonContext) -> None:
        """
        Save the current thread state from the current execution.
        That implies keeping a reference on symbolic and concrete
        registers.

        :param tt_ctx: current TritonContext to save
        :type tt_ctx: `TritonContext <https://triton.quarkslab.com/documentation/doxygen/py_TritonContext_page.html>`_
        """
        # Save symbolic registers
        self.sregs = tt_ctx.getSymbolicRegisters()
        # Save concrete registers
        for r in tt_ctx.getParentRegisters():
            self.cregs.update({r.getId(): tt_ctx.getConcreteRegisterValue(r)})


    def restore(self, tt_ctx: TritonContext) -> None:
        """
        Restore a thread state in the given TritonContext

        :param tt_ctx: context in which to restor the current thread state
        :type tt_ctx: `TritonContext <https://triton.quarkslab.com/documentation/doxygen/py_TritonContext_page.html>`_
        """
        # Restore concrete registers
        for rid, v in self.cregs.items():
            tt_ctx.setConcreteRegisterValue(tt_ctx.getRegister(rid), v)
        # Restore symbolic registers
        for rid, e in self.sregs.items():
            tt_ctx.assignSymbolicExpressionToRegister(e, tt_ctx.getRegister(rid))
