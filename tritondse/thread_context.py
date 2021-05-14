from enum import Enum, auto

from triton import TritonContext


class ThreadState(Enum):
    RUNNING = auto()  # Normal state
    DEAD    = auto()  # State after pthread_exit
    JOINING = auto()  # State after pthread_join
    LOCKED  = auto()  # State after a pthread_lock & co


class ThreadContext(object):
    """
    Thread data structure holding all information related to it.
    Purposely used to save registers and to restore them in a
    TritonContext.
    """

    def __init__(self, tid: int):
        """

        :param tid: thread id
        :type tid: int
        """
        self.cregs  = dict()                    # context of concrete registers
        self.sregs  = dict()                    # context of symbolic registers
        self._join_th_id = None                 # joined thread id
        self.tid    = tid                       # the thread id
        self.count  = 0                         # Number of instructions executed until scheduling
        self.state  = ThreadState.RUNNING
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


    def kill(self) -> None:
        """
        Kill the current thread. Called when exiting the thread.

        :return:
        """
        self.state = ThreadState.DEAD

    def is_dead(self) -> bool:
        """
        Returns whether the thread is killed or not

        :return: boolean indicating if the thread is dead or not
        """
        return self.state == ThreadState.DEAD

    def join_thread(self, th_id: int) -> None:
        """
        Put the thread in a join state where waits for
        another thread.

        :param th_id: id of the thread to join
        :return: None
        """
        self._join_th_id = th_id
        self.state = ThreadState.JOINING

    def is_waiting_to_join(self) -> bool:
        """
        Checks whether the thread is waiting to join
        another one.

        :return: boolean on whether it waits for another thread
        """
        return self.state == ThreadState.JOINING

    def cancel_join(self) -> None:
        """
        Cancel a join operation.

        :return: None
        """
        self._join_th_id = None
        self.state = ThreadState.RUNNING

    def is_main_thread(self) -> bool:
        """
        Returns whether or not it is the main thread
        (namely its id is 0)

        :return: bool
        """
        return self.tid == 0

    def is_running(self) -> bool:
        """
        Return if the thread is properly running or not.

        :return: True if the thread is running
        """
        return self.state == ThreadState.RUNNING
