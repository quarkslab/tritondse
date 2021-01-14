.. _label_callbacks:

Callback Mechanism
==================

The whole interaction from tritondse with the user-written code is performed
through the callback mechanism. Most callbacks does not expect return values
but provides as argument all necessary variables which enable changing the
state of the execution. Events that can be caught are:

* address reached
* instruction executed *(all of them)*
* memory address read or written
* register read or written
* function reached *(from its name)*
* end of an execution
* thread context switch
* new input creation *(before it gets appended in the pool of seeds)*

Only the new input creation, accept a modified input as return value.
That enable post-processing an input just before it enter the pool of seeds.
That is especially useful to recompute some fields etc.


CallbackManager
---------------

All callbacks are meant to be registered on the :py:obj:`CallbackManager`. That
object managed by :py:obj:`SymbolicExplorator` will be transmitted to every
:py:obj:`SymbolicExecutor` which will then be able to to catch all events. In
such context, callbacks will be triggered indifferently from any execution.
A user willing to do per-execution operation shall register an end of execution
to catch to switch from on execution to the other.

.. autoclass:: tritondse.callbacks.CallbackManager
    :members:
    :undoc-members:
    :exclude-members:


**Auxiliary enumerate:**

.. autoclass:: tritondse.callbacks.CbPos
    :members:
    :undoc-members:
    :exclude-members:



Probe Interface
---------------

The :py:obj:`ProbeInteface` is a very simple mechanism to register multiple callbacks
all at once by subclassing the interface. This interface expect a local attriubte ``cbs``
containing callback related informations.

.. autoclass:: tritondse.callbacks.ProbeInterface
    :members:
    :undoc-members:
    :exclude-members:

.. todo:: The probe interface might evolve in the upcoming versions

**Auxiliary enums**:

.. autoclass:: tritondse.callbacks.CbType
    :members:
    :undoc-members:
    :exclude-members:



Callback signatures
-------------------

.. autodata:: tritondse.callbacks.AddrCallback

.. autodata:: tritondse.callbacks.InstrCallback

.. autodata:: tritondse.callbacks.MemReadCallback

.. autodata:: tritondse.callbacks.MemWriteCallback

.. autodata:: tritondse.callbacks.NewInputCallback

.. autodata:: tritondse.callbacks.RegReadCallback

.. autodata:: tritondse.callbacks.RegWriteCallback

.. autodata:: tritondse.callbacks.RtnCallback

.. autodata:: tritondse.callbacks.SymExCallback

.. autodata:: tritondse.callbacks.ThreadCallback
