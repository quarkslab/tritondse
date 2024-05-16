JSON Parser
===========

Introduction
------------

The firmware is extracted from an IoT device that needs parsing some files
and in this use-case JSON files. Only the JSON parsing part of the firmware
is available. The MCU running this firmware is a STM32F412.

The goal is exploring the implementation with TritonDSE to try finding vulnerabilities.

.. raw:: html

    <div style="text-align: center;"><a href="../../_static/bugged_json_parser.bin"><i class="fa fa-download fa-lg"></i><br/>binary</a></div><br/>


Practical information
---------------------

* The code is ARM Thumb-2
* Base address is: 0x08000000
* Entrypoint is: 0x81dc46e | 1
* Exit point: (can be set to instruction just after)

The entrypoint is a call on the function parsing the JSON input.
Its prototype is the following:

.. code-block:: c

	int json_parser(char* buffer, int len, JSON_ctx* ctx);

While the two first parameters are straightforward. The third is an
object for which we don't know the exact structure.

Objectives
----------

* Load the firmware into a ``SymbolicExplorator`` (see the `Loaders` section of the tutorial).

* Try out different exploration strategies and visualize the resulting coverage in Lighthouse.

	- Different `CoverageStrategy` values.
	- Inject one one parameter at the time then multiple ones (see `How to inject arbitrary variables` in the `Seeds` section of the tutorial).

* Use `Sanitizers` to detect potential bugs.

Tips
----

The `JSON_ctx` structure contains two callback functions that you will need to stub (using hooks) to enable emulation to be carried to the end.

* Make sure both buffers ``buffer`` and ``ctx`` points to a recognizable memory area.


