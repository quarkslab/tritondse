TritonDSE
=========

TritonDSE is a Python library providing exploration capabilities to Triton
and some refinement easing its usage. This library is primarily designed
to perform pure emulation symbolic execution even though it can also be
applied under different settings. It works by performing an elementary
loading of the program and starts exploring from the entrypoint. The whole
exploration can be instrumented using a hook mechanism enabling to obtain
a handle on various events.

At the moment solely ELF and Linux are supported. But further development
can lead to more platform. Furthermore it provides facilities on the C
runtime and it has not been tested on other types of programs.

Difference with Triton
----------------------

TritonDSE goal is to provide higher-level primitives than `Triton <https://triton-library.github.io/>`_.
Indeed Triton is a low-level framework where one have to provide manually all instructions to be executed
symbolically. As such, TritonDSE provides the following features:

* Loader mechanism (based on `LIEF <https://lief-project.github.io/>`_, `cle <https://github.com/angr/cle>`_ or custom ones)
* Memory segmentation
* Coverage strategies (block, edge, path)
* Pointer coverage
* Automatic input injection on stdin, argv
* Input replay with QBDI
* input scheduling *(customizable)*
* sanitizer mechanism
* basic heap allocator
* some libc symbolic stubs

Installation
------------

Now that Triton is available through Pypi, all TritonDSE dependencies can be installed via pip.
Installing is as simple as:

.. code-block:: bash

    $ pip3 install tritondse



.. toctree::
    :caption: Getting Started
    :maxdepth: 1

    Getting Started <tutos/starting.ipynb>
    Hooks <tutos/hooks.ipynb>
    Seeds <tutos/seeds.ipynb>
    Sanitizers & Probes <tutos/sanitizers.ipynb>
    Loaders <tutos/loaders.ipynb>


.. toctree::
    :caption: Python API
    :maxdepth: 1
    :hidden:

    api/callbacks
    api/configuration
    api/coverage
    api/executor
    api/explorator
    api/exception
    api/loaders
    api/process
    api/sanitizers
    api/seeds
    api/tracing
    api/types
    api/workspace

.. toctree::
    :caption: Practicals
    :maxdepth: 1
    :hidden:

    practicals/toy_example/toy_example
    practicals/json_parser/json_parser
    practicals/crackme/crackme

.. toctree::
    :caption: Advanced API
    :maxdepth: 1
    :hidden:

    dev_doc/routines
    dev_doc/seedscheduling
