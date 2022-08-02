TritonDSE documentation
=======================

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

.. toctree::
   :maxdepth: 2

    Installation <installation>
    Getting Started <tutos/starting.ipynb>

.. toctree::
    :caption: Python API
    :maxdepth: 3

    api/program
    api/configuration
    api/callbacks
    api/process
    api/executor
    api/explorator
    api/sanitizers
    api/seeds
    api/workspace
    api/types

.. toctree::
    :caption: Tutorials
    :maxdepth: 3

    Hooks <tutos/hooks-removed.ipynb>

    Seeds <tutos/seeds.ipynb>
    Sanitizers & Probes <tutos/sanitizers.ipynb>
    Loaders <tutos/loaders.ipynb>

.. toctree::
    :caption: Practicals
    :maxdepth: 3

    practicals/toy_example
    practicals/json_parser
    practicals/crackme

.. toctree::
    :caption: Advanced API
    :maxdepth: 3

    dev_doc/coverage
    dev_doc/routines
    dev_doc/seedscheduling


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
