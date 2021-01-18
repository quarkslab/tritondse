Installation
============

Installing tritondse requires a valid Triton installation. Then all dependencies can
be installed through pip. Installation steps are:

* installing Triton (cf. `Documentation <https://triton.quarkslab.com/documentation/doxygen/index.html#install_sec>`_)
* installing tritondse with:

.. code-block:: bash

    $ cd tritondse
    $ pip3 install .

.. note:: As of the version 0.11 of LIEF, it can be installed in the same manner on an ARM, Aarch64
          Linux as it now chip a version of LIEF for that architecture. *(otherwise requires to be
          installed manually by compiling it)*.
