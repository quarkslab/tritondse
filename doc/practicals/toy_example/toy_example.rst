Toy Example
===========

These toy examples will present you various use-cases containing a bug to trigger.
The goal is to trigger them using the tritondse exploration.


0. Multiple input sources
-------------------------

TritonDSE supports injecting input on multiple locations. Use a `COMPOSITE` seed to inject
`stdin` and `argv` and explore the program to trigger the crash.

.. literalinclude:: src/0.c
   :language: c


1. Non standard input
---------------------

The goal here is to trigger the bug by symbolizing the content of a file.
Furthermore, `sscanf` is currently not supported by TritonDSE, you will need to provide
the emulation yourself.

Hint: In this case, `sscanf` behaves similarly to `atoi`. Check out the emulatoin of
`atoi` in `tritondse/routines.py`.

.. literalinclude:: src/1.c
   :language: c


2. Symbolic read
----------------

By default the exploration just negate branches, but does not try to perform
state coverage on pointer values *(as it raises a lot of test-cases potentially
not interesting)*. The goal here is to perform manual state coverage on pointer
values.

.. literalinclude:: src/2.c
   :language: c


3. Symbolic read & write
------------------------

Same principle here, except that triggering the bug require resolving some
kind of a pointer aliasing issue.

.. literalinclude:: src/3.c
   :language: c


4. String length
----------------

Symbolic execution hardly infers 'meta-properties' of data. For a string its length
is a meta-property that the symbolic executor does not know how to mutate. It can
be an issue when performing coverage.

.. literalinclude:: src/4.c
   :language: c


5. Off-by-One example
---------------------

Write a simple intrinsic function to obtain the stack buffer size
during exploration, and write a simple sanitizer for `strncpy` that
checks that no buffer overflow is taking place.

.. literalinclude:: src/5.c
   :language: c

