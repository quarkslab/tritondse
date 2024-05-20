Seed Scheduling
===============

Seed scheduling algorithm, are classes basically providing the next seed input
to execute. They can be given to the :py:obj:`tritondse.seeds_manager.SeedManager`
constructor. The scheduling of seeds might be different depending on the need.
All strategies should satisfy the interface defined by :py:obj:`tritondse.seeds_manager.SeedManager`.

.. autoclass:: tritondse.seed_scheduler.SeedScheduler
    :members:
    :undoc-members:
    :exclude-members:


Existing strategies
-------------------

.. automodule:: tritondse.seed_scheduler
    :members:
    :show-inheritance:
    :inherited-members:
    :undoc-members:
    :special-members: __len__
    :exclude-members: SeedScheduler
