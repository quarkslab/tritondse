# built-in imports
import json
import logging

# Local imports
from tritondse.seed import Seed
from tritondse.coverage import GlobalCoverage
from tritondse.workspace import Workspace


class SeedScheduler:
    """
    Abstract class for all seed selection strategies.
    This class provides the base methods that all
    subclasses should implement to be compliant with
    the interface.
    """

    def has_seed_remaining(self) -> bool:
        """
        Returns true if there are still seeds to be processed in the scheduler

        :returns: true if there are seeds to process
        """
        raise NotImplementedError()


    def add(self, seed: Seed) -> None:
        """
        Add a new seed in the scheduler

        :param seed: Seed to add in the scheduler
        :type seed: Seed
        """
        raise NotImplementedError()


    def update_worklist(self, coverage: GlobalCoverage) -> None:
        """
        Call after every execution.
        That function might help the scheduler with some of its internal states.
        For instance the scheduler is keep somes seed meant to cover an address
        which is now covered, it can just drop these seeds.

        :param coverage: global coverage of the exploration
        :type coverage: GlobalCoverage
        """
        raise NotImplementedError()


    def can_solve_models(self) -> bool:
        """
        Function called by the seed manager to know if it can
        start negating branches to discover new paths. Some seed
        scheduler might want to run concretely all inputs first
        before starting negating branches.

        :return: true if the :py:obj:`SeedManager` can negate branches
        """
        raise NotImplementedError()


    def pick(self) -> Seed:
        """
        Return the next seed to execute.

        :returns: seed to execute
        :rtype: Seed
        """
        raise NotImplementedError()


    def post_execution(self) -> None:
        """
        Called at the end of each execution after the generation of new seeds through SMT.
        Last thing called before starting the next iteration.
        """
        pass


    def post_exploration(self, workspace: Workspace) -> None:
        """
        Called at the end of the exploration to perform
        some clean-up or anything else.
        """
        pass



class WorklistAddressToSet(SeedScheduler):
    """
    This worklist classifies seeds by addresses. We map a seed X to an
    address Y, if the seed X has been generated to reach the address Y.
    When the method pick() is called, seeds covering a new address 'Y'
    are selected first. Otherwise anyone is taken.
    """
    def __init__(self, manager: 'SeedManager'):
        self.manager = manager
        self.cov = None
        self.worklist = dict() # {CovItem: set(Seed)}


    def __len__(self) -> int:
        """ Number of pending seeds to execute """
        count = 0
        for k, v in self.worklist.items():
            count += len(v)
        return count


    def has_seed_remaining(self) -> bool:
        """ Returns true if there are still seeds in the worklist """
        return len(self) != 0


    def add(self, seed: Seed) -> None:
        """ Add a seed to the worklist """
        for obj in seed.coverage_objectives:
            if obj in self.worklist:
                self.worklist[obj].add(seed)
            else:
                self.worklist[obj] = {seed}


    def update_worklist(self, coverage: GlobalCoverage) -> None:
        """ Update the coverage state of the woklist with the global one """
        self.cov = coverage


    def can_solve_models(self) -> bool:
        """
        Always true.
        This strategy always allows solving branches. As a consequence
        it might try to solve a branch already covered in a seed not run yet.
        But this enables iterating a seed only once.

        :returns: True
        """
        return True


    def pick(self) -> Seed:
        """ Return the next seed to execute

        :returns: next seed to execute (first one covering new addresses, otherwise any other)
        :rtype: Seed
        """
        seed_picked = None
        item_picked = None
        to_remove = set()

        for k, v in self.worklist.items():
            # If the set is empty remove the entry
            if not len(v):
                to_remove.add(k)
                continue

            # If the address has never been executed, return the seed
            if not self.cov.is_covered(k):
                seed_picked = v.pop()
                item_picked = k
                if not len(v):
                    to_remove.add(k)
                break

        # If all adresses has been executed, just pick a random seed
        if not seed_picked:
            for k, v in self.worklist.items():
                if v:
                    seed_picked = v.pop()
                    item_picked = k
                    if not len(v):
                        to_remove.add(k)
                    break

        # Pop the seed from all worklist[X] where it is
        for obj in seed_picked.coverage_objectives:
            if obj != item_picked:   # already poped it from item_picked thus only pop the other
                self.worklist[obj].remove(seed_picked)
                if not self.worklist[obj]:
                    to_remove.add(obj)

        # Garbage the worklist
        for i in to_remove:
            self.worklist.pop(i)

        return seed_picked


class WorklistRand(SeedScheduler):
    """
    Trivial strategy that returns any Seed without any classification.
    It uses a Set for insertion and pop (which is random) for picking seeds.
    """
    def __init__(self, manager: 'SeedManager'):
        self.worklist = set() # set(Seed)


    def __len__(self) -> int:
        """ Number of pending seeds to execute """
        return len(self.worklist)


    def has_seed_remaining(self) -> bool:
        """ Returns true if there are still seeds in the worklist """
        return len(self) != 0


    def add(self, seed: Seed) -> None:
        """ Add a seed to the worklist

        :param seed: Seed to add to this rand scheduler
        :type seed: Seed
        """
        self.worklist.add(seed)


    def update_worklist(self, coverage: GlobalCoverage) -> None:
        """ Update the coverage state of the worklist with the global one """
        self.cov = coverage


    def can_solve_models(self) -> bool:
        """ Always true """
        return True


    def pick(self) -> Seed:
        """
        Return the next seed to execute. The method pop() removes a random element
        from the set and returns the removed element. Unlike, a stack a
        random element is popped off the set.

        :returns: next seed to executre
        :rtype: Seed
        """
        return self.worklist.pop()


class FreshSeedPrioritizerWorklist(SeedScheduler):
    """
    Strategy that first execute all seeds without negating branches
    in order to get the most updated coverage and which then re-run
    all relevant seeds to negate their branches.

    This worklist works as follow:
        - return first fresh seeds first to get them executed (to improve coverage)
        - keep the seed in the worklist up until it gets dropped or thoroughtly processed
        - if no fresh seed is available, iterates seed that will generate coverage
    """
    def __init__(self, manager: 'SeedManager'):
        self.manager = manager
        self.fresh = []       # Seed never processed (list to make sure we can pop first one received)
        self.worklist = dict() # CovItem -> set(Seed)


    def __len__(self) -> int:
        """ Number of pending seeds to execute """
        s = set()
        for seeds in self.worklist.values():
            s.update(seeds)
        return len(self.fresh) + len(s)


    def has_seed_remaining(self) -> bool:
        """ Returns true if there are still seeds in the worklist """
        return len(self) != 0


    def add(self, seed: Seed) -> None:
        """ Add a seed to the worklist

        :param seed: seed to add to the scheduler
        :type seed: Seed
        """
        if seed.coverage_objectives:  # If the seed already have coverage objectives
            for item in seed.coverage_objectives:  # Add it in our worklist
                if item in self.worklist:
                    self.worklist[item].add(seed)
                else:
                    self.worklist[item] = {seed}
            # seed.coverage_objectives.clear()  # Flush the objectives
        else:  # Otherwise it is fresh
            self.fresh.append(seed)


    def update_worklist(self, coverage: GlobalCoverage) -> None:
        """ Update the coverage state of the worklist with the global one """
        # Iterate the worklist to see if some items have now been covered
        # and are thus not interesting anymore
        to_remove = [x for x in self.worklist if coverage.is_covered(x)]

        for item in to_remove:
            for seed in self.worklist.pop(item):
                seed.coverage_objectives.remove(item)
                if not seed.coverage_objectives:  # The seed cannot improve the coverage of anything
                    self.manager.drop_seed(seed)


    def can_solve_models(self) -> bool:
        """
        Returns True if there are no "fresh" seeds to execute.

        :returns: True if all fresh seeds have been executed.
        """
        return not self.fresh


    def pick(self) -> Seed:
        """ Return the next seed to execute """
        # Pop first fresh seed
        if self.fresh:
            return self.fresh.pop(0)  # Return first item as it is the older

        # Then pop seed meant to crash
        if ... in self.worklist:  # If we have specific seeds (mostly generated by sanitizers)
            it = self.worklist[...].pop()
            if not self.worklist[...]:  # Remove the key if now empty
                self.worklist.pop(...)
            return it

        # Then pop traditional coverage seeds
        k = list(self.worklist.keys())[0]      # arbitrary covitem
        seed = self.worklist[k].pop()          # remove first seed inside
        for it in seed.coverage_objectives:    # Remove the seed from all worklist[x]
            if it != k:                        # we already popped the item from k
                self.worklist[it].remove(seed) # remove the seed from that covitem set
            if not self.worklist[it]:          # remove the whole covitem if empty
                self.worklist.pop(it)
        return seed

    def post_execution(self) -> None:
        """
        Solely used to show intermediate statistics
        """
        logging.info(f"Seed Scheduler: worklist:{len(self)} Coverage objectives:{len(self.worklist)}  (fresh:{len(self.fresh)})")

    def post_exploration(self, workspace: Workspace) -> None:
        """
        At the end of the execution, print the worklist to know
        its state before exit.
        """
        workspace.save_metadata_file("coverage_objectives.json", json.dumps(list(self.worklist.keys())))
