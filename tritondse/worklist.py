#!/usr/bin/env python
# -*- coding: utf-8 -*-


class WorklistAddressToSet(object):
    """
    This worklist classifies seeds by addresses. We map a seed X to an
    address Y, if the seed X has been generated to reach the address Y.
    When the method pick() is called, we return a seed which will leads
    to reach new instructions according to the state coverage.
    """
    def __init__(self, config, coverage):
        self.config   = config
        self.coverage = coverage
        self.worklist = dict() # {addr : set(Seed)}


    def __len__(self):
        count = 0
        for k, v in self.worklist.items():
            count += len(v)
        return count


    def add(self, seed):
        if seed.target_addr in self.worklist:
            self.worklist[seed.target_addr].add(seed)
        else:
            self.worklist.update({seed.target_addr : set({seed})})


    def pick(self):
        default = None
        to_remove = set()

        for k, v in self.worklist.items():
            # If the set is empty remove the entry
            if v is None:
                to_remove.add(k)
                continue

            # If the address has never been executed, return the seed
            if k not in self.coverage.instructions:
                return v.pop()

            # If all adresses has been executed, just pick a random seed
            elif not default:
                default = v.pop()

        # Garbage the worklist
        for i in to_remove:
            del self.worklist[i]

        return default


# TODO
class WorklistDSF(object):
    def __init__(self, config, coverage):
        pass
    def __len__(self):
        pass
    def add(self, seed):
        pass
    def pick(self):
        pass


# TODO
class WorklistBFS(object):
    def __init__(self, config, coverage):
        pass
    def __len__(self):
        pass
    def add(self, seed):
        pass
    def pick(self):
        pass


# TODO
class WorklistRand(object):
    def __init__(self, config, coverage):
        pass
    def __len__(self):
        pass
    def add(self, seed):
        pass
    def pick(self):
        pass


# TODO
class WorklistFifo(object):
    def __init__(self, config, coverage):
        pass
    def __len__(self):
        pass
    def add(self, seed):
        pass
    def pick(self):
        pass


# TODO
class WorklistLifo(object):
    def __init__(self, config, coverage):
        pass
    def __len__(self):
        pass
    def add(self, seed):
        pass
    def pick(self):
        pass
