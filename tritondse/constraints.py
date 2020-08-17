#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Constraints(object):
    """
    This class is used to represent all constraints asked to the SMT solver
    during the symbolic exploration.
    """
    def __init__(self):
        self.hashes = set()


    def add_constraint(self, constraint):
        self.hashes.add(constraint.getHash())


    def merge(self, other):
        """ Merge an other instance of Constraints into this instance"""
        for crt_hash in other.hashes:
            self.hashes.add(crt_hash)


    def number_of_constraints_asked(self):
        return len(self.hashes)


    def save_on_disk(self, directory):
        with open(f'{directory}/constraints', 'w+') as fd:
            fd.write(repr(self.hashes))


    def already_asked(self, constraint):
        return (True if constraint.getHash() in self.hashes else False)
