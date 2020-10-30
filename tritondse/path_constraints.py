import os


class PathConstraintsHash(object):
    """
    This class is used to represent all paths taken during the symbolic
    exploration to avoid redundant constraints asking to the SMT solver.
    """
    def __init__(self):
        self.hashes = set()


    def add_hash_constraint(self, h):
        self.hashes.add(h)


    def merge(self, other):
        """ Merge an other instance of hash constraints into this instance"""
        for crt_hash in other.hashes:
            self.hashes.add(crt_hash)


    def number_of_constraints_asked(self):
        return len(self.hashes)


    def hash_already_asked(self, h):
        return (True if h in self.hashes else False)
