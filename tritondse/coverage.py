#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Coverage(object):
    """
    This class is used to represent the coverage of an execution.
    """
    def __init__(self):
        self.instructions = dict()


    def add_instruction(self, address : int, inc : int = 1):
        if address in self.instructions:
            self.instructions[address] += inc
        else:
            self.instructions[address] = inc


    def merge(self, other):
        """ Merge an other instance of Coverage into this instance"""
        for k, v in other.instructions.items():
            self.add_instruction(k, v)


    def number_of_instructions_covered(self):
        return len(self.instructions)


    def number_of_instructions_executed(self):
        count = 0
        for k, v in self.instructions.items():
            count += v
        return count


    def save_on_disk(self, directory):
        with open(f'{directory}/coverage', 'w+') as fd:
            fd.write(repr(self.instructions))


    def load_from_disk(self, directory):
        with open(f'{directory}/coverage', 'r') as fd:
            data = fd.read()
            if len(data):
                self.instructions = eval(data)
