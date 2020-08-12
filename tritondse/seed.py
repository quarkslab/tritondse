#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Seed(object):
    """
    This class is used to represent a seed input. The seed will be injected
    into stdin or argv according to the Triton DSE configuration.
    """
    def __init__(self, content = bytes()):
        self.content = bytes(content)


    def get_size(self):
        """
        Returns the size of the seed.
        """
        return len(self.content)
