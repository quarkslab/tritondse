#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib


class Seed(object):
    """
    This class is used to represent a seed input. The seed will be injected
    into stdin or argv according to the Triton DSE configuration.
    """
    def __init__(self, content=bytes()):
        self.content     = bytes(content)
        self.target_addr = None


    def __len__(self):
        return len(self.content)


    def __eq__(self, other):
        return self.content == other.content


    def __hash__(self):
        return hash(self.content)


    def get_size(self):
        """
        Returns the size of the seed.
        """
        return len(self.content)


    def get_hash(self):
        """
        Returns the md5 hash of the content
        """
        # Note
        #
        # HF mangling file : <crc64><crc64_reverse>.<size of seed in hexa>.honggfuzz.cov
        # It looks like there is no rule on the naming convention when providing a new seed to HF.
        # Keeping an MD5 one looks good.
        #
        m = hashlib.md5(self.content)
        return m.hexdigest()


class SeedFile(Seed):
    """
    This class is used to represent a seed input form a file.
    """
    def __init__(self, path):
        Seed.__init__(self)
        with open(path, 'rb') as f:
            self.content = f.read()
