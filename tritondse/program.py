#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lief
import sys
import os
import logging



class Program(object):
    """
    This class is used to represent a program.
    """
    def __init__(self, path : str, argv : list):
        lief.Logger.disable()

        if not os.path.isfile(path):
            logging.error('%s not found' %(path))
            sys.exit(-1)

        self.path   = path
        self.binary = lief.parse(self.path)
        self.argv   = argv


    def __str__(self):
        return self.binary.__str__()


    def get_entry_point(self):
        return self.binary.entrypoint


    def get_argc(self):
        return len(self.argv)


    def get_path(sefl):
        return self.path


    def set_path(sefl, path : str):
        self.path = path
