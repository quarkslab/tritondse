#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Installation of the tritondse module."""

import sys
from setuptools import setup, find_packages

setup(
    name="tritondse",
    version="0.1",
    description="A library of Dynamic Symbolic Exploration based the Triton library",
    packages=find_packages(),
    setup_requires=[],
    install_requires=[
        "triton-library",
        "lief",
        "pyQBDI",
        "cle",
        "quokka-project",
        "enum_tools"
    ],
    tests_require=[],
    license="qb",
    author="Quarkslab",
    classifiers=[
        'Topic :: Security',
        'Environment :: Console',
        'Operating System :: OS Independent',
    ],
    test_suite="",
    scripts=[]
)
