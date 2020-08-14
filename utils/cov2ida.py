#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This util is used to covert a metadata/coverage file into an IDA plugin.
#
# Usage:
#
#   $ ./cov2ida.py ./metadata/coverage
#
# Then, load idacov.py into IDA.

import sys

with open(sys.argv[1], 'r') as fd:
    coverage = eval(fd.read())

fd = open('idacov.py', 'w+')
for k, v in coverage.items():
    fd.write('idc.set_color(0x%x, idc.CIC_ITEM, 0x024022)\n' % (k))
fd.write('print("Taint applied.")')
fd.close()
