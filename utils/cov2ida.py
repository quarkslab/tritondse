#!/usr/bin/env python
## -*- coding: utf-8 -*-

import sys

with open(sys.argv[1], 'r') as fd:
    coverage = eval(fd.read())

fd = open('idacov.py', 'w+')
for k, v in coverage.items():
    fd.write('idc.set_color(0x%x, idc.CIC_ITEM, 0x024022)\n' % (k))
fd.write('print("Taint applied.")')
fd.close()
