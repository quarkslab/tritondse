#!/usr/bin/env python
## -*- coding: utf-8 -*-

import  ctypes
from    tritondse import *


def carry_around_add(a, b):
    c = a + b
    return (c & 0xffff) + (c >> 16)


def checksum_computation(execution, new_input_generated):
    base = 0
    for i in range(0x4): # number of packet with our initial seed
        # TODO: TCP, IGMP, UDP, etc.

        # IPv4: 0x0800
        if new_input_generated[base+12] == 0x08 and new_input_generated[base+13] == 0x00:
            new_input_generated[base+14] = 0x45 # size of the header fixed at 20 bytes
            new_input_generated[base+15] = 0x00 # ?

            p0 = ((new_input_generated[base+14] << 8) | new_input_generated[base+15])
            p1 = ((new_input_generated[base+16] << 8) | new_input_generated[base+17])
            p2 = ((new_input_generated[base+18] << 8) | new_input_generated[base+19])
            p3 = ((new_input_generated[base+20] << 8) | new_input_generated[base+21])
            p4 = ((new_input_generated[base+22] << 8) | new_input_generated[base+23])
            # p5 = checksum position
            p6 = ((new_input_generated[base+26] << 8) | new_input_generated[base+27])
            p7 = ((new_input_generated[base+28] << 8) | new_input_generated[base+29])
            p8 = ((new_input_generated[base+30] << 8) | new_input_generated[base+31])
            p9 = ((new_input_generated[base+32] << 8) | new_input_generated[base+33])

            checksum = 0x0000
            for i in [p0, p1, p2, p3, p4, p6, p7, p8, p9]:
                checksum = carry_around_add(checksum, i)

            checksum = ctypes.c_ushort(~checksum)
            new_input_generated[base+24] = checksum.value >> 8
            new_input_generated[base+25] = checksum.value & 0xff

        base += 600 # the size of a packet in our fuzzing_driver

    return new_input_generated


config = Config(debug=False)

config.symbolize_stdin     = True
config.execution_timeout   = 20
config.cb_post_model       = checksum_computation
config.program_argv        = [
    b'../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln',
    b'wlp0s20f3',
    b'5c:80:b6:96:d7:3c',
    b'192.168.1.45',
    b'255.255.255.0',
    b'192.168.1.255'
]

program = Program('../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln')
seed    = SeedFile('../programme_etalon_final/micro_http_server/private/frame.seed')
dse     = SymbolicExplorator(config, program, seed)

dse.explore()
