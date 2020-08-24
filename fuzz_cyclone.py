#!/usr/bin/env python
## -*- coding: utf-8 -*-

from tritondse import *
from scapy.all import Ether, IP, TCP, UDP


def checksum_computation(execution, new_input_generated):
    base = 0
    for i in range(0x4): # number of packet with our initial seed
        pkt_raw = new_input_generated[base:base+600]
        eth_pkt = Ether(pkt_raw)

        # Remove the checksum generated by the solver
        for proto in [IP, TCP, UDP]:
            if proto in eth_pkt:
                del eth_pkt[proto].chksum
        # Rebuild the Ethernet packet with scapy in order to recompute the checksum
        eth_pkt.build()

        # Rewrite the seed with the appropriate checksum
        count = 0
        for b in raw(eth_pkt):
            new_input_generated[base+count] = b
            count += 1

        base += 600 # the size of a packet in our fuzzing_driver

    return new_input_generated


config = Config(debug=False)

config.symbolize_stdin     = True
config.execution_timeout   = 20
config.thread_scheduling   = 800
config.cb_post_model       = checksum_computation
config.program_argv        = [
    b'./misc/micro_http_server_tt_fuzz_single_without_vuln',
    b'wlp0s20f3',
    b'5c:80:b6:96:d7:3c',
    b'192.168.1.45',
    b'255.255.255.0',
    b'192.168.1.255'
]

program = Program('./misc/micro_http_server_tt_fuzz_single_without_vuln')
#seed    = SeedFile('./misc/frame.seed')
seed     = SeedFile('./misc/frame.seed')

# Explore
#dse     = SymbolicExplorator(config, program, seed)
#dse.explore()

# One execution
ps = ProcessState(config)
execution = SymbolicExecutor(config, ps, program, seed)
execution.run()
execution.coverage.save_on_disk('/tmp')
