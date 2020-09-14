#!/usr/bin/env python3

import sys
import struct
from pathlib import Path

from tritondse import *
from scapy.all import Ether, IP, TCP, UDP


def hook_dumphexa(se: SymbolicExecutor, state: ProcessState, addr: Addr):
    buffer = state.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(0))
    size = state.tt_ctx.getConcreteRegisterValue(se.abi.get_arg_register(1))
    data = state.read_memory(buffer, size)
    with open("full_comm.cov", "ab") as f:
        f.write(struct.pack('<H', len(data))+data)


def trace_debug(se: SymbolicExecutor, state: ProcessState, instruction: 'Instruction'):
    print("[tid:%d] %#x: %s" %(instruction.getThreadId(), instruction.getAddress(), instruction.getDisassembly()))


def checksum_computation(se: SymbolicExecutor, state: ProcessState, new_input_generated: Input):
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


if __name__ == '__main__':
    config = Config(debug=False)
    config.symbolize_stdin      = True
    config.execution_timeout    = 0
    config.thread_scheduling    = 500
    config.time_inc_coefficient = 0.00001
    config.program_argv         = [
        b'../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln',
        b'wlp0s20f3',
        b'48:e2:44:f5:9b:01',
        b'10.0.13.86',
        b'255.255.255.0',
        b'10.0.13.254'
    ]

    try:
        program = Program('../programme_etalon_final/micro_http_server/micro_http_server_tt_fuzz_single_without_vuln')
    except FileNotFoundError as e:
        print(e)
        sys.exit(-1)

    seed = SeedFile('../programme_etalon_final/micro_http_server/misc/frame.seed')

    # Explore
    #dse = SymbolicExplorator(config, program, seed)
    #dse.callback_manager.register_new_input_callback(checksum_computation)
    #dse.explore()

    # One execution
    #ps = ProcessState(config)
    #execution = SymbolicExecutor(config, ps, program, seed)
    #execution.callback_manager.register_function_callback("dumphexa", hook_dumphexa)
    #execution.callback_manager.register_post_instuction_callback(trace_debug)
    #execution.run()

    # One execution with sanitizer
    ps = ProcessState(config)
    execution = SymbolicExecutor(config, ps, program, seed)
    execution.callback_manager.register_probe_callback(UAFSanitizer())
    execution.callback_manager.register_probe_callback(NullDerefSanitizer())
    execution.callback_manager.register_probe_callback(FormatStringSanitizer())
    execution.callback_manager.register_probe_callback(IntegerOverflowSanitizer())
    execution.run()
