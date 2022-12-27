import logging
import struct

from triton import Instruction
from triton import OPERAND
from triton import OPCODE

from tritondse import CbType
from tritondse import ProbeInterface
from tritondse import ProcessState
from tritondse import SymbolicExecutor
from tritondse import SymbolicExplorator


def read_mem_operand(ctx, mem_operand):
    base = mem_operand.getBaseRegister()
    index = mem_operand.getIndexRegister()
    disp = mem_operand.getDisplacement()
    scale = mem_operand.getScale()

    size = mem_operand.getSize()

    if index.getName() != 'unknown':
        # TODO Check why it returns unknown. Bug?
        addr = ctx.getConcreteRegisterValue(base) + scale.getValue() * ctx.getConcreteRegisterValue(index) + disp.getValue()
    else:
        addr = ctx.getConcreteRegisterValue(base) + disp.getValue()

    val = int.from_bytes(ctx.getConcreteMemoryAreaValue(addr, size), 'little')

    return addr, val


def write_mem_operand(ctx, mem_operand, value):
    base = mem_operand.getBaseRegister()
    index = mem_operand.getIndexRegister()
    disp = mem_operand.getDisplacement()
    scale = mem_operand.getScale()

    if index.getName() != 'unknown':
        # TODO Check why it returns unknown. Bug?
        addr = ctx.getConcreteRegisterValue(base) + scale.getValue() * ctx.getConcreteRegisterValue(index) + disp.getValue()
    else:
        addr = ctx.getConcreteRegisterValue(base) + disp.getValue()

    ctx.setConcreteMemoryAreaValue(addr, value.to_bytes('little'))

    return addr, value


def read_operand(pstate, operand):
    logging.debug(f'Read operand: {operand}')

    if operand.getType() == OPERAND.MEM:
        addr, val = read_mem_operand(pstate.tt_ctx, operand)
        logging.debug(f'Memory operand: [{addr}] = {val}')
    elif operand.getType() == OPERAND.REG:
        val = pstate.read_register(operand.getName())
        logging.debug(f'Register operand: {operand.getName()} = {val}')
    else:
        logging.error(f'Invalid operand type!')
        raise Exception()

    return val


def write_operand(pstate, operand, value):
    logging.debug(f'Write operand: {operand}')

    if operand.getType() == OPERAND.MEM:
        addr, val = write_mem_operand(pstate.tt_ctx, operand, value)
        logging.debug(f'Memory operand: [{addr}] = {val}')
    elif operand.getType() == OPERAND.REG:
        pstate.write_register(operand.getName(), value)
        logging.debug(f'Register operand: {operand.getName()} = {value}')
    else:
        logging.error(f'Invalid operand type!')
        raise Exception()


def mask(size):
    return (2**size)-1


def extract_sd(value):
    # Extract Scalar Double-Precision Floating-Point Value.

    sd_packed = value & 0xffffffffffffffff
    sd_unpacked = struct.unpack('>d', sd_packed.to_bytes(8, 'big'))[0]

    return sd_unpacked


def extract_si(value, size):
    # Extract Scalar Doubleword Integer Value.

    si_unpacked = value & mask(size)

    return si_unpacked


def extract_ss(value):
    # Extract Scalar Double-Precision Floating-Point Value.

    ss_packed = value & 0xffffffff
    ss_unpacked = struct.unpack('>f', ss_packed.to_bytes(4, 'big'))[0]

    return ss_unpacked


def insert_pd(value, sd_high, sd_low):
    # Pack Packed Double-Precision Floating-Point Value.

    sd_high_packed = int.from_bytes(struct.pack('>d', sd_high), 'big')
    sd_low_packed = int.from_bytes(struct.pack('>d', sd_low), 'big')
    pd = (sd_high_packed << 64) | sd_low_packed

    return (value & ~0xffffffffffffffffffffffffffffffff) | pd


def insert_ps(value, ss_high, ss_low):
    # Pack Packed Single-Precision Floating-Point Value.

    ss_high_packed = int.from_bytes(struct.pack('>f', ss_high), 'big')
    ss_low_packed = int.from_bytes(struct.pack('>f', ss_low), 'big')
    ps = (ss_high_packed << 32) | ss_low_packed

    return (value & ~0xffffffffffffffff) | ps


def insert_sd(value, sd):
    # Pack Scalar Double-Precision Floating-Point Value.

    sd_packed = int.from_bytes(struct.pack('>d', sd), 'big')

    return (value & ~0xffffffffffffffff) | sd_packed


def insert_si(value, si):
    # Pack Scalar Doubleword Integer Value.

    si_packed = int.from_bytes(struct.pack('>Q', si), 'big')

    return (value & ~0xffffffffffffffff) | si_packed


def insert_ss(value, ss):
    # Pack Scalar Single-Precision Floating-Point Value.

    ss_packed = int.from_bytes(struct.pack('>f', ss), 'big')

    return (value & ~0xffffffff) | ss_packed


def hook_addsd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # ADDSD—Add Scalar Double-Precision Floating-Point Values

    # Adds the low double-precision floating-point values from the second source operand and the first source operand
    # and stores the double-precision floating-point result in the destination operand.
    # The second source operand can be an XMM register or a 64-bit memory location. The first source and destination
    # operands are XMM registers.

    # ADDSD (128-bit Legacy SSE version)
    # DEST[63:0] DEST[63:0] + SRC[63:0]
    # DEST[MAXVL-1:64] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    dst_sd = extract_sd(dst_raw)
    src_sd = extract_sd(src_raw)

    result = dst_sd + src_sd

    dst_raw = insert_sd(dst_raw, result)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtsi2sd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTSI2SD—Convert Doubleword Integer to Scalar Double-Precision Floating-Point Value

    # Converts a signed doubleword integer (or signed quadword integer if operand size is 64 bits) in the “convert-from”
    # source operand to a double-precision floating-point value in the destination operand. The result is stored in the low
    # quadword of the destination operand, and the high quadword left unchanged. When conversion is inexact, the
    # value returned is rounded according to the rounding control bits in the MXCSR register.
    # The second source operand can be a general-purpose register or a 32/64-bit memory location. The first source and
    # destination operands are XMM registers.

    # CVTSI2SD
    # IF 64-Bit Mode And OperandSize = 64
    # THEN
    #   DEST[63:0] Convert_Integer_To_Double_Precision_Floating_Point(SRC[63:0]);
    # ELSE
    #   DEST[63:0] Convert_Integer_To_Double_Precision_Floating_Point(SRC[31:0]);
    # FI;
    # DEST[MAXVL-1:64] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_si = extract_si(src_raw, 64)

    dst_raw = insert_sd(dst_raw, src_si)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtsi2ss(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTSI2SS—Convert Doubleword Integer to Scalar Single-Precision Floating-Point Value

    # Converts a signed doubleword integer (or signed quadword integer if operand size is 64 bits) in the “convert-from”
    # source operand to a single-precision floating-point value in the destination operand (first operand). The “convert-
    # from” source operand can be a general-purpose register or a memory location. The destination operand is an XMM
    # register. The result is stored in the low doubleword of the destination operand, and the upper three doublewords
    # are left unchanged. When a conversion is inexact, the value returned is rounded according to the rounding control
    # bits in the MXCSR register or the embedded rounding control bits.

    # CVTSI2SS (128-bit Legacy SSE version)
    # IF 64-Bit Mode And OperandSize = 64
    # THEN
    #   DEST[31:0] Convert_Integer_To_Single_Precision_Floating_Point(SRC[63:0]);
    # ELSE
    #   DEST[31:0] Convert_Integer_To_Single_Precision_Floating_Point(SRC[31:0]);
    # FI;
    # DEST[MAXVL-1:32] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_si = extract_si(src_raw, 64)

    dst_raw = insert_ss(dst_raw, src_si)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtsd2ss(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTSD2SS—Convert Scalar Double-Precision Floating-Point Value to Scalar Single-Precision Floating-Point Value

    # Converts a double-precision floating-point value in the “convert-from” source operand (the second operand in
    # SSE2 version, otherwise the third operand) to a single-precision floating-point value in the destination operand.
    # When the “convert-from” operand is an XMM register, the double-precision floating-point value is contained in the
    # low quadword of the register. The result is stored in the low doubleword of the destination operand. When the
    # conversion is inexact, the value returned is rounded according to the rounding control bits in the MXCSR register.

    # CVTSD2SS (128-bit Legacy SSE version)
    # DEST[31:0] Convert_Double_Precision_To_Single_Precision_Floating_Point(SRC[63:0]);
    # (* DEST[MAXVL-1:32] Unmodified *)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_sd = extract_sd(src_raw)

    dst_raw = insert_ss(dst_raw, src_sd)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtss2sd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTSS2SD—Convert Scalar Single-Precision Floating-Point Value to Scalar Double-Precision Floating-Point Value

    # Converts a single-precision floating-point value in the “convert-from” source operand to a double-precision
    # floating-point value in the destination operand. When the “convert-from” source operand is an XMM register, the
    # single-precision floating-point value is contained in the low doubleword of the register. The result is stored in the
    # low quadword of the destination operand.

    # CVTSS2SD (128-bit Legacy SSE version)
    # DEST[63:0] Convert_Single_Precision_To_Double_Precision_Floating_Point(SRC[31:0]);
    # DEST[MAXVL-1:64] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_ss = extract_ss(src_raw)

    dst_raw = insert_sd(dst_raw, src_ss)

    write_operand(pstate, dst_op, dst_raw)


def hook_divss(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # DIVSS—Divide Scalar Single-Precision Floating-Point Values

    # Divides the low single-precision floating-point value in the first source operand by the low single-precision floating-
    # point value in the second source operand, and stores the single-precision floating-point result in the destination
    # operand. The second source operand can be an XMM register or a 32-bit memory location.

    # DIVSS (128-bit Legacy SSE version)
    # DEST[31:0] DEST[31:0] / SRC[31:0]
    # DEST[MAXVL-1:32] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    dst_ss = extract_ss(dst_raw)
    src_ss = extract_ss(src_raw)

    result = dst_ss / src_ss

    dst_raw = insert_sd(dst_raw, result)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvttsd2si(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTTSD2SI—Convert with Truncation Scalar Double-Precision Floating-Point Value to Signed Integer

    # Converts a double-precision floating-point value in the source operand (the second operand) to a signed double-
    # word integer (or signed quadword integer if operand size is 64 bits) in the destination operand (the first operand).
    # The source operand can be an XMM register or a 64-bit memory location. The destination operand is a general
    # purpose register. When the source operand is an XMM register, the double-precision floating-point value is
    # contained in the low quadword of the register.

    # (V)CVTTSD2SI (All versions)
    # IF 64-Bit Mode and OperandSize = 64
    # THEN
    #   DEST[63:0]  Convert_Double_Precision_Floating_Point_To_Integer_Truncate(SRC[63:0]);
    # ELSE
    #   DEST[31:0]  Convert_Double_Precision_Floating_Point_To_Integer_Truncate(SRC[63:0]);
    # FI;

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_sd = extract_sd(src_raw)

    src_si = int(src_sd)

    dst_raw = insert_si(dst_raw, src_si)

    write_operand(pstate, dst_op, dst_raw)


def hook_mulsd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # MULSD—Multiply Scalar Double-Precision Floating-Point Value

    # Multiplies the low double-precision floating-point value in the second source operand by the low double-precision
    # floating-point value in the first source operand, and stores the double-precision floating-point result in the destina-
    # tion operand. The second source operand can be an XMM register or a 64-bit memory location. The first source
    # operand and the destination operands are XMM registers.

    # MULSD (128-bit Legacy SSE version)
    # DEST[63:0] DEST[63:0] * SRC[63:0]
    # DEST[MAXVL-1:64] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    dst_sd = extract_sd(dst_raw)
    src_sd = extract_sd(src_raw)

    result = dst_sd * src_sd

    dst_raw = insert_sd(dst_raw, result)

    write_operand(pstate, dst_op, dst_raw)


def hook_mulpd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # MULPD—Multiply Packed Double-Precision Floating-Point Values

    # Multiply packed double-precision floating-point values from the first source operand with corresponding values in
    # the second source operand, and stores the packed double-precision floating-point results in the destination
    # operand.

    # MULPD (128-bit Legacy SSE version)
    # DEST[63:0] DEST[63:0] * SRC[63:0]
    # DEST[127:64] DEST[127:64] * SRC[127:64]
    # DEST[MAXVL-1:128] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    dst_l_sd = extract_sd(dst_raw)
    dst_h_sd = extract_sd(dst_raw >> 64)

    src_l_sd = extract_sd(src_raw)
    src_h_sd = extract_sd(src_raw >> 64)

    result_l = dst_l_sd * src_l_sd
    result_h = dst_h_sd * src_h_sd

    dst_raw = insert_pd(dst_raw, result_h, result_l)

    write_operand(pstate, dst_op, dst_raw)


def hook_divsd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # DIVSD—Divide Scalar Double-Precision Floating-Point Value

    # Divides the low double-precision floating-point value in the first source operand by the low double-precision
    # floating-point value in the second source operand, and stores the double-precision floating-point result in the desti-
    # nation operand. The second source operand can be an XMM register or a 64-bit memory location. The first source
    # and destination are XMM registers.

    # DIVSD (128-bit Legacy SSE version)
    # DEST[63:0] DEST[63:0] / SRC[63:0]
    # DEST[MAXVL-1:64] (Unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    dst_sd = extract_sd(dst_raw)
    src_sd = extract_sd(src_raw)

    result = dst_sd / src_sd

    dst_raw = insert_sd(dst_raw, result)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtdq2pd(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTDQ2PD—Convert Packed Doubleword Integers to Packed Double-Precision Floating-Point Values

    # Converts two, four or eight packed signed doubleword integers in the source operand (the second operand) to two,
    # four or eight packed double-precision floating-point values in the destination operand (the first operand).

    # CVTDQ2PD (128-bit Legacy SSE version)
    # DEST[63:0]  Convert_Integer_To_Double_Precision_Floating_Point(SRC[31:0])
    # DEST[127:64]  Convert_Integer_To_Double_Precision_Floating_Point(SRC[63:32])
    # DEST[MAXVL-1:128] (unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_l_si = extract_si(src_raw, 32)
    src_h_si = extract_si(src_raw >> 32, 32)

    dst_raw = insert_pd(dst_raw, src_h_si, src_l_si)

    write_operand(pstate, dst_op, dst_raw)


def hook_cvtpd2ps(se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
    # CVTPD2PS—Convert Packed Double-Precision Floating-Point Values to Packed Single-Precision Floating-Point Values

    # Converts two, four or eight packed double-precision floating-point values in the source operand (second operand)
    # to two, four or eight packed single-precision floating-point values in the destination operand (first operand).
    # When a conversion is inexact, the value returned is rounded according to the rounding control bits in the MXCSR
    # register or the embedded rounding control bits.

    # CVTPD2PS (128-bit Legacy SSE version)
    # DEST[31:0]  Convert_Double_Precision_To_Single_Precision_Floating_Point(SRC[63:0])
    # DEST[63:32]  Convert_Double_Precision_To_Single_Precision_Floating_Point(SRC[127:64])
    # DEST[127:64]  0
    # DEST[MAXVL-1:128] (unmodified)

    dst_op = inst.getOperands()[0]
    src_op = inst.getOperands()[1]

    dst_raw = read_operand(pstate, dst_op)
    src_raw = read_operand(pstate, src_op)

    src_l_sd = extract_sd(src_raw)
    src_h_sd = extract_sd(src_raw >> 64)

    dst_raw = insert_ps(dst_raw, src_h_sd, src_l_sd)

    write_operand(pstate, dst_op, dst_raw)


SSE_HOOKS = {
    OPCODE.X86.ADDSD: hook_addsd,
    OPCODE.X86.CVTDQ2PD: hook_cvtdq2pd,
    OPCODE.X86.CVTPD2PS: hook_cvtpd2ps,
    OPCODE.X86.CVTSD2SS: hook_cvtsd2ss,
    OPCODE.X86.CVTSI2SD: hook_cvtsi2sd,
    OPCODE.X86.CVTSI2SS: hook_cvtsi2ss,
    OPCODE.X86.CVTSS2SD: hook_cvtss2sd,
    OPCODE.X86.CVTTSD2SI: hook_cvttsd2si,
    OPCODE.X86.DIVSD: hook_divsd,
    OPCODE.X86.DIVSS: hook_divss,
    OPCODE.X86.MULPD: hook_mulpd,
    OPCODE.X86.MULSD: hook_mulsd,
}


class SSESupport(ProbeInterface):
    """
    Add support for missing floating-point SSE instructions in Triton.
    """
    NAME = "ssesupport-probe"

    def __init__(self):
        super(SSESupport, self).__init__()

        self._add_callback(CbType.POST_INST, self.handle_sse_instr)

    def handle_sse_instr(self, se: SymbolicExecutor, pstate: ProcessState, inst: Instruction):
        mnemonic = inst.getType()

        if mnemonic in SSE_HOOKS:
            logging.info(f'Hooking unsupported instruction: {inst}')
            SSE_HOOKS[mnemonic](se, pstate, inst)
