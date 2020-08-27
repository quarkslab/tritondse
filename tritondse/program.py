# built-in imports
from pathlib import Path
from typing import Optional, Generator, Tuple

# third party
import lief

# local imports
from tritondse.types import PathLike, Addr, Architecture
from tritondse.routines import *


lief.Logger.disable()

_arch_mapper = {lief.ARCHITECTURES.ARM: Architecture.ARM32,
                lief.ARCHITECTURES.ARM64: Architecture.AARCH64,
                lief.ARCHITECTURES.X86: Architecture.X86}


class Program(object):
    """
    Class representation of a program (loaded in memory)
    This class is wrapping LIEF to represent a program and to provide
    all the features allowing to load one regardless of its format.
    :raise: FileNotFoundError if the file is not properly recognized by lief
            or in the wrong architecture
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self._binary = lief.parse(str(self.path))
        self._arch = self._load_arch()

        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        if self._binary is None:  # lief has not been able to parse it
            raise FileNotFoundError(f"file {path} not recognised by lief")

        if self._arch is None:
            raise FileNotFoundError(f"binary {path} architecture unsupported {self._binary.abstract.header.architecture}")

    @property
    def entry_point(self) -> Addr:
        """
        Return the program entrypoint address as defined
        in the binary headers
        """
        return self._binary.entrypoint

    @property
    def architecture(self) -> Architecture:
        """
        Returns the architecture enum representing the target
        architecture as an enum object.
        """
        return self._arch

    @property
    def endianness(self) -> lief.ENDIANNESS:
        """
        Returns the endianness of the program as defined in the
        binary headers.
        :return: Endianness as defined by LIEF
        """
        return self._binary.abstract.header.endianness

    @property
    def format(self) -> lief.EXE_FORMATS:
        """
        Returns the binary format. Supported formats by lief are: ELF, PE, MachO
        :return: formats value as defined by lief
        """
        return self._binary.format

    def _load_arch(self) -> Optional[Architecture]:
        """
        Load architecture as an Architecture object.
        :return: Architecture or None if unsupported
        """
        arch = self._binary.abstract.header.architecture
        if arch in _arch_mapper:
            arch = _arch_mapper[arch]
            if arch == Architecture.X86:
                arch = Architecture.X86 if self._binary.abstract.header.is_32 else Architecture.X86_64
            return arch
        else:
            return None

    @property
    def relocation_enum(self):
        """
        Return the lief relocation enum associated with the current
        architecture of the binary.
        :return:
        """
        rel_map = {lief.ELF.ARCH.ARM: lief.ELF.RELOCATION_ARM,
                   lief.ELF.ARCH.AARCH64: lief.ELF.RELOCATION_AARCH64,
                   lief.ELF.ARCH.i386: lief.ELF.RELOCATION_i386,
                   lief.ELF.ARCH.x86_64: lief.ELF.RELOCATION_X86_64,
                   lief.ELF.ARCH.PPC: lief.ELF.RELOCATION_PPC,
                   lief.ELF.ARCH.PPC64: lief.ELF.RELOCATION_PPC64}
        return rel_map[self._binary.header.machine_type]

    def _is_glob_dat(self, rel: lief.ELF.Relocation) -> bool:
        rel_enum = self.relocation_enum
        if hasattr(rel_enum, "GLOB_DAT"):
            return rel_enum(rel.type) == getattr(rel_enum, "GLOB_DAT")
        else:
            return False  # Not GLOB_DAT relocation for this architecture

    def memory_segments(self) -> Generator[Tuple[Addr, bytes], None, None]:
        """
        Iterate over all memory segments of the program as loaded in memory.
        :return: Generator of tuples addrs and content
        :raise: NotImplementedError if the binary format cannot be loaded
        """
        if self.format == lief.EXE_FORMATS.ELF:
            for seg in self._binary.concrete.segments:
                if seg.type == lief.ELF.SEGMENT_TYPES.LOAD:
                    content = seg.content
                    if seg.virtual_size != len(seg.content):  # pad with zeros (as it might be .bss)
                        content += [0] * (seg.virtual_size-seg.physical_size)
                    yield seg.virtual_address, content
        else:
            raise NotImplementedError(f"memory segments not implemented for: {self.format.name}")

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported functions by the program. This function
        is a generator of tuples associating the function and its relocation
        address in the binary.
        :return: Generator of FunName, relocation address
        """
        if self.format == lief.EXE_FORMATS.ELF:
            try:
                # Iterate functions imported through PLT
                for rel in self._binary.concrete.pltgot_relocations:
                    yield rel.symbol.name, rel.address

                # Iterate functions imported via mandatory relocation (e.g: __libc_start_main)
                for rel in self._binary.dynamic_relocations:
                    if self._is_glob_dat(rel) and rel.has_symbol:
                        yield rel.symbol.name, rel.address
            except Exception:
                logging.error('Something wrong with the pltgot relocations')

        else:
            raise NotImplementedError(f"Imported functions relocations not implemented for: {self.format.name}")

    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported variable symbols. Yield for each of them the name and
        the relocation address in the binary.
        :return: Generator of symbol name, relocation address
        """
        if self.format == lief.EXE_FORMATS.ELF:
            rel_enum = self.relocation_enum
            # Iterate imported symbols
            for rel in self._binary.dynamic_relocations:
                if rel_enum(rel.type) == rel_enum.COPY and rel.has_symbol:
                    if rel.symbol.is_variable:
                        yield rel.symbol.name, rel.address
        else:
            raise NotImplementedError(f"Imported symbols relocations not implemented for: {self.format.name}")
