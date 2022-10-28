from __future__ import annotations

# built-in imports
from pathlib import Path
from typing import Optional, Generator, Tuple
import logging
from collections import namedtuple

# third party
import lief

# local imports
from tritondse.types import PathLike, Addr, Architecture, Platform, ArchMode, Perm, Endian
from tritondse.loader import Loader, LoadableSegment


_arch_mapper = {
    lief.ARCHITECTURES.ARM:   Architecture.ARM32,
    lief.ARCHITECTURES.ARM64: Architecture.AARCH64,
    lief.ARCHITECTURES.X86:   Architecture.X86
}

_plfm_mapper = {
    lief.EXE_FORMATS.ELF: Platform.LINUX,
    lief.EXE_FORMATS.PE: Platform.WINDOWS,
    lief.EXE_FORMATS.MACHO: Platform.MACOS
}


class Program(Loader):
    """
    Representation of a program (loaded in memory). This class is wrapping
    `LIEF <https://lief.quarkslab.com/doc/latest>`_ to represent a program
    and to provide all the features allowing to pseudo-load one regardless
    of its format.
    """

    EXTERN_SYM_BASE = 0x0f001000
    EXTERN_SYM_SIZE = 0x1000

    BASE_STACK = 0xf0000000
    END_STACK  = 0x70000000 # This is inclusive

    def __init__(self, path: PathLike):
        """
        :param path: Program path
        :type path: :py:obj:`tritondse.types.PathLike`
        :raise FileNotFoundError: if the file is not properly recognized by lief
                                  or in the wrong architecture
        """
        super(Program, self).__init__(path)
        self.path: Path = Path(path)  #: Binary file path
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self._binary = lief.parse(str(self.path))
        if self._binary is None:  # lief has not been able to parse it
            raise FileNotFoundError(f"file {path} not recognised by lief")

        self._arch = self._load_arch()
        if self._arch is None:
            raise FileNotFoundError(f"binary {path} architecture unsupported {self._binary.abstract.header.architecture}")

        try:
            self._plfm = _plfm_mapper[self._binary.format]
            # TODO: better refine for Android, iOS etc.
        except KeyError:
            self._plfm = None

        self._funs = {f.name: f for f in self._binary.concrete.functions}

    @property
    def name(self) -> str:
        """ Name of the loader"""
        return f"Program({self.path})"

    def endianess(self) -> Endian:
        # FIXME: Depending on architecture returning good endianess
        return Endian.LITTLE

    @property
    def entry_point(self) -> Addr:
        """
        Program entrypoint address as defined in the binary headers

        :rtype: :py:obj:`tritondse.types.Addr`
        """
        return self._binary.entrypoint


    @property
    def architecture(self) -> Architecture:
        """
        Architecture enum representing program architecture.

        :rtype: Architecture
        """
        return self._arch

    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary. Its solely based on the format
        of the file ELF, PE etc..

        :return: Platform
        """
        return self._plfm

    @property
    def endianness(self) -> lief.ENDIANNESS:
        """
        Endianness of the program as defined in binary headers.

        :rtype: lief.ENDIANNESS
        """
        return self._binary.abstract.header.endianness


    @property
    def format(self) -> lief.EXE_FORMATS:
        """
        Binary format. Supported formats by lief are: ELF, PE, MachO

        :rtype: lief.EXE_FORMATS
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
        LIEF relocation enum associated with the current
        architecture of the binary.

        :return: LIEF relocation enum
        :rtype: Union[lief.ELF.RELOCATION_AARCH64,
                      lief.ELF.RELOCATION_ARM,
                      lief.ELF.RELOCATION_PPC64,
                      lief.ELF.RELOCATION_PPC,
                      lief.ELF.RELOCATION_i386,
                      lief.ELF.RELOCATION_X86_64]
        """
        rel_map = {
            lief.ELF.ARCH.AARCH64: lief.ELF.RELOCATION_AARCH64,
            lief.ELF.ARCH.ARM:     lief.ELF.RELOCATION_ARM,
            lief.ELF.ARCH.PPC64:   lief.ELF.RELOCATION_PPC64,
            lief.ELF.ARCH.PPC:     lief.ELF.RELOCATION_PPC,
            lief.ELF.ARCH.i386:    lief.ELF.RELOCATION_i386,
            lief.ELF.ARCH.x86_64:  lief.ELF.RELOCATION_X86_64
        }
        return rel_map[self._binary.concrete.header.machine_type]


    def _is_glob_dat(self, rel: lief.ELF.Relocation) -> bool:
        """ Get whether the given relocation is of type GLOB_DAT.
        Used locally to find mandatory relocations
        """
        rel_enum = self.relocation_enum
        if hasattr(rel_enum, "GLOB_DAT"):
            return rel_enum(rel.type) == getattr(rel_enum, "GLOB_DAT")
        else:
            return False  # Not GLOB_DAT relocation for this architecture


    def memory_segments(self) -> Generator[LoadableSegment, None, None]:
        """
        Iterate over all memory segments of the program as loaded in memory.

        :return: Generator of tuples addrs and content
        :raise NotImplementedError: if the binary format cannot be loaded
        """
        if self.format == lief.EXE_FORMATS.ELF:
            for i, seg in enumerate(self._binary.concrete.segments):
                if seg.type == lief.ELF.SEGMENT_TYPES.LOAD:
                    content = bytearray(seg.content)
                    if seg.virtual_size != len(seg.content):  # pad with zeros (as it might be .bss)
                        content += bytearray([0]) * (seg.virtual_size - seg.physical_size)
                    yield LoadableSegment(seg.virtual_address, perms=Perm(int(seg.flags)), content=bytes(content), name=f"seg{i}")
        else:
            raise NotImplementedError(f"memory segments not implemented for: {self.format.name}")

        # Also return a specific map to put external symbols
        yield LoadableSegment(self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]")
        yield LoadableSegment(self.END_STACK, self.BASE_STACK-self.END_STACK, Perm.R | Perm.W, name="[stack]")

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported functions by the program. This function
        is a generator of tuples associating the function and its relocation
        address in the binary.

        :return: Generator of tuples function name and relocation address
        """
        if self.format == lief.EXE_FORMATS.ELF:
            try:
                # Iterate functions imported through PLT
                for rel in self._binary.concrete.pltgot_relocations:
                    yield rel.symbol.name, rel.address

                # Iterate functions imported via mandatory relocation (e.g: __libc_start_main)
                for rel in self._binary.dynamic_relocations:
                    if self._is_glob_dat(rel) and rel.has_symbol and not rel.symbol.is_variable:
                        yield rel.symbol.name, rel.address
            except Exception:
                logging.error('Something wrong with the pltgot relocations')

        else:
            raise NotImplementedError(f"Imported functions relocations not implemented for: {self.format.name}")


    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported variable symbols. Yield for each of them the name and
        the relocation address in the binary.

        :return: Generator of tuples with symbol name, relocation address
        """
        if self.format == lief.EXE_FORMATS.ELF:
            rel_enum = self.relocation_enum
            # Iterate imported symbols
            for rel in self._binary.dynamic_relocations:
                if rel.has_symbol:
                #if rel_enum(rel.type) == rel_enum.COPY and rel.has_symbol:
                    if rel.symbol.is_variable:
                        yield rel.symbol.name, rel.address
        else:
            raise NotImplementedError(f"Imported symbols relocations not implemented for: {self.format.name}")


    def find_function_addr(self, name: str) -> Optional[Addr]:
        """
        Search for the function name in fonctions of the binary.

        :param name: Function name
        :type name: str
        :return: Address of function if found
        :rtype: Addr
        """
        f = self._funs.get(name)
        return f.address if f else None


    @property
    def arch_mode(self) -> ArchMode:
        pass  # TODO: Richard
