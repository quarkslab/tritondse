from typing import Generator, Optional, Tuple
from pathlib import Path
import logging

from tritondse.loader import Loader, LoadableSegment
from tritondse.types import Addr, Architecture, PathLike, Platform, Perm
from triton import Instruction

import cle
import archinfo

_arch_mapper = {
    "ARMEL":   Architecture.ARM32,
    "AARCH64": Architecture.AARCH64,
    "AMD64":   Architecture.X86_64,
    "X86":   Architecture.X86,
}

_plfm_mapper = {
    "UNIX - System V": Platform.LINUX,
    "windows": Platform.WINDOWS,
    "macos": Platform.MACOS
}

class CleLoader(Loader):
    EXTERN_SYM_BASE = 0x01001000
    EXTERN_SYM_SIZE = 0x1000

    BASE_STACK = 0xf0000000
    END_STACK  = 0x70000000 # This is inclusive

    def __init__(self, path: PathLike, fcts_to_emulate={}):
        super(CleLoader, self).__init__(path)
        self.path: Path = Path(path)  #: Binary file path
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self.ld = cle.Loader(path)
        self.fcts_to_emulate = fcts_to_emulate


    @property
    def name(self) -> str:
        """ Name of the loader"""
        return f"CleLoader({self.path})"

    @property
    def architecture(self) -> Architecture:
        """
        Architecture enum representing program architecture.

        :rtype: Architecture
        """
        return _arch_mapper[self.ld.main_object.arch.name]


    @property
    def entry_point(self) -> Addr:
        """
        Program entrypoint address as defined in the binary headers

        :rtype: :py:obj:`tritondse.types.Addr`
        """
        return self.ld.main_object.entry


    def memory_segments(self) -> Generator[LoadableSegment, None, None]:
        """
        :return: Generator of tuples addrs and content
        """
        for obj in self.ld.all_objects:
            logging.debug(obj)
            for seg in obj.segments:
                segdata = self.ld.memory.load(seg.vaddr, seg.memsize)
                assert len(segdata) == seg.memsize
                perms = (Perm.R if seg.is_readable else 0) | (Perm.W if seg.is_writable else 0) | (Perm.X if seg.is_executable else 0) 
                logging.debug(f"Loading segment {seg} - perms:{perms}")
                yield LoadableSegment(seg.vaddr, perms, content=segdata, name=f"seg-{obj.binary_basename}")
        # Also return a specific map to put external symbols
        yield LoadableSegment(self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]")
        yield LoadableSegment(self.END_STACK, self.BASE_STACK-self.END_STACK+1, Perm.R | Perm.W, name="[stack]")

        # FIXME. Temporary solution to prevent crashes on access to the TLB e.g fs:28
        yield LoadableSegment(0, 0x1000, Perm.R | Perm.W, name="[fs]")

    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary.

        :return: Platform
        """
        return _plfm_mapper[self.ld.main_object.os]

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported functions by the program. This function
        is a generator of tuples associating the function and its relocation
        address in the binary.

        :return: Generator of tuples function name and relocation address
        """
        # TODO I think there's a problem here. We only deal with imports from the main binary
        for obj in self.ld.all_objects:
            if obj.binary_basename in self.fcts_to_emulate:
                for f in self.fcts_to_emulate[obj.binary_basename]:
                    reloc = self.ld.main_object.imports[f]
                    got_entry_addr = reloc.relative_addr + self.ld.main_object.mapped_base
                    yield f, got_entry_addr


    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported variable symbols. Yield for each of them the name and
        the relocation address in the binary.

        :return: Generator of tuples with symbol name, relocation address
        """
        # TODO I think there's a problem here. We only deal with imports from the main binary
        for s in self.ld.main_object.symbols:
            if s.resolved and s._type == cle.SymbolType.TYPE_OBJECT:
                logging.debug(f"CleLoader: hooking symbol {s.name} @ {s.relative_addr:#x} {s.resolved} {s.resolvedby} {s._type}")
                s_addr = s.relative_addr + self.ld.main_object.mapped_base
                yield s.name, s_addr
