from typing import Generator, Optional
from pathlib import Path

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

    def __init__(self, path: PathLike):
        super(CleLoader, self).__init__(path)
        self.path: Path = Path(path)  #: Binary file path
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self.ld = cle.Loader(path)


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
        In the case of a monolithic firmware, there is a single segment.
        The generator returns a single tuple with the load address and the content.

        :return: Generator of tuples addrs and content
        """
        for obj in self.ld.all_objects:
            print(obj)
            for seg in obj.segments:
                print(seg)
                segdata = self.ld.memory.load(seg.vaddr, seg.memsize)
                assert len(segdata) == seg.memsize
                # TODO perms
                yield LoadableSegment(seg.vaddr, perms=Perm.X | Perm.R | Perm.W, content=segdata, name=f"seg-{obj.binary_basename}")
        # Also return a specific map to put external symbols
        yield LoadableSegment(self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]")
        yield LoadableSegment(self.END_STACK, self.BASE_STACK-self.END_STACK, Perm.R | Perm.W, name="[stack]")

    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary.

        :return: Platform
        """
        return _plfm_mapper[self.ld.main_object.os]
