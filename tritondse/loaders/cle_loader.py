from typing import Generator, Optional, Tuple
from pathlib import Path
import logging

from tritondse.loaders import Loader, LoadableSegment
from tritondse.types import Addr, Architecture, PathLike, Platform, Perm

from tritondse.routines import SUPPORTED_ROUTINES

import cle

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
    EXTERN_SYM_BASE = 0x0f001000
    EXTERN_SYM_SIZE = 0x1000

    BASE_STACK = 0xf0000000
    END_STACK = 0x70000000  # This is inclusive

    def __init__(self, path: PathLike, ld_path: Optional[PathLike] = None):
        super(CleLoader, self).__init__(path)
        self.path: Path = Path(path)  #: Binary file path
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self._disable_vex_loggers()  # disable logs of pyvex

        self.ld_path = ld_path if ld_path is not None else ()
        self.ld = cle.Loader(str(path), ld_path=self.ld_path)
        self.longjmp_addr_cache = None

    def _disable_vex_loggers(self):
        for name, logger in logging.root.manager.loggerDict.items():
            if "pyvex" in name:
                logger.propagate = False

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

    @property
    def longjmp_addr(self) -> Addr:
        # TODO find a better way to do this
        # There is no generic way but since the plt layout is known we could do 
        # https://github.com/lief-project/LIEF/issues/762
        if not self.longjmp_addr_cache:
            import subprocess

            try:
                proc1 = subprocess.Popen(['objdump', '-D', f'{self.path}'], stdout=subprocess.PIPE)
                proc2 = subprocess.Popen(['grep', '<longjmp@plt>:'], stdin=proc1.stdout,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                proc1.stdout.close() # Allow proc1 to receive a SIGPIPE if proc2 exits.
                out, err = proc2.communicate()
                self.longjmp_addr_cache = int(out.split()[0], 16)
            except:
                self.longjmp_addr_cache = 0

        return self.longjmp_addr_cache

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
                if seg.__class__.__name__ != "ExternSegment":
                    # The format string in CLE is broken if the filesize is 0. This is a workaround.
                    logging.debug(f"Loading segment {seg} - perms:{perms}")
                yield LoadableSegment(seg.vaddr, perms, content=segdata, name=f"seg-{obj.binary_basename}")
        # Also return a specific map to put external symbols
        yield LoadableSegment(self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]")
        yield LoadableSegment(self.END_STACK, self.BASE_STACK-self.END_STACK+1, Perm.R | Perm.W, name="[stack]")

        # FIXME. Temporary solution to prevent crashes on access to the TLB e.g fs:28
        yield LoadableSegment(0, 0x2000, Perm.R | Perm.W, name="[fs]")

    # FIXME. Temporary solution to prevent crashes on access to the TLB e.g fs:28
    @property
    def cpustate(self):
        # NOTE: in Triton, the segment selector is used as the segment base and not as a selector into GDT.
        # i.e directly store the segment base into fs
        return {"fs": 0x1000}

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
        # For example if a library calls a libc function, we probably need to patch the library's GOT
        for obj in self.ld.all_objects:
            for fun in obj.imports:
                rtn_name = f"rtn_{fun}"
                if fun in SUPPORTED_ROUTINES:
                    reloc = obj.imports[fun]
                    got_entry_addr = reloc.relative_addr + obj.mapped_base
                    yield fun, got_entry_addr

        # Handle indirect functions.
        # Currently we only support indirect functions if there exists a stub for them in `routines.py`
        # Otherwise the program will crash because CLE doesn't perform the relocation for indirect functions.
        
        # We could perform the relocation ourself by writing to the got slot but we need a way to figure out 
        # the correct fptr to use.
        # In other words we should execute `resolver_fun` or parse it in some way to get the correct function ptr
        # to write to got_slot (write with self.ld.memory.pack_word(got_slot, func_ptr))
        for obj in self.ld.all_objects:
            for (resolver_func, got_rva) in obj.irelatives:
                got_slot = got_rva + obj.mapped_base
                sym = self.ld.find_symbol(resolver_func)
                if sym is None:
                    continue
                fun = sym.name
                if fun in SUPPORTED_ROUTINES:
                    yield fun, got_slot

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

    def find_function_addr(self, name: str) -> Optional[Addr]:
        """
        Search for the function name in fonctions of the binary.

        :param name: Function name
        :type name: str
        :return: Address of function if found
        :rtype: Addr
        """
        res = [x for x in self.ld.find_all_symbols(name) if x.is_function]
        return res[0].rebased_addr if res else None  # if multiple elements return the first
