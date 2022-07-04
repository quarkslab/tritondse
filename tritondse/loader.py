from __future__ import annotations

# built-in imports
from pathlib import Path
from typing import Optional, Generator, Tuple
import logging

# third party
import lief

# local imports
from tritondse.types import PathLike, Addr, Architecture, Platform, ArchMode
from tritondse.arch  import ARCHS


class Loader(object):
    """
    This class describes how to load the target program in memory.
    """

    def __init__(self):
        raise NotImplementedError()


    @property
    def entry_point(self) -> Addr:
        """
        Program entrypoint address as defined in the binary headers

        :rtype: :py:obj:`tritondse.types.Addr`
        """
        raise NotImplementedError()


    @property
    def architecture(self) -> Architecture:
        """
        Architecture enum representing program architecture.

        :rtype: Architecture
        """
        raise NotImplementedError()


    @property
    def arch_mode(self) -> ArchMode:
        """
        ArchMode enum representing the starting mode (e.g Thumb for ARM).

        :rtype: ArchMode
        """
        return None


    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary.

        :return: Platform
        """
        raise NotImplementedError()


    def memory_segments(self) -> Generator[Tuple[Addr, bytes], None, None]:
        """
        Iterate over all memory segments of the program as loaded in memory.

        :return: Generator of tuples addrs and content
        :raise NotImplementedError: if the binary format cannot be loaded
        """
        raise NotImplementedError()


    @property
    def cpustate(self) -> Dict[str, int]:
        """
        Provide the initial cpu state in the forma of a dictionary of
        {"register_name" : register_value}
        """
        return {}


    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported functions by the program. This function
        is a generator of tuples associating the function and its relocation
        address in the binary.

        :return: Generator of tuples function name and relocation address
        """
        return
        yield


    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported variable symbols. Yield for each of them the name and
        the relocation address in the binary.

        :return: Generator of tuples with symbol name, relocation address
        """
        return
        yield


    def find_function_addr(self, name: str) -> Optional[Addr]:
        """
        Search for the function name in fonctions of the binary.

        :param name: Function name
        :type name: str
        :return: Address of function if found
        :rtype: Addr
        """
        raise NotImplementedError()


class MonolithicLoader(Loader):
    def __init__(self, path, architecture, load_address, cpustate = {}, vmmap = None,\
            set_thumb = False, platform = None):

        self.path: Path = Path(path)  #: Binary file path
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self.bin_path = path
        self.load_address = load_address
        self._architecture = architecture
        self._platform = platform if platform else None
        self._cpustate = cpustate
        self.vmmap = vmmap if vmmap else None
        self._arch_mode = ArchMode.THUMB if set_thumb else None
        if  self._platform and (self._architecture, self._platform) in ARCHS:
            self._archinfo = ARCHS[(self._architecture, self._platform)]
        elif self._architecture in ARCHS:
            self._archinfo = ARCHS[self._architecture]
        else: 
            logging.error("Unknown architecture")
            assert False

    @property
    def architecture(self) -> Architecture:
        """
        Architecture enum representing program architecture.

        :rtype: Architecture
        """
        return self._architecture


    @property
    def arch_mode(self) -> ArchMode:
        """
        ArchMode enum representing the starting mode (e.g Thumb for ARM).

        :rtype: ArchMode
        """
        return self._arch_mode


    @property
    def entry_point(self) -> Addr:
        """
        Program entrypoint address as defined in the binary headers

        :rtype: :py:obj:`tritondse.types.Addr`
        """
        return self.cpustate[self._archinfo.pc_reg]


    def memory_segments(self) -> Generator[Tuple[Addr, bytes], None, None]:
        """
        In the case of a monolithic firmware, there is a single segment.
        The generator returns a single tuple with the load address and the content.

        :return: Generator of tuples addrs and content
        """
        with open(self.bin_path, "rb") as fd: 
            data = fd.read()
        yield self.load_address, data

        if self.vmmap:
            for (addr, buffer) in self.vmmap.items():
                yield addr, buffer


    @property
    def cpustate(self) -> Dict[str, int]:
        """
        Provide the initial cpu state in the format of a dictionary of
        {"register_name" : register_value}
        """
        return self._cpustate
