from __future__ import annotations

# built-in imports
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, List
from dataclasses import dataclass

# local imports
from tritondse.types import Addr, Architecture, Platform, ArchMode, Perm, Endian
from tritondse.arch import ARCHS
import tritondse.logging

logger = tritondse.logging.get()


@dataclass
class LoadableSegment:
    """ Represent a Segment to load in memory.
    It can either provide a content and will thus be
    initialized in a context or virtual.
    """
    address: int
    """ Virtual address where to load the segment """
    size: int = 0
    """ Size of the segment. If content is present use len(content)"""
    perms: Perm = Perm.R|Perm.W|Perm.X
    """ Permissions to assign the segment """
    content: Optional[bytes] = None
    """ Content of the segment """
    name: str = ""
    """ Name to give to the segment """


class Loader(object):
    """
    This class describes how to load the target program in memory.
    """
    def __init__(self, path: str):
        self.bin_path = Path(path)

    @property
    def name(self) -> str:
        """
        Name of the loader and target being loaded.

        :return: str of the loader name
        """
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
    def arch_mode(self) -> Optional[ArchMode]:
        """
        ArchMode enum representing the starting mode (e.g Thumb for ARM).
        if None, the default mode of the architecture will be used.

        :rtype: Optional[ArchMode]
        """
        return None

    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary.

        :return: Platform
        """
        return None

    @property
    def endianness(self) -> Endian:
        """
        Endianess of the loaded program

        :return: Endianness
        """
        raise NotImplementedError()

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:
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
        yield from ()

    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        """
        Iterate over all imported variable symbols. Yield for each of them the name and
        the relocation address in the binary.

        :return: Generator of tuples with symbol name, relocation address
        """
        yield from ()

    def find_function_addr(self, name: str) -> Optional[Addr]:
        """
        Search for the function name in fonctions of the binary.

        :param name: Function name
        :type name: str
        :return: Address of function if found
        :rtype: Addr
        """
        return None


class MonolithicLoader(Loader):
    """
    Monolithic loader. It helps loading raw firmware at a given address
    in DSE memory space, with the various attributes like architecture etc.
    """

    def __init__(self,
                 architecture: Architecture,
                 cpustate: Dict[str, int] = None,
                 maps: List[LoadableSegment] = None,
                 set_thumb: bool = False,
                 platform: Platform = None,
                 endianess: Endian = Endian.LITTLE):
        super(MonolithicLoader, self).__init__("")

        self._architecture = architecture
        self._platform = platform if platform else None
        self._cpustate = cpustate if cpustate else {}
        self.maps = maps
        self._arch_mode = ArchMode.THUMB if set_thumb else None
        self._endian = endianess
        if self._platform and (self._architecture, self._platform) in ARCHS:
            self._archinfo = ARCHS[(self._architecture, self._platform)]
        elif self._architecture in ARCHS:
            self._archinfo = ARCHS[self._architecture]
        else: 
            logger.error("Unknown architecture")
            assert False

    @property
    def name(self) -> str:
        """ Name of the loader"""
        return f"Monolithic({self.bin_path})"

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

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:
        """
        In the case of a monolithic firmware, there is a single segment.
        The generator returns a single tuple with the load address and the content.

        :return: Generator of tuples addrs and content
        """
        yield from self.maps

    @property
    def cpustate(self) -> Dict[str, int]:
        """
        Provide the initial cpu state in the format of a dictionary of
        {"register_name" : register_value}
        """
        return self._cpustate

    @property
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary.

        :return: Platform
        """
        return self._platform

    @property
    def endianness(self) -> Endian:
        """
        Endianess of the monolithic loader.
        (default is LITTLE)
        """
        return self._endian
