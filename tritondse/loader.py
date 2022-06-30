from __future__ import annotations

# built-in imports
from pathlib import Path
from typing import Optional, Generator, Tuple
import logging

# third party
import lief

# local imports
from tritondse.types import PathLike, Addr, Architecture, Platform
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
    def platform(self) -> Optional[Platform]:
        """
        Platform of the binary. Its solely based on the format
        of the file ELF, PE etc..

        :return: Platform
        """
        raise NotImplementedError()

    @property
    def endianness(self) -> lief.ENDIANNESS:
        """
        Endianness of the program as defined in binary headers.

        :rtype: lief.ENDIANNESS
        """
        raise NotImplementedError()


    @property
    def format(self) -> lief.EXE_FORMATS:
        """
        Binary format. Supported formats by lief are: ELF, PE, MachO

        :rtype: lief.EXE_FORMATS
        """
        raise NotImplementedError()


    def _load_arch(self) -> Optional[Architecture]:
        """
        Load architecture as an Architecture object.

        :return: Architecture or None if unsupported
        """
        raise NotImplementedError()


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
        raise NotImplementedError()


    def _is_glob_dat(self, rel: lief.ELF.Relocation) -> bool:
        """ Get whether the given relocation is of type GLOB_DAT.
        Used locally to find mandatory relocations
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
    def additional_options(self):
        """
        Provide additional options. Should make sure that the `from_loader`
        function in process_state.py know how to interpret them.
        """
        return {}


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
        self.set_thumb = set_thumb
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
    def additional_options(self):
        """
        Provide additional options. Should make sure that the `from_loader`
        function in process_state.py know how to interpret them.
        """
        if self.set_thumb: return {"set_thumb" : True}
        else: return {}


    @property
    def cpustate(self) -> Dict[str, int]:
        """
        Provide the initial cpu state in the forma of a dictionary of
        {"register_name" : register_value}
        """
        return self._cpustate
