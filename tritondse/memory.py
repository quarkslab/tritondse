from triton import TritonContext
import bisect
from typing import Optional, Union, Generator
from collections import namedtuple
import struct

from tritondse.types import Perm, Addr, ByteSize, Endian

MemMap = namedtuple('Map', "start size perm name")


class MapOverlapException(Exception):
    pass

class MemoryAccessViolation(Exception):
    """
    Exception triggered when accessing memory with
    the wrong permissions.
    """
    def __init__(self, addr: Addr, access: Perm, map_perm: Perm = None, memory_not_mapped: bool = False, perm_error: bool = False):
        super(MemoryAccessViolation, self).__init__()
        self.address = addr
        self._is_mem_unmapped = memory_not_mapped
        self._is_perm_error = perm_error
        self.access = access
        self.map_perm = map_perm

    def is_permission_error(self):
        return self._is_perm_error

    def is_memory_unmapped_error(self):
        return self._is_mem_unmapped

    def __str__(self):
        if self.is_permission_error():
            return f"MemoryAccessViolation(addr:{self.address:#08x}, access:{str(self.access)} on map:{str(self.map_perm)})"
        else:
            return f"MemoryAccessViolation(addr:{self.address:#08x}({str(self.access)}) unmapped)"

    def __repr__(self):
        return str(self)


STRUCT_MAP = {
    (True, 1): 'B',
    (False, 1): 'b',
    (True, 2): 'H',
    (False, 2): 'h',
    (True, 4): 'I',
    (False, 4): 'i',
    (True, 8): 'Q',
    (False, 8): 'q'
}

ENDIAN_MAP = {
    Endian.LITTLE: "<",
    Endian.BIG: ">"
}


class Memory(object):

    def __init__(self, ctx: TritonContext):
        self.ctx = ctx
        self._linear_map_addr = []  # List of [map_start, map_end, map_start, map_end ...]
        self._linear_map_map = []   # List of [MemMap,    None,    MemMap,    None    ...]
        self._segment_enabled = True
        self._endian = Endian.LITTLE
        self._endian_key = ENDIAN_MAP[self._endian]
        # self._maps = {}  # Addr: -> Map

    def set_endianess(self, en: Endian) -> None:
        """
        Set the endianness of memory accesses. By default
        endianess is little.

        :param en: Endian: Endianess to use.
        :return: None
        """
        self._endian = en
        self._endian_key = ENDIAN_MAP[self._endian]

    @property
    def _ptr_size(self) -> int:
        return self.ctx.getGprSize()

    def disable_segmentation(self) -> None:
        """
        Turn-off segmentation enforcing.
        :return: None
        """
        self._segment_enabled = False

    def enable_segmentation(self) -> None:
        """
        Turn-off segmentation enforcing.
        :return: None
        """
        self._segment_enabled = True

    def get_maps(self) -> Generator[MemMap, None, None]:
        yield from (x for x in self._linear_map_map if x)

    def map(self, start, size, perm: Perm = Perm.R | Perm.W | Perm.X, name="") -> MemMap:
        """
        Map the given address and size in memory with the given permission.

        :raise MapOverlapException: In the case the map overlap an existing mapping
        :param start: address to map
        :param size: size to map
        :param perm: permission
        :param name: name to given to the memory region
        :return: MemMap freshly mapped
        """
        def _map_idx(idx):
            self._linear_map_addr.insert(idx, start + size - 1)  # end address is included
            self._linear_map_addr.insert(idx, start)
            self._linear_map_map.insert(idx, None)
            memmap = MemMap(start, size, perm, name)
            self._linear_map_map.insert(idx, memmap)
            return memmap

        if not self._linear_map_addr:  # Nothing mapped yet
            return _map_idx(0)

        idx = bisect.bisect_left(self._linear_map_addr, start)

        if idx == len(self._linear_map_addr):  # It should be mapped at the end
            return _map_idx(idx)

        addr = self._linear_map_addr[idx]
        if (idx % 2) == 0:  # We are on a start address
            if start < addr and start+size <= addr:  # Can fit before
                return _map_idx(idx)
            else:  # there is an overlap
                raise MapOverlapException(f"0x{start:08x}:{size} overlap with map: 0x{addr:08x} (even)")
        else:  # We are on an end address
            prev = self._linear_map_addr[idx-1]
            raise MapOverlapException(f"0x{start:08x}:{size} overlap with map: 0x{prev:08x} (odd)")

    def unmap(self, addr: Addr) -> None:
        def _unmap_idx(idx):
            self._linear_map_addr.pop(idx) # Pop the start
            self._linear_map_addr.pop(idx) # Pop the end
            self._linear_map_map.pop(idx)  # Pop the object
            self._linear_map_map.pop(idx)  # Pop the None padding

        idx = bisect.bisect_left(self._linear_map_addr, addr)
        try:
            mapaddr = self._linear_map_addr[idx]
            if (idx % 2) == 0:  # We are on a start address (meaning we should be exactly on map start other unmapped)
                if addr == mapaddr:  # We are exactly on the map address
                    _unmap_idx(idx)
                else:
                    raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)
            else:  # We are on an end address
                _unmap_idx(idx-1)
        except IndexError:
            raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)

    def __setitem__(self, key: Addr, value: bytes):
        if isinstance(key, slice):
            raise TypeError("slice unsupported for __setitem__")
        else:
            self.write(key, value)

    def __getitem__(self, item: Union[Addr, slice]):
        if isinstance(item, slice):
            return self.read(item.start, item.stop)
        elif isinstance(item, int):
            return self.read(item, 1)

    def write(self, addr: Addr, data: bytes) -> None:
        if self._segment_enabled:
            map = self._get_map(addr, len(data))
            if map is None:
                raise MemoryAccessViolation(addr, Perm.W, memory_not_mapped=True)
            if Perm.W not in map.perm:
                raise MemoryAccessViolation(addr, Perm.W, map_perm=map.perm, perm_error=True)
        return self.ctx.setConcreteMemoryAreaValue(addr, data)

    def read(self, addr: Addr, size: ByteSize) -> bytes:
        if self._segment_enabled:
            map = self._get_map(addr, size)
            if map is None:
                raise MemoryAccessViolation(addr, Perm.R, memory_not_mapped=True)
            if Perm.R not in map.perm:
                raise MemoryAccessViolation(addr, Perm.R, map_perm=map.perm, perm_error=True)
        return self.ctx.getConcreteMemoryAreaValue(addr, size)

    def _get_map(self, ptr: Addr, size: ByteSize) -> Optional[MemMap]:
        """
        Internal function returning the MemMap object associated
        with any address. It returns None if part of the memory
        range falls out of a memory mapping.
        Complexity is O(log(n))

        :param ptr: address in memory
        :param size: size of the memory
        :return: True if mapped
        """
        idx = bisect.bisect_left(self._linear_map_addr, ptr)
        try:
            addr = self._linear_map_addr[idx]
            if (idx % 2) == 0:  # We are on a start address (meaning we should be exactly on map start other unmapped)
                end = self._linear_map_addr[idx+1]
                return self._linear_map_map[idx] if (ptr == addr and ptr+size <= end+1) else None
            else:  # We are on an end address
                start = self._linear_map_addr[idx-1]
                return self._linear_map_map[idx-1] if (start <= addr and ptr+size <= addr+1) else None  # fit into the map
        except IndexError:
            return None  # Either raised when linear_map is empty or the address is beyond everything that is mapped

    def get_map(self, addr: Addr, size: ByteSize = 1) -> Optional[MemMap]:
        """
        Find the MemMap associated with the given address and returns
        it if any.

        :param addr: Address of the map (or any map inside)
        :param size: size of bytes for which we want the map
        :return: MemMap if found
        """
        return self._get_map(addr, size)

    def is_mapped(self, ptr: Addr, size: ByteSize) -> bool:
        """
        The function checks whether the memory is mapped or not.
        The implementation return False if the memory chunk overlap
        on two memory regions.
        Complexity is O(log(n))

        :param ptr: address in memory
        :param size: size of the memory
        :return: True if mapped
        """
        return self._get_map(ptr, size) is not None

    def has_ever_been_written(self, ptr: Addr, size: ByteSize) -> bool:
        """
        Returns whether the given range of addresses has previously
        been written or not. (Do not take in account the memory mapping).

        :param ptr: The pointer to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :param size: Size of the memory range to check
        :return: True if all addresses have been defined
        """
        return self.ctx.isConcreteMemoryValueDefined(ptr, size)

    def read_uint(self, addr: Addr, size: ByteSize = 4):
        """
        Read in the process memory a **little-endian** integer of the ``size`` at ``addr``.

        :param addr: Address at which to read data
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: Number of bytes to read
        :type size: Union[str, :py:obj:`tritondse.types.ByteSize`]
        :return: Integer value read
        """
        data = self.read(addr, size)
        return struct.unpack(self._endian_key+STRUCT_MAP[(True, size)], data)[0]

    def read_sint(self, addr: Addr, size: ByteSize = 4):
        """
        Read in the process memory a **little-endian** integer of the ``size`` at ``addr``.

        :param addr: Address at which to read data
        :type addr: :py:obj:`tritondse.types.Addr`
        :param size: Number of bytes to read
        :type size: Union[str, :py:obj:`tritondse.types.ByteSize`]
        :return: Integer value read
        """
        data = self.read(addr, size)
        return struct.unpack(self._endian_key+STRUCT_MAP[(False, size)], data)[0]

    def read_ptr(self, addr: Addr) -> int:
        """
        Read in the process memory a little-endian integer of size :py:attr:`tritondse.ProcessState.ptr_size`

        :param addr: Address at which to read data
        :type addr: :py:obj:`tritondse.types.Addr`
        :return: Integer value read
        """
        return self.read_uint(addr, self._ptr_size)

    def read_char(self, addr: Addr):
        return self.read_sint(addr, 1)

    def read_uchar(self, addr: Addr):
        return self.read_uint(addr, 1)

    def read_int(self, addr: Addr):
        return self.read_sint(addr, 4)

    def read_word(self, addr: Addr):
        return self.read_uint(addr, 2)

    def read_dword(self, addr: Addr):
        return self.read_uint(addr, 4)

    def read_qword(self, addr: Addr):
        return self.read_uint(addr, 8)

    def read_long(self, addr: Addr):
        return self.read_sint(addr, 4)

    def read_ulong(self, addr: Addr):
        return self.read_uint(addr, 4)

    def read_long_long(self, addr: Addr):
        return self.read_sint(addr, 8)

    def read_ulong_long(self, addr: Addr):
        return self.read_uint(addr, 8)

    def read_string(self, addr: Addr) -> str:
        """ Read a string in process memory at the given address

        .. warning:: The memory read is unbounded. Thus the memory is iterated up until
                     finding a 0x0.

        :returns: the string read in memory
        :rtype: str
        """
        s = ""
        index = 0
        while True:
            val = self.read_uint(addr+index, 1)
            if not val:
                return s
            s += chr(val)
            index += 1

    def write_int(self, addr: Addr, value: int, size: ByteSize = 4):
        """
        Write in the process memory the given integer value of the given size at
        a specific address.

        :param addr: Address at which to read data
        :type addr: :py:obj:`tritondse.types.Addr`
        :param value: data to write represented as an integer
        :type value: int
        :param size: Number of bytes to read
        :type size: :py:obj:`tritondse.types.ByteSize`
        """
        self.write(addr, struct.pack(self._endian_key+STRUCT_MAP[(value >= 0, size)], value))


    def write_ptr(self, addr: Addr, value: int) -> None:
        """
        Similar to :py:meth:`write_int` but the size is automatically adjusted
        to be ``ptr_size``.

        :param addr: address where to write data
        :type addr: :py:obj:`tritondse.types.Addr`
        :param value: pointer value to write
        :type value: int
        """
        self.write_int(addr, value, self._ptr_size)

    def write_char(self, addr: Addr, value: int):
        self.write_int(addr, value, 1)

    def write_word(self, addr: Addr, value: int):
        self.write_int(addr, value, 2)

    def write_dword(self, addr: Addr, value: int):
        self.write_int(addr, value, 4)

    def write_qword(self, addr: Addr, value: int):
        self.write_int(addr, value, 8)

    def write_long(self, addr: Addr, value: int):
        return self.write_int(addr, value, 4)

    def write_long_long(self, addr: Addr, value: int):
        return self.write_int(addr, value, 8)
