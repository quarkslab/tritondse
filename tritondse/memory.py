from triton import TritonContext
import bisect
from typing import Optional, Union, Generator, List
from collections import namedtuple
import struct
from contextlib import contextmanager

from tritondse.types import Perm, Addr, ByteSize, Endian

MemMap = namedtuple('Map', "start size perm name")


class MapOverlapException(Exception):
    """
    Exception raised when trying to map a memory area where some of
    the addresses overlap with an already mapped area.
    """
    pass


class MemoryAccessViolation(Exception):
    """
    Exception triggered when accessing memory with
    the wrong permissions.
    """
    def __init__(self, addr: Addr, access: Perm, map_perm: Perm = None, memory_not_mapped: bool = False, perm_error: bool = False):
        """
        :param addr: address where the violation occured
        :param access: type of access performed
        :param map_perm: permission of the memory page of `address`
        :param memory_not_mapped: whether the address was mapped or not
        :param perm_error: whether it is a permission error
        """
        super(MemoryAccessViolation, self).__init__()
        self.address: Addr = addr
        """ address where the violation occurred"""
        self._is_mem_unmapped = memory_not_mapped
        self._is_perm_error = perm_error
        self.access: Perm = access
        """Access type that was performed"""
        self.map_perm: Optional[Perm] = map_perm
        """Permissions of the memory map associated to the address"""

    def is_permission_error(self) -> bool:
        """True if the exception was caused by a permission issue"""
        return self._is_perm_error

    def is_memory_unmapped_error(self) -> bool:
        """
        Return true if the exception was raised due to an access
        to an area not mapped
        """
        return self._is_mem_unmapped

    def __str__(self) -> str:
        if self.is_permission_error():
            return f"(addr:{self.address:#08x}, access:{str(self.access)} on map:{str(self.map_perm)})"
        else:
            return f"({str(self.access)}: {self.address:#08x} unmapped)"

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
    """
    Memory representation of the current :py:class:`ProcessState` object.
    It wraps all interaction with Triton's memory context to provide high-level
    function. It adds a segmentation and memory permission model at the top
    of Triton. It also overrides __getitem__ and the slice mechanism to be able
    read and write concrete memory values in a Pythonic manner.
    """

    def __init__(self, ctx: TritonContext, endianness: Endian = Endian.LITTLE):
        """
        :param ctx: TritonContext to interface with
        """
        self.ctx: TritonContext = ctx
        """Underlying Triton context"""
        self._linear_map_addr = []  # List of [map_start, map_end, map_start, map_end ...]
        self._linear_map_map = []   # List of [MemMap,    None,    MemMap,    None    ...]
        self._segment_enabled = True
        self._endian = endianness
        self._endian_key = ENDIAN_MAP[self._endian]
        self._mem_cbs_enabled = True
        # self._maps = {}  # Addr: -> Map

    def set_endianess(self, en: Endian) -> None:
        """
        Set the endianness of memory accesses. By default,
        endianess is little.

        :param en: Endian: Endianess to use.
        :return: None
        """
        self._endian = en
        self._endian_key = ENDIAN_MAP[self._endian]

    @property
    def _ptr_size(self) -> int:
        return self.ctx.getGprSize()

    @property
    def segmentation_enabled(self) -> bool:
        """
        returns whether segmentation enforcing is enabled

        :return: True if segmentation is enabled
        """
        return self._segment_enabled

    def disable_segmentation(self) -> None:
        """
        Turn-off segmentation enforcing.
        """
        self._segment_enabled = False

    def enable_segmentation(self) -> None:
        """
        Turn-off segmentation enforcing.
        """
        self._segment_enabled = True

    def set_segmentation(self, enabled: bool) -> None:
        """
        Set the segmentation enforcing with the given boolean.
        """
        self._segment_enabled = enabled

    @contextmanager
    def without_segmentation(self, disable_callbacks=False) -> Generator['Memory', None, None]:
        """
        Context manager enabling manipulating temporarily the memory
        without considering the memory permissions.
        E.g: It enables writing data in a memory mapped in RX
        :param disable_callbacks: Whether to disable memory callbacks that could have been set
        :return:
        """
        self.disable_segmentation()
        cbs = self._mem_cbs_enabled
        self._mem_cbs_enabled = not disable_callbacks
        yield self
        self._mem_cbs_enabled = cbs
        self.enable_segmentation()

    def callbacks_enabled(self) -> bool:
        """
        Return whether memory callbacks are enabled.

        :return: True if callbacks are enabled
        """
        return self._mem_cbs_enabled

    def get_maps(self) -> Generator[MemMap, None, None]:
        """
        Iterate all the memory maps defined, including all memory
        areas allocated on the heap.

        :return: generator of all :py:class:`MemMap` objects
        """
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
        """
        Unmap the :py:class:`MemMap` object mapped at the address.
        The address can be within the map and not requires pointing
        at the head.

        :param addr: address to unmap
        :return: None
        """
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

    def mprotect(self, addr: Addr, perm: Perm) -> None:
        """
        Update the map at the given address with permissions provided in argument.

        :param addr: address of the map of which to change permission
        :param perm: permission to assign
        :return: None
        """
        idx = bisect.bisect_left(self._linear_map_addr, addr)
        try:
            if (idx % 2) == 0:  # We are on a start address (meaning we should be exactly on map start other unmapped)
                map = self._linear_map_map[idx]
                self._linear_map_map[idx] = MemMap(map.start, map.size, perm, map.name)  # replace map with new perms
            else:  # We are on an end address
                map = self._linear_map_map[idx-1]
                self._linear_map_map[idx-1] = MemMap(map.start, map.size, perm, map.name)  # replace map with new perms
        except IndexError:
            raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)

    def __setitem__(self, key: Addr, value: bytes) -> None:
        """
        Assign the given value at the address given by the key.
        The value must be bytes but can be multiple bytes.
        Warning: You cannot use the slice API on this function.

        :param key: address to write to
        :param value: content to write
        :raise MemoryAccessViolation: in case of invalid access
        """
        if isinstance(key, slice):
            raise TypeError("slice unsupported for __setitem__")
        else:
            self.write(key, value)

    def __getitem__(self, item: Union[Addr, slice]) -> bytes:
        """
        Read the memory at the given address. If the key
        is an integer reads a single byte. If the key is
        a slice: read addr+size bytes in memory.

        :param item: address, or address:size to read
        :return: memory content
        :raise MemoryAccessViolation: if the access is invalid
        """
        if isinstance(item, slice):
            return self.read(item.start, item.stop)
        elif isinstance(item, int):
            return self.read(item, 1)

    def write(self, addr: Addr, data: bytes) -> None:
        """
        Write the given `data` bytes at `addr` address.

        :param addr: address where to write
        :param data: data to write
        :return: None
        """
        if self._segment_enabled:
            map = self._get_map(addr, len(data))
            if map is None:
                raise MemoryAccessViolation(addr, Perm.W, memory_not_mapped=True)
            if Perm.W not in map.perm:
                raise MemoryAccessViolation(addr, Perm.W, map_perm=map.perm, perm_error=True)
        return self.ctx.setConcreteMemoryAreaValue(addr, data)

    def read(self, addr: Addr, size: ByteSize) -> bytes:
        """
        Read `size` bytes at `addr` address.

        :param addr: address to read
        :param size: size of content to read
        :return: bytes read
        """
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

    def find_map(self, name: str) -> Optional[List[MemMap]]:
        """
        Find a map given its name.

        :param name: Map name
        :return: MemMap if found
        """
        l = []
        for map in (x for x in self._linear_map_map if x):
            if map.name == name:
                l.append(map)
        return l

    def map_from_name(self, name: str) -> MemMap:
        """
        Return a map from its name. This function assumes
        the map is present.

        :raise AssertionError: If the map is not found
        :param name: Map name
        :return: MemMap
        """
        for map in (x for x in self._linear_map_map if x):
            if map.name == name:
                return map
        assert False

    def is_mapped(self, ptr: Addr, size: ByteSize = 1) -> bool:
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
        :raise struct.error: If value can't fit in `size`
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
        :raise struct.error: If value can't fit in `size`
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

    def read_char(self, addr: Addr) -> int:
        """
        Read a char in memory (1-byte) following endianess.

        :param addr: address to read
        :return: char value as int
        """
        return self.read_sint(addr, 1)

    def read_uchar(self, addr: Addr) -> int:
        """
        Read an unsigned char in memory (1-byte) following endianness.

        :param addr: address to read
        :return: unsigned char value as int
        """
        return self.read_uint(addr, 1)

    def read_int(self, addr: Addr) -> int:
        """
        Read a signed integer in memory (4-byte) following endianness.

        :param addr: address to read
        :return: signed integer value as int
        """
        return self.read_sint(addr, 4)

    def read_word(self, addr: Addr) -> int:
        """
        Read signed word in memory (2-byte) following endianness.

        :param addr: address to read
        :return: signed word value as int
        """
        return self.read_uint(addr, 2)

    def read_dword(self, addr: Addr) -> int:
        """
        Read signed double word in memory (4-byte) following endianness.

        :param addr: address to read
        :return: dword value as int
        """
        return self.read_uint(addr, 4)

    def read_qword(self, addr: Addr) -> int:
        """
        Read signed qword in memory (8-byte) following endianness.

        :param addr: address to read
        :return: qword value as int
        """
        return self.read_uint(addr, 8)

    def read_long(self, addr: Addr) -> int:
        """
        Read 'C style' long in memory (4-byte) following endianness.

        :param addr: address to read
        :return: value as int
        """
        return self.read_sint(addr, 4)

    def read_ulong(self, addr: Addr) -> int:
        """
        Read unsigned long in memory (4-byte) following endianness.

        :param addr: address to read
        :return: unsigned long value as int
        """
        return self.read_uint(addr, 4)

    def read_long_long(self, addr: Addr) -> int:
        """
        Read long long in memory (8-byte) following endianness.

        :param addr: address to read
        :return: long long value as int
        """
        return self.read_sint(addr, 8)

    def read_ulong_long(self, addr: Addr) -> int:
        """
        Read unsigned long long in memory (8-byte) following endianness.

        :param addr: address to read
        :return: unsigned long long value as int
        """
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
        :param value: data to write represented as an integer
        :param size: Number of bytes to read
        :raise struct.error: If integer value cannot fit in `size`
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
        :raise struct.error: If integer value cannot fit in a pointer size
        """
        self.write_int(addr, value, self._ptr_size)

    def write_char(self, addr: Addr, value: int) -> None:
        """
        Write the integer value as a single byte in memory.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a byte (>255)
        """
        self.write_int(addr, value, 1)

    def write_word(self, addr: Addr, value: int) -> None:
        """
        Write the word (2-byte) in memory following endianess.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a word
        """
        self.write_int(addr, value, 2)

    def write_dword(self, addr: Addr, value: int) -> None:
        """
        Write the word (4-byte) in memory following endianess.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a dword
        """
        self.write_int(addr, value, 4)

    def write_qword(self, addr: Addr, value: int) -> None:
        """
        Write the qword (8-byte) in memory following endianess.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a qword
        """
        self.write_int(addr, value, 8)

    def write_long(self, addr: Addr, value: int) -> None:
        """
        Write a "C style" long (4-byte) in memory following endianess.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a long
        """
        return self.write_int(addr, value, 4)

    def write_long_long(self, addr: Addr, value: int) -> None:
        """
        Write the "C style" long long (8-byte) in memory following endianess.

        :param addr: address to write
        :param value: integer value
        :raise struct.error: If integer value do not fit in a long long
        """
        return self.write_int(addr, value, 8)
