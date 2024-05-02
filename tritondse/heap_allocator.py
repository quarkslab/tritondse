# local imports
from tritondse.types import Addr, ByteSize, Perm
from tritondse.memory import Memory
from tritondse.exception import AllocatorException
import tritondse.logging

logger = tritondse.logging.get('heapallocator')


class HeapAllocator(object):
    """
    Custom tiny heap allocator. Used by built-ins routines like malloc/free.
    This allocation manager also provides an API enabling checking whether
    a pointer is allocated freed etc.

    .. warning:: This allocator is very simple and does not perform any
                 coalescing of freed memory areas. Thus, it may not correctly
                 model the behavior of libc allocator.
    """

    def __init__(self, start: Addr, end: Addr, memory: Memory):
        """
        Class constructor. Takes heap bounds as parameter.

        :param start: Where the heap area can start
        :type start: :py:obj:`tritondse.types.Addr`
        :param end: Where the heap area must end
        :type start: :py:obj:`tritondse.types.Addr`
        :param memory: Memory: Memory object on which to perform allocations
        """
        # Range of the memory mapping
        self.start: Addr = start
        #: Starting address of the heap

        self.end: Addr = end
        #: Ending address of the heap

        self._curr_offset: Addr = self.start  #: Heap current offset address
        self._memory = memory

        # Pools memory
        self.alloc_pool = dict()    # {ptr: MemMap}
        self.free_pool = dict()     # {size: set(MemMap ...)}

        # TODO: For a to-the-moon allocator, we could merge freed chunks. Like 4 chunks of 1 byte into one chunk of 4 bytes.
        # TODO: For a to-the-moon allocator, we could split a big chunk into two chunks when asking an allocation.

    def alloc(self, size: ByteSize) -> Addr:
        """
        Performs an allocation of the given byte size.

        :param size: Byte size to allocate
        :type size: :py:obj:`tritondse.types.ByteSize`
        :raise AllocatorException: if not memory is available
        :return: The pointer address allocated
        :rtype: :py:obj:`tritondse.types.Addr`
        """

        if size <= 0:
            logger.error(f"Heap: invalid allocation size {size}")
            return 0

        ptr = None
        for sz in sorted(x for x in self.free_pool if x >= size):
            # get the free chunk
            ptr = self.free_pool[sz].pop().start

            # If the set is empty after the pop(), remove the entry
            if not self.free_pool[sz]:
                del self.free_pool[sz]
            break

        if ptr is None:     # We did not find reusable freed ptr
            ptr = self._curr_offset
            self._curr_offset += size

        # Now we can allocate the chunk
        mmap = self._memory.map(ptr, size, Perm.R | Perm.W, 'heap')
        self.alloc_pool.update({ptr: mmap})

        return ptr

    def free(self, ptr: Addr) -> None:
        """
        Free the given memory chunk.

        :param ptr: Address to free
        :type ptr: :py:obj:`tritondse.types.Addr`
        :raise AllocatorException: if the pointer has already been freed or if it has never been allocated
        """
        if self.is_ptr_freed(ptr):
            raise AllocatorException('Double free or corruption!')

        if not self.is_ptr_allocated(ptr):
            raise AllocatorException(f'Invalid pointer ({hex(ptr)})')

        # Add the chunk into our free_pool
        memmap = self.alloc_pool[ptr]
        if memmap.size in self.free_pool:
            self.free_pool[memmap.size].add(memmap)
        else:
            self.free_pool[memmap.size] = {memmap}

        # Remove the chunk from our alloc_pool
        self._memory.unmap(ptr)
        del self.alloc_pool[ptr]

    def is_ptr_allocated(self, ptr: Addr) -> bool:
        """
        Check whether a given address has been allocated

        :param ptr: Address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :return: True if pointer points to an allocated memory region
        """
        return self._memory.is_mapped(ptr, 1)

    def is_ptr_freed(self, ptr: Addr) -> bool:
        """
        Check whether a given pointer has recently been freed.

        :param ptr: Address to check
        :type ptr: :py:obj:`tritondse.types.Addr`
        :return: True if pointer has been freed, False otherwise
        """
        # FIXME: This function is linear in the size of chunks. Can make it logarithmic
        for size, chunks in self.free_pool.items():
            for chunk in chunks:
                if chunk.start <= ptr < chunk.start + size:
                    return True
        return False
