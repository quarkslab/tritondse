#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tritondse.types import Addr, ByteSize



class AllocatorException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)



class HeapAllocator(object):
    """
    Custom tiny heap allocator. Used by built-ins
    routines like malloc/free. This allocation manager
    also enables better subsequent queries.
    """

    def __init__(self, start: Addr, end: Addr):
        '''
        Class constructor. Takes heap bounds as parameter.

        :param start Addr: Where the heap area can start
        :param end Addr: Where the heap area must be end
        '''
        # Range of the memory mapping
        self.start = start
        self.end   = end
        self.base  = self.start

        # Pools memory
        self.alloc_pool   = dict() # {ptr: size}
        self.free_pool    = dict() # {size: set(ptr, ...)}

        # TODO: For a to-the-moon allocator, we could merge freed chunks. Like 4 chunks of 1 byte into one chunk of 4 bytes.
        # TODO: For a to-the-moon allocator, we could split a big chunk into two chunks when asking an allocation.


    def __ptr_from_free_to_alloc(self, size: int) -> Addr:
        # Pop an available pointer
        ptr = self.free_pool[size].pop()

        # If the set is empty after the pop(), remove the entry
        if not self.free_pool[size]:
            del self.free_pool[size]

        # Move this chunk into our alloc_pool
        if ptr in self.alloc_pool:
            raise AllocatorException('This pointer is already provided')
        self.alloc_pool.update({ptr: size})

        return ptr


    def alloc(self, size: ByteSize) -> Addr:
        """
        Performs an allocation of the given byte size.

        :param size: Byte size to allocate
        :raise: AllocatorException if not memory is available
        """
        # First, check if we have an available chunk in our
        # free_pool which have the same size
        if size in self.free_pool and self.free_pool[size]:
            return self.__ptr_from_free_to_alloc(size)

        # If we have nothing available in our free_pool with the same size, check
        # if we have a chunk with a bigger size. We will take the best choice (bigger
        # but not to much).
        for k, v in sorted(self.free_pool.items()):
            # Should never be equal, but why not?
            if k >= size and v:
                return self.__ptr_from_free_to_alloc(k)

        # If we have nothing available in our free_pool at all, just allocate
        # a new chunk
        ptr = self.base
        self.alloc_pool.update({ptr: size})
        self.base += size
        if self.base >= self.end:
            raise AllocatorException('Out of Memory')

        return ptr


    def free(self, ptr: Addr) -> None:
        """
        Free the given memory chunk.

        :param ptr: Address to free
        :raise: AllocatorException if the pointer has already been
                freed or if it has never been allocated
        """
        if self.is_ptr_freed(ptr):
            raise AllocatorException('Double free or corruption!')

        if not self.is_ptr_allocated(ptr):
            raise AllocatorException(f'Invalid pointer ({hex(ptr)})')

        # Add the chunk into our free_pool
        size = self.alloc_pool[ptr]
        if size in self.free_pool:
            self.free_pool[size].add(ptr)
        else:
            self.free_pool.update({size: set({ptr})})

        # Remove the chunk from our alloc_pool
        del self.alloc_pool[ptr]


    def is_ptr_allocated(self, ptr: Addr) -> bool:
        """
        Check whether a given address has been allocated

        :param ptr: Address to check
        :return: True if pointer points to an allocated memory region
        """
        for chunk, size in self.alloc_pool.items():
            if ptr >= chunk and ptr < chunk + size:
                return True
        return False


    def is_ptr_freed(self, ptr: Addr) -> bool:
        """
        Check whether a given pointer has recently been freed.

        :param ptr: Address to check
        :return: True if pointer has been freed, False otherwise
        """
        for size, chunks in self.free_pool.items():
            for chunk in chunks:
                if ptr >= chunk and ptr < chunk + size:
                    return True
        return False
