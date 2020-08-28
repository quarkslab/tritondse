#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tritondse.types import Addr



class AllocException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)



class HeapAllocator(object):
    def __init__(self, start: Addr, end: Addr):
        '''
        This class is used to represent the heap allocation manager.

        :param start int: Where the heap area can start
        :param end int: Where the heap area must be end
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
            raise AllocException('This pointer is already provided')
        self.alloc_pool.update({ptr: size})

        return ptr


    def alloc(self, size: int) -> Addr:
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
            raise AllocException('Out of Memory')

        return ptr


    def free(self, ptr: Addr):
        if not ptr in self.alloc_pool:
            raise AllocException('This pointer is not allocated or not alligned on the head chunk')

        # Add the chunk into our free_pool
        size = self.alloc_pool[ptr]
        if size in self.free_pool:
            self.free_pool[size].add(ptr)
        else:
            self.free_pool.update({size: set({ptr})})

        # Remove the chunk from our alloc_pool
        del self.alloc_pool[ptr]


    def is_ptr_allocated(self, ptr: Addr) -> bool:
        for chunk, size in self.alloc_pool.items():
            if ptr >= chunk and ptr < chunk + size:
                return True
        return False


    def is_ptr_freed(self, ptr: Addr) -> bool:
        for size, chunks in self.free_pool.items():
            for chunk in chunks:
                if ptr >= chunk and ptr < chunk + size:
                    return True
        return False
