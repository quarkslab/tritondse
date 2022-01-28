from enum import IntEnum


class Permission(IntEnum):
    NONE  = 0
    READ  = 1
    WRITE = 2
    EXEC  = 3


class Range:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def overlaps(self, other_range: 'Range') -> bool:
        return self.start <= other_range.start < self.end or \
               self.start < other_range.end <= self.end

class Segment:

    def __init__(self, range_: Range, permissions: Permission):
        self.range = range_
        self.permissions = permissions


class Module:

    def __init__(self, name: str):
        self.name = name
        self.range = None
        self.segments = dict()

    def add(self, segment: Segment) -> None:
        if not self.range:
            self.range = Range(segment.range.start, segment.range.end)

        if segment.range.start < self.range.start:
            self.range.start = segment.range.start

        if self.range.end < segment.range.end:
            self.range.end = segment.range.end

        if not segment.range.start in self.segments:
            self.segments[segment.range.start] = segment
