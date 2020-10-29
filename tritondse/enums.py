from enum import Enum, auto


class Enums(Enum):
    ARGV          = auto()
    CONCRETIZE    = auto()
    STDIN         = auto()
    SYMBOLIZE     = auto()


class CoverageStrategy(Enum):
    CODE_COVERAGE = auto()
    PATH_COVERAGE = auto()
    EDGE_COVERAGE = auto()
