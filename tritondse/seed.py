import os
import hashlib
from enum    import Enum
from pathlib import Path


class SeedStatus(Enum):
    """ All kind of seed """
    NEW     = 0
    OK_DONE = 1
    CRASH   = 2
    HANG    = 3



class Seed(object):
    """
    This class is used to represent a seed input. The seed will be injected
    into stdin or argv according to the Triton DSE configuration.
    """
    def __init__(self, content=bytes(), status=SeedStatus.NEW):
        self.content = bytes(content)
        self.coverage_objectives = set()  # set of coverage items that the seed is meant to cover
        self.target = set()               # CovItem informational field indicate the item the seed was generated for
        self._status = status


    def is_bootstrap_seed(self) -> bool:
        """ Returns true if the seed is the initial seed """
        return self.content == b""


    def is_fresh(self) -> bool:
        """ Returns true if the seed is a new one (never been executed) """
        return not self.coverage_objectives


    @property
    def status(self) -> SeedStatus:
        """ Returns the status of the seed """
        return self._status


    @status.setter
    def status(self, value: SeedStatus) -> None:
        """ Sets the status of the seed """
        self._status = value


    def __len__(self):
        """ Returns the size of the seed """
        return len(self.content)


    def __eq__(self, other):
        """ Returns true if the seed is equal to another seed """
        return self.content == other.content


    def __hash__(self):
        return hash(self.content)


    def get_size(self) -> int:
        """ Returns the size of the seed """
        return len(self.content)


    def get_hash(self) -> int:
        """ Returns the md5 hash of the content """
        m = hashlib.md5(self.content)
        return m.hexdigest()


    @property
    def filename(self):
        """ Returns the file name of the seed """
        return f'{self.get_hash()}.{self.get_size():08x}.tritondse.cov'


    @staticmethod
    def from_file(path: str, status: SeedStatus = SeedStatus.NEW) -> 'Seed':
        """ Returns a fresh instance of a seed from a file """
        return Seed(Path(path).read_bytes(), status)
