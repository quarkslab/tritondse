import os
import hashlib
from enum    import Enum
from pathlib import Path


class SeedStatus(Enum):
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
        self.content     = bytes(content)
        self.target_addr = None
        self._status = status

    def is_bootstrap_seed(self) -> bool:
        return self.content == b""

    @property
    def status(self) -> SeedStatus:
        return self._status


    @status.setter
    def status(self, value: SeedStatus) -> None:
        self._status = value


    def __len__(self):
        return len(self.content)


    def __eq__(self, other):
        return self.content == other.content


    def __hash__(self):
        return hash(self.content)


    def get_size(self):
        """
        Returns the size of the seed.
        """
        return len(self.content)


    def get_hash(self):
        """
        Returns the md5 hash of the content
        """
        # Note
        #
        # HF mangling file : <crc64><crc64_reverse>.<size of seed in hexa>.honggfuzz.cov
        # It looks like there is no rule on the naming convention when providing a new seed to HF.
        # Keeping an MD5 one looks good.
        #
        m = hashlib.md5(self.content)
        return m.hexdigest()

    @property
    def filename(self):
        """
        Return the file name of the seed
        """
        # TODO: Handle HF mangling?
        return f'{self.get_hash()}.{self.get_size():08x}.tritondse.cov'

    @staticmethod
    def from_file(path: str, status: SeedStatus = SeedStatus.NEW) -> 'Seed':
        return Seed(Path(path).read_bytes(), status)
