import hashlib
from enum    import Enum
from pathlib import Path
from tritondse.types import PathLike


class SeedStatus(Enum):
    """
     Seed status enum.
     Enables giving a status to a seed during its execution.
     At the end of a :py:obj:`SymbolicExecutor` run one of these
     status must have set to the seed.
     """
    NEW     = 0
    OK_DONE = 1
    CRASH   = 2
    HANG    = 3



class Seed(object):
    """
    Seed input.
    Holds the bytes buffer of the content a status after execution
    but also some metadata of code portions it is meant to cover.
    """
    def __init__(self, content=bytes(), status=SeedStatus.NEW):
        """
        :param content: content of the input. By default is b"" *(and is thus considered as a bootstrap seed)*
        :type content: bytes
        :param status: status of the seed if already known
        :type status: SeedStatus
        """
        self.content: bytes  = bytes(content)  #: content of the seed
        self.coverage_objectives = set()  # set of coverage items that the seed is meant to cover
        self.target = set()               # CovItem informational field indicate the item the seed was generated for
        self._status = status


    def is_bootstrap_seed(self) -> bool:
        """
        A bootstrap seed is an empty seed (b""). It will received a
        specific processing in the engine as its size will be automatically
        adapted to the size read (in stdin for instance)

        :returns: true if the seed is a bootstrap seed
        """
        return self.content == b""


    def is_fresh(self) -> bool:
        """
        A fresh seed is never been executed. Its is recognizable
        as it does not contain any coverage objectives.

        :returns: True if the seed has never been executed
        """
        return not self.coverage_objectives


    @property
    def status(self) -> SeedStatus:
        """
        Status of the seed.

        :rtype: SeedStatus"""
        return self._status


    @status.setter
    def status(self, value: SeedStatus) -> None:
        """ Sets the status of the seed """
        self._status = value


    def __len__(self) -> int:
        """
        Size of the content of the seed.

        :rtype: int
        """
        return len(self.content)


    def __eq__(self, other) -> bool:
        """
        Equality check based on content.

        :returns: true if content of both seeds are equal """
        return self.content == other.content


    def __hash__(self):
        """
        Seed hash function overriden to base itself on content.
        That enable storing seed in dictionnaries directly based
        on their content to discriminate them.

        :rtype: int
        """
        return hash(self.content)


    def get_size(self) -> int:
        """
        Size of the seed content in bytes

        :rtype: int
        """
        return len(self.content)


    def get_hash(self) -> str:
        """
        MD5 hash of the seed content

        :rtype: str
        """
        m = hashlib.md5(self.content)
        return m.hexdigest()


    @property
    def filename(self):
        """
        Standardized filename based on hash and size.
        That does not mean the file exists or anything.

        :returns: formatted intended filename of the seed
        :rtype: str
        """
        return f'{self.get_hash()}.{self.get_size():08x}.tritondse.cov'


    @staticmethod
    def from_file(path: PathLike, status: SeedStatus = SeedStatus.NEW) -> 'Seed':
        """
        Read a seed from a file. The status can optionally given
        as it cannot be determined from the file.

        :param path: seed path
        :type path: :py:obj:`tritondse.types.PathLike`
        :param status: status of the seed if any, otherwise :py:obj:`SeedStatus.NEW`
        :type status: SeedStatus

        :returns: fresh seed instance
        :rtype: Seed
        """
        return Seed(Path(path).read_bytes(), status)
