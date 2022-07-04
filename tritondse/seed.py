import hashlib
import ast
import logging 
from enum    import Enum
from pathlib import Path
from tritondse.types import PathLike
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


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


class SeedType(Enum):
    """
    Seed type enum
    Raw seeds are just bytes Seed(b"AAAAA\x00BBBBB")
    Composite can describe how to inject the input more precisely 
    """
    RAW       = 0
    COMPOSITE = 1


class CompositeField(Enum):
    """
    Enum representing the different Fields present in CompositeData 
    """
    ARGV        = 0
    FILE        = 1
    VARIABLE    = 2


@dataclass()
class SymbolicCompositeData:
    argv: Optional[List] = None
    files: Optional[Dict] = None
    variables: Optional[Dict] = None # Currently only used to manually inject variables


@dataclass(frozen=True)
class CompositeData:
    argv: Optional[List[str]] = None
    files: Optional[Dict[str, bytes]] = None
    variables: Optional[Dict[str, bytes]] = None # Currently only used to manually inject variables

    def __bytes__(self):
        serialized = b"{'argv': "
        serialized += str(self.argv).encode() 

        serialized += b", 'files': "
        if self.files:
            sorted_files_dict = dict(sorted(self.files.items()))
            c = str(sorted_files_dict).encode()
            serialized += c
        else: 
            serialized += str(self.files).encode() 

        serialized += b", 'variables': "
        if self.variables:
            sorted_var_dict = dict(sorted(self.variables.items()))
            c = str(sorted_var_dict).encode()
            serialized += c
        else: 
            serialized += str(self.variables).encode() 

        serialized += b"}"
        return serialized

    def from_bytes(data_bytes: bytes) -> 'CompositeData':
        seed_dict = ast.literal_eval(data_bytes.decode())
        if "argv" not in seed_dict or "files" not in seed_dict:
            logging.error('Failed to convert bytes to CompositeData')
            assert False

        return CompositeData(argv=seed_dict["argv"], files=seed_dict["files"])

    def __hash__(self):
        return hash(bytes(self))


class Seed(object):
    """
    Seed input.
    Holds the bytes buffer of the content a status after execution
    but also some metadata of code portions it is meant to cover.
    """
    def __init__(self, content: Union[bytes, CompositeData] = bytes(), status=SeedStatus.NEW):
        """
        :param content: content of the input. By default is b"" *(and is thus considered as a bootstrap seed)*
        :type content: bytes
        :param status: status of the seed if already known
        :type status: SeedStatus
        """
        self.content = content
        self.coverage_objectives = set()  # set of coverage items that the seed is meant to cover
        self.target = set()               # CovItem informational field indicate the item the seed was generated for
        self._status = status
        self._type = SeedType.COMPOSITE if isinstance(content, CompositeData) else SeedType.RAW


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

        :rtype: SeedStatus
        """
        return self._status


    @property
    def type(self) -> SeedType:
        """
        Type of the seed.

        :rtype: SeedType
        """
        return self._type


    @status.setter
    def status(self, value: SeedStatus) -> None:
        """ Sets the status of the seed """
        self._status = value

    def is_status_set(self) -> bool:
        """ Checks whether a status has already been assigned to the seed. """
        return self.status != SeedStatus.NEW

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


    def __bytes__(self):
        """
        Return a representation of the seed's content in bytes.

        :rtype: int
        """
        tag = b"COMPOSITE\n" if isinstance(self.content, CompositeData) else b"RAW\n"
        return tag + bytes(self.content)


    def __hash__(self):
        """
        Seed hash function overriden to base itself on content.
        That enable storing seed in dictionnaries directly based
        on their content to discriminate them.

        :rtype: int
        """
        return hash(self.content)

    @property
    def hash(self) -> str:
        """
        MD5 hash of the seed content

        :rtype: str
        """
        m = hashlib.md5(bytes(self))
        return m.hexdigest()

    @property
    def size(self) -> int:
        """
        Size of the seed content in bytes

        :rtype: int
        """
        return len(bytes(self))

    @property
    def filename(self):
        """
        Standardized filename based on hash and size.
        That does not mean the file exists or anything.

        :returns: formatted intended filename of the seed
        :rtype: str
        """
        return f'{self.hash}.{self.size:08x}.tritondse.cov'


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
        file_lines = Path(path).read_bytes().split(b"\n")
        seed_type, seed_content = file_lines[0], file_lines[1]
        if seed_type == b"RAW":
            return Seed(seed_content, status)
        elif seed_type == b"COMPOSITE":
            return Seed(CompositeData.from_bytes(seed_content), status)
        else: 
            logging.error('Seed.from_file: Invalid file')
            assert False
