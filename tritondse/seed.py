import hashlib
import base64
import json
from enum import Enum
from pathlib import Path
from tritondse.types import PathLike, SymExType
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field
import enum_tools.documentation


@enum_tools.documentation.document_enum
class SeedStatus(Enum):
    """
    Seed status enum.
    Enables giving a status to a seed during its execution.
    At the end of a :py:obj:`SymbolicExecutor` run one of these
    status must have set to the seed.
    """
    NEW = 0      # doc: The input seed is new (has not been executed yet)
    OK_DONE = 1  # doc: The input seed has been executed and terminated correctly
    CRASH = 2    # doc: The input seed crashed in some ways
    HANG = 3     # doc: The input seed made the program to hang


@enum_tools.documentation.document_enum
class SeedFormat(Enum):
    """
    Seed format enum
    Raw seeds are just bytes Seed(b"AAAAA\x00BBBBB")
    Composite can describe how to inject the input more precisely 
    """
    RAW = 0        # doc: plain bytes input seed
    COMPOSITE = 1  # doc: complex input object


@dataclass(frozen=True)
class CompositeData:
    argv: List[bytes] = field(default_factory=list)
    "list of argv values"
    files: Dict[str, bytes] = field(default_factory=dict)
    "dictionnary of files and the associated content (stdin is one of them)"
    variables: Dict[str, bytes] = field(default_factory=dict)
    "user defined variables, that the use must take care to inject at right location"

    def _to_json(self):
        data = {
            'argv': [base64.b64encode(v).decode() for v in self.argv],
            'files': {k: (base64.b64encode(v).decode() if isinstance(v, bytes) else v) for k, v in self.files.items()},
            'variables': {k: (base64.b64encode(v).decode() if isinstance(v, bytes) else v) for k, v in self.variables.items()},
        }
        return json.dumps(data, indent=2)

    def __bytes__(self):
        return self._to_json().encode()

    @staticmethod
    def from_dict(json_data: dict) -> 'CompositeData':
        argv = [base64.b64decode(v) for v in json_data['argv']]
        files = {k: (base64.b64decode(v) if isinstance(v, str) else v) for k, v in json_data['files'].items()}
        variables = {k: (base64.b64decode(v) if isinstance(v, str) else v) for k, v in json_data['variables'].items()}
        return CompositeData(argv=argv, files=files, variables=variables)

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
        self.meta_fname = []
        self.target = None                # CovItem informational field indicate the item the seed was generated for
        self._status = status
        self._type = SeedFormat.COMPOSITE if isinstance(content, CompositeData) else SeedFormat.RAW

    def is_composite(self) -> bool:
        """Returns wether the seed is a composite seed or not. """
        return self._type == SeedFormat.COMPOSITE

    def is_raw(self) -> bool:
        """Returns wether the seed is a raw seed or not. """
        return self._type == SeedFormat.RAW

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
    def format(self) -> SeedFormat:
        """
        Format of the seed.

        :rtype: SeedFormat
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
        return len(bytes(self.content))

    def __eq__(self, other) -> bool:
        """
        Equality check based on content.

        :returns: true if content of both seeds are equal """
        return self.content == other.content

    def bytes(self) -> bytes:
        return bytes(self)

    def __bytes__(self) -> bytes:
        """
        Return a representation of the seed's content in bytes.

        :rtype: bytes
        """
        return bytes(self.content)

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
        return f"{self.hash}_{self.size:04x}_{'_'.join(self.meta_fname)}.tritondse.cov"

    @staticmethod
    def from_bytes(raw_seed: bytes, status: SeedStatus = SeedStatus.NEW) -> 'Seed':
        """
        Parse a seed from its byte representation. If its a composite one
        it will parse the bytes as JSON and create the CompositeData accordingly.

        :param raw_seed: bytes: raw bytes of the seed
        :param status: status of the seed if any, otherwise :py:obj:`SeedStatus.NEW`
        :type status: SeedStatus

        :returns: fresh seed instance
        :rtype: Seed
        """
        try:
            data = json.loads(raw_seed)

            if not isinstance(data, dict):  # it might happen that files contains only digit which is a valid JSON
                return Seed(raw_seed, status)

            if 'files' in data and 'argv' in data:
                return Seed(CompositeData.from_dict(data), status)
            else:  # Else still consider file as raw bytes
                return Seed(raw_seed, status)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Seed(raw_seed, status)

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
        raw = Path(path).read_bytes()
        return Seed.from_bytes(raw, status)

    # Utility function for composite seeds
    def is_file_defined(self, name: str) -> bool:
        if self.is_composite():
            return name in self.content.files
        else:
            return False

    def get_file_input(self, name: str) -> bytes:
        """
        Return the bytes associated to a given file within
        a composite seed.

        :raise KeyError: if the name cannot be found in the seed.
        :param name: name of the file to retrieve
        :return: bytes of the file content
        """
        return self.content.files[name]
