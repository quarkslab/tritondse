# built-in imports
from __future__ import annotations
import shutil
from pathlib import Path
import logging
from typing import Generator, Optional, Union
import time

# local imports
from tritondse.types import PathLike
from tritondse.seed import Seed, SeedStatus


class Workspace(object):
    """
    Class to abstract the file tree of the current exploration workspace.
    A user willing to save additional files in the workspace is invited
    to do it from the workspace API as it somehow abstract the exact
    location of it.
    """

    DEFAULT_WORKSPACE = "/tmp/triton_workspace"
    CORPUS_DIR = "corpus"
    CRASH_DIR = "crashes"
    HANG_DIR = "hangs"
    WORKLIST_DIR = "worklist"
    METADATA_DIR = "metadata"
    BIN_DIR = "bin"
    LOG_FILE = "tritondse.log"

    def __init__(self, root_dir: PathLike):
        """
        :param root_dir: Root directory of the workspace. Created if not existing
        :type root_dir: :py:obj:`tritondse.types.PathLike`
        """
        if not root_dir:  # If no workspace was provided create a unique temporary one
            self.root_dir = Path(self.DEFAULT_WORKSPACE) / str(time.time()).replace(".", "")
            self.root_dir.mkdir(parents=True)
        else:
            self.root_dir = Path(root_dir)
            if not self.root_dir.exists():  # Create the directory in case it was not existing
                self.root_dir.mkdir(parents=True)
        self.root_dir = self.root_dir.resolve()

    def initialize(self, flush: bool = False) -> None:
        """
        Initialize the workspace by creating all required subfolders
        if not already existing.

        :param flush: if True deletes all files contained in the workspace
        :type flush: bool
        """

        for dir in (self.root_dir / x for x in [self.CORPUS_DIR, self.CRASH_DIR, self.HANG_DIR, self.WORKLIST_DIR, self.METADATA_DIR, self.BIN_DIR]):
            if not dir.exists():
                logging.debug(f"Creating the {dir} directory")
                dir.mkdir(parents=True)
            else:
                if flush:
                    shutil.rmtree(dir)
                    dir.mkdir()


    def get_metadata_file(self, name: str) -> Optional[str]:
        """
        Read a metadata file from the workspace on disk.
        Data is read as a string. If the given file does not
        exists, None is returned

        :param name: file name (can also be a path)
        :type name: str
        :returns: File content as string if existing
        :rtype: Optional[str]
        """
        p = self.root_dir / name
        if p.exists():
            return p.read_text()
        else:
            return None


    def get_metadata_file_path(self, name: str) -> Path:
        """
        Get a file path in the workspace directory that the user
        can write into. Might be called for the user to write on
        its own the file content. If name is a file tree, all parent
        directories are created.

        :param name: filename wanted
        :type name: str
        :return: absolute filepath (regardless of whether it exists or not)
        """
        p = self.root_dir / name
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        return p

    def get_binary_directory(self) -> Path:
        """
        Get the directory containing the executable (and its dependencies).
        :return: Path of the directory
        """
        return self.root_dir / self.BIN_DIR


    def save_metadata_file(self, name: str, content: Union[str, bytes]) -> None:
        """
        Save ``content`` in a file ``name`` in the metadata directory.
        The name should be a file name not a path.

        :param name: file name
        :type name: str
        :param content: content of the file to write
        :type content: Union[str, bytes]
        """
        p = (self.root_dir / self.METADATA_DIR) / name
        if isinstance(content, str):
            p.write_text(content)
        else:
            p.write_bytes(content)


    def _iter_seeds(self, directory: str, st: SeedStatus) -> Generator[Seed, None, None]:
        """ Iterate over seeds """
        for file in (self.root_dir/directory).glob("*.cov"):
            yield Seed(file.read_bytes(), st)


    def iter_corpus(self) -> Generator[Seed, None, None]:
        """
        Iterate over the corpus files as Seed object.

        :returns: generator of Seed object
        :rtype: Generator[Seed, None, None]
        """
        yield from self._iter_seeds(self.CORPUS_DIR, SeedStatus.OK_DONE)


    def iter_crashes(self) -> Generator[Seed, None, None]:
        """
        Iterate over the crashes files as Seed object.

        :returns: generator of Seed object
        :rtype: Generator[Seed, None, None]
        """
        yield from self._iter_seeds(self.CRASH_DIR, SeedStatus.CRASH)


    def iter_hangs(self) -> Generator[Seed, None, None]:
        """
        Iterate over the hang files as Seed object.

        :returns: generator of Seed object
        :rtype: Generator[Seed, None, None]
        """
        yield from self._iter_seeds(self.HANG_DIR, SeedStatus.HANG)


    def iter_worklist(self) -> Generator[Seed, None, None]:
        """
        Iterate over the worklist files as Seed object.
        Worklist are all the pending seeds

        :returns: generator of Seed object
        :rtype: Generator[Seed, None, None]
        """
        yield from self._iter_seeds(self.WORKLIST_DIR, SeedStatus.NEW)


    def save_seed(self, seed: Seed) -> None:
        """
        Save the current seed in the workspace directory matching its status.

        :param seed: Seed to save
        :type seed: Seed
        """
        mapper = {SeedStatus.NEW: self.WORKLIST_DIR,
                  SeedStatus.OK_DONE: self.CORPUS_DIR,
                  SeedStatus.HANG: self.HANG_DIR,
                  SeedStatus.CRASH: self.CRASH_DIR}
        p = (self.root_dir / mapper[seed.status]) / seed.filename
        p.write_bytes(bytes(seed))


    def update_seed_location(self, seed: Seed) -> None:
        """
        Move a worklist seed to its final location according to its (new) status.
        Typically used to move a seed from pending ones to corpus or crash once it
        is fully consumed.

        :param seed: seed to move
        :type seed: Seed
        """
        old_p = (self.root_dir / self.WORKLIST_DIR) / seed.filename
        try:
            old_p.unlink()  # Remove the seed from the worklist
        except:
            pass  # FIXME: Not meant to get here
        self.save_seed(seed)


    def save_file(self, rel_path: PathLike, content: Union[str, bytes], override: bool = False):
        """
        Save a, arbitrary file in the workspace by providing the relative path
        of the file. If ``override`` is True, erase the previous file if any.

        :param rel_path: relative path of the file
        :type rel_path: :py:obj:`tritondse.types.PathLike`
        :param content: content to write
        :type content: Union[str, bytes]
        :param override: whether to override or not an existing file
        :type override: bool
        """
        p = self.root_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or override:
            if isinstance(content, str):
                p.write_text(content)
            elif isinstance(content, bytes):
                p.write_bytes(content)
            else:
                assert False

    @property
    def logfile_path(self):
        return self.root_dir / self.LOG_FILE
