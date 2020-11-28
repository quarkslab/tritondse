# built-in imports
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
    Class to abstract the file tree of the current
    exploration workspace
    """

    DEFAULT_WORKSPACE = "/tmp/triton_workspace"

    CORPUS_DIR = "corpus"
    CRASH_DIR = "crashes"
    HANG_DIR = "hangs"
    WORKLIST_DIR = "worklist"
    METADATA_DIR = "metadata"

    def __init__(self, root_dir: PathLike):
        if not root_dir:  # If no workspace was provided create a unique temporary one
            self.root_dir = Path(self.DEFAULT_WORKSPACE) / str(int(time.time()))
            self.root_dir.mkdir(parents=True)
        else:
            self.root_dir = Path(root_dir)
            if not self.root_dir.exists():  # Create the directory in case it was not existing
                self.root_dir.mkdir(parents=True)

    def initialize(self, flush: bool = False):
        """
        Initialize the workspace and create directories if needed

        :param flush: if True deletes all files contained in the workspace
        :return: None
        """

        for dir in (self.root_dir / x for x in [self.CORPUS_DIR, self.CRASH_DIR, self.HANG_DIR, self.WORKLIST_DIR, self.METADATA_DIR]):
            if not dir.exists():
                logging.debug(f"Creating the {dir} directory")
                dir.mkdir(parents=True)
            else:
                if flush:
                    shutil.rmtree(dir)
                    dir.mkdir()


    def get_metadata_file(self, name) -> Optional[str]:
        """ Get the metadata content from disk """
        p = self.root_dir / name
        if p.exists():
            return p.read_text()
        else:
            return None


    def get_metadata_file_path(self, name: str) -> Path:
        """
        Return a file path in the workspace directory that the user can write into.

        :param name: filename wanted
        :return: full filepath
        """
        return self.root_dir / name


    def save_metadata_file(self, name: str, content: str) -> None:
        """ Save metadata on disk """
        p = (self.root_dir / self.METADATA_DIR) / name
        p.write_text(content)


    def _iter_seeds(self, directory: str, st: SeedStatus) -> Generator[Seed, None, None]:
        """ Iterate over seeds """
        for file in (self.root_dir/directory).glob("*.cov"):
            yield Seed(file.read_bytes(), st)


    def iter_corpus(self) -> Generator[Seed, None, None]:
        """ Iterate over the corpus """
        yield from self._iter_seeds(self.CORPUS_DIR, SeedStatus.OK_DONE)


    def iter_crashes(self) -> Generator[Seed, None, None]:
        """ Iterate over crashes """
        yield from self._iter_seeds(self.CRASH_DIR, SeedStatus.CRASH)


    def iter_hangs(self) -> Generator[Seed, None, None]:
        """ Iterate over hangs """
        yield from self._iter_seeds(self.HANG_DIR, SeedStatus.HANG)


    def iter_worklist(self) -> Generator[Seed, None, None]:
        """ Iterate over the worklist """
        yield from self._iter_seeds(self.WORKLIST_DIR, SeedStatus.NEW)


    def save_seed(self, seed: Seed) -> None:
        """ Save the current seed in the directory matching its status """
        mapper = {SeedStatus.NEW: self.WORKLIST_DIR,
                  SeedStatus.OK_DONE: self.CORPUS_DIR,
                  SeedStatus.HANG: self.HANG_DIR,
                  SeedStatus.CRASH: self.CRASH_DIR}
        p = (self.root_dir / mapper[seed.status]) / seed.filename
        p.write_bytes(seed.content)


    def update_seed_location(self, seed: Seed) -> None:
        """ Move a worklist seed to its final location according to its new status """
        old_p = (self.root_dir / self.WORKLIST_DIR) / seed.filename
        try:
            old_p.unlink()  # Remove the seed from the worklist
        except:
            pass
        self.save_seed(seed)


    def save_file(self, rel_path: str, content: Union[str, bytes], override=False):
        """ Save a file on disk """
        p = self.root_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or override:
            if isinstance(content, str):
                p.write_text(content)
            elif isinstance(content, bytes):
                p.write_bytes(content)
            else:
                assert False
