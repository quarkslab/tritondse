from .config                import Config
from .program               import Program
from .process_state         import ProcessState
from .coverage              import CoverageStrategy
from .symbolic_executor     import SymbolicExecutor
from .symbolic_explorator   import SymbolicExplorator, ExplorationStatus
from .seed                  import Seed, SeedStatus

from triton import VERSION

TRITON_VERSION = f"v{VERSION.MAJOR}.{VERSION.MINOR}"
