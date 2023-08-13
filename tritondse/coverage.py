# built-in imports
from __future__ import annotations
import hashlib
import struct
from pathlib import Path
from typing import List, Generator, Tuple, Set, Union, Dict, Optional
from collections import Counter
from enum import IntFlag, Enum, auto
import pickle
import enum_tools.documentation

# third-party imports
from triton import AST_NODE

# local imports
from tritondse.types import Addr, PathConstraint, PathBranch, SolverStatus, PathHash, Edge, SymExType
import tritondse.logging

logger = tritondse.logging.get()

CovItem = Union[Addr, Edge, PathHash, Tuple[PathHash, Edge]]
"""
Variant type representing a coverage item.
It can be:

* an address :py:obj:`tritondse.types.Addr` for block coverage
* an edge :py:obj:`tritondse.types.Edge` for edge coverage
* a string :py:obj:`tritondse.types.PathHash` for path coverage
* a tuple of both a Pathhash and an edge
"""

@enum_tools.documentation.document_enum
class CoverageStrategy(str, Enum):
    """
    Coverage strategy (metric) enum.
    This enum will change whether a given branch have
    to be solved or not.
    """

    BLOCK = "block"   # doc: block coverage, only tracks new basic blocks covered
    EDGE = "edge"     # doc: edge coverage, tracks CFGs edges covered
    PATH = "path"     # doc: tracks any new path covered
    PREFIXED_EDGE = "PREFIXED_EDGE"  # doc: edge coverage but also taking in account path prefix)


@enum_tools.documentation.document_enum
class BranchSolvingStrategy(IntFlag):
    """
    Branch strategy enumerate.
    It defines the manner with which branches are checked with SMT
    on a single trace, namely a :py:obj:`CoverageSingleRun`. For a
    given branch that has not been covered strategies are:

    * ``ALL_NOT_COVERED``: check by SMT all occurences
    * ``FIRST_LAST_NOT_COVERED``: check only the first and last occurence in the trace
    """
    ALL_NOT_COVERED = auto()         # doc: check by SMT all occurences of a given branch (true by default)
    FIRST_LAST_NOT_COVERED = auto()  # doc: check by SMT the first and last occurence of a given branch
    UNSAT_ONCE = auto()              # doc: if a branch is UNSAT do not try solving it again
    TIMEOUT_ONCE = auto()            # doc: if a branch is TIMEOUT do not try solving it again
    TIMEOUT_ALWAYS = auto()          # doc: always try solving again a TIMEOUT branch (incompatible with :py:enum:mem:`TIMEOUT_ONCE`
    COVER_SYM_DYNJUMP = auto()       # doc: try covering dynamic jumps on a symbolic register or memory value
    COVER_SYM_READ = auto()          # doc: try enumerating values for symbolic reads
    COVER_SYM_WRITE = auto()         # doc: try enumerating values for symbolic writes
    SOUND_MEM_ACCESS = auto()        # doc: enables adding a constraint when using a symbolic read/write or jump
    MANUAL = auto()                  # doc: disable automatic branch solving after an execution (has to be done manually in callbacks)


class CoverageSingleRun(object):
    """
    Coverage produced by a **Single Execution**
    Depending on the strategy given to the constructor
    it stores different data.
    """

    def __init__(self, strategy: CoverageStrategy):
        """
        :param strategy: Strategy to employ
        :type strategy: CoverageStrategy
        """
        self.strategy: CoverageStrategy = strategy  #: Coverage strategy

        # For instruction coverage
        self.covered_instructions: Dict[Addr, int] = Counter()
        """ Instruction coverage. Counter for code coverage) """

        self.covered_items: Dict[CovItem, int] = Counter()
        """ Stores covered items whatever they are """
        self.not_covered_items: Set[CovItem] = set()
        self._not_covered_items_mirror: Dict[CovItem, List[str]] = {}  # solely used for prefixed-edge
        """ CovItems not covered in the trace. It thus represent what can be
        covered by the trace (input). We call it coverage objectives."""

        # For path coverage
        self._current_path: List[Addr] = []
        """ List of addresses forming the path currently being taken """
        self._current_path_hash = hashlib.md5()

    def add_covered_address(self, address: Addr) -> None:
        """
        Add an instruction address covered.
        *(Called by :py:obj:`SymbolicExecutor` for each
        instruction executed)*

        :param address: The address of the instruction
        :type address: :py:obj:`tritondse.types.Addr`
        """
        self.covered_instructions[address] += 1

    def add_covered_dynamic_branch(self, source: Addr, target: Addr) -> None:
        """
        Add a dynamic branch covered. The branch will be encoded according to the
        coverage strategy.

        :param source: Address of the dynamic jump
        :param target: Target address on which the jump is performed
        :return:
        """
        if self.strategy == CoverageStrategy.BLOCK:
            pass # Target address will be covered anyway

        if self.strategy == CoverageStrategy.EDGE:
            self.covered_items[(source, target)] += 1
            self.not_covered_items.discard((source, target))    # Remove it from non-taken if it was inside

        if self.strategy == CoverageStrategy.PATH:
            self._current_path.append(target)
            self._current_path_hash.update(struct.pack("<Q", target))
            self.covered_items[self._current_path_hash.hexdigest()] += 1

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            # Add covered as covered
            self.covered_items[("", (source, target))] += 1
            # update the current path hash etc
            self._current_path.append(target)
            self._current_path_hash.update(struct.pack("<Q", target))

    def add_covered_branch(self, program_counter: Addr, taken_addr: Addr, not_taken_addr: Addr) -> None:
        """
        Add a branch to our covered branches list. Each branch is encoded according
        to the coverage strategy. For code coverage, the branch encoding is the
        address of the instruction. For edge coverage, the branch encoding is the
        tupe (src address, dst address). For path coverage, the branch encoding
        is the MD5 of the conjunction of all taken branch addresses.

        :param program_counter: The address taken in by the branch
        :type program_counter: :py:obj:`tritondse.types.Addr`
        :param taken_addr: Target address of branch taken
        :type taken_addr: Addr
        :param not_taken_addr: Target address of branch **not** taken
        :type not_taken_addr: Addr
        """

        if self.strategy == CoverageStrategy.BLOCK:
            self.covered_items[taken_addr] += 1
            self.not_covered_items.discard(taken_addr)    # remove address from non-covered if inside
            if not_taken_addr not in self.covered_items:  # Keep the address that has not been covered (and could have)
                self.not_covered_items.add(not_taken_addr)

        if self.strategy == CoverageStrategy.EDGE:
            taken_tuple, not_taken_tuple = (program_counter, taken_addr), (program_counter, not_taken_addr)
            self.covered_items[taken_tuple] += 1
            self.not_covered_items.discard(taken_tuple)    # Remove it from non-taken if it was inside
            if not_taken_tuple not in self.covered_items:  # Add the not taken tuple in non-covered
                self.not_covered_items.add(not_taken_tuple)

        if self.strategy == CoverageStrategy.PATH:
            self._current_path.append(taken_addr)

            # Compute the hash of the not taken path and add it to non-covered paths
            not_taken_path_hash = self._current_path_hash.copy()
            not_taken_path_hash.update(struct.pack('<Q', not_taken_addr))
            self.not_covered_items.add(not_taken_path_hash.hexdigest())

            # Update the current path hash and add it to hashes
            self._current_path_hash.update(struct.pack("<Q", taken_addr))
            self.covered_items[self._current_path_hash.hexdigest()] += 1

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            taken_tuple, not_taken_tuple = (program_counter, taken_addr), (program_counter, not_taken_addr)
            _, not_taken = (self._current_path_hash.hexdigest(), taken_tuple), (self._current_path_hash.hexdigest(), not_taken_tuple)
            gtaken, gnot_taken = ("", taken_tuple), ("", not_taken_tuple)

            # Add covered as covered
            self.covered_items[gtaken] += 1

            # Find all items in not_covered that have this edge
            if taken_tuple in self._not_covered_items_mirror:               # if a not_covered match this edge
                for prefix in self._not_covered_items_mirror[taken_tuple]:  # iterate all the prefixes
                    self.not_covered_items.discard((prefix, taken_tuple))   # and discard them
                self._not_covered_items_mirror.pop(taken_tuple)             # finally discard the entry

            # look if not_taken edge not in covered
            if gnot_taken not in self.covered_items:
                self.not_covered_items.add(not_taken)
                if not_taken[1] not in self._not_covered_items_mirror:
                    self._not_covered_items_mirror[not_taken[1]] = [not_taken[0]]
                else:
                    self._not_covered_items_mirror[not_taken[1]].append(not_taken[0])

            # update the current path hash etc
            self._current_path.append(taken_addr)
            self._current_path_hash.update(struct.pack("<Q", taken_addr))

    @property
    def unique_instruction_covered(self) -> int:
        """
        :return: The number of unique instructions covered
        """
        return len(self.covered_instructions)

    @property
    def unique_covitem_covered(self) -> int:
        """
        :return: The number of unique edges covered
        """
        return len(self.covered_items)

    @property
    def total_instruction_executed(self) -> int:
        """
        :return: The number of total instruction executed
        """
        return sum(self.covered_instructions.values())

    def post_execution(self) -> None:
        """
        Function is called after each execution
        for post processing or clean-up. *(Not
        doing anythin at the moment)*
        """
        pass

    def is_covered(self, item: CovItem) -> bool:
        """
        Return whether the item has been covered or not.
        **The item should match the strategy**

        :param item: An address, an edge or a path
        :type item: CovItem
        :return: bool
        """
        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            try:
                return ('', item[1]) in self.covered_items
            except TypeError:  # in case of ellipsis
                return False
        else:
            return item in self.covered_items

    def pp_item(self, covitem: CovItem) -> str:
        """
        Pretty print a CovItem according the coverage strategy

        :param covitem: An address, an edge or a path
        :return: str
        """
        if self.strategy == CoverageStrategy.BLOCK:
            return f"0x{covitem:08x}"
        elif self.strategy == CoverageStrategy.EDGE:
            return f"(0x{covitem[0]:08x}-0x{covitem[1]:08x})"
        elif self.strategy == CoverageStrategy.PATH:
            return covitem[:10]  # already a hash str
        elif self.strategy == CoverageStrategy.PREFIXED_EDGE:
            return f"({covitem[0][:6]}_0x{covitem[1][0]:08x}-0x{covitem[1][1]:08x})"

    def difference(self, other: CoverageSingleRun) -> Set[CovItem]:
        if self.strategy == other.strategy:
            return self.covered_items.keys() - other.covered_items.keys()
        else:
            logger.error("Trying to make difference of coverage with different strategies")
            return set()

    def __sub__(self, other) -> Set[CovItem]:
        return self.difference(other)


class GlobalCoverage(CoverageSingleRun):
    """
    Global Coverage.
    Represent the overall coverage of the exploration.
    It is filled by iteratively call merge with the
    :py:obj:`CoverageSingleRun` objects created during
    exploration.
    """

    COVERAGE_FILE = "coverage.json"

    def __init__(self, strategy: CoverageStrategy, branch_strategy: BranchSolvingStrategy):
        """
        :param strategy: Coverage strategy to use
        :type strategy: CoverageStrategy
        :param branch_strategy: Branch checking strategies
        :type branch_strategy: BranchSolvingStrategy
        """
        super().__init__(strategy)
        self.branch_strategy = branch_strategy

        # Keep pending items to be covered (code, edge, path)
        self.pending_coverage: Set[CovItem] = set()
        """ Set of pending coverage items. These are items for which a branch
        as already been solved and 
        """

        self.uncoverable_items: Dict[CovItem, SolverStatus] = {}
        """ CovItems that are determined not to be coverable. """

        self.covered_symbolic_pointers: Set[Addr] = set()
        """ Set of addresses for which pointers have been enumerated """

    def iter_new_paths(self, path_constraints: List[PathConstraint]) -> Generator[Tuple[SymExType, List[PathConstraint], PathBranch, CovItem, int], Optional[SolverStatus], None]:
        """
        The function iterate the given path predicate and yield PatchConstraint to
        consider as-is and PathBranch representing the new branch to take. It acts
        as a black-box so that the SeedManager does not have to know what strategy
        is being used under the hood. From an implementation perspective the goal
        of the function is to manipulate the path WITHOUT doing any SMT related things.

        :param path_constraints: list of path constraint to iterate
        :return: generator of path constraint and branches to solve. The first tuple
                 item is a list of PathConstraint to add in the path predicate and the second
                 is the branch to solve (but not to keep in path predicate)
        """
        if BranchSolvingStrategy.MANUAL in self.branch_strategy:
            logger.info(f'Branch solving strategy set to MANUAL.')
            return

        pending_csts = []
        current_hash = hashlib.md5()  # Current path hash for PATH coverage

        # NOTE: When we arrive here the CoverageSingleRun associated with the path_constraints
        # has already been merge. Thus covered, pending etc do include ones of the CoverageSingleRuns

        not_covered_items = self._get_items_trace(path_constraints)  # Map of CovItem -> [idx1, idx2, ..., .] (occurence in list)
        # is_ok_with_branch_strategy = lambda covitem, idx: True if self.strategy == CoverageStrategy.PATH else (idx in occurence_map[covitem])

        # Re-iterate through all path constraints to solve them concretely (with knowledge of what is beyond in the trace)
        for i, pc in enumerate(path_constraints):
            if pc.isMultipleBranches():     # If there is a condition
                for branch in pc.getBranchConstraints():  # Get all branches
                    # Get the constraint of the branch which has not been taken.
                    if not branch['isTaken']:
                        covitem = self._get_covitem(current_hash, branch)
                        generic_covitem = ('', covitem[1]) if self.strategy == CoverageStrategy.PREFIXED_EDGE else covitem
                        #print(f"Covitem: {covitem}: {covitem not in self.covered_items} | {covitem not in self.pending_coverage} | {covitem not in self.uncoverable_items} | {i in not_covered_items.get(covitem, [])} | {i} | {not_covered_items.get(covitem)}")

                        # Not covered in: previous runs | yet to be covered by a seed already SAT | not uncoverable | parts of items to solve
                        if generic_covitem not in self.covered_items and \
                           generic_covitem not in self.pending_coverage and \
                           covitem not in self.uncoverable_items and \
                           i in not_covered_items.get(covitem, []):

                            # Send the branch to solve to the function iterating
                            res = yield SymExType.CONDITIONAL_JMP, pending_csts, branch, covitem, i

                            # If path SAT add it to pending coverage
                            if res == SolverStatus.SAT:
                                self.pending_coverage.add(generic_covitem)

                            elif res == SolverStatus.UNSAT:
                                if BranchSolvingStrategy.UNSAT_ONCE in self.branch_strategy:
                                    self.uncoverable_items[covitem] = res
                                elif self.strategy in [CoverageStrategy.PATH, CoverageStrategy.PREFIXED_EDGE]:
                                    self.uncoverable_items[covitem] = res  # paths, and prefixed-edge ensure to be unique thus drop if unsat

                            elif res == SolverStatus.TIMEOUT:
                                if BranchSolvingStrategy.TIMEOUT_ONCE in self.branch_strategy:
                                    self.uncoverable_items[covitem] = res

                            elif res == SolverStatus.UNKNOWN:
                                pass

                            else: # status == None
                                logger.debug(f'Branch skipped!')

                            pending_csts = []  # reset pending constraint added

                    else:
                        pass  # Branch was taken do nothing

                # Add it the path predicate constraints and update current path hash
                pending_csts.append(pc)
                current_hash.update(struct.pack("<Q", pc.getTakenAddress()))

            else:
                cmt = pc.getComment()

                if (cmt.startswith("dyn-jmp") and BranchSolvingStrategy.COVER_SYM_DYNJUMP in self.branch_strategy) or \
                   (cmt.startswith("sym-read") and BranchSolvingStrategy.COVER_SYM_READ in self.branch_strategy) or \
                   (cmt.startswith("sym-write") and BranchSolvingStrategy.COVER_SYM_WRITE in self.branch_strategy):
                    typ, offset, addr = cmt.split(":")
                    typ = SymExType(typ)
                    offset, addr = int(offset), int(addr)
                    if addr not in self.covered_symbolic_pointers:  # if the address pointer has never been covered
                        pred = pc.getTakenPredicate()
                        if pred.getType() == AST_NODE.EQUAL:
                            p1, p2 = pred.getChildren()
                            if p2.getType() == AST_NODE.BV:
                                logger.info(f"Try to enumerate value {offset}:0x{addr:02x}: {p1}")
                                res = yield typ, pending_csts, p1, (addr, p2.evaluate()), i
                                self.covered_symbolic_pointers.add(addr)  # add the pointer in covered regardless of result
                            else:
                                logger.warning(f"memory constraint unexpected pattern: {pred}")
                        else:
                            logger.warning(f"memory constraint unexpected pattern: {pred}")

                    if BranchSolvingStrategy.SOUND_MEM_ACCESS in self.branch_strategy:
                        pending_csts.append(pc)  # if sound add the mem dereference as a constraint in path predicate
                        # NOTE: in both case the branch is not taken in account in the current_path_hash

                else:  # Routines, or user-defined constraints thus add it all the time.
                    pending_csts.append(pc)

    def _get_covitem(self, path_hash, branch: PathBranch) -> CovItem:
        src, dst = branch['srcAddr'], branch['dstAddr']

        # Check if the target is new with regards to the strategy
        if self.strategy == CoverageStrategy.BLOCK:
            return dst
        elif self.strategy == CoverageStrategy.EDGE:
            return src, dst
        elif self.strategy == CoverageStrategy.PATH:
            # Have to fork the hash of the current pc for each branch we want to revert
            forked_hash = path_hash.copy()
            forked_hash.update(struct.pack("<Q", dst))
            return forked_hash.hexdigest()
        elif self.strategy == CoverageStrategy.PREFIXED_EDGE:
            return path_hash.hexdigest(), (src, dst)
        else:
            assert False

    def _get_items_trace(self, path_constraints: List[PathConstraint]) -> Dict[CovItem, List[int]]:
        """
        Iterate the all trace and retrieve all covered and not covered CovItem. For non covered one
        it filter instances to check.
        """
        not_covered = {}
        current_hash = hashlib.md5()  # Current path hash for PATH coverage
        for i, pc in enumerate(path_constraints):
            if pc.isMultipleBranches():     # If there is a condition
                for branch in pc.getBranchConstraints():  # Get all branches
                    if not branch['isTaken']:
                        covitem = self._get_covitem(current_hash, branch)
                        if covitem in not_covered:
                            not_covered[covitem].append(i)
                        else:
                            not_covered[covitem] = [i]
                current_hash.update(struct.pack("<Q", pc.getTakenAddress()))  # compute current path hash along the way
            else:
                pass  # Ignore all other dynamic constraints in path computation

        # Now filter the map according to the branch solving strategy
        if BranchSolvingStrategy.FIRST_LAST_NOT_COVERED in self.branch_strategy:
            if self.strategy == CoverageStrategy.PREFIXED_EDGE:
                # Black magic
                m = {("", e): [] for h, e in not_covered.keys()}  # Map: ("", edge) -> []
                for (h, e), v in not_covered.items():          # fill map with all occurences edges regardless of path
                    m[("", e)].extend(v)
                for k in m.keys():                             # iterate the result and only keep min and max occurence
                    idxs = m[k]
                    if len(idxs) > 2:
                        m[k] = [min(idxs), max(idxs)]
                for k in not_covered.keys():                   # Push back resulting list in not_covered items
                    not_covered[k] = m[('', k[1])]

            else:  # Straightforward
                for k in not_covered.keys():
                    l = not_covered[k]
                    if len(l) > 2:
                        not_covered[k] = [l[0], l[-1]]  # Only keep first and last iteration
        else: # ALL_NOT_COVERED
            pass  # Keep all occurences
        return not_covered


    def merge(self, other: CoverageSingleRun) -> None:
        """
        Merge a CoverageSingeRun instance into this instance

        :param other: The CoverageSingleRun to merge into self
        :type other: CoverageSingleRun
        """
        assert self.strategy == other.strategy

        # Update instruction coverage for code coverage (in all cases keep code coverage)
        self.covered_instructions.update(other.covered_instructions)

        # Remove covered items from pending ones
        self.pending_coverage.difference_update(other.covered_items)

        # Update covered items
        self.covered_items.update(other.covered_items)

        # Update non-covered ones
        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            # More complex as not_covered as covitem: (hash, edge) while covered has covitems: ("", edge)

            # remove self not covered that are now covered
            for _, edge in other.covered_items.keys():
                if edge in self._not_covered_items_mirror:
                    for prefix in self._not_covered_items_mirror[edge]:  # iterate all the prefixes
                        self.not_covered_items.discard((prefix, edge))  # and discard them
                    self._not_covered_items_mirror.pop(edge)  # finally discard the entry

            # Only add other not covered if still not covered
            for prefix, edge in other.not_covered_items:
                if ("", edge) not in self.covered_items:
                    self.not_covered_items.add((prefix, edge))
                    if edge not in self._not_covered_items_mirror:
                        self._not_covered_items_mirror[edge] = [prefix]
                    else:
                        self._not_covered_items_mirror[edge].append(prefix)

        else: # Straightfoward set difference
            self.not_covered_items.update(other.not_covered_items - self.covered_items.keys())


    def can_improve_coverage(self, other: CoverageSingleRun) -> bool:
        """
        Check if some of the non-covered are not already in the global coverage
        Used to know if an input is relevant to keep or not

        :param other: The CoverageSingleRun to check against our global coverage state
        :return: bool
        """
        return bool(self.new_items_to_cover(other))


    def can_cover_symbolic_pointers(self, execution: 'SymbolicExecutor') -> bool:
        """
        Determines if this execution has symbolic memory accesses to enumerate. If so we may want
        to enumerate them even though 
        """
        path_constraints = execution.pstate.get_path_constraints()

        for pc in path_constraints:
            if not pc.isMultipleBranches():     # If there isn't a condition i.e it's a sym ptr access
                cmt = pc.getComment()

                if (cmt.startswith("dyn-jmp") and BranchSolvingStrategy.COVER_SYM_DYNJUMP in self.branch_strategy) or \
                   (cmt.startswith("sym-read") and BranchSolvingStrategy.COVER_SYM_READ in self.branch_strategy) or \
                   (cmt.startswith("sym-write") and BranchSolvingStrategy.COVER_SYM_WRITE in self.branch_strategy):
                    typ, offset, addr = cmt.split(":")
                    typ = SymExType(typ)
                    offset, addr = int(offset), int(addr)
                    if addr not in self.covered_symbolic_pointers:  # if the address pointer has never been covered
                        return True
        return False


    def new_items_to_cover(self, other: CoverageSingleRun) -> Set[CovItem]:
        """
        Return all coverage items (addreses, edges, paths) that the given CoverageSingleRun
        can cover if it is possible to negate their branches

        :param other: The CoverageSingleRun to check with our global coverage state
        :return: A set of CovItem
        """
        assert self.strategy == other.strategy
        # Take not covered_items (potential candidates) substract already covered items, uncoverable and pending ones.
        # Resulting covitem are really new ones that the trace brings
        return other.not_covered_items - self.covered_items.keys() - self.uncoverable_items.keys() - self.pending_coverage

    def improve_coverage(self, other: CoverageSingleRun) -> bool:
        """
        Checks if the given object do cover new covitem than the current
        coverage. More concretely it performs the difference between the
        two covered dicts. If ``other`` contains new items return True.

        :param other: coverage on which to check coverage
        :return: Whether the coverage covers new items
        """
        return bool(other.covered_items.keys() - self.covered_items.keys())

    @staticmethod
    def from_file(file: Union[str, Path]) -> 'GlobalCoverage':
        with open(file, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_file(self, file: Union[str, Path]) -> None:
        copy = self._current_path_hash
        self._current_path_hash = None
        with open(file, "wb") as f:
            pickle.dump(self, f)
        self._current_path_hash = copy


    def post_exploration(self, workspace: 'Workspace') -> None:
        """ Function called at the very end of the exploration.
        It saves the coverage in the workspace.

        :param workspace: Workspace in which to save coverage
        :type workspace: Workspace
        """
        # Save the coverage
        self.to_file(workspace.get_metadata_file_path(self.COVERAGE_FILE))

    def clone(self) -> 'GlobalCoverage':
        cov2 = GlobalCoverage(self.strategy, self.branch_strategy)

        # Copy items from the CoverageSingleRun
        cov2.covered_instructions = Counter({k: v for k, v in self.covered_instructions.items()})
        cov2.covered_items = Counter({k: v for k, v in self.covered_items.items()})
        cov2.not_covered_items = {x for x in self.not_covered_items}
        cov2._not_covered_items_mirror = {k: v for k, v in self._not_covered_items_mirror.items()}
        cov2._current_path = self._current_path[:]
        self._current_path: List[Addr] = []
        self._current_path_hash = self._current_path_hash.copy()

        # Copy items from the global coverage
        cov2.pending_coverage = {x for x in self.pending_coverage}
        cov2.uncoverable_items = {k: v for k, v in self.uncoverable_items.items()}
        cov2.covered_symbolic_pointers = {x for x in self.covered_symbolic_pointers}
        return cov2
