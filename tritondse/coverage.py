# built-in imports
import os
from enum import IntEnum
import json
import logging
from collections import Counter
import hashlib
import struct
from typing import List, Generator, Tuple

# local imports
from tritondse.workspace import Workspace
from tritondse.types     import Addr, PathConstraint, PathBranch, Solver


class CoverageStrategy(IntEnum):
    CODE_COVERAGE = 0
    PATH_COVERAGE = 1
    EDGE_COVERAGE = 2


class CoverageSingleRun(object):
    """
    This class is used to represent the coverage of an execution.
    """

    def __init__(self, strategy: CoverageStrategy):
        self.strategy = strategy

        # For instruction coverage
        self.instructions = Counter()

        # For edge coverage
        self.edges = Counter()

        # For path coverage
        self.paths = set()
        self.current_path = []  # Hold all addresses currently forming the path taken
        self.current_path_hash = hashlib.md5()

    def add_covered_address(self, address: Addr):
        self.instructions[address] += 1

    def add_covered_branch(self, program_counter: Addr, pc: PathConstraint) -> None:
        if pc.isMultipleBranches():
            taken = pc.getTakenAddress()
            if self.strategy == CoverageStrategy.EDGE_COVERAGE:
                self.edges[(program_counter, taken)] += 1
            if self.strategy == CoverageStrategy.PATH_COVERAGE:
                self.current_path.append(taken)
                # Update the current path hash and add it to hashes
                self.current_path_hash.update(struct.pack("<Q", taken))
                self.paths.add(self.current_path_hash.hexdigest())
        else:
            pass  # otherwise, unconditional we are not interested

    @property
    def unique_instruction_covered(self) -> int:
        return len(self.instructions)

    @property
    def total_instruction_executed(self) -> int:
        return sum(self.instructions.values())


    def post_execution(self) -> None:
        pass


class GlobalCoverage(CoverageSingleRun):
    """
    This class is used to represent the coverage of an execution.
    """

    INSTRUCTION_COVERAGE_FILE = "instruction_coverage.json"
    PATH_COVERAGE_FILE = "path_coverage.json"

    def __init__(self, strategy: CoverageStrategy, workspace: Workspace):
        super().__init__(strategy)
        self.workspace = workspace

        # Load the coverage from the workspace (if it exists)
        self.load_coverage()

        # Keep pending items to be covered (code, edge, path)
        self.pending_coverage = set()


    def iter_new_paths(self, path_constraints: List[PathConstraint]) -> Generator[Tuple[List[PathConstraint], PathBranch, Addr], Solver, None]:
        """
        The function iterate the given path predicate and yield PatchConstraint to
        consider as-is and PathBranch representing the new branch to take. It acts
        as a black-box so that the SeedManager does not have to know what strategy
        is being used under the hood. From an impementation perspective the goal
        of the function is to manipulate the path WITHOUT doing any SMT related things.

        .. todo:: Need to implement strategies for a given target returning, all
           occurences, only the first, only the last etc. At the moment only the first.

        :param path_constraints: list of path constraint to iterate
        :return: generator of path constraint and branches to solve. The first tuple
        item is a list of PathConstraint to add in the path predicate and the second
        is the branch to solve (but not to keep in path predicate)
        """
        pending_csts = []
        current_hash = hashlib.md5()  # Current path hash for PATH coverage

        for pc in path_constraints:         # Iterate through all path constraints
            if pc.isMultipleBranches():     # If there is a condition
                for branch in pc.getBranchConstraints():  # Get all branches
                    # Get the constraint of the branch which has not been taken.
                    if not branch['isTaken']:
                        src, dst = branch['srcAddr'], branch['dstAddr']

                        # Check if the target is new with regards to the strategy
                        if self.strategy == CoverageStrategy.CODE_COVERAGE:
                            item = dst
                            new = item not in self.instructions and item not in self.pending_coverage

                        elif self.strategy == CoverageStrategy.EDGE_COVERAGE:
                            item = (src, dst)
                            new = item not in self.edges and item not in self.pending_coverage

                        elif self.strategy == CoverageStrategy.PATH_COVERAGE:
                            # Have to fork the hash of the current pc for each branch we want to revert
                            forked_hash = current_hash.copy()
                            forked_hash.update(struct.pack("<Q", dst))
                            item = forked_hash.hexdigest()
                            new = item not in self.paths and item not in self.pending_coverage
                        else:
                            assert False

                        # If the not taken branch is new wrt coverage
                        if new:
                            res = yield pending_csts, pc, dst
                            if res == Solver.SAT:  # If path was satisfiable add it to pending coverage
                                self.pending_coverage.add(item)

                            pending_csts = []  # reset pending constraint added

                    else:
                        pass  # Branch was taken do nothing
                # Add it the path preodicate constraints and update current path hash
                pending_csts.append(pc)
                current_hash.update(pc.getTakenAddress())
            else:
                pass   # RMQ: Do nothing on unconditional jumps?


    def merge(self, other: CoverageSingleRun) -> None:
        """ Merge an other instance of Coverage into this instance"""
        assert self.strategy == other.strategy

        # Update instruction coverage for code coverage (in all cases keep code coverage)
        self.instructions.update(other.instructions)

        # Update pending
        if self.strategy == CoverageStrategy.CODE_COVERAGE:
            self.pending_coverage.difference_update(other.instructions)

        # Update instruction coverage for edge
        if self.strategy == CoverageStrategy.EDGE_COVERAGE:
            self.edges.update(other.edges)
            self.pending_coverage.difference_update(other.edges)

        # Update instruction coverage for path constraints
        if self.strategy == CoverageStrategy.PATH_COVERAGE:
            self.paths.update(other.paths)
            self.pending_coverage.difference_update(other.paths)


    def save_coverage(self) -> None:
        """ Save the coverage in the workspace"""
        # Save instruction coverage
        if self.instructions:
            self.workspace.save_metadata_file(self.INSTRUCTION_COVERAGE_FILE, json.dumps(self.instructions, indent=2))

        # Save path coverage
        if self.paths:
            self.workspace.save_metadata_file(self.PATH_COVERAGE_FILE, json.dumps(list(self.paths)))


    def load_coverage(self) -> None:
        """ Load the coverage from the workspace """
        # Load instruction coverage
        data = self.workspace.get_metadata_file(self.INSTRUCTION_COVERAGE_FILE)
        if data:
            logging.debug(f"Loading the existing instruction coverage from: {self.INSTRUCTION_COVERAGE_FILE}")
            self.instructions = json.loads(data)

        # Load path coverage
        data = self.workspace.get_metadata_file(self.PATH_COVERAGE_FILE)
        if data:
            logging.debug(f"Loading the existing path coverage from: {self.PATH_COVERAGE_FILE}")
            self.paths = set(json.loads(data))


    def post_exploration(self) -> None:
        """ Function called at the very end of the exploration """
        self.save_coverage()
