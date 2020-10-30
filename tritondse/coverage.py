# built-in imports
import os
from enum import Enum, auto
import json
import logging

# local imports
from tritondse.workspace        import Workspace
from tritondse.path_constraints import PathConstraintsHash


class CoverageStrategy(Enum):
    CODE_COVERAGE = auto()
    PATH_COVERAGE = auto()
    EDGE_COVERAGE = auto()


class CoverageSingleRun(object):
    """
    This class is used to represent the coverage of an execution.
    """

    def __init__(self, strategy: CoverageStrategy):
        self.strategy = strategy

        # For instruction coverage
        self.instructions = dict()

        # For path coverage
        self.path_constraints = PathConstraintsHash()

        # TODO: Doing different things depending on coverage strategy


    def add_instruction(self, address: int, inc: int = 1):
        if address in self.instructions:
            self.instructions[address] += inc
        else:
            self.instructions[address] = inc


    def number_of_instructions_covered(self):
        return len(self.instructions)


    def number_of_instructions_executed(self):
        count = 0
        for k, v in self.instructions.items():
            count += v
        return count


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


    def merge(self, other: CoverageSingleRun):
        """ Merge an other instance of Coverage into this instance"""
        for k, v in other.instructions.items():
            self.add_instruction(k, v)


    def save_coverage(self) -> None:
        """ Save the coverage in the workspace"""
        # Save instruction coverage
        if self.instructions:
            self.workspace.save_metadata_file(self.INSTRUCTION_COVERAGE_FILE, json.dumps(self.instructions, indent=2))

        # Save path coverage
        if self.path_constraints.hashes:
            self.workspace.save_metadata_file(self.PATH_COVERAGE_FILE, json.dumps(list(self.path_constraints.hashes)))


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
            self.path_constraints.hashes = set(json.loads(data))


    def post_exploration(self) -> None:
        """ Function called at the very end of the exploration """
        self.save_coverage()
