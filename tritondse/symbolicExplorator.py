#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import os
import glob
import threading
import copy

from triton                     import *
from tritondse.config           import Config
from tritondse.coverage         import Coverage
from tritondse.processState     import ProcessState
from tritondse.program          import Program
from tritondse.seed             import Seed, SeedFile
from tritondse.enums            import Enums
from tritondse.symbolicExecutor import SymbolicExecutor


class SymbolicExplorator(object):
    """
    This class is used to represent the symbolic exploration.
    """
    def __init__(self, config : Config, program : Program, seed : Seed = Seed()):
        self.program            = program
        self.config             = config
        self.initial_seed       = seed
        self.coverage           = Coverage()
        self.worklist           = set()
        self.corpus             = set()
        self.crash              = set()
        self.stop               = False
        self.total_exec         = 0
        self.ts                 = time.time()
        self.constraints_asked  = set()

        self.__init_dirs__()

        # Define the first seed
        self.worklist.add(self.initial_seed)


    def __load_seed_from_file__(self, path):
        logging.debug('Loading %s' %(path))
        return SeedFile(path)


    def __init_dirs__(self):
        # --------- Initialize METADATA --------------------------------------
        if not os.path.isdir(self.config.metadata_dir):
            logging.debug('Creating the %s directory' %(self.config.metadata_dir))
            os.mkdir(self.config.metadata_dir)
        else:
            logging.debug('Loading the existing metadata directory from %s' %(self.config.metadata_dir))
            # Loading coverage
            with open(f'{self.config.metadata_dir}/coverage', 'w+') as fd:
                self.coverage.instructions = fd.read()
            # Loading constraints
            with open(f'{self.config.metadata_dir}/constraints', 'w+') as fd:
                self.constraints_asked = fd.read()

        # --------- Initialize WORKLIST --------------------------------------
        if not os.path.isdir(self.config.worklist_dir):
            logging.debug('Creating the %s directory' %(self.config.worklist_dir))
            os.mkdir(self.config.worklist_dir)
        else:
            logging.debug('Checking the existing worklist directory from %s' %(self.config.worklist_dir))
            for path in glob.glob('%s/*.cov' %(self.config.worklist_dir)):
                self.worklist.add(self.__load_seed_from_file__(path))

        # --------- Initialize CORPUS ----------------------------------------
        if not os.path.isdir(self.config.corpus_dir):
            logging.debug('Creating the %s directory' %(self.config.corpus_dir))
            os.mkdir(self.config.corpus_dir)
        else:
            logging.debug('Checking the existing corpus directory from %s' %(self.config.corpus_dir))
            for path in glob.glob('%s/*.cov' %(self.config.corpus_dir)):
                self.corpus.add(self.__load_seed_from_file__(path))

        # --------- Initialize CRASH -----------------------------------------
        if not os.path.isdir(self.config.crash_dir):
            logging.debug('Creating the %s directory' %(self.config.crash_dir))
            os.mkdir(self.config.crash_dir)
        else:
            logging.debug('Checking the existing crash directory from %s' %(self.config.crash_dir))
            for path in glob.glob('%s/*.cov' %(self.config.crash_dir)):
                self.crash.add(self.__load_seed_from_file__(path))


    def __time_delta__(self):
        return time.time() - self.ts


    def __pick_seed__(self):
        # TODO: Définir ici toutes les strategies de recherche.
        # Exemple:
        #   - dfs
        #   - bfs
        #   - rand
        #   - lifo
        #   - fifo
        return self.worklist.pop()


    def __save_seed_on_disk__(self, directory, seed):
        # Init the mangling
        name = f'{directory}/{seed.get_hash()}.00000000.tritondse.cov'

        # Save it to the disk
        with open(name, 'wb+') as fd:
            fd.write(seed.content)

        # Add the seed to the current corpus
        self.corpus.add(seed)

        return name


    def __save_metadata_on_disk__(self):
        # Save coverage
        self.coverage.save_on_disk(self.config.metadata_dir)
        # Save constraints
        with open(f'{self.config.metadata_dir}/constraints', 'w+') as fd:
            fd.write(repr(self.constraints_asked))


    def __get_new_input__(self, execution):
        # Set of new inputs
        inputs = list()

        # Get path constraints from the last execution
        pco = execution.pstate.tt_ctx.getPathConstraints()

        # Get the astContext
        astCtxt = execution.pstate.tt_ctx.getAstContext()

        # We start with any input. T (Top)
        previousConstraints = astCtxt.equal(astCtxt.bvtrue(), astCtxt.bvtrue())

        # Go through the path constraints
        for pc in pco:

            # If there is a condition
            if pc.isMultipleBranches():

                # Get all branches
                branches = pc.getBranchConstraints()
                for branch in branches:

                    # Get the constraint of the branch which has not been taken.
                    if not branch['isTaken']:

                        # Check timeout
                        if self.config.exploration_timeout and self.__time_delta__() >= self.config.exploration_timeout:
                            return inputs

                        # Create the constraint
                        constraint = astCtxt.land([previousConstraints, branch['constraint']])

                        # Only ask for a model if the constraints has never been asked
                        if constraint.getHash() not in self.constraints_asked:
                            ts = time.time()
                            model = execution.pstate.tt_ctx.getModel(constraint)
                            te = time.time()
                            logging.info('Query to the solver (new path). Solving time: %f seconds' % (te - ts))

                            if model:
                                # TODO Note: branch[] contains information that can help the SeedManager to classify its
                                # seeds insertion:
                                #
                                #   - branch['srcAddr'] -> The location of the branch instruction
                                #   - branch['dstAddr'] -> The destination of the jump if and only if branch['isTaken'] is True.
                                #                          Otherwise, the destination address is the next linear instruction.

                                # The seed size is the number of symbolic variables.
                                symvars = execution.pstate.tt_ctx.getSymbolicVariables()
                                content = bytearray(len(symvars))
                                # For each byte of the seed, we assign the value provided by the solver.
                                # If the solver provide no model for some bytes of the seed, their value
                                # stay unmodified.
                                for k, v in model.items():
                                    content[k] = v.getValue()
                                # Create the Seed object and assign the new model
                                seed = Seed(bytes(content))
                                inputs.append(seed)
                                # Save the constraint as already asked.
                                self.constraints_asked.add(constraint.getHash())

            # Update the previous constraints with true branch to keep a good path.
            previousConstraints = astCtxt.land([previousConstraints, pc.getTakenPredicate()])

        return inputs


    def worker(self, seed):
        """ Worker thread """
        # Inc number of trace execution
        self.total_exec += 1

        if self.config.exploration_timeout and self.__time_delta__() >= self.config.exploration_timeout:
            logging.info('Exploration timout')
            self.stop = True
            return

        # Execute the binary with seeds
        execution = SymbolicExecutor(self.config, ProcessState(self.config), self.program, seed)
        execution.run()

        # Update instructions covered
        self.coverage.merge(execution.coverage)

        # Save the current seed into the corpus directory
        logging.info('Corpus dumped into %s' % (self.__save_seed_on_disk__(self.config.corpus_dir, seed)))

        # Generate new inputs
        logging.info('Getting models, please wait...')
        inputs = self.__get_new_input__(execution)
        for m in inputs:
            if m not in self.corpus and m and m not in self.crash:
                self.worklist.add(m)

        # Save metadata on disk
        self.__save_metadata_on_disk__()

        logging.info('Worklist size: %d' % (len(self.worklist)))
        logging.info('Corpus size: %d' % (len(self.corpus)))
        logging.info('Total time: %f seconds' % (self.__time_delta__()))


    def explore(self):
        while self.worklist and not self.stop:
            # Take an input
            seed = self.__pick_seed__()

            # Execution into a thread
            t = threading.Thread(
                    name='\033[0;%dm[exec:%08d]' % ((31 + (self.total_exec % 4)), self.total_exec),
                    target=self.worker,
                    args=[seed],
                    daemon=True
                )
            t.start()
            try:
                t.join()
            except KeyboardInterrupt:
                logging.warning("keyboard interrupt, stop symbolic exploration")
                self.stop = True

        # Whatever happens, save worklist on disk
        for seed in self.worklist:
            logging.info('Save seed from the worklist into %s' % (self.__save_seed_on_disk__(self.config.worklist_dir, seed)))
