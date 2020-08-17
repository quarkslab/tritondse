#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
import os
import time

from tritondse.config      import Config
from tritondse.seed        import Seed, SeedFile
from tritondse.coverage    import Coverage
from tritondse.constraints import Constraints



class SeedsManager:
    """
    This class is used to represent the seeds management.
    """
    def __init__(self, config : Config, seed : Seed = Seed()):
        self.config         = config
        self.initial_seed   = seed
        self.coverage       = Coverage()
        self.constraints    = Constraints()
        self.worklist       = set()
        self.corpus         = set()
        self.crash          = set()

        self.__init_dirs()

        # Define the first seed
        self.worklist.add(self.initial_seed)


    def __load_seed_from_file(self, path):
        logging.debug('Loading %s' %(path))
        return SeedFile(path)


    def __init_dirs(self):
        # --------- Initialize METADATA --------------------------------------
        if not os.path.isdir(self.config.metadata_dir):
            logging.debug('Creating the %s directory' %(self.config.metadata_dir))
            os.mkdir(self.config.metadata_dir)
        else:
            logging.debug('Loading the existing metadata directory from %s' %(self.config.metadata_dir))

            # Loading coverage
            with open(f'{self.config.metadata_dir}/coverage', 'w+') as fd:
                coverage = fd.read()

            # Loading constraints
            with open(f'{self.config.metadata_dir}/constraints', 'w+') as fd:
                constraints = fd.read()

            # Eval Python objects
            if len(coverage):
                self.coverage.instructions = eval(coverage)

            if len(constraints):
                self.constraints.hashes = eval(constraints)


        # --------- Initialize WORKLIST --------------------------------------
        if not os.path.isdir(self.config.worklist_dir):
            logging.debug('Creating the %s directory' %(self.config.worklist_dir))
            os.mkdir(self.config.worklist_dir)
        else:
            logging.debug('Checking the existing worklist directory from %s' %(self.config.worklist_dir))
            for path in glob.glob('%s/*.cov' %(self.config.worklist_dir)):
                self.worklist.add(self.__load_seed_from_file(path))


        # --------- Initialize CORPUS ----------------------------------------
        if not os.path.isdir(self.config.corpus_dir):
            logging.debug('Creating the %s directory' %(self.config.corpus_dir))
            os.mkdir(self.config.corpus_dir)
        else:
            logging.debug('Checking the existing corpus directory from %s' %(self.config.corpus_dir))
            for path in glob.glob('%s/*.cov' %(self.config.corpus_dir)):
                self.corpus.add(self.__load_seed_from_file(path))


        # --------- Initialize CRASH -----------------------------------------
        if not os.path.isdir(self.config.crash_dir):
            logging.debug('Creating the %s directory' %(self.config.crash_dir))
            os.mkdir(self.config.crash_dir)
        else:
            logging.debug('Checking the existing crash directory from %s' %(self.config.crash_dir))
            for path in glob.glob('%s/*.cov' %(self.config.crash_dir)):
                self.crash.add(self.__load_seed_from_file(path))


    def __save_seed_on_disk(self, directory, seed):
        # Init the mangling
        name = f'{directory}/{seed.get_hash()}.%08x.tritondse.cov' %(seed.get_size())

        # Save it to the disk
        with open(name, 'wb+') as fd:
            fd.write(seed.content)

        return name


    def __remove_seed_on_disk(self, directory, seed):
        # Init the mangling
        name = f'{directory}/{seed.get_hash()}.%08x.tritondse.cov' %(seed.get_size())
        if os.path.exists(name):
            os.remove(name)


    def __save_metadata_on_disk(self):
        # Save coverage
        self.coverage.save_on_disk(self.config.metadata_dir)
        # Save constraints
        self.constraints.save_on_disk(self.config.metadata_dir)


    def __get_new_inputs(self, execution):
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

                        # Create the constraint
                        constraint = astCtxt.land([previousConstraints, branch['constraint']])

                        # Only ask for a model if the constraints has never been asked
                        if self.constraints.already_asked(constraint) == False:
                            ts = time.time()
                            model = execution.pstate.tt_ctx.getModel(constraint)
                            te = time.time()
                            logging.info('Query to the solver. Solving time: %f seconds' % (te - ts))

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
                                self.constraints.add_constraint(constraint)

            # Update the previous constraints with true branch to keep a good path.
            previousConstraints = astCtxt.land([previousConstraints, pc.getTakenPredicate()])

        return inputs


    def pick_seed(self):
        # TODO: Définir ici toutes les strategies de recherche.
        # Exemple:
        #   - dfs
        #   - bfs
        #   - rand
        #   - lifo
        #   - fifo
        return self.worklist.pop()


    def post_execution(self, execution, seed):
        # Update instructions covered
        self.coverage.merge(execution.coverage)

        # Add the seed to the current corpus
        self.corpus.add(seed)

        # Save the current seed into the corpus directory
        logging.info('Corpus dumped into %s' % (self.__save_seed_on_disk(self.config.corpus_dir, seed)))

        # Remove the seed from the worklist directory
        self.__remove_seed_on_disk(self.config.worklist_dir, seed)

        # Generate new inputs
        logging.info('Getting models, please wait...')
        inputs = self.__get_new_inputs(execution)
        for m in inputs:
            # Check if we already have executed this new seed
            if m not in self.corpus and m and m not in self.crash:
                self.worklist.add(m)
                logging.info('Seed dumped into %s' % (self.__save_seed_on_disk(self.config.worklist_dir, m)))

        # Save metadata on disk
        self.__save_metadata_on_disk()

        logging.info('Worklist size: %d' % (len(self.worklist)))
        logging.info('Corpus size: %d' % (len(self.corpus)))
        logging.info('Total of instructions covered: %d' % (self.coverage.number_of_instructions_covered()))
