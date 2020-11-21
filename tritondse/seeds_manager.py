# built-in imports
import logging
import time
from typing import Generator
from collections import Counter

# local imports
from tritondse.config    import Config
from tritondse.seed      import Seed, SeedStatus
from tritondse.callbacks import CallbackManager
from tritondse.coverage  import GlobalCoverage
from tritondse.worklist  import WorklistAddressToSet, FreshSeedPrioritizerWorklist
from tritondse.workspace import Workspace
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.types     import Solver, Model



class SeedManager:
    """
    This class is used to represent the seeds management.
    """
    def __init__(self, config: Config, callbacks: CallbackManager, coverage: GlobalCoverage, workspace: Workspace):
        self.config           = config
        self.workspace        = workspace
        self.coverage         = coverage
        # self.worklist         = WorklistAddressToSet(config, self.coverage) # TODO: Use the appropriate worklist according to config and the strategy wanted
        self.worklist         = FreshSeedPrioritizerWorklist(self)
        self.cbm              = callbacks
        self.corpus           = set()
        self.crash            = set()
        self.hangs            = set()

        self.__load_seed_workspace()

        self._stat_branch_reverted = Counter()
        self._stat_branch_fail = Counter()
        self._yolo_map = set()  # CovItem

    def __load_seed_workspace(self):
        # Load seed from the corpus
        for seed in self.workspace.iter_corpus():
            self.corpus.add(seed)
        # Load hangs
        for seed in self.workspace.iter_hangs():
            self.hangs.add(seed)
        # Load crashes
        for seed in self.workspace.iter_crashes():
            self.crash.add(seed)
        # Load worklist
        for seed in self.workspace.iter_worklist():
            self.worklist.add(seed)


    def is_new_seed(self, seed: Seed) -> bool:
        return sum(seed in x for x in [self.corpus, self.crash, self.hangs]) == 0


    def add_seed_queue(self, seed: Seed) -> None:
        """
        Add a seed to to appropriate internal queue depending
        on its status.
        :param seed: Seed to add in internal queue
        :return: None
        """
        # Add the seed to the appropriate list
        if seed.status == SeedStatus.NEW:
            self.worklist.add(seed)
        elif seed.status == SeedStatus.OK_DONE:
            self.corpus.add(seed)
        elif seed.status == SeedStatus.HANG:
            self.hangs.add(seed)
        elif seed.status == SeedStatus.CRASH:
            self.crash.add(seed)
        else:
            assert False

    # def post_execution(self, execution: SymbolicExecutor, seed: Seed) -> None:
    #     # Update instructions covered from the last execution into our exploration coverage
    #     self.coverage.merge(execution.coverage)
    #
    #     # Update the current seed queue
    #     if seed.status == SeedStatus.NEW:
    #         logging.error(f"seed not meant to be NEW at the end of execution ({seed.filename})")
    #     else:
    #         self.add_seed_queue(seed)
    #
    #     # Iterate all pending seeds to be added in the right location
    #     for s in execution.pending_seeds:
    #         self._add_seed(s)  # will add the seed in both internal queues & workspace
    #
    #     # Move the current seed into the right directory (and remove it from worklist)
    #     self.workspace.update_seed_location(seed)
    #     logging.info(f'Seed {seed.filename} dumped [{seed.status.name}]')
    #
    #     self._generate_new_inputs(execution)
    #
    #     logging.info('Worklist size: %d' % (len(self.worklist)))
    #     logging.info('Corpus size: %d' % (len(self.corpus)))
    #     logging.info(f'Unique instructions covered: {self.coverage.unique_instruction_covered}')


    def post_execution(self, execution: SymbolicExecutor, seed: Seed) -> None:
        # Update instructions covered from the last execution into our exploration coverage
        self.coverage.merge(execution.coverage)
        self.worklist.update_worklist(execution.coverage)

        # Iterate all pending seeds to be added in the right location
        for s in execution.pending_seeds:
            if not s.coverage_objectives:      # If they don't have objectives set the Ellipsis wildcard
                s.coverage_objectives.add(...)
            self._add_seed(s)  # will add the seed in both internal queues & workspace

        # Update the current seed queue
        if seed.status == SeedStatus.NEW:
            logging.error(f"seed not meant to be NEW at the end of execution ({seed.get_hash()}) (dropped)")
            self.drop_seed(seed)

        elif seed.status in [SeedStatus.HANG, SeedStatus.CRASH]:
            self.add_seed_queue(seed)
            # NOTE: Do not perform further processing on the seed (like generating inputs from it)

        elif seed.status == SeedStatus.OK_DONE:
            if self.coverage.can_improve_coverage(execution.coverage):
                items = self.coverage.new_items_to_cover(execution.coverage)
                seed.coverage_objectives =items  # Set its new objectives

                if self.worklist.can_solve_models():     # No fresh seeds pending thus can solve model
                    logging.info(f'Seed {seed.get_hash()} generate new coverage')
                    self._generate_new_inputs(execution)
                else:
                    logging.info(f"Seed {seed.get_hash()} push back in worklist (to unstack fresh)")
                    seed.status = SeedStatus.NEW  # Reset its status for later run

                self.add_seed_queue(seed)     # will be pushed in worklist, or corpus
            else:
                self.corpus.add(seed)
                # Move the current seed into the right directory (and remove it from worklist)
                self.workspace.update_seed_location(seed)
                logging.warning(f'Seed {seed.get_hash()} dumped cannot generate new coverage [{seed.status.name}]')

        else:
            assert False

        logging.info(f"Worklist: {len(self.worklist)} (fresh:{len(self.worklist.fresh)}) Corpus:{len(self.corpus)} Crash:{len(self.crash)}")
        logging.info(f"Coverage objectives:{len(self.worklist.worklist)} instruction: {self.coverage.unique_instruction_covered} edges:{self.coverage.unique_edge_covered}")


    def _generate_new_inputs(self, execution: SymbolicExecutor):
        # Generate new inputs
        for new_input in self.__iter_new_inputs(execution):
            # Check if we already have executed this new seed
            if self.is_new_seed(new_input):
                self.worklist.add(new_input)
                self.workspace.save_seed(new_input)
                logging.info(f'New seed model {new_input.filename} dumped [{new_input.status.name}]')
            else:
                logging.debug(f"New seed {new_input.filename} has already been generated")


    def __iter_new_inputs(self, execution: SymbolicExecutor) -> Generator[Seed, None, None]:
        # Get the astContext
        actx = execution.pstate.tt_ctx.getAstContext()

        # We start with any input. T (Top)
        path_predicate = [actx.equal(actx.bvtrue(), actx.bvtrue())]

        # Define a limit of branch constraints
        smt_queries = 0

        # Solver status
        status = None
        path_generator = self.coverage.iter_new_paths(execution.pstate.tt_ctx.getPathConstraints())

        try:
            while True:
                # If smt_queries_limit is zero: unlimited queries
                # If smt_queries_limit is negative: no query
                if self.config.smt_queries_limit < 0:
                    logging.info(f'The configuration is defined as: no query')
                    break

                p_prefix, branch, covitem = path_generator.send(status)

                # Add path_prefix in path predicate
                path_predicate.extend(x.getTakenPredicate() for x in p_prefix)

                # Yolo solve without path_predicate
                s = self._yolo_solve(execution, branch, covitem)
                if s:
                    yield s

                # Create the constraint
                constraint = actx.land(path_predicate + [branch['constraint']])

                # Solve the constraint
                ts = time.time()
                model, status = execution.pstate.tt_ctx.getModel(constraint, status=True)
                status = Solver(status)
                smt_queries += 1
                logging.info(f'Query n°{smt_queries}, solve:{self.coverage.pp_item(covitem)} (time: {time.time() - ts:.02f}s) [{self.pp_smt_status(status)}]')

                if status == Solver.SAT:
                    self._stat_branch_reverted[covitem] += 1  # Update stats
                    if covitem in self._stat_branch_fail:
                        self._stat_branch_fail.pop(covitem)

                    new_seed = self._mk_new_seed(execution, execution.seed, model)
                    # Trick to keep track of which target a seed is meant to cover
                    new_seed.coverage_objectives.add(covitem)
                    yield new_seed  # Yield the seed to get it added in the worklist
                elif status == Solver.UNSAT:
                    self._stat_branch_fail[covitem] += 1
                    pass
                elif status == Solver.TIMEOUT:
                    pass
                    # while status == Solver.TIMEOUT:
                    #     limit = int(len(constraint) / 2)
                    #     if limit < 1:
                    #         break
                    #     ts = time.time()
                    #     model, status = execution.pstate.tt_ctx.getModel(actx.land(constraint[limit:]), status=True)
                    #     te = time.time()
                    #     smt_queries += 1
                    #     logging.info(f'Sending query n°{smt_queries} to the solver. Solving time: {te - ts:.02f} seconds. Status: {status}')

                elif status == Solver.UNKNOWN:
                    pass
                else:
                    assert False

                # Check if we reached the limit of query
                if self.config.smt_queries_limit and smt_queries >= self.config.smt_queries_limit:
                    logging.info(f'Limit of query reached. Stop asking for models')
                    break

        except StopIteration:  # We have iterated the whole path generator
            pass


    def _yolo_solve(self, execution, branch, covitem):
        if covitem in self._yolo_map:
            return  # do not try to revert the branch if we already succeeded

        # Solve the constraint
        model, status = execution.pstate.tt_ctx.getModel(branch['constraint'], status=True)
        status = Solver(status)
        logging.info(f'Yolo query solve:{self.coverage.pp_item(covitem)} [{self.pp_smt_status(status)}]')

        if status == Solver.SAT:
            self._yolo_map.add(covitem)
            new_seed = self._mk_new_seed(execution, execution.seed, model)
            # Trick to keep track of which target a seed is meant to cover
            new_seed.coverage_objectives.add(covitem)
            return new_seed  # Yield the seed to get it added in the worklist
        else:
            return None


    def _mk_new_seed(self, exec: SymbolicExecutor, seed: Seed, model: Model) -> Seed:
        # Current content before getting model
        content = bytearray(seed.content)
        # For each byte of the seed, we assign the value provided by the solver.
        # If the solver provide no model for some bytes of the seed, their value
        # stay unmodified (with their current value).
        for k, v in sorted(model.items()):
            content[k] = v.getValue()

        # Calling callback if user defined one
        for cb in self.cbm.get_new_input_callback():
            cont = cb(exec, exec.pstate, content)
            # if the callback return a new input continue with that one
            content = cont if cont is not None else content

        # Create the Seed object and assign the new model
        return Seed(bytes(content))


    def pick_seed(self):
        return self.worklist.pick()


    def seeds_available(self) -> bool:
        return self.worklist.has_seed_remaining()


    def is_seed_new(self, seed: Seed) -> bool:
        """ Return True whether the seed is entirely new for the SeedManager or not """
        return seed is not None and seed not in self.corpus and seed not in self.crash and seed not in self.hangs


    def add_new_seed(self, seed: Seed) -> None:
        if self.is_new_seed(seed):
            self._add_seed(seed)
            logging.debug(f'Seed {seed.filename} dumped [{seed.status.name}]')
        else:
            logging.debug(f"seed {seed} is already known (not adding it)")


    def _add_seed(self, seed: Seed) -> None:
        """ Add the seed in both internal queues but also workspace """
        self.add_seed_queue(seed)
        self.workspace.save_seed(seed)


    def drop_seed(self, seed: Seed) -> None:
        logging.info(f"droping seed {seed.get_hash()} as it cannot improve coverage anymore")
        seed.status = SeedStatus.OK_DONE
        self.add_seed_queue(seed)  # Will put it in the corpus
        self.workspace.update_seed_location(seed)  # Will put it in the corpus in files


    def post_exploration(self):
        # Do things you would do at the very end of exploration
        # (or just before it becomes idle)
        count = sum(x for x in self._stat_branch_reverted.values())
        count_fail = sum(x for x in self._stat_branch_fail.values())
        logging.info(f"Branches reverted: {count} (unique: {len(self._stat_branch_reverted)})")
        logging.info(f"Branches still fail: {count_fail} (unique: {len(self._stat_branch_fail)})")
        self.worklist.post_exploration()


    def pp_smt_status(self, status: Solver):
        mapper = {Solver.SAT: 92, Solver.UNSAT: 91, Solver.TIMEOUT: 93, Solver.UNKNOWN: 95}
        return f"\033[{mapper[status]}m{status.name}\033[0m"
