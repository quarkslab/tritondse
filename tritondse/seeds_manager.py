# built-in imports
import logging
import time
from typing  import Generator


# local imports
from tritondse.config    import Config
from tritondse.seed      import Seed, SeedStatus
from tritondse.callbacks import CallbackManager
from tritondse.coverage  import GlobalCoverage
from tritondse.worklist  import WorklistAddressToSet
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
        self.worklist         = WorklistAddressToSet(config, self.coverage) # TODO: Use the appropriate worklist according to config and the strategy wanted
        self.cbm              = callbacks
        self.corpus           = set()
        self.crash            = set()
        self.hangs            = set()

        self.__load_seed_workspace()


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


    def post_execution(self, execution: SymbolicExecutor, seed: Seed) -> None:
        # Update instructions covered from the last execution into our exploration coverage
        self.coverage.merge(execution.coverage)

        # Add the seed to the appropriate list
        if seed.status == SeedStatus.NEW:
            logging.error(f"seed not meant to be NEW at the end of execution ({seed.filename})")
        elif seed.status == SeedStatus.OK_DONE:
            self.corpus.add(seed)
        elif seed.status == SeedStatus.HANG:
            self.hangs.add(seed)
        elif seed.status == SeedStatus.CRASH:
            self.crash.add(seed)
        else:
            assert False

        # Move the current seed into the right directory (and remove it from worklist)
        self.workspace.update_seed_location(seed)
        logging.info(f'Seed {seed.filename} dumped [{seed.status.name}]')

        # Generate new inputs
        logging.info('Getting models, please wait...')
        for new_input in self.__iter_new_inputs(execution):
            # Check if we already have executed this new seed
            if self.is_new_seed(new_input):
                self.worklist.add(new_input)
                self.workspace.save_seed(new_input)
                logging.info(f'New seed model {new_input.filename} dumped [{new_input.status.name}]')
            else:
                logging.debug(f"New seed {new_input.filename} has already been generated")

        logging.info('Worklist size: %d' % (len(self.worklist)))
        logging.info('Corpus size: %d' % (len(self.corpus)))
        logging.info(f'Unique instructions covered: {self.coverage.unique_instruction_covered}')


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
                p_prefix, branch, dst_addr = path_generator.send(status)

                # Add path_prefix in path predicate
                path_predicate.extend(x.getTakenPredicate() for x in p_prefix)

                # Create the constraint
                constraint = actx.land(path_predicate + [branch['constraint']])

                # Solve the constraint
                ts = time.time()
                model, status = execution.pstate.tt_ctx.getModel(constraint, status=True)
                status = Solver(status)
                smt_queries += 1
                logging.info(f'solver query n°{smt_queries}, time: {time.time() - ts:.02f} seconds. Status: {status.name}')

                if status == Solver.SAT:
                    new_seed = self._mk_new_seed(execution, execution.seed, model)
                    # Trick to keep track of which target a seed is meant to cover
                    new_seed.target_addr = dst_addr
                    yield new_seed  # Yield the seed to get it added in the worklist
                elif status == Solver.UNSAT:
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
                if self.config.smt_queries_limit and smt_queries >= self.config.smt_queries_limit:  # FIXME: breaking if smt_queries_limit is negative
                    break
        except StopIteration:  # We have iterated the whole path generator
            pass


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


    def add_seed(self, seed):
        if seed is not None and seed not in self.corpus and seed not in self.crash:
            self.worklist.add(seed)
            self.workspace.save_seed(seed)
            logging.info(f'Seed {seed.filename} dumped [{seed.status.name}]')


    def post_exploration(self):
        # Do things you would do at the very end of exploration
        # (or just before it becomes idle)
        pass
