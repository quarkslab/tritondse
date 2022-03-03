# built-in imports
import logging
import time
import json
from typing import Generator, Optional
from collections import Counter

# local imports
from tritondse.seed              import Seed, SeedStatus
from tritondse.callbacks         import CallbackManager
from tritondse.coverage          import GlobalCoverage, CovItem
from tritondse.worklist          import WorklistAddressToSet, FreshSeedPrioritizerWorklist, SeedScheduler
from tritondse.workspace         import Workspace
from tritondse.symbolic_executor import SymbolicExecutor
from tritondse.types             import SolverStatus, SymExType



class SeedManager:
    """
    Seed Manager.
    This class is in charge of providing the next seed to execute by prioritizing
    them. It also holds various sets of pending seeds, corpus, crashes etc and
    manages them in the workspace.

    It contains basically 2 types of seeds which are:

    * pending seeds (kept in the seed scheduler). These are the seeds that might
      be selected to be run
    * seed consumed (corpus, crash, hangs) which are seeds not meant to be re-executed
      as they cannot lead to new paths, all candidate paths are UNSAT etc.
    """
    def __init__(self, coverage: GlobalCoverage, workspace: Workspace, smt_queries_limit: int, scheduler: SeedScheduler = None, callback_manager: CallbackManager = None):
        """
        :param coverage: global coverage object. The instance will be updated by the seed manager
        :type coverage: GlobalCoverage
        :param workspace: workspace instance object.
        :type workspace: Workspace
        :param smt_queries_limit: maximum number of queries for  a given execution
        :type smt_queries_limit: int
        :param scheduler: seed scheduler object to use as scheduling strategy
        :type scheduler: SeedScheduler
        """
        self.smt_queries_limit = smt_queries_limit
        self.workspace = workspace
        self.coverage  = coverage
        if scheduler is None:
            self.worklist = FreshSeedPrioritizerWorklist(self)
        else:
            self.worklist = scheduler
        self.cbm = callback_manager

        self.corpus = set()
        self.crash  = set()
        self.hangs  = set()

        self.__load_seed_workspace()

        self._solv_count = 0
        self._solv_time_sum = 0
        self._solv_status = {SolverStatus.SAT: 0, SolverStatus.UNSAT: 0, SolverStatus.UNKNOWN: 0, SolverStatus.TIMEOUT: 0}
        self._stat_branch_reverted = Counter()
        self._stat_branch_fail = Counter()
        self._current_solv_time = 0

    def __load_seed_workspace(self):
        """ Load seed from the workspace """
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
        """
        Check if a seed is a new one (not into corpus, crash and hangs)

        :param seed: The seed to test
        :type seed: Seed
        :return: True if the seed is a new one

        .. warning:: That function does not check that the seed
                     is not in the pending seeds queue
        """
        return sum(seed in x for x in [self.corpus, self.crash, self.hangs]) == 0


    def add_seed_queue(self, seed: Seed) -> None:
        """
        Add a seed to to appropriate internal queue depending
        on its status. If it is new it is added in pending seed,
        if OK, HANG or CRASH it the appropriate set.
        **Note that the seed is not written in the workspace**

        :param seed: Seed to add in an internal queue
        :type seed: Seed
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


    def post_execution(self, execution: SymbolicExecutor, seed: Seed, solve_new_path: int = True) -> float:
        """
        Function called after each execution. It updates the global
        code coverage object, and tries to generate new paths through
        SMT in accordance with the seed scheduling strategy.

        :param execution: The current execution
        :type execution: SymbolicExecutor
        :param seed: The seed of the execution
        :type seed: Seed
        :param solve_new_path: Whether or not to solve constraint to find new paths
        :type solve_new_path: bool
        :return: Total SMT solving time
        """

        # Update instructions covered from the last execution into our exploration coverage
        self.coverage.merge(execution.coverage)
        self.worklist.update_worklist(execution.coverage)

        # if the seed have target checks that we covered it
        if seed.target:
            color = ("YES", 92) if execution.coverage.is_covered(seed.target) else ("NO", 91)
            logging.info(f"Seed covered its target: \033[{color[1]}m{color[0]}\033[0m")

        # reset the current solving time
        self._current_solv_time = 0

        # Iterate all pending seeds to be added in the right location
        for s in execution.pending_seeds:
            if self.is_new_seed(s):
                if not s.coverage_objectives:      # If they don't have objectives set the Ellipsis wildcard
                    s.coverage_objectives.add(...)
                self._add_seed(s)  # will add the seed in both internal queues & workspace
            else:
                logging.warning(f"dropping enqueued seed: {s.hash} (already seen)")

        # Update the current seed queue
        if seed.status == SeedStatus.NEW:
            logging.error(f"seed not meant to be NEW at the end of execution ({seed.hash}) (dropped)")
            self.drop_seed(seed)

        elif seed.status in [SeedStatus.HANG, SeedStatus.CRASH]:
            self.archive_seed(seed)
            # NOTE: Do not perform further processing on the seed (like generating inputs from it)

        elif seed.status == SeedStatus.OK_DONE:
            if self.coverage.can_improve_coverage(execution.coverage):
                items = self.coverage.new_items_to_cover(execution.coverage)
                seed.coverage_objectives = items  # Set its new objectives

                if self.worklist.can_solve_models() and solve_new_path:     # No fresh seeds pending thus can solve model
                    logging.info(f'Seed {seed.hash} generate new coverage')
                    self._generate_new_inputs(execution)
                    self.archive_seed(seed)
                else:
                    logging.info(f"Seed {seed.hash} push back in worklist (to unstack fresh)")
                    seed.status = SeedStatus.NEW  # Reset its status for later run
                    self.add_seed_queue(seed)  # will be pushed back in worklist
            else:
                self.archive_seed(seed)
                logging.info(f'Seed {seed.hash} archived cannot generate new coverage [{seed.status.name}]')

        else:
            assert False

        logging.info(f"Corpus:{len(self.corpus)} Crash:{len(self.crash)}")
        self.worklist.post_execution()
        logging.info(f"Coverage instruction:{self.coverage.unique_instruction_covered} covitem:{self.coverage.unique_covitem_covered}")
        return self._current_solv_time

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
        actx = execution.pstate.actx

        # We start with any input. T (Top)
        path_predicate = [actx.equal(actx.bvtrue(), actx.bvtrue())]

        # Define a limit of branch constraints
        smt_queries = 0

        # Solver status
        status = None
        path_constraints = execution.pstate.get_path_constraints()
        total_len = len(path_constraints)
        path_generator = self.coverage.iter_new_paths(path_constraints)

        try:
            while True:
                # If smt_queries_limit is zero: unlimited queries
                # If smt_queries_limit is negative: no query
                if self.smt_queries_limit < 0:
                    logging.info(f'The configuration is defined as: no query')
                    break

                typ, p_prefix, branch, covitem, ith = path_generator.send(status)

                # Create edge in case of conditional branch, for all the other the edge shall be already set
                edge = (branch['srcAddr'], branch['dstAddr']) if typ == SymExType.CONDITIONAL_JMP else covitem

                # Call on_branch_solving, if one replies False does not solve the branch
                cb_result = all(cb(execution, execution.pstate, edge, typ) for cb in self.cbm.get_on_solving_callback())

                # Add path_prefix in path predicate (regardless on whether we solve the item or not)
                path_predicate.extend(x.getTakenPredicate() for x in p_prefix)

                # Skip processing the current path in case the result of the
                # callbacks return False.
                if not cb_result:
                    status = None
                    continue

                # Create the constraint
                if typ in [SymExType.DYNAMIC_JMP, SymExType.SYMBOLIC_READ, SymExType.SYMBOLIC_WRITE]:
                    expr, (addr, tgt) = branch, covitem   # branch and covitem have a different meaning here
                    ts = time.time()
                    results = execution.pstate.solve_enumerate_expression(expr, path_predicate, [tgt], execution.config.smt_enumeration_limit)  # enumerate values

                    # all stats updates
                    solve_time = time.time() - ts
                    count = len(results)
                    self._solv_count += count
                    self._solv_time_sum += solve_time
                    status = SolverStatus.SAT if count else SolverStatus.UNSAT
                    self._solv_status[status] += count

                    results = [(x[0], (addr, x[1])) for x in results]   # extract results
                    logging.info(f'pc:{ith}/{total_len} | Query n°{smt_queries}-{smt_queries+count}, enumerate:{expr} (time: {solve_time:.02f}s) values:[{count}:{self._pp_smt_status(status)}]')
                    smt_queries += count+1  # for the unsat

                elif typ == SymExType.CONDITIONAL_JMP:
                    constraint = actx.land(path_predicate + [branch['constraint']])

                    # Solve the constraint
                    ts = time.time()
                    status, model = execution.pstate.solve(constraint, with_pp=False)  # Do not use path predicate as we are iterating it
                    solve_time = time.time() - ts
                    self._update_solve_stats(covitem, status, solve_time)
                    results = [(model, covitem)]
                    smt_queries += 1
                    logging.info(f'pc:{ith}/{total_len} | Query n°{smt_queries}, solve:{self.coverage.pp_item(covitem)} (time: {solve_time:.02f}s) [{self._pp_smt_status(status)}]')
                else:
                    assert False

                if status == SolverStatus.SAT:
                    for model, covitem in results:
                        new_seed = execution.mk_new_seed_from_model(model)
                        # Trick to keep track of which target a seed is meant to cover
                        new_seed.coverage_objectives.add(covitem)
                        new_seed.target = covitem if typ == SymExType.CONDITIONAL_JMP else None
                        yield new_seed  # Yield the seed to get it added in the worklist
                else:
                    pass

                # Check if we reached the limit of query
                if self.smt_queries_limit and smt_queries >= self.smt_queries_limit:
                    logging.info(f'Limit of query reached. Stop asking for models')
                    break

        except StopIteration:  # We have iterated the whole path generator
            pass

    def _update_solve_stats(self, covitem: CovItem, status: SolverStatus, solving_time: float):
        self._solv_count += 1
        self._solv_time_sum += solving_time
        self._current_solv_time += solving_time
        self._solv_status[status] += 1
        if status == SolverStatus.SAT:
            self._stat_branch_reverted[covitem] += 1  # Update stats
            if covitem in self._stat_branch_fail:
                self._stat_branch_fail.pop(covitem)
        elif status == SolverStatus.UNSAT:
            self._stat_branch_fail[covitem] += 1


    def pick_seed(self) -> Optional[Seed]:
        """
        Get the next seed to be executed by querying it
        in the seed scheduler.

        :returns: Seed to execute from the pending seeds
        :rtype: Seed
        """
        return self.worklist.pick()


    def seeds_available(self) -> bool:
        """
        Checks whether or not there is still pending seeds to process.

        :returns: True if seeds are still pending
        """
        return self.worklist.has_seed_remaining()


    def add_new_seed(self, seed: Seed) -> None:
        """
        Add the given seed in the manager.
        The function uses its type to know where to add the seed.

        :param seed: seed to add
        :type seed: Seed
        """
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
        """
        Drop a seed that is not of interest anymore.
        The function thus switch its status to ``OK_DONE``
        and move it in the corpus. *(the seed is not removed
        from the corpus)*

        :param seed: seed object to drop
        :type seed: Seed
        """
        self.archive_seed(seed)

    def archive_seed(self, seed: Seed, status: SeedStatus = None) -> None:
        """
        Send a seed in the corpus. As such, the seed
        is not meant to be used anymore (for finding
        new seeds).

        :param seed: seed object
        :type seed: Seed
        :param status: optional status to assign the seed
        :type status: SeedStatus
        """
        if status:
            seed.status = status
        self.add_seed_queue(seed)  # Will put it in the corpus
        self.workspace.update_seed_location(seed)  # Will put it in the corpus in files

    def post_exploration(self) -> None:
        """
        Function called at the end of exploration. It perform
        some stats printing, but would also perform any clean
        up tasks. *(not meant to be called by the user)*
        """
        # Do things you would do at the very end of exploration
        # (or just before it becomes idle)
        stats = {
            "total_solving_time": self._solv_time_sum,
            "total_solving_attempt": self._solv_count,
            "branch_reverted": {str(k): v for k, v in self._stat_branch_reverted.items()}, # convert covitem to str whatever it is
            "branch_not_solved": {str(k): v for k, v in self._stat_branch_fail.items()},  # convert covitem to str whatever it is
            "UNSAT": self._solv_status[SolverStatus.UNSAT],
            "SAT": self._solv_status[SolverStatus.SAT],
            "TIMEOUT": self._solv_status[SolverStatus.TIMEOUT]
        }
        self.workspace.save_metadata_file("solving_stats.json", json.dumps(stats, indent=2))
        logging.info(f"Branches reverted: {len(self._stat_branch_reverted)}  Branches still fail: {len(self._stat_branch_fail)}")
        self.worklist.post_exploration(self.workspace)


    def _pp_smt_status(self, status: SolverStatus):
        """ The pretty print function of the solver status """
        mapper = {SolverStatus.SAT: 92, SolverStatus.UNSAT: 91, SolverStatus.TIMEOUT: 93, SolverStatus.UNKNOWN: 95}
        return f"\033[{mapper[status]}m{status.name}\033[0m"
