import glob
import json
import os
import time

from ..Optimization import Tools
from ..Visualization import ParetoVisualizer
import numpy as np
from typing import List, Optional, Union
from pymoo.indicators.hv import HV

# Hypervolume reference point. Distance uses the actual constraint (1.5). Complexity uses
# one MORE than the actual constraint (15) -- complexity is an integer, so a mechanism sitting
# exactly on the constraint boundary would contribute zero hypervolume (and get dropped by
# plot_HV's strict "<" comparison entirely) if the reference point were 15.0 instead of 16.0.
DEFAULT_REF = np.array([1.5, 16.0])

# The smallest possible closed-loop, single-DOF planar linkage is a four-bar (the fixed crank
# plus one more link back to a second ground pivot) -- nothing below 4 joints is a mechanism
# in the sense this challenge is about. compute_F treats anything smaller as invalid, the same
# way it already treats an over-complexity mechanism as invalid via the DEFAULT_REF check.
MIN_COMPLEXITY = 4


def make_empty_submission():
    return {
        'Problem 1': [],
        'Problem 2': [],
        'Problem 3': [],
        'Problem 4': [],
        'Problem 5': [],
        'Problem 6': []
    }


class SubmissionTools(Tools):
    """A Tools evaluator specialized for this challenge problem: still callable exactly like
    Tools (and still needs .compile()) for raw distance evaluation, with extra methods layered
    on top for scoring and visualizing populations of *submission-format* mechanisms -- dicts
    with 'x0', 'edges', 'fixed_joints', and optionally 'target_idx' -- against the
    distance/complexity hypervolume convention this challenge problem is graded on."""

    def __init__(self, timesteps=200, max_size=20, scaled=False, device='cpu'):
        super().__init__(timesteps=timesteps, max_size=max_size, scaled=scaled, device=device)
        self.visualizer = ParetoVisualizer(timesteps=timesteps, max_size=max_size, scaled=scaled)

    def compute_F(self, population: List[dict], target_curve: np.ndarray) -> np.ndarray:
        """Distance and complexity for every mechanism in a population. Returns an (N, 2) array."""
        if len(population) == 0:
            return np.zeros((0, 2))

        x0s = [np.array(p['x0']) for p in population]
        edges = [np.array(p['edges']) for p in population]
        fixed_joints = [np.array(p['fixed_joints']) for p in population]
        target_idx = [p.get('target_idx', None) for p in population]

        # Complexity is simply the number of joints (nodes) in each mechanism.
        # x0 holds only the free joints (nodes 2+); add back the 2 fixed crank nodes.
        complexity = np.array([x0.shape[0] + 2 for x0 in x0s], dtype=float)

        # Below MIN_COMPLEXITY there's no closed-loop mechanism to solve for (e.g. a bare
        # motor with no free joints still "solves", tracing the crank tip's circle -- but
        # that's not a linkage this challenge is scoring). Above self.max_size, the solver's
        # preallocated arrays can't even hold the mechanism and raise. Skip solving either
        # case and just score them as unconditionally invalid (infinite distance), the same
        # treatment every other invalid mechanism already gets everywhere else in this class.
        in_range = np.logical_and(complexity >= MIN_COMPLEXITY, complexity <= self.max_size)

        if not np.all(in_range):
            n_too_small = int(np.sum(complexity < MIN_COMPLEXITY))
            n_too_large = int(np.sum(complexity > self.max_size))
            print(f"Warning: {n_too_small + n_too_large} mechanism(s) in this population have "
                  f"an out-of-range complexity ({n_too_small} below the {MIN_COMPLEXITY}-joint "
                  f"minimum, {n_too_large} above the {self.max_size}-joint solver limit) and "
                  f"will be scored as invalid (infinite distance).")

        distances = np.full(len(population), np.inf)
        if np.any(in_range):
            idx = np.where(in_range)[0]
            solved = np.array(self(x0s=[x0s[i] for i in idx],
                                   edges=[edges[i] for i in idx],
                                   fixed_joints=[fixed_joints[i] for i in idx],
                                   target_curve=target_curve,
                                   target_idx=[target_idx[i] for i in idx]))
            distances[idx] = solved

        return np.stack([distances, complexity], axis=1)

    def hypervolume(self, population: List[dict], target_curve: np.ndarray,
                     ref: np.ndarray = DEFAULT_REF) -> float:
        """The scored hypervolume for a population -- exactly what evaluate_submission computes
        per problem, but usable directly on any population (e.g. to compare intermediate results)."""
        F = self.compute_F(population, target_curve)
        if len(F) == 0:
            return 0.0
        valid = np.logical_and(F[:, 0] <= ref[0], F[:, 1] <= ref[1])
        if not np.any(valid):
            return 0.0
        return float(HV(ref)(F[valid]))

    def best_of_each_size(self, population: List[dict], target_curve: np.ndarray,
                           sizes=range(4, 16)) -> List[dict]:
        """Reduce a population to the submission convention: at most one mechanism per size
        (whichever has the lowest distance), for the given sizes."""
        if len(population) == 0:
            return []
        F = self.compute_F(population, target_curve)
        best = []
        for s in sizes:
            idx = np.where(F[:, 1] == s)[0]
            if len(idx) == 0:
                continue
            best.append(population[idx[np.argmin(F[idx, 0])]])
        return best

    def plot_population(self, population: List[dict], target_curve: np.ndarray,
                         ref: np.ndarray = DEFAULT_REF, title: Optional[str] = None, ax=None):
        """Distance/complexity Pareto front for one population."""
        F = self.compute_F(population, target_curve)
        ax = self.visualizer.plot_HV(F, ref=ref, objective_labels=['Distance', 'Complexity'], ax=ax)
        # Standardize the axes across every plot so fronts are directly comparable: complexity
        # never goes below 4 (the smallest allowed size), distance never below 0.
        ax.set_xlim(left=4)
        ax.set_ylim(bottom=0)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_population_detail(self, population: List[dict], target_curve: np.ndarray):
        """Visualize every Pareto-efficient mechanism in a population: its drawing, its traced
        curve vs. the target, and where it sits on the distance/complexity front."""
        F = self.compute_F(population, target_curve)
        self.visualizer.plot_pareto_efficient(
            F=F,
            population=population,
            target_curve=target_curve,
            objective_labels=['Distance', 'Complexity']
        )


def _mechanism_to_jsonable(mechanism: dict) -> dict:
    """Convert one mechanism's arrays to plain nested lists for JSON serialization."""
    out = {
        'x0': np.asarray(mechanism['x0']).tolist(),
        'edges': np.asarray(mechanism['edges']).tolist(),
        'fixed_joints': np.asarray(mechanism['fixed_joints']).tolist(),
    }
    target_idx = mechanism.get('target_idx')
    if target_idx is not None:
        out['target_idx'] = int(target_idx)
    return out


def _mechanism_from_jsonable(mechanism: dict) -> dict:
    """Inverse of _mechanism_to_jsonable: nested lists back to numpy arrays."""
    out = {
        'x0': np.array(mechanism['x0'], dtype=float),
        'edges': np.array(mechanism['edges'], dtype=int),
        'fixed_joints': np.array(mechanism['fixed_joints'], dtype=int),
    }
    target_idx = mechanism.get('target_idx')
    if target_idx is not None:
        out['target_idx'] = int(target_idx)
    return out


def submission_to_jsonable(submission: dict) -> dict:
    """A full submission dict, with every mechanism's arrays converted to plain lists."""
    return {problem: [_mechanism_to_jsonable(m) for m in population]
            for problem, population in submission.items()}


def submission_from_jsonable(data: dict) -> dict:
    """Inverse of submission_to_jsonable."""
    return {problem: [_mechanism_from_jsonable(m) for m in population]
            for problem, population in data.items()}


def save_submission(submission: dict, path: str) -> None:
    """Save a submission dict as JSON (mechanism arrays become plain nested lists)."""
    with open(path, 'w') as f:
        json.dump(submission_to_jsonable(submission), f)


def load_submission(path: str) -> dict:
    """Load a submission previously saved with save_submission -- or, for backward
    compatibility, an older submission saved with np.save (a .npy path)."""
    if path.endswith('.npy'):
        return np.load(path, allow_pickle=True).item()
    with open(path) as f:
        return submission_from_jsonable(json.load(f))


def save_egg_result(problem_num: int, population: List[dict], solutions_dir: str = 'solutions') -> str:
    """Save one egg's solved population to its own timestamped file in solutions_dir, in the
    single-problem convention consolidate_solutions expects. Returns the path saved to."""
    os.makedirs(solutions_dir, exist_ok=True)
    egg_submission = make_empty_submission()
    egg_submission[f'Problem {problem_num}'] = population

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(solutions_dir, f'egg_{problem_num}_{timestamp}.json')
    save_submission(egg_submission, save_path)
    return save_path


def consolidate_solutions(tools: 'SubmissionTools', target_curves: np.ndarray,
                           solutions_dir: str = 'solutions') -> dict:
    """Pool every file save_egg_result has saved (across every egg and every run/re-run) and
    reduce each egg's pool down to the submission convention: one mechanism per size, keeping
    whichever is best (lowest distance) at each size (via tools.best_of_each_size)."""
    final_submission = make_empty_submission()

    for problem_num in range(1, 7):
        files = (glob.glob(os.path.join(solutions_dir, f'egg_{problem_num}_*.json')) +
                 glob.glob(os.path.join(solutions_dir, f'egg_{problem_num}_*.npy')))  # .npy: older saves

        pooled = []
        for f in files:
            saved = load_submission(f)
            pooled.extend(saved[f'Problem {problem_num}'])

        final_submission[f'Problem {problem_num}'] = tools.best_of_each_size(
            pooled, target_curve=target_curves[problem_num - 1])

    return final_submission


def evaluate_submission(
    submission: Union[dict, str],
    target_curves: Union[np.ndarray, str] = 'target_curves.npy') -> float:

    optimization_tools = SubmissionTools(
        # timesteps=200 matches target_curves.npy's own resolution, and is a deliberate
        # speed/accuracy tradeoff, not a bug: both distance and validity (does the mechanism
        # lock at some crank angle?) are only ever checked at these 200 sampled angles. A
        # mechanism can therefore score as valid here while actually locking, or scoring
        # slightly differently, at a crank angle that falls between samples -- denser sampling
        # only shrinks how often this happens, it doesn't eliminate it, so we don't chase it
        # here. This is the same resolution students see in the starter notebook.
        timesteps=200,
        max_size=20,
        scaled=True, # scale-invariant distance: mechanism scale is standardized by the fixed crank, not by curve size
        device='cpu'
    )
    optimization_tools.compile()

    if isinstance(submission, str):
        submission = load_submission(submission)
    if isinstance(target_curves, str):
        target_curves = np.load(target_curves)

    scores = []
    for problem in range(6):
        problem_key = f'Problem {problem + 1}'
        population = []
        for i, item in enumerate(submission.get(problem_key, [])):
            if i >= 12:
                print(f"Warning: More than 12 designs submitted for {problem_key}. Only the first 12 will be evaluated.")
                break
            if 'x0' not in item or 'edges' not in item or 'fixed_joints' not in item:
                # Invalid entry, skip
                continue
            population.append(item)

        scores.append(optimization_tools.hypervolume(population, target_curves[problem]))

    return {'Overall Score': float(np.mean(scores)), 'Score Breakdown': {
        f'Problem {i + 1}': float(scores[i]) for i in range(6)
    }}
