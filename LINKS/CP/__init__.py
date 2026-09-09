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

        distances = np.array(self(x0s=x0s, edges=edges, fixed_joints=fixed_joints,
                                   target_curve=target_curve, target_idx=target_idx))

        # Complexity is simply the number of joints (nodes) in each mechanism.
        # x0 holds only the free joints (nodes 2+); add back the 2 fixed crank nodes.
        complexity = np.array([x0.shape[0] + 2 for x0 in x0s], dtype=float)

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


def evaluate_submission(
    submission: Union[dict, str],
    target_curves: Union[np.ndarray, str] = 'target_curves.npy') -> float:

    optimization_tools = SubmissionTools(
        timesteps=200,
        max_size=20,
        scaled=True, # scale-invariant distance: mechanism scale is standardized by the fixed crank, not by curve size
        device='cpu'
    )
    optimization_tools.compile()

    if isinstance(submission, str):
        submission = np.load(submission, allow_pickle=True).item()
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
