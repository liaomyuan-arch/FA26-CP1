"""Reproducible hybrid optimizer for the FA26 CP1 linkage challenge.

The instructor-provided LINKS package remains the source of truth for mechanism
generation, simulation, distance, gradients, validity, and scoring.  This file
only orchestrates those pieces and adds checkpointing, search, validation, and
reporting around them.

Typical usage (from the repository root):

    python cp1_optimizer.py quick
    python cp1_optimizer.py full
    python cp1_optimizer.py validate
    python cp1_optimizer.py visualize
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Match the authoritative notebook: native Windows uses CPU JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LINKS.CP import (
    SubmissionTools,
    evaluate_submission,
    load_submission,
    make_empty_submission,
    save_submission,
)
from LINKS.Geometry import CurveEngine
from LINKS.Kinematics import MechanismSolver
from LINKS.Optimization import DifferentiableTools, MechanismRandomizer


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ARCHIVE_PATH = RESULTS_DIR / "candidate_archive.json"
SUMMARY_PATH = RESULTS_DIR / "run_summary.json"
FINAL_PATH = ROOT / "my_submission.json"
TARGETS_PATH = ROOT / "target_curves.npy"
MIN_JOINTS = 4
MAX_JOINTS = 15
MAX_DISTANCE = 1.5
PROBLEM_KEYS = [f"Problem {i}" for i in range(1, 7)]


@dataclass(frozen=True)
class SearchConfig:
    name: str
    problem_numbers: tuple[int, ...]
    sizes: tuple[int, ...]
    random_topologies: int
    population_size: int
    generations: int
    gradient_steps: int
    gradient_lr: float
    base_seed: int
    perturb_scale: float


QUICK_CONFIG = SearchConfig(
    name="quick",
    problem_numbers=(1,),
    sizes=(4, 6, 8),
    random_topologies=1,
    population_size=8,
    generations=3,
    gradient_steps=20,
    gradient_lr=1e-2,
    base_seed=260901,
    perturb_scale=0.12,
)

FULL_CONFIG = SearchConfig(
    name="full",
    problem_numbers=(1, 2, 3, 4, 5, 6),
    sizes=tuple(range(MIN_JOINTS, MAX_JOINTS + 1)),
    random_topologies=2,
    population_size=16,
    generations=10,
    gradient_steps=100,
    gradient_lr=1e-2,
    base_seed=260926,
    perturb_scale=0.18,
)


@dataclass(frozen=True)
class TargetedConfig:
    problem_number: int = 2
    sizes: tuple[int, ...] = (5, 6, 7, 8)
    random_restarts_per_size: int = 32
    max_topology_groups: int = 20
    population_size: int = 48
    generations: int = 50
    gradient_finalists: int = 6
    gradient_steps: int = 250
    gradient_lr: float = 8e-3
    base_seeds: tuple[int, ...] = (271021, 271057, 271091, 271129)
    perturb_scale: float = 0.22


PROBLEM2_TARGETED_CONFIG = TargetedConfig()


def mechanism_to_jsonable(mechanism: dict[str, Any]) -> dict[str, Any]:
    out = {
        "x0": np.asarray(mechanism["x0"], dtype=float).tolist(),
        "edges": np.asarray(mechanism["edges"], dtype=int).tolist(),
        "fixed_joints": np.asarray(mechanism["fixed_joints"], dtype=int).tolist(),
    }
    target_idx = mechanism.get("target_idx")
    if target_idx is not None:
        out["target_idx"] = int(target_idx)
    return out


def mechanism_from_jsonable(mechanism: dict[str, Any]) -> dict[str, Any]:
    out = {
        "x0": np.asarray(mechanism["x0"], dtype=float),
        "edges": np.asarray(mechanism["edges"], dtype=int).reshape(-1, 2),
        "fixed_joints": np.asarray(mechanism["fixed_joints"], dtype=int).reshape(-1),
    }
    if mechanism.get("target_idx") is not None:
        out["target_idx"] = int(mechanism["target_idx"])
    return out


def normalized_mechanism(mechanism: dict[str, Any]) -> dict[str, Any]:
    out = mechanism_from_jsonable(mechanism_to_jsonable(mechanism))
    if out.get("target_idx") is None:
        out["target_idx"] = int(out["x0"].shape[0] + 1)
    return out


def mechanism_key(mechanism: dict[str, Any], include_positions: bool = True) -> str:
    normalized = mechanism_to_jsonable(normalized_mechanism(mechanism))
    if not include_positions:
        normalized.pop("x0")
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def complexity(mechanism: dict[str, Any]) -> int:
    return int(np.asarray(mechanism["x0"]).shape[0] + 2)


def clone_with_x(mechanism: dict[str, Any], x0: np.ndarray) -> dict[str, Any]:
    out = normalized_mechanism(mechanism)
    out["x0"] = np.asarray(x0, dtype=float).copy()
    return out


class CandidateArchive:
    """Best known mechanism for every (problem, complexity), saved immediately."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in PROBLEM_KEYS}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            for problem in PROBLEM_KEYS:
                self.records[problem] = raw.get("candidates", {}).get(problem, {})

    def seed_submission(self, submission_path: Path, tools: SubmissionTools, targets: np.ndarray) -> None:
        if not submission_path.exists():
            return
        submission = load_submission(str(submission_path))
        for problem_number, problem in enumerate(PROBLEM_KEYS, start=1):
            population = submission.get(problem, [])
            if not population:
                continue
            values = tools.compute_F(population, targets[problem_number - 1])
            for mech, (distance, joints) in zip(population, np.asarray(values)):
                if np.isfinite(distance):
                    self.consider(
                        problem_number,
                        mech,
                        float(distance),
                        method="existing submission",
                        seed=None,
                        save=False,
                    )
        self.save()

    def consider(
        self,
        problem_number: int,
        mechanism: dict[str, Any],
        distance: float,
        method: str,
        seed: int | None,
        save: bool = True,
    ) -> bool:
        problem = f"Problem {problem_number}"
        joints = complexity(mechanism)
        key = str(joints)
        previous = self.records[problem].get(key)
        if not np.isfinite(distance):
            return False
        if previous is not None and float(previous["distance"]) <= float(distance):
            return False
        self.records[problem][key] = {
            "distance": float(distance),
            "complexity": joints,
            "method": method,
            "seed": seed,
            "mechanism": mechanism_to_jsonable(normalized_mechanism(mechanism)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if save:
            self.save()
        return True

    def records_for_problem(self, problem_number: int) -> list[dict[str, Any]]:
        records = self.records[f"Problem {problem_number}"]
        return [records[key] for key in sorted(records, key=int)]

    def mechanisms_for_problem(self, problem_number: int) -> list[dict[str, Any]]:
        return [mechanism_from_jsonable(record["mechanism"]) for record in self.records_for_problem(problem_number)]

    def get(self, problem_number: int, joints: int) -> dict[str, Any] | None:
        record = self.records[f"Problem {problem_number}"].get(str(joints))
        return None if record is None else mechanism_from_jsonable(record["mechanism"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "description": "Best instructor-evaluated candidate at each problem/complexity.",
            "candidates": self.records,
        }
        temp = self.path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        temp.replace(self.path)


def evaluate_population(
    tools: SubmissionTools,
    mechanisms: list[dict[str, Any]],
    target_curve: np.ndarray,
) -> np.ndarray:
    if not mechanisms:
        return np.zeros(0, dtype=float)
    return np.asarray(
        tools(
            x0s=[np.asarray(m["x0"], dtype=float) for m in mechanisms],
            edges=[np.asarray(m["edges"], dtype=int) for m in mechanisms],
            fixed_joints=[np.asarray(m["fixed_joints"], dtype=int) for m in mechanisms],
            target_curve=target_curve,
            target_idx=[m.get("target_idx") for m in mechanisms],
        ),
        dtype=float,
    )


def moving_target_indices(mechanism: dict[str, Any]) -> list[int]:
    n = complexity(mechanism)
    fixed = set(np.asarray(mechanism["fixed_joints"], dtype=int).tolist())
    # Node 1 is the crank tip. Include it for completeness, although non-circular
    # targets will normally select a later moving joint.
    return [idx for idx in range(1, n) if idx not in fixed]


def select_target_joint(
    tools: SubmissionTools,
    mechanism: dict[str, Any],
    target_curve: np.ndarray,
) -> tuple[dict[str, Any], float]:
    candidates = []
    for target_idx in moving_target_indices(mechanism):
        candidate = normalized_mechanism(mechanism)
        candidate["target_idx"] = target_idx
        candidates.append(candidate)
    distances = evaluate_population(tools, candidates, target_curve)
    best = int(np.argmin(distances))
    return candidates[best], float(distances[best])


def seeded_evolution(
    tools: SubmissionTools,
    mechanism: dict[str, Any],
    target_curve: np.ndarray,
    rng: np.random.Generator,
    population_size: int,
    generations: int,
    perturb_scale: float,
) -> tuple[dict[str, Any], float]:
    """Elitist real-valued evolutionary search for one valid topology.

    This follows Demo_Optimization's GA idea, while keeping the topology fixed so
    offspring do not overwhelmingly violate the dyadic one-DOF constraint (the
    failure mode documented in Starter Notebook Sample Solution 2).
    """
    base = normalized_mechanism(mechanism)
    x_base = np.asarray(base["x0"], dtype=float)
    span = max(float(np.ptp(x_base, axis=0).max()), 0.5)

    positions = [x_base.copy()]
    while len(positions) < population_size:
        scale = perturb_scale * span * (0.35 + 0.9 * rng.random())
        positions.append(np.clip(x_base + rng.normal(0.0, scale, x_base.shape), -4.0, 4.0))

    def score(xs: list[np.ndarray]) -> np.ndarray:
        return evaluate_population(tools, [clone_with_x(base, x) for x in xs], target_curve)

    distances = score(positions)
    best_idx = int(np.argmin(distances))
    best_x = positions[best_idx].copy()
    best_d = float(distances[best_idx])

    for generation in range(generations):
        order = np.argsort(distances)
        elite_count = max(2, population_size // 4)
        elites = [positions[int(i)].copy() for i in order[:elite_count]]
        next_positions = elites.copy()
        decay = 1.0 - 0.75 * generation / max(generations, 1)
        sigma = perturb_scale * span * decay
        while len(next_positions) < population_size:
            parent_a = elites[int(rng.integers(0, len(elites)))]
            parent_b = elites[int(rng.integers(0, len(elites)))]
            blend = rng.uniform(-0.15, 1.15, size=(x_base.shape[0], 1))
            child = blend * parent_a + (1.0 - blend) * parent_b
            mask = rng.random(x_base.shape) < 0.35
            child = child + mask * rng.normal(0.0, sigma, x_base.shape)
            next_positions.append(np.clip(child, -4.0, 4.0))
        positions = next_positions
        distances = score(positions)
        generation_idx = int(np.argmin(distances))
        generation_d = float(distances[generation_idx])
        if generation_d < best_d:
            best_d = generation_d
            best_x = positions[generation_idx].copy()

    return clone_with_x(base, best_x), best_d


def seeded_evolution_multi(
    tools: SubmissionTools,
    mechanisms: list[dict[str, Any]],
    target_curve: np.ndarray,
    rng: np.random.Generator,
    population_size: int,
    generations: int,
    perturb_scale: float,
) -> tuple[dict[str, Any], float]:
    """Topology-preserving evolution from several valid position seeds.

    All inputs must share edges, fixed joints, and target index.  Offspring use a
    mixture of arithmetic crossover, differential mutation, coordinate crossover,
    and Gaussian mutation; connectivity is never mutated, so structural validity is
    preserved and only geometric locking needs to be rejected by the instructor tool.
    """
    if not mechanisms:
        raise ValueError("At least one seed mechanism is required")
    base = normalized_mechanism(mechanisms[0])
    seed_positions = [np.asarray(mech["x0"], dtype=float).copy() for mech in mechanisms]
    all_points = np.concatenate(seed_positions, axis=0)
    span = max(float(np.ptp(all_points, axis=0).max()), 0.5)

    positions = [position.copy() for position in seed_positions[:population_size]]
    while len(positions) < population_size:
        parent = seed_positions[int(rng.integers(0, len(seed_positions)))]
        scale = perturb_scale * span * (0.2 + 0.8 * rng.random())
        positions.append(np.clip(parent + rng.normal(0.0, scale, parent.shape), -4.0, 4.0))

    def score(xs: list[np.ndarray]) -> np.ndarray:
        return evaluate_population(tools, [clone_with_x(base, x) for x in xs], target_curve)

    distances = score(positions)
    best_idx = int(np.argmin(distances))
    best_x = positions[best_idx].copy()
    best_d = float(distances[best_idx])

    for generation in range(generations):
        order = np.argsort(distances)
        elite_count = max(4, population_size // 5)
        parent_count = max(elite_count, population_size // 2)
        elites = [positions[int(i)].copy() for i in order[:elite_count]]
        parents = [positions[int(i)].copy() for i in order[:parent_count]]
        next_positions = elites.copy()
        progress = generation / max(1, generations - 1)
        sigma = perturb_scale * span * (1.0 - 0.88 * progress)

        while len(next_positions) < population_size:
            operator = rng.random()
            if operator < 0.45 and len(parents) >= 3:
                # Differential mutation followed by binomial coordinate crossover.
                ids = rng.choice(len(parents), size=3, replace=False)
                a, b, c = (parents[int(i)] for i in ids)
                donor = a + rng.uniform(0.45, 0.9) * (b - c)
                anchor = parents[int(rng.integers(0, len(parents)))]
                mask = rng.random(anchor.shape) < rng.uniform(0.35, 0.8)
                mask[int(rng.integers(0, anchor.shape[0])), int(rng.integers(0, 2))] = True
                child = np.where(mask, donor, anchor)
            else:
                # Extrapolating arithmetic crossover between strong parents.
                parent_a = parents[int(rng.integers(0, len(parents)))]
                parent_b = parents[int(rng.integers(0, len(parents)))]
                blend = rng.uniform(-0.25, 1.25, size=(parent_a.shape[0], 1))
                child = blend * parent_a + (1.0 - blend) * parent_b

            mutation_mask = rng.random(child.shape) < (0.42 - 0.22 * progress)
            child = child + mutation_mask * rng.normal(0.0, sigma, child.shape)
            next_positions.append(np.clip(child, -4.0, 4.0))

        positions = next_positions
        distances = score(positions)
        generation_idx = int(np.argmin(distances))
        generation_d = float(distances[generation_idx])
        if generation_d < best_d:
            best_d = generation_d
            best_x = positions[generation_idx].copy()

    return clone_with_x(base, best_x), best_d


def gradient_refine(
    grad_tools: DifferentiableTools,
    mechanism: dict[str, Any],
    target_curve: np.ndarray,
    steps: int,
    initial_lr: float,
) -> tuple[dict[str, Any], float, int]:
    """Monotone Adam refinement using the instructor's exact gradient."""
    mech = normalized_mechanism(mechanism)
    x = np.asarray(mech["x0"], dtype=float).copy()
    edges = np.asarray(mech["edges"], dtype=int)
    fixed = np.asarray(mech["fixed_joints"], dtype=int)
    target_idx = int(mech["target_idx"])
    preprocessed = grad_tools.get_preprocessed(x, edges, fixed)

    distance, gradient = grad_tools(
        x0s=x,
        preprocessed=preprocessed,
        target_curve=target_curve,
        target_idx=target_idx,
    )
    distance = float(distance)
    best_x, best_d = x.copy(), distance
    moment = np.zeros_like(x)
    variance = np.zeros_like(x)
    lr = initial_lr
    accepted = 0

    for iteration in range(1, steps + 1):
        gradient = np.asarray(gradient, dtype=float)
        if not np.isfinite(distance) or not np.isfinite(gradient).all():
            break
        moment = 0.9 * moment + 0.1 * gradient
        variance = 0.999 * variance + 0.001 * np.square(gradient)
        m_hat = moment / (1.0 - 0.9**iteration)
        v_hat = variance / (1.0 - 0.999**iteration)
        direction = m_hat / (np.sqrt(v_hat) + 1e-8)

        accepted_step = False
        trial_lr = lr
        for _ in range(5):
            trial_x = np.clip(x - trial_lr * direction, -4.0, 4.0)
            trial_distance, trial_gradient = grad_tools(
                x0s=trial_x,
                preprocessed=preprocessed,
                target_curve=target_curve,
                target_idx=target_idx,
            )
            trial_distance = float(trial_distance)
            if np.isfinite(trial_distance) and trial_distance <= distance + 1e-9:
                x = trial_x
                distance = trial_distance
                gradient = trial_gradient
                lr = min(trial_lr * 1.03, initial_lr)
                accepted += 1
                accepted_step = True
                if distance < best_d:
                    best_x, best_d = x.copy(), distance
                break
            trial_lr *= 0.5
        if not accepted_step:
            lr *= 0.5
            moment *= 0.5
            variance *= 0.5
            if lr < 1e-6:
                break

    return clone_with_x(mech, best_x), best_d, accepted


def topology_library(
    archive: CandidateArchive,
    original_submission: dict[str, list[dict[str, Any]]],
    joints: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for problem_number in range(1, 7):
        archived = archive.get(problem_number, joints)
        if archived is not None:
            candidates.append(archived)
        for mech in original_submission.get(f"Problem {problem_number}", []):
            if complexity(mech) == joints:
                candidates.append(normalized_mechanism(mech))

    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(mechanism_key(candidate, include_positions=False), candidate)
    return list(unique.values())


def generate_random_mechanisms(
    randomizer: MechanismRandomizer,
    joints: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    np.random.seed(seed)
    random.seed(seed)
    generated = []
    attempts = 0
    while len(generated) < count and attempts < max(10, count * 8):
        attempts += 1
        try:
            generated.append(
                normalized_mechanism(
                    randomizer(
                        n=joints,
                        n_tests=16,
                        max_tries=30,
                        max_skeleton_tries=40,
                    )
                )
            )
        except RuntimeError:
            continue
    return generated


def truncate_mechanism(mechanism: dict[str, Any], target_joints: int) -> dict[str, Any] | None:
    """Truncate an incrementally numbered dyadic mechanism to a smaller prefix."""
    source = normalized_mechanism(mechanism)
    if complexity(source) < target_joints or target_joints < MIN_JOINTS:
        return None
    edges = np.asarray(source["edges"], dtype=int)
    keep_edges = edges[np.logical_and(edges[:, 0] < target_joints, edges[:, 1] < target_joints)]
    fixed = np.asarray(source["fixed_joints"], dtype=int)
    keep_fixed = fixed[fixed < target_joints]
    return {
        "x0": np.asarray(source["x0"], dtype=float)[: target_joints - 2].copy(),
        "edges": keep_edges.copy(),
        "fixed_joints": keep_fixed.copy(),
        "target_idx": target_joints - 1,
    }


def targeted_seed_pool(
    archive: CandidateArchive,
    baseline: dict[str, list[dict[str, Any]]],
    randomizer: MechanismRandomizer,
    tools: SubmissionTools,
    target_curve: np.ndarray,
    joints: int,
    config: TargetedConfig,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []

    # Exact-size archive and submission mechanisms from every problem provide known-valid
    # topology/position seeds, while higher-complexity archive entries provide incremental
    # dyadic prefixes that can reveal useful smaller structures.
    for problem_number in range(1, 7):
        problem_key = f"Problem {problem_number}"
        archived = archive.get(problem_number, joints)
        if archived is not None:
            raw.append(archived)
        raw.extend(
            normalized_mechanism(mech)
            for mech in baseline.get(problem_key, [])
            if complexity(mech) == joints
        )
        for record in archive.records_for_problem(problem_number):
            if int(record["complexity"]) > joints:
                truncated = truncate_mechanism(
                    mechanism_from_jsonable(record["mechanism"]), joints
                )
                if truncated is not None:
                    raw.append(truncated)

    # Multiple deterministic seed streams deliberately retain different starting positions
    # even when MechanismRandomizer produces the same connectivity more than once.
    per_stream = int(math.ceil(config.random_restarts_per_size / len(config.base_seeds)))
    for base_seed in config.base_seeds:
        stream_seed = base_seed + joints * 100
        raw.extend(generate_random_mechanisms(randomizer, joints, per_stream, stream_seed))
    raw = raw[: len(raw) - max(0, len(raw) - (120 + config.random_restarts_per_size))]

    selected: list[dict[str, Any]] = []
    for candidate in raw:
        try:
            chosen, distance = select_target_joint(tools, candidate, target_curve)
        except Exception:
            continue
        if np.isfinite(distance):
            selected.append(chosen)
    return selected


def pareto_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict distance/complexity Pareto front, ordered by complexity."""
    front = []
    best_distance = math.inf
    for record in sorted(records, key=lambda item: (int(item["complexity"]), float(item["distance"]))):
        distance = float(record["distance"])
        if distance < best_distance - 1e-10 and distance <= MAX_DISTANCE:
            front.append(record)
            best_distance = distance
    return front[:12]


def validate_mechanism_schema(mechanism: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = {"x0", "edges", "fixed_joints"} - set(mechanism)
    if missing:
        return [f"missing required fields: {sorted(missing)}"]
    x0 = np.asarray(mechanism["x0"], dtype=float)
    edges = np.asarray(mechanism["edges"])
    fixed = np.asarray(mechanism["fixed_joints"])
    if x0.ndim != 2 or x0.shape[1:] != (2,):
        errors.append(f"x0 must have shape (N-2, 2), got {x0.shape}")
        return errors
    joints = x0.shape[0] + 2
    if not MIN_JOINTS <= joints <= MAX_JOINTS:
        errors.append(f"complexity {joints} outside [{MIN_JOINTS}, {MAX_JOINTS}]")
    if not np.isfinite(x0).all():
        errors.append("x0 contains NaN/Inf")
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        errors.append(f"edges must have shape (E, 2), got {edges.shape}")
    elif not np.issubdtype(edges.dtype, np.integer):
        errors.append("edges must contain integers")
    else:
        edges_i = edges.astype(int)
        if len(edges_i) and (edges_i.min() < 0 or edges_i.max() >= joints):
            errors.append("edge index outside mechanism")
        if np.any(edges_i[:, 0] == edges_i[:, 1]):
            errors.append("self edge present")
        normalized_edges = [tuple(sorted(edge)) for edge in edges_i.tolist()]
        if (0, 1) in normalized_edges:
            errors.append("implicit crank edge (0,1) must not be submitted")
        if len(set(normalized_edges)) != len(normalized_edges):
            errors.append("duplicate edge present")
    if fixed.ndim != 1 or not np.issubdtype(fixed.dtype, np.integer):
        errors.append("fixed_joints must be a 1D integer array")
    else:
        fixed_i = fixed.astype(int)
        if len(fixed_i) and (fixed_i.min() < 2 or fixed_i.max() >= joints):
            errors.append("fixed joint outside [2, N-1]; node 0 is implicit")
        if len(np.unique(fixed_i)) != len(fixed_i):
            errors.append("duplicate fixed joint")
    target_idx = mechanism.get("target_idx", joints - 1)
    if not isinstance(target_idx, (int, np.integer)) or not 1 <= int(target_idx) < joints:
        errors.append(f"target_idx {target_idx!r} outside [1, N-1]")
    return errors


def validate_submission_strict(
    submission: dict[str, list[dict[str, Any]]],
    targets: np.ndarray,
    tools: SubmissionTools,
    solver: MechanismSolver,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_valid = True
    for problem_number, problem in enumerate(PROBLEM_KEYS, start=1):
        population = submission.get(problem, [])
        if not population:
            all_valid = False
        if len(population) > 12:
            all_valid = False
        for position, mechanism in enumerate(population):
            schema_errors = validate_mechanism_schema(mechanism)
            distance = math.inf
            simulation_valid = False
            if not schema_errors:
                try:
                    mech = normalized_mechanism(mechanism)
                    solution = np.asarray(
                        solver(mech["x0"], mech["edges"], mech["fixed_joints"]), dtype=float
                    )
                    simulation_valid = solution.shape[1:] == (200, 2) and np.isfinite(solution).all()
                    distance = float(tools.compute_F([mech], targets[problem_number - 1])[0, 0])
                except Exception as exc:  # validation must report, not hide, any failure
                    schema_errors.append(f"simulation/evaluation raised {type(exc).__name__}: {exc}")
            valid = (
                not schema_errors
                and simulation_valid
                and np.isfinite(distance)
                and distance <= MAX_DISTANCE
            )
            all_valid &= valid
            rows.append(
                {
                    "problem": problem_number,
                    "submission_index": position,
                    "complexity": complexity(mechanism),
                    "distance": distance,
                    "valid": bool(valid),
                    "simulation_valid": bool(simulation_valid),
                    "errors": schema_errors,
                }
            )
    official_score = evaluate_submission(submission, targets)
    report = {"all_valid": bool(all_valid), "rows": rows, "official_score": official_score}
    return rows, report


def build_submission(archive: CandidateArchive) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    submission = make_empty_submission()
    fronts: dict[str, list[dict[str, Any]]] = {}
    for problem_number, problem in enumerate(PROBLEM_KEYS, start=1):
        front = pareto_records(archive.records_for_problem(problem_number))
        fronts[problem] = front
        submission[problem] = [mechanism_from_jsonable(record["mechanism"]) for record in front]
    return submission, fronts


def create_visualizations(
    submission: dict[str, list[dict[str, Any]]],
    targets: np.ndarray,
    tools: SubmissionTools,
    solver: MechanismSolver,
) -> list[str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_engine = CurveEngine(normalize_scale=True, device="cpu")
    output_paths: list[str] = []
    for problem_number, problem in enumerate(PROBLEM_KEYS, start=1):
        population = submission.get(problem, [])
        if not population:
            continue
        values = np.asarray(tools.compute_F(population, targets[problem_number - 1]))
        columns = 3
        rows = int(math.ceil(len(population) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
        for ax, mech, (distance, joints) in zip(axes.flat, population, values):
            normalized = normalized_mechanism(mech)
            curves = solver(normalized["x0"], normalized["edges"], normalized["fixed_joints"])
            traced = np.asarray(curves[int(normalized["target_idx"])])
            aligned, aligned_target, _ = curve_engine.optimal_alignment(
                traced, targets[problem_number - 1]
            )
            ax.plot(aligned_target[:, 0], aligned_target[:, 1], color="navy", lw=2.5, label="target")
            ax.plot(aligned[:, 0], aligned[:, 1], color="darkorange", lw=1.8, label="trajectory")
            ax.set_title(f"{int(joints)} joints, d={float(distance):.4f}")
            ax.axis("equal")
            ax.axis("off")
        for ax in axes.flat[len(population) :]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.suptitle(f"Problem {problem_number}: retained Pareto mechanisms")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = RESULTS_DIR / f"problem_{problem_number}_curves.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        output_paths.append(str(path))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for problem_number, ax in enumerate(axes.flat, start=1):
        population = submission.get(f"Problem {problem_number}", [])
        if population:
            tools.plot_population(
                population,
                targets[problem_number - 1],
                title=f"Problem {problem_number}",
                ax=ax,
            )
        else:
            ax.text(0.5, 0.5, "No retained mechanism", ha="center", va="center")
            ax.axis("off")
    fig.tight_layout()
    path = RESULTS_DIR / "pareto_summary.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    output_paths.append(str(path))
    return output_paths


def run_search(config: SearchConfig) -> dict[str, Any]:
    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGETS_PATH)
    original_submission = load_submission(str(FINAL_PATH)) if FINAL_PATH.exists() else make_empty_submission()

    tools = SubmissionTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    tools.compile()
    grad_tools = DifferentiableTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    grad_tools.compile()
    randomizer = MechanismRandomizer(
        min_size=MIN_JOINTS,
        max_size=MAX_JOINTS,
        timesteps=200,
        device="cpu",
    )
    solver = MechanismSolver(timesteps=200, max_size=20, device="cpu")
    solver.compile()

    archive = CandidateArchive(ARCHIVE_PATH)
    archive.seed_submission(FINAL_PATH, tools, targets)
    log_rows: list[dict[str, Any]] = []

    for problem_number in config.problem_numbers:
        target_curve = targets[problem_number - 1]
        print(f"\n=== Problem {problem_number} ===", flush=True)
        for joints in config.sizes:
            seed = config.base_seed + problem_number * 1000 + joints
            rng = np.random.default_rng(seed)
            # Reuse a bounded, deterministic selection of prior good topologies.  Without
            # this cap, the archive grows across problems and later eggs become needlessly
            # slower even though many saved entries share nearly equivalent structures.
            seeds = topology_library(archive, original_submission, joints)[:3]
            seeds.extend(generate_random_mechanisms(randomizer, joints, config.random_topologies, seed))
            unique: dict[str, dict[str, Any]] = {}
            for candidate in seeds:
                unique.setdefault(mechanism_key(candidate, include_positions=False), candidate)
            seeds = list(unique.values())
            print(f"size={joints:2d}: {len(seeds)} valid topology seed(s)", flush=True)

            best_mech: dict[str, Any] | None = None
            best_distance = math.inf
            best_method = ""
            for topology_number, candidate in enumerate(seeds):
                selected, initial_distance = select_target_joint(tools, candidate, target_curve)
                evolved, evolved_distance = seeded_evolution(
                    tools,
                    selected,
                    target_curve,
                    rng,
                    config.population_size,
                    config.generations,
                    config.perturb_scale,
                )
                # Position changes can make another moving joint the better output.
                evolved, evolved_distance = select_target_joint(tools, evolved, target_curve)
                if evolved_distance < best_distance:
                    best_mech, best_distance = evolved, evolved_distance
                    best_method = "valid topology search + seeded evolution"
                log_rows.append(
                    {
                        "mode": config.name,
                        "problem": problem_number,
                        "complexity": joints,
                        "seed": seed,
                        "topology": topology_number,
                        "initial_distance": initial_distance,
                        "evolved_distance": evolved_distance,
                    }
                )

            if best_mech is None:
                print("  no candidate generated", flush=True)
                continue
            refined, refined_distance, accepted_steps = gradient_refine(
                grad_tools,
                best_mech,
                target_curve,
                config.gradient_steps,
                config.gradient_lr,
            )
            method = best_method
            if refined_distance < best_distance:
                best_mech, best_distance = refined, refined_distance
                method += " + gradient refinement"

            # Canonical re-evaluation through the same SubmissionTools path used by scoring.
            canonical_distance = float(tools.compute_F([best_mech], target_curve)[0, 0])
            improved = archive.consider(
                problem_number,
                best_mech,
                canonical_distance,
                method=method,
                seed=seed,
            )
            print(
                f"  best={canonical_distance:.6f} accepted_grad_steps={accepted_steps} "
                f"archive={'updated' if improved else 'kept prior'}",
                flush=True,
            )

    submission, fronts = build_submission(archive)
    output_path = RESULTS_DIR / "quick_submission.json" if config.name == "quick" else FINAL_PATH
    save_submission(submission, str(output_path))
    rows, validation_report = validate_submission_strict(submission, targets, tools, solver)
    validation_path = RESULTS_DIR / f"{config.name}_validation.json"
    with validation_path.open("w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, indent=2, allow_nan=False)
    if config.name == "full" and not validation_report["all_valid"]:
        raise RuntimeError(f"Refusing to finish with an invalid submission; see {validation_path}")

    visualizations = create_visualizations(submission, targets, tools, solver)
    finished = time.time()
    summary = {
        "mode": config.name,
        "config": asdict(config),
        "runtime_seconds": finished - started,
        "submission_path": str(output_path),
        "archive_path": str(ARCHIVE_PATH),
        "validation_path": str(validation_path),
        "visualizations": visualizations,
        "retained_counts": {problem: len(submission[problem]) for problem in PROBLEM_KEYS},
        "pareto_records": fronts,
        "validation": validation_report,
        "search_log": log_rows,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(validation_report["official_score"], indent=2), flush=True)
    print(f"Runtime: {finished - started:.1f}s", flush=True)
    return summary


def visualize_one_problem(
    population: list[dict[str, Any]],
    target_curve: np.ndarray,
    tools: SubmissionTools,
    solver: MechanismSolver,
    problem_number: int,
    path: Path,
) -> None:
    curve_engine = CurveEngine(normalize_scale=True, device="cpu")
    values = np.asarray(tools.compute_F(population, target_curve))
    columns = 3
    rows = int(math.ceil(len(population) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    for ax, mech, (distance, joints) in zip(axes.flat, population, values):
        normalized = normalized_mechanism(mech)
        curves = solver(normalized["x0"], normalized["edges"], normalized["fixed_joints"])
        traced = np.asarray(curves[int(normalized["target_idx"])])
        aligned, aligned_target, _ = curve_engine.optimal_alignment(traced, target_curve)
        ax.plot(aligned_target[:, 0], aligned_target[:, 1], color="navy", lw=2.5, label="target")
        ax.plot(aligned[:, 0], aligned[:, 1], color="darkorange", lw=1.8, label="trajectory")
        ax.set_title(f"{int(joints)} joints, d={float(distance):.4f}")
        ax.axis("equal")
        ax.axis("off")
    for ax in axes.flat[len(population) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"Problem {problem_number}: targeted-search Pareto mechanisms")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_problem2_targeted(config: TargetedConfig = PROBLEM2_TARGETED_CONFIG) -> dict[str, Any]:
    """Intensive, score-gated search that may change only Problem 2."""
    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(TARGETS_PATH)
    target_curve = targets[config.problem_number - 1]
    baseline = load_submission(str(FINAL_PATH))
    baseline_json = json.dumps(
        {problem: [mechanism_to_jsonable(mech) for mech in population]
         for problem, population in baseline.items()},
        sort_keys=True,
    )
    save_submission(baseline, str(RESULTS_DIR / "problem2_targeted_baseline.json"))

    tools = SubmissionTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    tools.compile()
    grad_tools = DifferentiableTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    grad_tools.compile()
    solver = MechanismSolver(timesteps=200, max_size=20, device="cpu")
    solver.compile()
    randomizer = MechanismRandomizer(
        min_size=MIN_JOINTS,
        max_size=MAX_JOINTS,
        timesteps=200,
        device="cpu",
    )
    archive = CandidateArchive(ARCHIVE_PATH)
    archive.seed_submission(FINAL_PATH, tools, targets)

    baseline_score = evaluate_submission(baseline, targets)
    baseline_problem2 = float(baseline_score["Score Breakdown"]["Problem 2"])
    baseline_overall = float(baseline_score["Overall Score"])
    baseline_values = {
        complexity(mech): float(tools.compute_F([mech], target_curve)[0, 0])
        for mech in baseline["Problem 2"]
    }
    search_log: list[dict[str, Any]] = []

    for joints in config.sizes:
        print(f"\n=== Targeted Problem 2, size {joints} ===", flush=True)
        seeds = targeted_seed_pool(
            archive, baseline, randomizer, tools, target_curve, joints, config
        )
        groups: dict[str, list[dict[str, Any]]] = {}
        initial_group_distance: dict[str, float] = {}
        for candidate in seeds:
            key = mechanism_key(candidate, include_positions=False)
            groups.setdefault(key, []).append(candidate)
            distance = float(tools.compute_F([candidate], target_curve)[0, 0])
            initial_group_distance[key] = min(initial_group_distance.get(key, math.inf), distance)

        ranked_group_keys = sorted(groups, key=lambda key: initial_group_distance[key])
        ranked_group_keys = ranked_group_keys[: config.max_topology_groups]
        print(
            f"  {len(seeds)} finite seeds, {len(groups)} topology/target groups, "
            f"evolving best {len(ranked_group_keys)} groups",
            flush=True,
        )

        evolved_candidates: list[tuple[float, dict[str, Any], str]] = []
        for group_index, key in enumerate(ranked_group_keys):
            group_seed = config.base_seeds[group_index % len(config.base_seeds)] + joints * 1000 + group_index
            rng = np.random.default_rng(group_seed)
            evolved, evolved_distance = seeded_evolution_multi(
                tools,
                groups[key],
                target_curve,
                rng,
                config.population_size,
                config.generations,
                config.perturb_scale,
            )
            evolved, evolved_distance = select_target_joint(tools, evolved, target_curve)
            evolved_candidates.append((evolved_distance, evolved, key))
            print(
                f"  group {group_index + 1:2d}/{len(ranked_group_keys)}: "
                f"{initial_group_distance[key]:.6f} -> {evolved_distance:.6f}",
                flush=True,
            )

        evolved_candidates.sort(key=lambda item: item[0])
        finalists = evolved_candidates[: config.gradient_finalists]
        best_distance = math.inf
        best_mechanism: dict[str, Any] | None = None
        best_steps = 0
        for finalist_index, (pre_gradient, candidate, key) in enumerate(finalists):
            refined, refined_distance, accepted_steps = gradient_refine(
                grad_tools,
                candidate,
                target_curve,
                config.gradient_steps,
                config.gradient_lr,
            )
            canonical = float(tools.compute_F([refined], target_curve)[0, 0])
            search_log.append(
                {
                    "complexity": joints,
                    "finalist": finalist_index,
                    "topology_key": key,
                    "pre_gradient_distance": pre_gradient,
                    "post_gradient_distance": canonical,
                    "accepted_gradient_steps": accepted_steps,
                }
            )
            if canonical < best_distance:
                best_distance = canonical
                best_mechanism = refined
                best_steps = accepted_steps

        prior_archive = archive.records[f"Problem {config.problem_number}"].get(str(joints))
        prior_distance = math.inf if prior_archive is None else float(prior_archive["distance"])
        improved = False
        if best_mechanism is not None and best_distance < prior_distance - 1e-10:
            improved = archive.consider(
                config.problem_number,
                best_mechanism,
                best_distance,
                method="targeted multi-seed topology-preserving evolution + gradient refinement",
                seed=config.base_seeds[0] + joints,
            )
        print(
            f"  size {joints}: prior={prior_distance:.6f}, best={best_distance:.6f}, "
            f"accepted_grad_steps={best_steps}, archive={'updated' if improved else 'unchanged'}",
            flush=True,
        )

    # Preserve every non-target problem byte-for-byte at the JSON-data level.
    candidate_submission = copy.deepcopy(baseline)
    problem2_front = pareto_records(archive.records_for_problem(config.problem_number))
    candidate_submission["Problem 2"] = [
        mechanism_from_jsonable(record["mechanism"]) for record in problem2_front
    ]
    for problem in ("Problem 1", "Problem 3", "Problem 4", "Problem 5", "Problem 6"):
        before = [mechanism_to_jsonable(mech) for mech in baseline[problem]]
        after = [mechanism_to_jsonable(mech) for mech in candidate_submission[problem]]
        if before != after:
            raise RuntimeError(f"Targeted run unexpectedly changed {problem}")

    candidate_score = evaluate_submission(candidate_submission, targets)
    candidate_problem2 = float(candidate_score["Score Breakdown"]["Problem 2"])
    candidate_overall = float(candidate_score["Overall Score"])
    score_improved = candidate_overall > baseline_overall + 1e-10

    validation_report: dict[str, Any] | None = None
    committed_to_submission = False
    if score_improved:
        save_submission(candidate_submission, str(FINAL_PATH))
        reloaded = load_submission(str(FINAL_PATH))
        _, validation_report = validate_submission_strict(reloaded, targets, tools, solver)
        if not validation_report["all_valid"]:
            save_submission(baseline, str(FINAL_PATH))
            raise RuntimeError("Targeted submission failed independent validation; baseline restored")
        reloaded_score = validation_report["official_score"]
        if float(reloaded_score["Overall Score"]) <= baseline_overall + 1e-10:
            save_submission(baseline, str(FINAL_PATH))
            raise RuntimeError("Reloaded targeted score did not improve; baseline restored")
        committed_to_submission = True
        visualize_one_problem(
            reloaded["Problem 2"],
            target_curve,
            tools,
            solver,
            2,
            RESULTS_DIR / "problem_2_targeted_curves.png",
        )

    # Report same-size improvements from the complete archive, not only from the final
    # Pareto front: an improved size can still be dominated by an even better smaller design.
    final_values = {
        int(record["complexity"]): float(record["distance"])
        for record in archive.records_for_problem(config.problem_number)
    }
    improvements = {
        str(joints): {
            "before": baseline_values.get(joints),
            "after": final_values.get(joints, baseline_values.get(joints)),
            "absolute_improvement": (
                None if baseline_values.get(joints) is None or final_values.get(joints) is None
                else baseline_values[joints] - final_values[joints]
            ),
        }
        for joints in config.sizes
    }
    summary = {
        "config": asdict(config),
        "runtime_seconds": time.time() - started,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "problem2_hypervolume_improvement": candidate_problem2 - baseline_problem2,
        "overall_score_improvement": candidate_overall - baseline_overall,
        "score_improved": score_improved,
        "submission_updated": committed_to_submission,
        "same_complexity_improvements": improvements,
        "problem2_pareto_records": problem2_front,
        "validation": validation_report,
        "search_log": search_log,
        "baseline_json_sha256": hashlib.sha256(baseline_json.encode("utf-8")).hexdigest(),
    }
    with (RESULTS_DIR / "problem2_targeted_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps({key: summary[key] for key in (
        "baseline_score", "candidate_score", "problem2_hypervolume_improvement",
        "overall_score_improvement", "submission_updated")}, indent=2), flush=True)
    return summary


def validate_existing() -> dict[str, Any]:
    targets = np.load(TARGETS_PATH)
    tools = SubmissionTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    tools.compile()
    solver = MechanismSolver(timesteps=200, max_size=20, device="cpu")
    solver.compile()
    submission = load_submission(str(FINAL_PATH))
    _, report = validate_submission_strict(submission, targets, tools, solver)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "final_validation.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def visualize_existing() -> list[str]:
    targets = np.load(TARGETS_PATH)
    tools = SubmissionTools(timesteps=200, max_size=20, scaled=True, device="cpu")
    tools.compile()
    solver = MechanismSolver(timesteps=200, max_size=20, device="cpu")
    solver.compile()
    submission = load_submission(str(FINAL_PATH))
    paths = create_visualizations(submission, targets, tools, solver)
    print("\n".join(paths))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("quick", "full", "problem2", "validate", "visualize"))
    args = parser.parse_args()
    if args.mode == "quick":
        run_search(QUICK_CONFIG)
    elif args.mode == "full":
        run_search(FULL_CONFIG)
    elif args.mode == "problem2":
        run_problem2_targeted()
    elif args.mode == "validate":
        validate_existing()
    else:
        visualize_existing()


if __name__ == "__main__":
    main()
