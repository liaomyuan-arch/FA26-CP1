# CP1 solution workflow

`CP1_StarterNotebook.ipynb` remains the authoritative assignment specification and is not
modified.  The added `cp1_optimizer.py` orchestrates the instructor-provided `LINKS` code.

## Notebook and scoring summary

- A submission has keys `Problem 1` through `Problem 6`, each containing at most 12 mechanisms.
- Each mechanism contains `x0`, `edges`, `fixed_joints`, and optionally `target_idx`.
- `x0` has shape `(N-2, 2)` because crank nodes 0 and 1 are implicit.
- The crank edge `(0, 1)` and fixed ground joint 0 are also implicit and must not be submitted.
- Valid complexity is 4 through 15 total joints.
- Simulation and distance use 200 crank samples.
- Distance is scale-normalized and must be at most 1.5 to contribute.
- Per-problem score is distance/complexity hypervolume with reference point `(1.5, 16)`.
- The final score is the mean of the six per-problem hypervolumes.

## Instructor components used

- Generation: `LINKS.Optimization.MechanismRandomizer`
- Simulation/structural ordering: `LINKS.Kinematics.MechanismSolver`
- Distance and validity: `LINKS.CP.SubmissionTools` / `LINKS.Optimization.Tools`
- Gradients: `LINKS.Optimization.DifferentiableTools`
- Alignment and curve overlays: `LINKS.Geometry.CurveEngine`
- Hypervolume and final evaluation: `SubmissionTools.hypervolume` and `evaluate_submission`
- JSON I/O: `save_submission` and `load_submission`

## Sample solutions in the starter notebook

1. Sample Solution 1 generates one valid topology at each size and applies gradient descent
   only to joint positions.  It is fast but sensitive to the single random topology and local
   optimum.
2. Sample Solution 2 uses a mixed-variable NSGA-II search at a fixed size over connectivity,
   fixed joints, target joint, and positions.  Valid randomizer seeds are essential; arbitrary
   connectivity mutation usually creates invalid mechanisms, which the notebook explicitly
   identifies as its main bottleneck.
3. Sample Solution 3 uses a 15-slot mixed-variable NSGA-II representation and prunes unused
   nodes, making distance and complexity simultaneous objectives.  It can collapse to only one
   or two sizes, so pooling independent searches remains useful.

## Added hybrid pipeline

The added pipeline preserves the assignment definitions and uses:

1. valid-by-construction random topology generation at every size 4–15;
2. reuse of every previously good topology as a seed;
3. exhaustive moving-target-joint selection for each seed;
4. elitist seeded evolutionary search over positions while holding a valid topology fixed;
5. monotone Adam refinement using the instructor's exact differentiable objective;
6. immediate best-per-problem/per-size checkpointing in `results/candidate_archive.json`;
7. strict independent simulation/schema/distance validation;
8. Pareto filtering and exact JSON serialization;
9. target-versus-trajectory plots and a Pareto summary in `results/`.

This structure avoids changing any instructor evaluation function and avoids the documented
failure mode where arbitrary connectivity mutation destroys nearly every offspring.

## Commands

Use the assignment environment (the requirements specify JAX 0.5.3 and pymoo 0.6.1):

```powershell
python cp1_optimizer.py quick
python cp1_optimizer.py full
python cp1_optimizer.py problem2
python cp1_optimizer.py validate
python cp1_optimizer.py visualize
```

`quick` runs a small end-to-end test on Problem 1 and writes
`results/quick_submission.json`.  `full` searches all six problems and writes the final
`my_submission.json`.  `problem2` performs the intensive, score-gated targeted search over
Problem 2 complexities 5–8; it preserves every other problem and only replaces the submission
when the independently reloaded official score increases.  All search modes are deterministic
for their recorded seeds and resume from the candidate archive.
