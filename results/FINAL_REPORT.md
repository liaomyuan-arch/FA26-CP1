# CP1 final optimization report

All values below were recalculated after reloading `my_submission.json` from disk with the
instructor's `SubmissionTools`/`evaluate_submission` implementation.  Strict validation also
ran every retained mechanism through `MechanismSolver` at 200 crank angles.  Every retained
mechanism passed schema, structural, finite-simulation, complexity, and distance checks.

| Problem | Joints | Distance | Valid | Method |
|---:|---:|---:|:---:|---|
| 1 | 4 | 0.317134 | Yes | Valid topology + evolution + gradient |
| 1 | 5 | 0.117268 | Yes | Valid topology + evolution + gradient |
| 1 | 7 | 0.048791 | Yes | Valid topology + evolution + gradient |
| 1 | 10 | 0.037656 | Yes | Valid topology + evolution + gradient |
| 2 | 4 | 1.409625 | Yes | Valid topology + evolution + gradient |
| 2 | 5 | 1.181974 | Yes | Valid topology + evolution + gradient |
| 2 | 6 | 1.178673 | Yes | Valid topology + evolution + gradient |
| 2 | 7 | 0.896074 | Yes | Valid topology + evolution + gradient |
| 2 | 8 | 0.451755 | Yes | Valid topology + evolution + gradient |
| 2 | 9 | 0.062879 | Yes | Valid topology + evolution + gradient |
| 3 | 4 | 0.458699 | Yes | Valid topology + evolution + gradient |
| 3 | 5 | 0.427489 | Yes | Valid topology + evolution + gradient |
| 3 | 6 | 0.255919 | Yes | Valid topology + evolution + gradient |
| 3 | 10 | 0.182323 | Yes | Valid topology + evolution + gradient |
| 3 | 11 | 0.152118 | Yes | Valid topology + evolution + gradient |
| 4 | 4 | 0.191289 | Yes | Valid topology + evolution + gradient |
| 4 | 5 | 0.183625 | Yes | Valid topology + evolution + gradient |
| 4 | 6 | 0.167753 | Yes | Valid topology + evolution + gradient |
| 4 | 8 | 0.131487 | Yes | Valid topology + evolution + gradient |
| 4 | 10 | 0.117529 | Yes | Valid topology + evolution + gradient |
| 5 | 4 | 0.378838 | Yes | Valid topology + evolution + gradient |
| 5 | 5 | 0.256089 | Yes | Valid topology + evolution + gradient |
| 5 | 6 | 0.243358 | Yes | Valid topology + evolution + gradient |
| 5 | 7 | 0.241486 | Yes | Valid topology + evolution + gradient |
| 5 | 8 | 0.233411 | Yes | Valid topology + evolution + gradient |
| 5 | 11 | 0.185807 | Yes | Valid topology + evolution + gradient |
| 6 | 4 | 0.739103 | Yes | Valid topology + evolution + gradient |
| 6 | 5 | 0.556959 | Yes | Valid topology + evolution + gradient |
| 6 | 6 | 0.333359 | Yes | Valid topology + evolution + gradient |
| 6 | 7 | 0.121109 | Yes | Valid topology + evolution + gradient |
| 6 | 10 | 0.107949 | Yes | Valid topology + evolution + gradient |

## Scores and retained counts

| Problem | Retained Pareto mechanisms | Hypervolume |
|---:|---:|---:|
| 1 | 4 | 17.076019 |
| 2 | 6 | 12.441754 |
| 3 | 5 | 15.147222 |
| 4 | 5 | 16.321434 |
| 5 | 6 | 15.250963 |
| 6 | 5 | 15.359558 |
| **Average** | **31 total** | **15.266158** |

## Reproducibility

- QUICK mode: base seed `260901`; Problems/sizes: Problem 1 at 4, 6, and 8 joints.
- FULL mode: base seed `260926`; all six problems and every size 4 through 15.
- Per size: two newly generated valid topologies plus up to three archived topology seeds.
- Evolution: population 16, 10 generations, perturbation scale 0.18.
- Local refinement: up to 100 monotone Adam steps, initial learning rate 0.01.
- Full optimization runtime: 141.7 seconds on CPU in this environment.
- The candidate archive stores the best discovered mechanism at every searched complexity,
  including mechanisms excluded from the final JSON because another mechanism dominates them.

Problem 2 remains the clearest opportunity for improvement: its 4–8 joint designs are valid
but relatively far from the target, while its 9-joint design is already strong.  Additional
valid-topology restarts focused on sizes 5–8 should provide the largest likely hypervolume gain.
