# Individual Reflection Report Draft — FA26 CP1 Linkage Synthesis

> The report text runs from Section 1 through Section 7. The verification and preparation
> notes after the horizontal rule are not intended to be submitted as part of the report.

## 1. Introduction and Problem Understanding

The objective of CP1 was to synthesize planar linkages whose selected output joint traces each
of six target curves. A mechanism was represented by its non-crank joint positions, edges,
fixed joints, and target-joint index. The motor crank was fixed by convention, so nodes 0 and 1,
the crank edge, and ground joint 0 were implicit rather than design variables. The submitted
mechanisms had to contain 4–15 joints and produce a curve distance no greater than 1.5 under the
instructor’s 200-angle simulation and scale-normalized distance calculation.

This was not simply a minimum-error problem. Mechanism complexity was the second objective,
and each problem was scored by the hypervolume of its distance-versus-joint-count Pareto set.
The reference point was a distance of 1.5 and a complexity of 16. Therefore, a slightly less
accurate five-joint linkage could be more useful than a very accurate fifteen-joint linkage. I
needed to search for accurate mechanisms while preserving useful tradeoffs across sizes.

## 2. Initial Approach and Instructor Baselines

The starter notebook presented three baselines. The first generated one valid mechanism at
each size and used gradients to optimize only its joint positions. This was inexpensive and
showed how `DifferentiableTools` could improve a fixed design, but the result depended strongly
on the randomly selected topology and initial geometry. Gradient descent could not leave that
topology or reliably escape a poor local region.

The second baseline used mixed-variable NSGA-II at a fixed size. Its variables included
connectivity bits, continuous positions, fixed-joint flags, and a target-joint integer. This
representation was flexible, but the notebook also demonstrated its main weakness: arbitrary
connectivity crossover or mutation usually did not produce a valid dyadic, single-degree-of-
freedom mechanism. Seeding with `MechanismRandomizer` was much better than naive random
initialization, but most structural offspring still wasted evaluations by becoming invalid or
locking.

The third baseline used a fifteen-node representation and pruned disconnected nodes so that
distance and complexity could be optimized simultaneously. This made NSGA-II a true
multi-objective search, but the population often collapsed to only one or two sizes. I treated
these methods as useful components rather than copying one of them unchanged. NSGA-II’s
Pareto-ranking logic matched the assignment, but its generic mixed-variable operators were not
well matched to the sparse set of valid linkage graphs.

## 3. Optimization Approach

My final pipeline was:

**valid topology generation → target-joint selection → evolutionary position search →
gradient refinement → validation → Pareto filtering → JSON submission.**

The main domain-informed decision was to preserve valid mechanism structure during the
continuous search. `MechanismRandomizer` generated mechanisms that already followed the
implicit-crank convention and could be solved as dyadic one-DOF linkages. For each candidate,
I evaluated every non-fixed moving joint as the possible output and retained the best
`target_idx`. Once a topology and target joint were selected, the evolutionary search changed
only the joint coordinates. This reduced the active design space to (2(N-2)) continuous
variables and prevented a position mutation from changing connectivity.

The evolutionary stage provided global exploration. The initial full run used multiple valid
topologies, a population of 16, ten generations, and position perturbations and crossover. Its
purpose was to move between different geometric regions for the same linkage graph, including
regions that a single gradient trajectory would not reach. The best evolutionary candidates
were then passed to `DifferentiableTools`. I used Adam-style gradient updates with trial-step
rejection: a step was accepted only if the instructor distance stayed finite and did not
increase; otherwise, the learning rate was reduced. In this arrangement, evolution handled
large nonlocal changes while gradients provided precise local improvement.

All objective values came from the instructor code. `MechanismSolver` performed kinematic
simulation, `SubmissionTools` supplied the scored scale-normalized distance and hypervolume,
and `DifferentiableTools` supplied the corresponding gradient. I did not alter any of these
definitions. I also maintained a candidate archive containing the best instructor-evaluated
mechanism found for each problem and complexity. A candidate could replace an archive entry
only if its recomputed distance was lower at the same joint count. The archive was written
after each improvement, which made later runs resumable and prevented a good result from being
lost to a worse stochastic run.

Before submission, I removed dominated mechanisms. I then independently checked array shapes,
edge and fixed-joint indices, target indices, complexity, finite 200-angle simulations, and the
distance threshold. Finally, I saved the JSON, reloaded it from disk, and ran the instructor’s
`evaluate_submission` function again. This separation between search and canonical validation
was important because an optimizer can encounter locking or non-finite geometries even when a
topology is structurally legal.

[PERSONAL NOTE: briefly describe which part of implementing or debugging this pipeline required
the most attention from you, without changing the technical description above.]

## 4. Experiments, Challenges, and What Did Not Work Well

I first ran a QUICK test on Problem 1 to verify generation, optimization, archiving,
serialization, scoring, and plotting. The initial full run then searched all sizes 4–15 for all
six problems. It used two new random topologies per size, up to three archived topology seeds,
a population of 16, ten generations, and up to 100 gradient steps. This run took 141.7 seconds
on CPU and produced an average hypervolume of 15.266158.

The clearest weakness was Problem 2. Its initial distances were 1.181974 at five joints,
1.178673 at six, 0.896074 at seven, and 0.451755 at eight. Its hypervolume was 12.441754, much
lower than the other five problems. These mechanisms were valid, so the issue was not a failed
simulation or formatting error. The result instead showed that my broad search had not found
good low-complexity geometries for this curve. The initial budget was intentionally modest and
spread over 72 problem-size combinations, so it used relatively few topologies, starts, and
generations at any one combination. The large performance variation among random topologies
also confirmed the sensitivity to initialization described in the starter notebook.

At that point, I could not conclude that Problem 2 inherently required a complicated linkage.
The broad run had only shown that its current low-complexity candidates were poor. I therefore
kept the validated submission as a baseline and allocated a larger search budget only to
Problem 2 sizes 5–8.

[PERSONAL NOTE: state what made you decide that Problem 2, rather than another problem, deserved
the extra computation.]

## 5. Targeted Problem 2 Improvement

The targeted run used 32 random restarts per size and reused the candidate archive. It also
formed smaller valid seeds by truncating the incrementally numbered prefixes of archived
dyadic mechanisms when possible. After selecting the best moving output joint, candidates were
grouped by topology and target joint. I evolved up to 20 groups per size with a population of
48 for 50 generations. Within a group, connectivity stayed fixed while the coordinates were
changed using differential mutation, coordinate crossover, extrapolating arithmetic
crossover, and Gaussian mutation. I then gradient-refined six finalists for as many as 250
steps. Four deterministic seed streams were recorded, and the targeted run took 214.1 seconds
on CPU.

The results changed my interpretation of the initial Problem 2 performance. The five-joint
distance decreased from 1.181974 to 0.058808; the six-joint result decreased from 1.178673 to
0.059267; the seven-joint result decreased from 0.896074 to 0.047998; and the eight-joint result
decreased from 0.451755 to 0.058742. For the winning five-joint candidate, the evolutionary
stage reached 0.062461 before gradient refinement reduced it to 0.058808. This is direct
evidence that the global and local stages were complementary in that run.

The result also showed why the final submission had to be Pareto-filtered rather than filled
with every improved mechanism. Although the six- and eight-joint designs were much better than
their baselines, they were dominated by smaller mechanisms: the five-joint distance was lower
than the six-joint distance, and the seven-joint distance was lower than the eight-joint
distance. The final Problem 2 front therefore retained only the 4-, 5-, and 7-joint designs.
Problem 2 hypervolume increased from 12.441754 to 16.040776, and the overall average increased
from 15.266158 to 15.865995.

This experiment supports a limited conclusion: for this implementation and these saved seeds,
the poor initial Problem 2 score was mainly a search-allocation and initialization problem,
not proof that the target required high complexity. I cannot separate the individual effects
of the larger population, additional topology groups, custom operators, and longer refinement
because they were changed together.

[PERSONAL NOTE: describe your reaction to the five-joint improvement or which part of the
before/after trajectory was most noticeable to you.]

## 6. Final Results

The final submission contained 28 Pareto mechanisms. The score after reloading and validating
the JSON was:

| Problem | Retained mechanisms | Hypervolume |
|---:|---:|---:|
| 1 | 4 | 17.076019 |
| 2 | 3 | 16.040776 |
| 3 | 5 | 15.147222 |
| 4 | 5 | 16.321434 |
| 5 | 6 | 15.250963 |
| 6 | 5 | 15.359558 |
| **Average** | **28 total** | **15.865995** |

Every retained mechanism passed the schema checks, structural solve, finite 200-angle
simulation, complexity limit, and distance limit. Problems 1 and 3–6 were unchanged during the
targeted Problem 2 run.

## 7. Reflection and Future Work

The most useful domain knowledge in my approach was not an analytic linkage formula; it was
the decision to treat valid dyadic connectivity as a scarce resource. Holding a valid topology
fixed made the evolutionary operators productive instead of allowing most offspring to fall
outside the feasible mechanism class. Searching sizes in an outer loop was less elegant than
a single variable-size multi-objective algorithm, but it gave direct control over computational
effort and made the archive easy to interpret.

One next step would be to design topology operators that remain valid by construction. For
example, mutation could add, remove, or redirect complete dyadic construction steps instead of
flipping arbitrary adjacency bits. A graph grammar or construction-sequence representation
could make crossover meaningful while preserving one degree of freedom. I would also separate
the targeted changes in an ablation study: topology count, population size, custom crossover,
and gradient budget should be varied independently. That would distinguish which change
produced the largest gain.

The search budget could also be allocated adaptively. Sizes whose archive value is already
dominated do not need the same number of evaluations as sizes near the Pareto front. Convergence
histories and improvement rates could trigger more restarts only where the estimated
hypervolume gain is high. The topology groups are independent, so parallel execution would be
a natural speed improvement.

Finally, curve distance is only one part of a physical linkage design. The assignment did not
score link interference, joint limits, transmission angle, sensitivity to dimensional error,
forces, or manufacturability. It also used scale-normalized curve matching and 200 sampled
crank angles. A practical machine or robotic mechanism would require denser motion checks and
additional objectives or constraints for these physical effects. The same hybrid framework
could still be used, but feasibility would need to reflect the actual hardware rather than
only the traced shape.

[PERSONAL NOTE: add one specific future extension you would personally choose first and why.]

---

# Author Verification and Preparation Notes

## Factual claims to verify personally before submission

1. Confirm that the written-report page limit and formatting are still “under 3 pages,
   single-spaced, excluding figures.”
2. Confirm the final overall score is `15.865995` and the six hypervolumes match the table in
   `results/final_validation.json`.
3. Confirm that the final submission contains 28 mechanisms and that Problem 2 contains the
   three Pareto mechanisms at 4, 5, and 7 joints.
4. Confirm the targeted same-size distances: 5 joints `0.058808`, 6 joints `0.059267`, 7 joints
   `0.047998`, and 8 joints `0.058742`.
5. Confirm that the 6- and 8-joint Problem 2 candidates are archived but excluded from the
   submission because they are dominated.
6. Confirm the initial full-run configuration: two random topologies per size, population 16,
   ten generations, and up to 100 gradient steps.
7. Confirm the targeted configuration: 32 random restarts, at most 20 topology/target groups,
   population 48, 50 generations, six gradient finalists, and up to 250 steps.
8. Confirm the recorded CPU runtimes of 141.7 seconds for the initial full run and 214.1 seconds
   for the targeted Problem 2 run. These are environment-specific and should not be presented
   as general performance benchmarks.
9. Confirm that all distances and scores came from the unmodified instructor evaluator and
   that the final JSON was reloaded before final validation.
10. Confirm that Problems 1 and 3–6 were unchanged by the targeted Problem 2 run.

## Personal details that would make the report more individual

1. In Section 3, add the specific implementation or debugging task you found most difficult.
2. In Section 4, explain how you chose Problem 2 for the targeted run using your own decision
   process rather than only the numerical result.
3. In Section 5, describe what you noticed when you first compared the old and new Problem 2
   trajectories.
4. In Section 7, name the one extension you would implement first if given another day.
5. Optionally add one sentence about how you checked that you could explain the code without
   relying on the optimizer as a black box.

## Recommended existing figures and captions

1. **`results/pareto_summary.png`**  
   *Caption:* “Final distance–complexity Pareto fronts for all six target curves. Shaded area is
   the hypervolume relative to the assignment reference point; only non-dominated mechanisms
   are retained.”

2. **`results/problem_2_targeted_curves.png`**  
   *Caption:* “Aligned target and output trajectories for the final Problem 2 Pareto set. The
   targeted run found close matches with only five and seven joints; the four-joint linkage
   remains as the low-complexity endpoint.”

3. **Optional supporting figure: `results/problem_1_curves.png`**  
   *Caption:* “Problem 1 target and aligned trajectories across the retained complexity levels,
   illustrating how additional joints improve curve accuracy while changing the Pareto tradeoff.”

### Useful figure that does not currently exist

A compact Problem 2 before/after figure would strengthen Section 5. It could use
`results/problem2_targeted_baseline.json` and `my_submission.json` to show the old and new
five- and seven-joint trajectories or to plot their four before/after distance pairs. This
figure can be generated directly from saved results, but it is not currently in the repository.

## Implementation points to understand before an in-class explanation

- Why the crank nodes and crank edge are implicit, and why `x0` has `N-2` rows.
- How `dyadic_path`/`sort_mechanism` distinguish valid solve order from under- or
  over-constrained connectivity.
- Why a structurally valid mechanism can still lock during its crank rotation and return a
  non-finite distance.
- How scale normalization, cyclic correspondence, reversal, and rotation affect curve distance.
- Why arbitrary adjacency-bit mutation has such a low probability of producing useful
  offspring.
- How the topology-preserving differential and arithmetic operators modify positions without
  modifying edges or fixed joints.
- Why evolutionary search and gradient refinement solve different parts of the search problem.
- How `target_idx` is selected and why a different moving joint can be the best output.
- How the candidate archive prevents regression and supports resuming stochastic searches.
- How Pareto dominance removed the improved 6- and 8-joint Problem 2 designs.
- How hypervolume rewards both lower error and lower complexity, and why the reference point is
  `(1.5, 16)`.
- What the final validator checks beyond simply calling the distance function.
