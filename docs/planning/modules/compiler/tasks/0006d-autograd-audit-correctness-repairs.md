# Compiler 0006D — Repair batched MATMUL gradients and fail-closed sibling paths

## Status

Complete

## Change class and goal

Class C: Compiler autograd semantics and fail-closed preflight. Repair the two confirmed
project-review defects within existing contracts, without adding derivative capability.

## Findings and scope

1. `LinearAlgebraGradientRules.apply`, matrix/vector right gradient: `T(left) @ gradient`
   misinterprets a batched cotangent as a matrix. `[2,3,4] @ [4]` fails when differentiating the
   vector; coincident `[2,2,2] @ [2]` dimensions can conceal a wrong result. Restore the missing
   column axis, multiply, squeeze, then reduce batch dimensions.
2. `AutogradPreflight` records non-differentiable and fail-closed paths together and ignores a
   blocked route when another route connects the same target. BatchNorm scale-gradient then
   input-gradient requests can silently omit saved-statistic contributions. Required fail-closed
   paths must reject even beside a differentiable path; intentionally non-differentiable
   comparison/mask/index roles must retain their existing behavior.

Owned production: `LinearAlgebraGradientRules`, `AutogradPreflight`, and only if necessary
`FirstOrderGradientCoverage`, under `modules/compiler/.../compiler`. Owned validation: focused
Compiler tests and a bounded public CPU numerical regression under `testing/integration-tests`
if that is the smallest existing execution seam. No Compiler dependency on Engine/backends.
Review formula/preflight Javadoc and relevant compiler/autograd explanation. No new BatchNorm
saved-statistic rule, BPTT, Conv3d adjoint, rewrite policy, numerical relaxation, or API.

## Required reading and contracts

Read `AGENTS.md`, root `ARCHITECTURE.md` scope index, Core invariants, Module-ownership routing,
Dependency rules, and Testing requirements. Read `docs/architecture/contracts/compiler-autograd.md`
headings `modules/compiler`, `Compiler-owned automatic differentiation`, `Training graph model`,
and `Compile lifecycle`; Model MATMUL/broadcast and BatchNorm saved-output family Javadocs.
Inspect the affected sources, `GradientRulesTest`, `AutogradPreflightTest`, functional-gradient
tests, and any existing integration fixture reused for execution.

## Acceptance and validation

- Numerical batched matrix/vector gradients cover unequal dimensions, coincident dimensions,
  multiple batch axes, nonuniform explicit seeds, both operand roles, and second-stage closure.
  For `A[2,2,2] = 1..8` and unit output cotangent, vector gradient is `[16,20]`, not `[36,36]`.
- The two-stage `sum(batchNormTraining(...).output()) -> scale -> x` request fails closed on its
  saved-statistic path before constructing an incomplete second derivative. Test mixed saved/
  ordinary paths, relevant disconnected policies, and legal non-differentiable mask branches.
- Reject only required paths in the selected derivative subgraph, not unrelated outputs or
  intentionally non-differentiable boundaries. Preserve existing first-order BatchNorm.
- Add focused regressions that fail on the baseline. Run them during development, then one
  `./gradlew :modules:compiler:test :modules:compiler:javadoc` run after stabilization and the
  selected integration class if added. Final repository checkpoint belongs to coordinator.
- Independent Class C review checks path selection, higher-order implications, formulas,
  documentation, and numerical regression evidence, reusing successful suites.
- Report files, evidence, documentation/Javadoc impact, remaining issues, and completion status.

## Dependencies and integration

- Depends on: completed Compiler 0006/0006B11; existing profile-free Model semantics; verified
  project-review findings A1/A2. Not Draft 0006C or 0007.
- Conflicts with: other Compiler derivative/preflight/formula writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: `ea6f81fa` plus this planning-only brief; production base `728680ea`.
- Integration order: independent of Tuning 0005; merge each only after its own review.
- Integration validation: focused/module evidence, independent review, combined repository
  `test javadoc` checkpoint, affected integration tests, and diff checks.
- Shared-document integration owner: coordinator; executor does not edit roadmap/master plans.

Launch verification (2026-10-13): source review and read-only diagnostic reproduced both defects;
the current contract already requires correct MATMUL formulas and rejection of unsupported
derivative paths. User authorizes repair; predecessors are Complete; no competing Compiler
writer exists. Use a clean implementation context and isolated worktree.

## Integration record

Implementation `e6fb9d28` integrated as `5dead33c` after independent review.
292 Compiler tests, four numerical CPU integration tests and Javadoc; independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository checkpoint passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's scope.

Status: Complete
