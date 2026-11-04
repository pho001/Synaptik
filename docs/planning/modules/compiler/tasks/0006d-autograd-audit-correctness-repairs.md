# [Compiler](../../../../glossary.md#term-compiler) 0006D — Repair batched [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [gradients](../../../../glossary.md#term-gradient) and [fail-closed](../../../../glossary.md#term-fail-closed) sibling paths

> **Editorial reading key — not part of the historical record.**
>
> A derivative route is positional as well as identity-based: batched matrix/vector gradients need
> the correct inserted/squeezed axes before genuine batch reduction. Separately, an unsupported
> required sibling route cannot be discarded merely because another route reaches the same target.
> That is different from intentionally non-differentiable masks or indices. Compiler repairs
> formulas and preflight; the numerical CPU regression is downstream evidence, not a new
> backend-owned differentiation rule.
>
> Current reading: [pre-capture differentiation and functional requests](../../../../reference/compiler/autograd.md#current-package-private-pre-capture-autograd).

## Status

Complete

## Change class and goal

Class C: [Compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) semantics and [fail-closed](../../../../glossary.md#term-fail-closed) [preflight](../../../../glossary.md#term-preflight). Repair the two confirmed
project-review defects within existing contracts, without adding [derivative](../../../../glossary.md#term-derivative) capability.

## Findings and [scope](../../../../glossary.md#term-scope)

1. `LinearAlgebraGradientRules.apply`, [matrix](../../../../glossary.md#term-matrix-meanings)/[vector](../../../../glossary.md#term-vector) right [gradient](../../../../glossary.md#term-gradient): `T(left) @ gradient`
   misinterprets a batched cotangent as a matrix. <span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent))</span>`[2,3,4] @ [4]` fails when differentiating the
   vector; coincident `[2,2,2] @ [2]` dimensions can conceal a wrong result. Restore the missing
   column axis, multiply, squeeze, then reduce batch dimensions.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
2. `AutogradPreflight` records non-differentiable and fail-closed paths together and ignores a
   blocked route when another route connects the same target. BatchNorm scale-gradient then
   input-gradient requests can silently omit saved-statistic contributions. Required fail-closed
   paths must reject even beside a differentiable path; intentionally non-differentiable
   comparison/mask/index roles must retain their existing behavior.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [target glossary](../../../../glossary.md#term-target); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>

Owned production: `LinearAlgebraGradientRules`, `AutogradPreflight`, and only if necessary
`FirstOrderGradientCoverage`, under `modules/compiler/.../compiler`. Owned [validation](../../../../glossary.md#term-validation): focused
[Compiler](../../../../glossary.md#term-compiler) tests and a bounded public [CPU](../../../../glossary.md#term-cpu) numerical [regression](../../../../glossary.md#term-regression) under `testing/integration-tests`
if that is the smallest existing execution seam. No Compiler [dependency](../../../../glossary.md#term-dependency) on [Engine](../../../../glossary.md#term-engine)/[backends](../../../../glossary.md#backend).
Review formula/[preflight](../../../../glossary.md#term-preflight) [Javadoc](../../../../glossary.md#term-javadoc) and relevant compiler/[autograd](../../../../glossary.md#autograd) explanation. No new BatchNorm
saved-[statistic](../../../../glossary.md#term-statistic) rule, [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [Conv3d](../../../../glossary.md#conv3d--ncdhw) [adjoint](../../../../glossary.md#term-adjoint), rewrite policy, numerical relaxation, or [API](../../../../glossary.md#term-application-programming-interface-api).

## Required reading and contracts

Read `AGENTS.md`, root `ARCHITECTURE.md` [scope](../../../../glossary.md#term-scope) index, Core invariants, [Module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings) routing,
[Dependency](../../../../glossary.md#term-dependency) rules, and Testing requirements. Read `docs/architecture/contracts/compiler-autograd.md`
headings `modules/compiler`, `Compiler-owned automatic differentiation`, `Training graph model`,
and `Compile lifecycle`; [Model](../../../../glossary.md#term-model) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/broadcast and BatchNorm saved-output family Javadocs.
Inspect the affected sources, `GradientRulesTest`, `AutogradPreflightTest`, functional-[gradient](../../../../glossary.md#term-gradient)
tests, and any existing integration [fixture](../../../../glossary.md#term-fixture) reused for execution.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Numerical batched matrix/vector gradients cover unequal dimensions, coincident dimensions,
  multiple batch axes, nonuniform explicit seeds, both operand roles, and second-stage closure.
  For <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [dimensions glossary](../../../../glossary.md#dimension); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [axes glossary](../../../../glossary.md#term-axis); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`A[2,2,2] = 1..8` and unit output [cotangent](../../../../glossary.md#cotangent), [vector](../../../../glossary.md#term-vector) [gradient](../../../../glossary.md#term-gradient) is `[16,20]`, not `[36,36]`.
- The two-stage `sum(batchNormTraining(...).output()) -> scale -> x` request fails closed on its
  saved-statistic path before constructing an incomplete second derivative. Test mixed saved/
  ordinary paths, relevant disconnected policies, and legal non-differentiable mask branches.<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [second derivative glossary](../../../../glossary.md#term-second-derivative); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Reject only required paths in the selected derivative subgraph, not unrelated outputs or
  intentionally non-differentiable boundaries. Preserve existing first-order BatchNorm.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- Add focused regressions that fail on the baseline. [Run](../../../../glossary.md#term-run) them during development, then one
  `./gradlew :modules:compiler:test :modules:compiler:javadoc` run after stabilization and the
  selected integration class if added. Final repository checkpoint belongs to coordinator.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Independent Class C review checks path selection, higher-order implications, formulas,
  documentation, and numerical regression evidence, reusing successful suites.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Report files, evidence, documentation/[Javadoc](../../../../glossary.md#term-javadoc) impact, remaining issues, and completion status.

## Dependencies and integration

- Depends on: completed Compiler 0006/0006B11; existing profile-free Model semantics; verified
  project-review findings A1/A2. Not Draft 0006C or 0007.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Conflicts with: other [Compiler](../../../../glossary.md#term-compiler) [derivative](../../../../glossary.md#term-derivative)/[preflight](../../../../glossary.md#term-preflight)/formula writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: `ea6f81fa` plus this [planning](../../../../glossary.md#term-planning)-only brief; production base `728680ea`.
- Integration order: independent of Tuning 0005; merge each only after its own review.
- Integration [validation](../../../../glossary.md#term-validation): focused/[module](../../../../glossary.md#term-module) evidence, independent review, combined repository
  `test javadoc` [checkpoint](../../../../glossary.md#term-checkpoint), affected integration tests, and diff checks.
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator; executor does not edit roadmap/master plans.

Launch verification (2026-10-13): source review and read-only diagnostic reproduced both defects;
the current contract already requires correct [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) formulas and rejection of unsupported
derivative paths. User authorizes repair; predecessors are Complete; no competing Compiler
writer exists. Use a clean implementation context and isolated worktree.

## Integration record

Implementation `e6fb9d28` integrated as `5dead33c` after independent review.
292 [Compiler](../../../../glossary.md#term-compiler) tests, four numerical [CPU](../../../../glossary.md#term-cpu) integration tests and [Javadoc](../../../../glossary.md#term-javadoc); independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
