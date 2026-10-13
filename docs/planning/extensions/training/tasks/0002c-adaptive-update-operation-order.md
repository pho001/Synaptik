# Training 0002C — Preserve documented Adam update operation order

## Status

Ready

## Goal and class

Class B numerical-behavior repair under the existing optimizer contract. Compute the adaptive
update as `(learningRate * correctedM) / denominator` in FLOAT32 and FLOAT64, validating the
scaled numerator and final update. Current quotient-first evaluation can reject finite
documented updates and disagrees with the independent test oracle.

## Scope and contracts

Own `TrainingParameter.stageAdaptiveElement32/64`, focused `AdamOptimizerTest` regressions and
affected Javadoc/explanation. No optimizer API, state layout, coefficient policy, rollback,
snapshot, or persistence changes. Low types continue to use the same FLOAT32 master path.
Read AGENTS; root scope index, ownership/dependency rules; `extensions-training.md` headings
`extensions/training` and `Optimizer/training lifecycle`; affected source/tests and optimizer guide.

## Acceptance and validation

- Reproduce Adam and AdamW two-step FLOAT32 case: initial scalar 1, learning rate/epsilon
  `1e-30`, beta1 `0.5`, beta2 `0`, no decay, gradients `1e19` then `0`. Second documented
  update is finite while quotient-first overflows. Add analogous FLOAT64 regression.
- Test both widths/algorithms, numeric result and committed counters/state. Preserve genuine
  non-finite-intermediate rejection and complete failure atomicity; do not weaken finite checks.
- Document the already specified operation order at the affected implementation contract.
- Focused regression, then `./gradlew :extensions:training:test :extensions:training:javadoc`
  once after stabilization. No full repository rerun; independent review reuses this evidence.
- Commit owned files only, no push; report files/tests/doc impact and Complete/Incomplete status.

## Dependencies and integration

- Depends on: Training 0002B Complete; project-review E1.
- Conflicts with: Training optimizer/formula/state writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea`, plus planning-only repair briefs.
- Integration order: independent of Compiler/Prepare/Tuning repairs.
- Integration validation: module tests/Javadoc, independent numerical/documentation review,
  coordinator's combined repository checkpoint.
- Shared-document integration owner: coordinator; no worker roadmap/master-plan edits.

Launch verified 2026-10-13: public CPU diagnostic confirms finite rejection, source and contract
agree on the bounded repair, predecessors complete, no competing Training writer. Use clean
implementation context and isolated worktree. Data/Checkpoint are not prerequisites.
