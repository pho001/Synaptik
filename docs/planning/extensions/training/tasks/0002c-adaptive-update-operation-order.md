# Training 0002C — Preserve documented [Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw) update [operation](../../../../glossary.md#operation) order

## Status

Complete

## Goal and class

Class B numerical-behavior repair under the existing [optimizer](../../../../glossary.md#term-optimizer) contract. Compute the adaptive
update as `(learningRate * correctedM) / denominator` in [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64), validating the
scaled numerator and final update. Current quotient-first [evaluation](../../../../glossary.md#term-evaluation) can reject [finite](../../../../glossary.md#term-finite)
documented updates and disagrees with the independent [test oracle](../../../../glossary.md#term-test-oracle).

## [Scope](../../../../glossary.md#term-scope) and contracts

Own `TrainingParameter.stageAdaptiveElement32/64`, focused `AdamOptimizerTest` regressions and
affected [Javadoc](../../../../glossary.md#term-javadoc)/explanation. No [optimizer](../../../../glossary.md#term-optimizer) [API](../../../../glossary.md#term-application-programming-interface-api), state [layout](../../../../glossary.md#layout), coefficient policy, [rollback](../../../../glossary.md#term-rollback),
[snapshot](../../../../glossary.md#term-snapshot), or [persistence](../../../../glossary.md#term-persistence) changes. Low types continue to use the same [FLOAT32](../../../../glossary.md#term-float32) master path.
Read AGENTS; root scope index, [ownership](../../../../glossary.md#term-ownership-meanings)/[dependency](../../../../glossary.md#term-dependency) rules; `extensions-training.md` headings
`extensions/training` and `Optimizer/training lifecycle`; affected source/tests and optimizer guide.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Reproduce [Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw) and [AdamW](../../../../glossary.md#adaptive-optimizer--adam-adamw) two-step [FLOAT32](../../../../glossary.md#term-float32) case: initial [scalar](../../../../glossary.md#term-scalar) 1, [learning rate](../../../../glossary.md#term-learning-rate)/[epsilon](../../../../glossary.md#epsilon)
  `1e-30`, beta1 `0.5`, beta2 `0`, no decay, [gradients](../../../../glossary.md#term-gradient) `1e19` then `0`. Second documented
  update is finite while quotient-first overflows. Add analogous FLOAT64 regression.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Test both widths/algorithms, numeric result and committed counters/state. Preserve genuine
  non-finite-intermediate rejection and complete failure atomicity; do not weaken finite checks.<span data-glossary-navigation="true"> ([failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
- Document the already specified [operation](../../../../glossary.md#operation) order at the affected implementation contract.
- Focused [regression](../../../../glossary.md#term-regression), then `./gradlew :extensions:training:test :extensions:training:javadoc`
  once after stabilization. No full repository rerun; independent review reuses this evidence.
- [Commit](../../../../glossary.md#term-commit) owned files only, no push; report files/tests/doc impact and Complete/Incomplete status.

## Dependencies and integration

- Depends on: Training 0002B Complete; project-review E1.
- Conflicts with: Training [optimizer](../../../../glossary.md#term-optimizer)/formula/state writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea`, plus [planning](../../../../glossary.md#term-planning)-only repair briefs.
- Integration order: independent of [Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/Tuning repairs.
- Integration validation: module tests/Javadoc, independent numerical/documentation review,
  coordinator's combined repository checkpoint.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator; no worker roadmap/master-plan edits.

Launch verified 2026-10-13: public [CPU](../../../../glossary.md#term-cpu) diagnostic confirms [finite](../../../../glossary.md#term-finite) rejection, source and contract
agree on the bounded repair, predecessors complete, no competing Training writer. Use clean
implementation context and isolated worktree. Data/[Checkpoint](../../../../glossary.md#term-checkpoint) are not prerequisites.

## Integration record

Implementation `97e66ee3` integrated as `1f9ea4bd` after independent review.
11 focused / 68 [module](../../../../glossary.md#term-module) tests and [Javadoc](../../../../glossary.md#term-javadoc); independent code/documentation APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
