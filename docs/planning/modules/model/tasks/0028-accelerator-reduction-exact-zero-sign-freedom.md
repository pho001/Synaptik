# Task 0028: ACCELERATOR Reduction Exact-Zero Sign Freedom

## Status

Complete

Readiness was verified from Model 0027, Config 0006, Engine 0018, CPU 0017, and Metal 0019.
The corrected mandatory Metal 0017 probe found one existing-contract mismatch before any production
edit, so Metal 0017 stayed Blocked and no backend change was active. Model 0026 remained an
independent FLOAT16 Draft. This task executed as the sole Ready frontier from clean base `92b4fffc`
plus its planning-only authorization.

## Change class

Class C — this changes the Model-owned cross-module numerical allowed-result set used by capability,
conformance, preparation, execution, and cache identity. It changes no executable backend capability
or public configuration API, and requires a clean implementation context plus independent targeted
architecture/documentation review.

## Goal

Make the smallest operation-specific `ACCELERATOR` semantic refinement needed for arithmetic
`FLOAT32` `SUM`, `MEAN`, and `SUM_TO_SHAPE`: a final exact-zero arithmetic result may use either
zero sign. Keep `STRICT_IEEE` unchanged, preserve all-and-only declared terms, and preserve every
copy, identity, empty-domain, count, mapping, classification, and nonzero result rule.

## Chosen observable contract

The freedom is final-result-only; per-step exact-zero sign freedom is unnecessary and is not added.
For each output coordinate under `ACCELERATOR`:

- derive exactly the declared ordered-domain term multiset after axes, masking, and target-Shape
  mapping, then use the existing permitted full binary trees, per-step `FLOAT32` rounding, and
  row-scoped DAZ/FTZ;
- a `SUM` or arithmetic `SUM_TO_SHAPE` cell with at least two declared terms may publish either
  signed zero only when the selected permitted tree's root addition produces exact zero; every
  nonzero, infinity, and NaN result remains governed by the existing contract;
- `MEAN` must divide a permitted sum by the declared positive selected count in `FLOAT32`; only when
  that mandatory quotient is exact zero may the published quotient use either zero sign;
- an empty multi-axis point form, equal-Shape `SUM_TO_SHAPE`, or any other one-term `SUM`/
  `SUM_TO_SHAPE` cell performs no addition and preserves the selected input bits; no synthetic
  positive-zero identity may be inserted to manufacture arithmetic or zero-sign freedom;
- empty-domain `SUM` remains its exact positive-zero identity, zero-count `MEAN` remains NaN, and
  all existing masked-empty and special-value rules remain unchanged.

Existing FTZ already permits either sign only when a finite subnormal arithmetic result is flushed.
This task separately permits either sign for an exact-zero arithmetic root or final mean quotient;
it grants no per-step sign choice, tolerance, term loss, term duplication, new term, added identity,
reciprocal multiply, algebraic rewrite, reduced precision, cross-node contraction, or generic fast
math. The rule applies to the Model operation family rather than naming Metal or any backend form.

## Scope

- Amend the sole normative numerical-profile table in the foundational Model contract with the
  final-result rule and explicit identity/one-term/empty-domain exclusions.
- Amend ADR 0019 to record why result-level freedom is sufficient and why broader per-step freedom,
  a Metal exception, and a generic tolerance are rejected.
- Align `AggregateReductionKind` profile Javadocs, including the strict per-kind signed-zero rules;
  review public Tensor reduction Javadocs and update only passages whose profile wording would
  otherwise be incomplete or contradictory.
- Update focused Model/module-boundary, Tensor API, glossary, and numerical-profile explanations so
  authority, current semantics, and lack of backend capability remain clear.
- Use a disposable raw-bit proof to compare final-root freedom with hypothetical per-step
  exact-zero sign choices over bounded small trees, signed zeros, cancellation, subnormals,
  infinities, and NaNs; remove it after recording the conclusion.

## Non-goals

No Model evaluator or numerical-profile Java enum; no Config, Planning, Compiler, Prepare, Runtime,
Trace, Engine, CPU, Metal, native, schema, codec, capability, route, or cache implementation change.
No permanent test-only evaluator or source-text/Javadoc assertion. Do not mark Metal 0017 unblocked,
promote Metal 0020, or alter MATMUL/convolution/scan/extrema/comparison/unary semantics. Do not add
FLOAT16 or change Model 0026.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — Model remains the sole semantic owner and later layers may not reinterpret the result set.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  and [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — operation meanings and the sole normative profile table are Model-owned.
- [ADR 0019 — Explicit numerical profiles](../../../../design/decisions/0019-explicit-numerical-profiles.md)
  — retain the graph-wide, opt-in, operation-specific bounded design and its gross-error exclusions.

If the result-level rule cannot admit every bounded proof outcome without changing an intermediate,
identity, term, classification, or nonzero result, stop and report the counterexample; do not widen
the task silently.

## Dependencies and integration

- Depends on: Model 0027; retained corrected blocker evidence from Metal 0017
- Conflicts with: Model 0026; Metal 0020; any edit to the numerical-profile table, ADR 0019,
  reduction Javadocs, or shared profile documentation
- Parallel group: None
- Common base revision: `92b4fffc` plus this planning-only authorization
- Integration order: First; before Metal 0020 and Metal 0018
- Integration validation: accelerator reduction exact-zero semantic-authority checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `docs/architecture/contracts/foundational-modules.md` — sole normative reduction row and
  unchanged global exclusions.
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — bounded refinement rationale and
  rejected broader alternatives.
- `AggregateReductionKind` and affected `Tensor` reduction Javadocs — profile-indexed observable
  semantics without backend or algorithm claims.
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, and targeted
  `docs/glossary.md` entries — focused explanatory alignment.
- Existing `ReductionSemanticsTest`, `TensorNumericReductionTest`,
  `TensorMultiAxisReductionTest`, `TensorMaskedReductionTest`, and
  `TensorSumToShapeExpressionTest` — unchanged Model vocabulary/construction regression coverage;
  add no text-pinning test.
- Model/Metal master plans and `docs/planning/roadmap.md` — completion/frontier synchronization only
  after implementation and review.

## Acceptance criteria

1. The sole normative table states exactly the final-result contract above; `STRICT_IEEE`, every
   non-`FLOAT32` type, and every other operation-family row are unchanged.
2. The bounded raw-bit proof finds no nonzero or classification outcome that requires per-step
   exact-zero sign freedom; it covers one-, two-, and larger-term trees and is removed afterward.
3. All and only declared terms remain mandatory. Axis order, masking, target-Shape mapping, per-step
   rounding, DAZ/FTZ, and the final positive-count MEAN divide remain mandatory.
4. One-term and identity/copy forms preserve represented input bits; equal-Shape `SUM_TO_SHAPE` is
   bit-preserving; empty SUM and zero-count MEAN retain their current exact outcomes. No added
   identity turns a copy into arithmetic.
5. NaN/infinity classification, finite nonzero results, gross-error exclusions, and lack of any
   tolerance or backend-specific exception remain explicit.
6. Architecture, ADR, Javadocs, Tensor/API explanation, and targeted glossary use one consistent
   rule and do not imply that Model evaluates values or that any backend supports the wider set.
7. Focused Model tests, Model Javadoc, architecture tests, Markdown/status/frontier checks, and diff
   checks pass; independent Class C review has no unresolved finding.

## Validation

Worker validation:

```bash
python3 /tmp/synaptik-model-0028-zero-sign-proof.py
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
rm -f /tmp/synaptik-model-0028-zero-sign-proof.py
git diff --check
```

The proof must compare raw-bit allowed sets, distinguish exact zero from FTZ, and report bounded
case/tree counts. Inspect generated `AggregateReductionKind` and affected Tensor Javadocs. Confirm
no executable Java statement, test source, Gradle, configuration, backend, native, Runtime, Prepare,
Compiler, Engine, or Trace file changed; existing tests validate unchanged construction behavior,
not documentation text.

Integration/repository validation: after independent review, rerun only changed documentation and
architecture checks plus `git diff --check`; no backend suite is required because this task adds no
capability.

## Follow-up

- Metal 0020 is now Ready after this task completed, independent approval passed, and its base,
  dependencies, conflicts, contract, and mandatory full fresh corrected probe were reverified.
- Metal 0018 remains Draft after Metal 0020; Model 0026 remains an independent Draft.

## Documentation and review impact

This changes an authoritative numerical contract, ADR rationale, public operation Javadocs, and
user-facing semantic explanation. A separate clean Class C review/documentation context is
mandatory. It must review the raw-bit proof, narrowness of root-only freedom, strict and identity
preservation, all-and-only/count invariants, gross-error exclusions, single-owner wording, links,
glossary impact, and absence of backend implementation or capability drift.

## Result

Implemented the final-result-only exact-zero sign refinement at `fc003ab8`. The foundational
contract, ADR 0019, `AggregateReductionKind` and `Tensor` Javadocs, Tensor API, module-boundary
explanation, and targeted glossary entries now consistently preserve strict behavior, all terms,
bit-preserving identity/copy forms, the positive-count MEAN divide, and every gross-error exclusion.
No backend capability, native schema, cache identity, executable Java statement, test source,
Gradle, Config, Planning, Compiler, Prepare, Runtime, Trace, Engine, CPU, or Metal file changed.

The disposable raw-bit proof covered 2,427 one- through four-term multisets and 147,471 permitted
ordered binary trees over signed zeros, cancellation, subnormals, infinities, and NaNs. Root-only
exact-zero sign freedom and hypothetical per-step freedom produced identical final SUM and MEAN
allowed-result sets, so no broader freedom was needed; the proof artifact was removed.

The combined Model test, Model Javadoc, and architecture validation passed all 23 Gradle tasks.
Markdown, exact-path, whitespace, and diff checks passed. Independent Class C review returned
`APPROVE` with zero findings. Metal 0017 remains Blocked under its historical contract; Metal 0020
is the sole Ready frontier, while Metal 0018 and Model 0026 remain Draft.

Status: Complete
