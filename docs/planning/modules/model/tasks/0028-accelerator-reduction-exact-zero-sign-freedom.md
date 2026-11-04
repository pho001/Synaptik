# Task 0028: [ACCELERATOR](../../../../glossary.md#term-accelerator) Reduction Exact-Zero Sign Freedom

## Status

Complete

Readiness was verified from [Model](../../../../glossary.md#term-model) 0027, Config 0006, [Engine](../../../../glossary.md#term-engine) 0018, [CPU](../../../../glossary.md#term-cpu) 0017, and [Metal](../../../../glossary.md#term-metal) 0019.
The corrected mandatory Metal 0017 probe found one existing-contract mismatch before any production
edit, so Metal 0017 stayed Blocked and no [backend](../../../../glossary.md#backend) change was active. Model 0026 remained an
independent [FLOAT16](../../../../glossary.md#term-float16) Draft. This task executed as the sole [Ready](../../../../glossary.md#term-ready) frontier from clean base `92b4fffc`
plus its [planning](../../../../glossary.md#term-planning)-only authorization.

## Change class

Class C — this changes the [Model](../../../../glossary.md#term-model)-owned cross-[module](../../../../glossary.md#term-module) numerical allowed-result set used by capability,
[conformance](../../../../glossary.md#term-conformance), preparation, execution, and [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings). It changes no executable [backend capability](../../../../glossary.md#backend-capability)
or public configuration [API](../../../../glossary.md#term-application-programming-interface-api), and requires a clean implementation context plus independent targeted
architecture/documentation review.

## Goal

Make the smallest [operation](../../../../glossary.md#operation)-specific [`ACCELERATOR`](../../../../glossary.md#term-accelerator) semantic refinement needed for arithmetic
[`FLOAT32`](../../../../glossary.md#term-float32) `SUM`, [`MEAN`](../../../../glossary.md#term-mean), and `SUM_TO_SHAPE`: a final exact-zero arithmetic result may use either
zero sign. Keep `STRICT_IEEE` unchanged, preserve all-and-only declared terms, and preserve every
copy, [identity](../../../../glossary.md#term-identity-meanings), empty-domain, count, mapping, classification, and nonzero result rule.

## Chosen observable contract

The freedom is final-result-only; per-step exact-zero sign freedom is unnecessary and is not added.
For each output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) under [`ACCELERATOR`](../../../../glossary.md#term-accelerator):

- derive exactly the declared ordered-domain term multiset after axes, masking, and target-Shape
  mapping, then use the existing permitted full binary trees, per-step <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>[`FLOAT32`](../../../../glossary.md#term-float32) rounding, and
  row-scoped DAZ/FTZ;<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz))</span>
- a `SUM` or arithmetic `SUM_TO_SHAPE` cell with at least two declared terms may publish either
  signed zero only when the selected permitted tree's root addition produces exact zero; every
  nonzero, infinity, and NaN result remains governed by the existing contract;<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan))</span>
- [`MEAN`](../../../../glossary.md#term-mean) must divide a permitted sum by the declared positive selected count in `FLOAT32`; only when
  that mandatory quotient is exact zero may the published quotient use either zero sign;
- an empty multi-[axis](../../../../glossary.md#term-axis) point form, equal-[Shape](../../../../glossary.md#shape) `SUM_TO_SHAPE`, or any other one-term `SUM`/
  `SUM_TO_SHAPE` cell performs no addition and preserves the selected input bits; no synthetic
  positive-zero identity may be inserted to manufacture arithmetic or zero-sign freedom;
- empty-domain `SUM` remains its exact positive-zero [identity](../../../../glossary.md#term-identity-meanings), zero-count `MEAN` remains NaN, and
  all existing masked-empty and special-value rules remain unchanged.

Existing [FTZ](../../../../glossary.md#term-ftz) already permits either sign only when a [finite](../../../../glossary.md#term-finite) [subnormal](../../../../glossary.md#term-subnormal) arithmetic result is flushed.
This task separately permits either sign for an exact-zero arithmetic root or final mean quotient;
it grants no per-step sign choice, [tolerance](../../../../glossary.md#term-tolerance), term [loss](../../../../glossary.md#loss), term duplication, new term, added identity,
reciprocal multiply, algebraic rewrite, reduced [precision](../../../../glossary.md#term-precision), cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension), or generic fast
math. The rule applies to the [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) family rather than naming [Metal](../../../../glossary.md#term-metal) or any [backend](../../../../glossary.md#backend) form.

## [Scope](../../../../glossary.md#term-scope)

- Amend the sole normative numerical-profile table in the foundational Model contract with the
  final-result rule and explicit identity/one-term/empty-domain exclusions.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Amend ADR 0019 to record why result-level freedom is sufficient and why broader per-step freedom,
  a Metal exception, and a generic tolerance are rejected.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Metal glossary](../../../../glossary.md#term-metal); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Align `AggregateReductionKind` profile Javadocs, including the strict per-kind signed-zero rules;
  review public Tensor reduction Javadocs and update only passages whose profile wording would
  otherwise be incomplete or contradictory.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Update focused Model/module-boundary, Tensor API, glossary, and numerical-profile explanations so
  authority, current semantics, and lack of backend capability remain clear.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Use a disposable raw-bit proof to compare final-root freedom with hypothetical per-step
  exact-zero sign choices over bounded small trees, signed zeros, cancellation, subnormals,
  infinities, and NaNs; remove it after recording the conclusion.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

## Non-goals

No [Model](../../../../glossary.md#term-model) evaluator or numerical-profile Java enum; no Config, [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime),
[Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), native, [schema](../../../../glossary.md#term-schema), [codec](../../../../glossary.md#term-codec), capability, route, or [cache](../../../../glossary.md#term-cache) implementation change.
No permanent test-only evaluator or source-text/[Javadoc](../../../../glossary.md#term-javadoc) assertion. Do not mark Metal 0017 unblocked,
promote Metal 0020, or alter [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[scan](../../../../glossary.md#term-cumulative-scan)/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/comparison/[unary](../../../../glossary.md#term-unary) semantics. Do not add
[FLOAT16](../../../../glossary.md#term-float16) or change Model 0026.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  — Model remains the sole semantic owner and later layers may not reinterpret the result set.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  and [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — operation meanings and the sole normative profile table are Model-owned.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [ADR 0019 — Explicit numerical profiles](../../../../design/decisions/0019-explicit-numerical-profiles.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
  — retain the graph-wide, opt-in, operation-specific bounded design and its gross-error exclusions.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

If the result-level rule cannot admit every bounded proof outcome without changing an intermediate,
[identity](../../../../glossary.md#term-identity-meanings), term, classification, or nonzero result, stop and report the counterexample; do not widen
the task silently.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0027; retained corrected blocker evidence from [Metal](../../../../glossary.md#term-metal) 0017
- Conflicts with: Model 0026; Metal 0020; any edit to the numerical-profile table, ADR 0019,
  reduction Javadocs, or shared profile documentation<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Parallel group: None
- Common base revision: `92b4fffc` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: First; before Metal 0020 and Metal 0018
- Integration [validation](../../../../glossary.md#term-validation): [accelerator](../../../../glossary.md#term-accelerator) reduction exact-zero semantic-authority [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `docs/architecture/contracts/foundational-modules.md` — sole normative reduction row and
  unchanged global exclusions.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — bounded refinement rationale and
  rejected broader alternatives.
- `AggregateReductionKind` and affected [`Tensor`](../../../../glossary.md#tensor) reduction Javadocs — profile-indexed observable
  semantics without backend or algorithm claims.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, and targeted
  `docs/glossary.md` entries — focused explanatory [alignment](../../../../glossary.md#term-alignment).
- Existing `ReductionSemanticsTest`, `TensorNumericReductionTest`,
  `TensorMultiAxisReductionTest`, `TensorMaskedReductionTest`, and
  `TensorSumToShapeExpressionTest` — unchanged Model vocabulary/construction regression coverage;
  add no text-pinning test.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- [Model](../../../../glossary.md#term-model)/[Metal](../../../../glossary.md#term-metal) master plans and `docs/planning/roadmap.md` — completion/frontier synchronization only
  after implementation and review.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Acceptance criteria

1. The sole [normative](../../../../glossary.md#term-normative) table states exactly the final-result contract above; `STRICT_IEEE`, every
   non-[`FLOAT32`](../../../../glossary.md#term-float32) type, and every other [operation](../../../../glossary.md#operation)-family row are unchanged.
2. The bounded raw-bit proof finds no nonzero or classification outcome that requires per-step
   exact-zero sign freedom; it covers one-, two-, and larger-term trees and is removed afterward.
3. All and only declared terms remain mandatory. Axis order, masking, target-Shape mapping, per-step
   rounding, DAZ/FTZ, and the final positive-count MEAN divide remain mandatory.<span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [target glossary](../../../../glossary.md#term-target); [rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [MEAN glossary](../../../../glossary.md#term-mean))</span>
4. One-term and [identity](../../../../glossary.md#term-identity-meanings)/copy forms preserve represented input bits; equal-[Shape](../../../../glossary.md#shape) `SUM_TO_SHAPE` is
   bit-preserving; empty SUM and zero-count MEAN retain their current exact outcomes. No added
   identity turns a copy into arithmetic.
5. NaN/infinity classification, finite nonzero results, gross-error exclusions, and lack of any
   tolerance or backend-specific exception remain explicit.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance); [backend glossary](../../../../glossary.md#backend))</span>
6. Architecture, ADR, Javadocs, Tensor/API explanation, and targeted glossary use one consistent
   rule and do not imply that Model evaluates values or that any backend supports the wider set.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>
7. Focused Model tests, Model Javadoc, architecture tests, Markdown/status/frontier checks, and diff
   checks pass; independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
python3 /tmp/synaptik-model-0028-zero-sign-proof.py
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
rm -f /tmp/synaptik-model-0028-zero-sign-proof.py
git diff --check
```

The proof must compare raw-bit allowed sets, distinguish exact zero from [FTZ](../../../../glossary.md#term-ftz), and report bounded
case/tree counts. Inspect generated `AggregateReductionKind` and affected [Tensor](../../../../glossary.md#tensor) Javadocs. Confirm
no executable Java statement, test source, [Gradle](../../../../glossary.md#term-gradle), configuration, [backend](../../../../glossary.md#backend), native, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare),
[Compiler](../../../../glossary.md#term-compiler), [Engine](../../../../glossary.md#term-engine), or [Trace](../../../../glossary.md#trace) file changed; existing tests validate unchanged construction behavior,
not documentation text.

Integration/repository validation: after independent review, rerun only changed documentation and
architecture checks plus `git diff --check`; no backend suite is required because this task adds no
capability.

## Follow-up

- Metal 0020 is now Ready after this task completed, independent approval passed, and its base,
  dependencies, conflicts, contract, and mandatory full fresh corrected probe were reverified.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- [Metal](../../../../glossary.md#term-metal) 0018 remains Draft after Metal 0020; [Model](../../../../glossary.md#term-model) 0026 remains an independent Draft.

## Documentation and review impact

This changes an authoritative numerical contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) rationale, public [operation](../../../../glossary.md#operation) Javadocs, and
user-facing semantic explanation. A separate clean Class C review/documentation context is
mandatory. It must review the raw-bit proof, narrowness of root-only freedom, strict and [identity](../../../../glossary.md#term-identity-meanings)
preservation, all-and-only/count invariants, gross-error exclusions, single-[owner](../../../../glossary.md#term-owner) wording, links,
glossary impact, and absence of [backend](../../../../glossary.md#backend) implementation or capability drift.

## Result

Implemented the final-result-only exact-zero sign refinement at `fc003ab8`. The foundational
contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0019, `AggregateReductionKind` and [`Tensor`](../../../../glossary.md#tensor) Javadocs, Tensor [API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module)-boundary
explanation, and targeted glossary entries now consistently preserve strict behavior, all terms,
bit-preserving [identity](../../../../glossary.md#term-identity-meanings)/copy forms, the positive-count [MEAN](../../../../glossary.md#term-mean) divide, and every gross-error exclusion.
No [backend capability](../../../../glossary.md#backend-capability), native [schema](../../../../glossary.md#term-schema), [cache](../../../../glossary.md#term-cache) identity, executable Java statement, test source,
[Gradle](../../../../glossary.md#term-gradle), Config, [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), or [Metal](../../../../glossary.md#term-metal) file changed.

The disposable raw-bit proof covered 2,427 one- through four-term multisets and 147,471 permitted
ordered [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) trees over signed zeros, [cancellation](../../../../glossary.md#term-cancellation), [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and NaNs. Root-only
exact-zero sign freedom and hypothetical per-step freedom produced identical final SUM and MEAN
allowed-result sets, so no broader freedom was needed; the proof [artifact](../../../../glossary.md#term-artifact) was removed.

The combined [Model](../../../../glossary.md#term-model) test, Model [Javadoc](../../../../glossary.md#term-javadoc), and architecture [validation](../../../../glossary.md#term-validation) passed all 23 Gradle tasks.
Markdown, exact-path, whitespace, and diff checks passed. Independent Class C review returned
`APPROVE` with zero findings. Metal 0017 remains Blocked under its historical contract; Metal 0020
is the sole [Ready](../../../../glossary.md#term-ready) frontier, while Metal 0018 and Model 0026 remain Draft.

Status: Complete
