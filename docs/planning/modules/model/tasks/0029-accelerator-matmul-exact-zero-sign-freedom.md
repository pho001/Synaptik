# Task 0029: ACCELERATOR MATMUL Exact-Zero Sign Freedom

## Status

Complete

Completion verification: task 0029 completed at `308267837a17c2fdbeb8b90ad381d1f899980786`
from clean base `0625a18727ac2cf59220cfa7db5de93e1cac2651`. Its disposable proof covered
1,823 cases, 168,461 ordered reassociation trees, 1,895,389 legal unfused/FMA plans, and
3,232 baseline root states. The 23-task Model/Javadoc/architecture validation, Markdown and diff
checks, and intended documentation/Javadoc-only scope passed. Independent Class C review
reconstructed the proof and returned `APPROVE` with zero findings. Metal 0018 remains Blocked
under its unchanged historical contract; Metal 0021 is the independently authorized successor.

## Change class

Class C — this changes the Model-owned cross-module numerical allowed-result set used by
capability, conformance, preparation, execution, and cache identity. It adds no backend capability
or public configuration API and requires a clean implementation context plus independent targeted
architecture/documentation review.

## Goal

Make the smallest backend-neutral `ACCELERATOR` refinement that admits the observed `FLOAT32`
`MATMUL` result: after one complete nonempty declared contraction has produced exact zero under an
otherwise permitted evaluation, the published MATMUL cell may use either zero sign. Keep
`STRICT_IEEE`, convolution, every declared multiply/add/FMA and intermediate result, empty
contractions, nonzero values, and result classifications unchanged.

## Chosen observable contract

Split the current shared `MATMUL`/`CONV2D`/`CONV3D` table row. Change only `MATMUL`; the two
convolution kinds retain the current reassociation/FMA plus DAZ/FTZ contract without new exact-zero
sign freedom. For each `ACCELERATOR` FLOAT32 MATMUL output coordinate:

- derive all and only the declared pairwise products for the contraction dimension and evaluate one
  currently permitted complete contraction, including existing reassociation, corresponding
  multiply-add FMA placement, FLOAT32 rounding, and row-scoped DAZ/FTZ;
- preserve each declared multiplication's result, each FMA/addition result before publication, and
  every existing sign rule; no product or intermediate exact zero gains a sign choice;
- only after the complete nonempty contraction's otherwise permitted result is exact zero may the
  published MATMUL cell use either zero sign; this includes a one-term contraction without adding
  a positive-zero accumulator or converting the multiplication into an FMA;
- preserve the exact positive-zero identity for an empty contraction, every finite nonzero result,
  infinity/NaN classification, and existing NaN-representation freedom.

The rule changes no subnormal into zero except through existing FTZ, grants no term loss or added
identity, and applies to Model MATMUL rather than a rank, backend, library selector, or topology.

## Scope

- Split and amend only the MATMUL part of the sole normative numerical-profile contraction row;
  retain the convolution row's current meaning verbatim in substance.
- Amend ADR 0019 with the final-publication decision and rejection of per-product, per-intermediate,
  shared-convolution, backend-specific, and tolerance alternatives.
- Align `MatmulKind` and public `Tensor.matmul`/linear-composition Javadocs. Review `Conv2dKind`,
  `Conv3dKind`, and affected Tensor convolution Javadocs and make their unchanged boundary explicit
  only where the row split would otherwise leave ambiguity.
- Align the focused module-boundary, Tensor API, glossary, and numerical-profile explanations.
- Use a disposable raw-bit proof over bounded contractions to show that final-only sign freedom
  admits the reported one-term result without changing any nonzero/classification result and that
  hypothetical product/intermediate sign freedom adds no observable final result needed here.

## Non-goals

No Model evaluator or profile Java enum; no Config, Planning, Compiler, Prepare, Runtime, Trace,
Engine, CPU, Metal, native, schema, codec, capability, route, or cache implementation change. No
per-product or intermediate zero-sign freedom, term loss/duplication/invention, implicit identity,
tolerance, reduced precision, reciprocal substitution, algebraic rewrite, new FMA, or cross-node
contraction. The implementation did not change convolution, reduction, scan, extrema, comparison,
unary, non-FLOAT32, or empty-contraction semantics. It did not reopen Metal 0018, promote the
then-Draft Metal 0021 successor, or change Model 0026.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — Model remains the sole semantic owner; later layers may realize but not reinterpret the set.
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  and [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — operation meanings and the sole normative profile table are Model-owned.
- [ADR 0019 — Explicit numerical profiles](../../../../design/decisions/0019-explicit-numerical-profiles.md)
  — retain graph-wide, opt-in, operation-specific bounds and all gross-error exclusions.

If final-publication-only freedom cannot admit the reported result without changing a product,
intermediate, term, identity, classification, or nonzero result, stop and report the counterexample;
do not widen the task silently.

## Dependencies and integration

- Depends on: Model 0027; Model 0028; retained Metal 0018 blocker evidence
- Conflicts with: Model 0026; Metal 0021; any edit to the numerical-profile table, ADR 0019,
  MATMUL/convolution Javadocs, or shared profile documentation
- Parallel group: None
- Common base revision: `0625a18727ac2cf59220cfa7db5de93e1cac2651` plus this planning-only authorization
- Integration order: First; before Metal 0021
- Integration validation: accelerator MATMUL exact-zero semantic-authority checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `docs/architecture/contracts/foundational-modules.md` — split contraction rows, narrow MATMUL
  publication rule, unchanged convolution and global exclusions.
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — evidence, decision rationale,
  alternatives, and consequences.
- `MatmulKind`, `Tensor` MATMUL/linear Javadocs; targeted convolution Javadocs — observable profile
  semantics without backend or algorithm claims.
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, and targeted
  `docs/glossary.md` entries — focused explanatory alignment.
- Existing MATMUL/convolution Model tests — unchanged vocabulary/construction coverage; add no
  source-text test or permanent numerical evaluator.
- Model/Metal master plans and `docs/planning/roadmap.md` — completion/frontier synchronization only
  after implementation and review.

## Acceptance criteria

1. The sole normative table gives only nonempty FLOAT32 ACCELERATOR MATMUL final-publication
   exact-zero sign freedom; strict, convolution, empty contraction, and every other row/type stay
   unchanged.
2. Every declared product appears exactly once. Existing reassociation, legal FMA placements,
   FLOAT32 rounding, DAZ/FTZ, product signs, and intermediate signs remain the complete evaluation
   freedoms; no positive-zero accumulator or other identity is added.
3. A one-term `+0.0f * -1.0f` contraction permits published `+0` or `-0`, while the underlying
   multiply remains `-0`; finite nonzero values and NaN/infinity classification do not widen.
4. The disposable proof covers one- through four-term contractions, signed zeros, cancellation,
   subnormals, normal boundaries, infinities, and NaNs across reassociation and legal FMA choices;
   it reports raw-bit/class results and is removed.
5. ADR, Javadocs, API/architecture explanation, and targeted glossary consistently distinguish the
   MATMUL-only final-publication rule from unchanged convolution and backend capability.
6. Focused Model tests, Model Javadoc, architecture tests, Markdown/status/frontier checks, and diff
   checks pass; independent Class C review has no unresolved finding.

## Validation

Worker validation:

```bash
python3 /tmp/synaptik-model-0029-matmul-zero-sign-proof.py
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
rm -f /tmp/synaptik-model-0029-matmul-zero-sign-proof.py
git diff --check
```

The proof must distinguish multiplication/intermediate bits from final publication, distinguish
exact zero from FTZ, enumerate legal reassociation/FMA placements, and report bounded case/tree
counts. Inspect generated MATMUL and targeted convolution Javadocs. Confirm no executable Java
statement, test source, Gradle, configuration, backend, native, Runtime, Prepare, Compiler, Engine,
or Trace file changed.

Integration/repository validation: after independent review, rerun changed documentation and
architecture checks plus `git diff --check`; do not repeat successful Java tests unless executable
Java changed or evidence became stale.

## Follow-up

- Metal 0021 has been reverified against the completed contract and promoted separately to the
  sole Ready frontier from the clean integrated task-0029 revision. It must run a completely fresh
  full real-M3 oracle from the beginning before any production edit. Historical Metal 0018 remains
  Blocked.

## Documentation and review impact

This changes an authoritative numerical contract, ADR rationale, public operation Javadocs, and
user-facing semantic explanation. A separate clean Class C review/documentation context is
mandatory. It reviews proof completeness, MATMUL-only scope, final-publication placement, strict/
convolution/empty preservation, all-term and no-identity invariants, gross-error exclusions,
terminology, links, glossary impact, and absence of backend implementation or capability drift.

## Result

Complete at `308267837a17c2fdbeb8b90ad381d1f899980786`. The normative profile table now separates
MATMUL from convolution and permits either sign only at publication of a complete nonempty
otherwise-permitted exact-zero FLOAT32 ACCELERATOR MATMUL result. Products, additions, FMA
results, empty contraction, nonzero values, classifications, strict behavior, and convolution
remain unchanged.

The removed disposable proof reported 1,823 cases, 168,461 ordered reassociation trees,
1,895,389 legal unfused/FMA evaluation plans, and 3,232 baseline root states. The 23-task
Model/Javadoc/architecture validation and Markdown/diff scope checks passed. Independent Class C
review reconstructed the proof and returned `APPROVE` with zero findings. No executable Java
statement, test source, backend, native, schema, identity, configuration, or build file changed.

Status: Complete
