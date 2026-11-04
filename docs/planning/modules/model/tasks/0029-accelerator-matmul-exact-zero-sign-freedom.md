# Task 0029: [ACCELERATOR](../../../../glossary.md#term-accelerator) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Exact-Zero Sign Freedom

**Editorial reading key (current context, not a status update).**

This record documents a historical Model contract refinement, not a backend capability or status
update. In the current API, a caller invokes `left.matmul(right)` to construct a storage-free
expression: operand descriptions and ordered input references establish a contraction, the shared
axis whose paired products contribute to each output cell. Construction neither reads operand values
nor executes or owns result buffers. Model defines numerical meaning; execution providers realize
supported routes without redefining it. Semantic responsibility is not storage ownership.

Here, “publication” means the completed cell’s observable value, not an intermediate arithmetic
result. For nonempty FLOAT32 `ACCELERATOR` MATMUL, positive zero multiplied by negative one still
produces negative zero, but that one-term cell may publish either zero sign, without an added
accumulator or FMA. Empty contractions remain positive zero; `STRICT_IEEE`, convolution, nonzero
results and classifications retain their boundaries. Follow the current [numerical meaning and
result
metadata](../../../../reference/tensor/matmul-linear.md#data-type-numerical-meaning-and-result-metadata)
explanation to distinguish expression construction from evaluation and locate the final-publication
rule; it does not extend to a subsequent linear-projection bias addition.

**Zero-sign reading illustration.** For a one-cell, one-term FLOAT32 contraction, let $a=+0$ and
$b=-1$. The product has representation $-0$; the historical permission lets the completed nonempty
cell publish either zero sign. That permission is not a second arithmetic operation: it adds no
accumulator, fusion or bias. An empty contraction has no product at all and retains its separate
positive-zero identity.

## Status

Complete

Completion verification: task 0029 completed at `308267837a17c2fdbeb8b90ad381d1f899980786`
from clean base `0625a18727ac2cf59220cfa7db5de93e1cac2651`. Its disposable proof covered
1,823 cases, 168,461 ordered [reassociation](../../../../glossary.md#term-reassociation) trees, 1,895,389 legal unfused/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) plans, and
3,232 baseline root states. The 23-task [Model](../../../../glossary.md#term-model)/[Javadoc](../../../../glossary.md#term-javadoc)/architecture [validation](../../../../glossary.md#term-validation), Markdown and diff
checks, and intended documentation/Javadoc-only [scope](../../../../glossary.md#term-scope) passed. Independent Class C review
reconstructed the proof and returned `APPROVE` with zero findings. [Metal](../../../../glossary.md#term-metal) 0018 remains Blocked
under its unchanged historical contract; Metal 0021 is the independently authorized successor.

## Change class

Class C — this changes the [Model](../../../../glossary.md#term-model)-owned cross-[module](../../../../glossary.md#term-module) numerical allowed-result set used by
capability, [conformance](../../../../glossary.md#term-conformance), preparation, execution, and [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings). It adds no [backend capability](../../../../glossary.md#backend-capability)
or public configuration [API](../../../../glossary.md#term-application-programming-interface-api) and requires a clean implementation context plus independent targeted
architecture/documentation review.

## Goal

Make the smallest [backend](../../../../glossary.md#backend)-neutral [`ACCELERATOR`](../../../../glossary.md#term-accelerator) refinement that admits the observed [`FLOAT32`](../../../../glossary.md#term-float32)
[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) result: after one complete nonempty declared [contraction](../../../../glossary.md#contraction-dimension) has produced exact zero under an
otherwise permitted [evaluation](../../../../glossary.md#term-evaluation), the published MATMUL cell may use either zero sign. Keep
`STRICT_IEEE`, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), every declared multiply/add/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) and intermediate result, empty
contractions, nonzero values, and result classifications unchanged.

## Chosen observable contract

Split the current shared [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul)/[`CONV2D`](../../../../glossary.md#convolution--conv2d)/[`CONV3D`](../../../../glossary.md#conv3d--ncdhw) table row. Change only `MATMUL`; the two
[convolution](../../../../glossary.md#term-channels-first-convolution-layers) kinds retain the current [reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) plus [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) contract without new exact-zero
sign freedom. For each [`ACCELERATOR`](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) MATMUL output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix):

- derive all and only the declared pairwise products for the contraction dimension and evaluate one
  currently permitted complete contraction, including existing reassociation, corresponding
  multiply-add FMA placement, FLOAT32 rounding, and row-scoped DAZ/FTZ;<span data-glossary-navigation="true"> ([contraction dimension glossary](../../../../glossary.md#contraction-dimension); [contraction glossary](../../../../glossary.md#contraction-dimension); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- preserve each declared multiplication's result, each FMA/addition result before publication, and
  every existing sign rule; no product or intermediate exact zero gains a sign choice;<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- only after the complete nonempty contraction's otherwise permitted result is exact zero may the
  published MATMUL cell use either zero sign; this includes a one-term contraction without adding
  a positive-zero accumulator or converting the multiplication into an FMA;<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- preserve the exact positive-zero identity for an empty contraction, every finite nonzero result,
  infinity/NaN classification, and existing NaN-representation freedom.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan))</span>

The rule changes no [subnormal](../../../../glossary.md#term-subnormal) into zero except through existing FTZ, grants no term [loss](../../../../glossary.md#loss) or added
[identity](../../../../glossary.md#term-identity-meanings), and applies to [Model](../../../../glossary.md#term-model) MATMUL rather than a [rank](../../../../glossary.md#term-rank), [backend](../../../../glossary.md#backend), library selector, or [topology](../../../../glossary.md#term-topology).

## [Scope](../../../../glossary.md#term-scope)

- Split and amend only the MATMUL part of the sole normative numerical-profile contraction row;
  retain the convolution row's current meaning verbatim in substance.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [normative glossary](../../../../glossary.md#term-normative); [contraction glossary](../../../../glossary.md#contraction-dimension); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Amend ADR 0019 with the final-publication decision and rejection of per-product, per-intermediate,
  shared-convolution, backend-specific, and tolerance alternatives.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Align `MatmulKind` and public `Tensor.matmul`/[linear](../../../../glossary.md#term-linear-meanings)-composition Javadocs. Review `Conv2dKind`,
  `Conv3dKind`, and affected Tensor convolution Javadocs and make their unchanged boundary explicit
  only where the row split would otherwise leave ambiguity.
- Align the focused [module](../../../../glossary.md#term-module)-boundary, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), glossary, and numerical-profile explanations.
- Use a disposable raw-bit proof over bounded contractions to show that final-only sign freedom
  admits the reported one-term result without changing any nonzero/classification result and that
  hypothetical product/intermediate sign freedom adds no observable final result needed here.

## Non-goals

No [Model](../../../../glossary.md#term-model) evaluator or profile Java enum; no Config, [Planning](../../../../glossary.md#term-planning), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Trace](../../../../glossary.md#trace),
[Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), native, [schema](../../../../glossary.md#term-schema), [codec](../../../../glossary.md#term-codec), capability, route, or [cache](../../../../glossary.md#term-cache) implementation change. No
per-product or intermediate zero-sign freedom, term [loss](../../../../glossary.md#loss)/duplication/invention, implicit [identity](../../../../glossary.md#term-identity-meanings),
[tolerance](../../../../glossary.md#term-tolerance), reduced [precision](../../../../glossary.md#term-precision), reciprocal substitution, algebraic rewrite, new [FMA](../../../../glossary.md#term-fused-multiply-add-fma), or cross-[node](../../../../glossary.md#node)
[contraction](../../../../glossary.md#contraction-dimension). The implementation did not change [convolution](../../../../glossary.md#term-channels-first-convolution-layers), reduction, [scan](../../../../glossary.md#term-cumulative-scan), [extrema](../../../../glossary.md#saturation-pole-and-extremum), comparison,
[unary](../../../../glossary.md#term-unary), non-[FLOAT32](../../../../glossary.md#term-float32), or empty-contraction semantics. It did not reopen Metal 0018, promote the
then-Draft Metal 0021 successor, or change Model 0026.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  — Model remains the sole semantic owner; later layers may realize but not reinterpret the set.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>
- [Foundational modules — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  and [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — operation meanings and the sole normative profile table are Model-owned.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [ADR 0019 — Explicit numerical profiles](../../../../design/decisions/0019-explicit-numerical-profiles.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
  — retain graph-wide, opt-in, operation-specific bounds and all gross-error exclusions.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

If final-[publication](../../../../glossary.md#term-publication)-only freedom cannot admit the reported result without changing a product,
intermediate, term, [identity](../../../../glossary.md#term-identity-meanings), classification, or nonzero result, stop and report the counterexample;
do not widen the task silently.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0027; Model 0028; retained [Metal](../../../../glossary.md#term-metal) 0018 blocker evidence
- Conflicts with: Model 0026; Metal 0021; any edit to the numerical-profile table, ADR 0019,
  MATMUL/convolution Javadocs, or shared profile documentation<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Parallel group: None
- Common base revision: `0625a18727ac2cf59220cfa7db5de93e1cac2651` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: First; before Metal 0021
- Integration [validation](../../../../glossary.md#term-validation): [accelerator](../../../../glossary.md#term-accelerator) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) exact-zero semantic-authority [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `docs/architecture/contracts/foundational-modules.md` — split contraction rows, narrow MATMUL
  publication rule, unchanged convolution and global exclusions.<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [publication glossary](../../../../glossary.md#term-publication); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- `docs/design/decisions/0019-explicit-numerical-profiles.md` — evidence, decision rationale,
  alternatives, and consequences.
- `MatmulKind`, [`Tensor`](../../../../glossary.md#tensor) MATMUL/linear Javadocs; targeted convolution Javadocs — observable profile
  semantics without backend or algorithm claims.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- `docs/architecture/module-boundaries.md`, `docs/api/tensor-api.md`, and targeted
  `docs/glossary.md` entries — focused explanatory [alignment](../../../../glossary.md#term-alignment).
- Existing MATMUL/convolution Model tests — unchanged vocabulary/construction coverage; add no
  source-text test or permanent numerical evaluator.
- [Model](../../../../glossary.md#term-model)/[Metal](../../../../glossary.md#term-metal) master plans and `docs/planning/roadmap.md` — completion/frontier synchronization only
  after implementation and review.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Acceptance criteria

1. The sole normative table gives only nonempty FLOAT32 ACCELERATOR MATMUL final-publication
   exact-zero sign freedom; strict, convolution, empty contraction, and every other row/type stay
   unchanged.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [FLOAT32 glossary](../../../../glossary.md#term-float32); [ACCELERATOR glossary](../../../../glossary.md#term-accelerator); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [publication glossary](../../../../glossary.md#term-publication); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
2. Every declared product appears exactly once. Existing reassociation, legal FMA placements,
   FLOAT32 rounding, DAZ/FTZ, product signs, and intermediate signs remain the complete evaluation
   freedoms; no positive-zero accumulator or other identity is added.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [evaluation glossary](../../../../glossary.md#term-evaluation); [accumulator glossary](../../../../glossary.md#accumulator-type); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
3. A one-term `+0.0f * -1.0f` [contraction](../../../../glossary.md#contraction-dimension) permits published `+0` or `-0`, while the underlying
   multiply remains `-0`; [finite](../../../../glossary.md#term-finite) nonzero values and [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) classification do not widen.
4. The disposable proof covers one- through four-term contractions, signed zeros, cancellation,
   subnormals, normal boundaries, infinities, and NaNs across reassociation and legal FMA choices;
   it reports raw-bit/class results and is removed.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
5. ADR, Javadocs, API/architecture explanation, and targeted glossary consistently distinguish the
   MATMUL-only final-publication rule from unchanged convolution and backend capability.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
6. Focused Model tests, Model Javadoc, architecture tests, Markdown/status/frontier checks, and diff
   checks pass; independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
python3 /tmp/synaptik-model-0029-matmul-zero-sign-proof.py
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
python3 /tmp/validate_synaptik_markdown.py
rm -f /tmp/synaptik-model-0029-matmul-zero-sign-proof.py
git diff --check
```

The proof must distinguish multiplication/intermediate bits from final [publication](../../../../glossary.md#term-publication), distinguish
exact zero from [FTZ](../../../../glossary.md#term-ftz), enumerate legal [reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) placements, and report bounded case/tree
counts. Inspect generated [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and targeted [convolution](../../../../glossary.md#term-channels-first-convolution-layers) Javadocs. Confirm no executable Java
statement, test source, [Gradle](../../../../glossary.md#term-gradle), configuration, [backend](../../../../glossary.md#backend), native, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Compiler](../../../../glossary.md#term-compiler), [Engine](../../../../glossary.md#term-engine),
or [Trace](../../../../glossary.md#trace) file changed.

Integration/repository validation: after independent review, rerun changed documentation and
architecture checks plus `git diff --check`; do not repeat successful Java tests unless executable
Java changed or evidence became stale.

## Follow-up

- Metal 0021 has been reverified against the completed contract and promoted separately to the
  sole Ready frontier from the clean integrated task-0029 revision. It must run a completely fresh
  full real-M3 oracle from the beginning before any production edit. Historical Metal 0018 remains
  Blocked.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Ready glossary](../../../../glossary.md#term-ready); [run glossary](../../../../glossary.md#term-run); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Documentation and review impact

This changes an authoritative numerical contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) rationale, public [operation](../../../../glossary.md#operation) Javadocs, and
user-facing semantic explanation. A separate clean Class C review/documentation context is
mandatory. It reviews proof completeness, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-only [scope](../../../../glossary.md#term-scope), final-[publication](../../../../glossary.md#term-publication) placement, strict/
[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/empty preservation, all-term and no-[identity](../../../../glossary.md#term-identity-meanings) invariants, gross-error exclusions,
terminology, links, glossary impact, and absence of [backend](../../../../glossary.md#backend) implementation or capability drift.

## Result

Complete at `308267837a17c2fdbeb8b90ad381d1f899980786`. The [normative](../../../../glossary.md#term-normative) profile table now separates
[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) from [convolution](../../../../glossary.md#term-channels-first-convolution-layers) and permits either sign only at [publication](../../../../glossary.md#term-publication) of a complete nonempty
otherwise-permitted exact-zero [FLOAT32](../../../../glossary.md#term-float32) [ACCELERATOR](../../../../glossary.md#term-accelerator) MATMUL result. Products, additions, [FMA](../../../../glossary.md#term-fused-multiply-add-fma)
results, empty [contraction](../../../../glossary.md#contraction-dimension), nonzero values, classifications, strict behavior, and convolution
remain unchanged.

The removed disposable proof reported 1,823 cases, 168,461 ordered [reassociation](../../../../glossary.md#term-reassociation) trees,
1,895,389 legal unfused/FMA [evaluation](../../../../glossary.md#term-evaluation) plans, and 3,232 baseline root states. The 23-task
[Model](../../../../glossary.md#term-model)/[Javadoc](../../../../glossary.md#term-javadoc)/architecture [validation](../../../../glossary.md#term-validation) and Markdown/diff [scope](../../../../glossary.md#term-scope) checks passed. Independent Class C
review reconstructed the proof and returned `APPROVE` with zero findings. No executable Java
statement, test source, [backend](../../../../glossary.md#backend), native, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), configuration, or build file changed.

Status: Complete
