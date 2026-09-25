# Task 0030: Total Recursive ACCELERATOR Numerical Floor

## Status

Ready

## Change class

Class C — this replaces the cross-module numerical meaning of `ACCELERATOR`, supersedes an
accepted architecture decision, and closes every current Model operation family under one
recursive contract without changing the public profile vocabulary or backend capability.

## Goal

Keep public `NumericalProfile` exactly `STRICT_IEEE` and `ACCELERATOR`, leave `STRICT_IEEE`
unchanged, and make `ACCELERATOR` a total `FLOAT32` superset through three small recursive floors:
exact/discrete behavior, primitive floating evaluation, and all-terms-once aggregation. Composite
operations inherit their existing Model formula. Add no policy object, placement selector,
determinism mode, conformance-envelope registry, or per-operation backend exception layer.

## Scope

- Replace the sole operation-row table with the recursive contract and add ADR 0021, superseding
  ADR 0019's closed-table choice while retaining its graph-wide identity, ownership, monotonicity,
  and fail-closed capability decisions.
- Audit every concrete `OperationKind` and public Tensor composition. Make each formula, guard,
  selection, identity, divisor, state transition, and special-value result explicit enough to
  apply the floors without backend invention.
- State once that Compiler-generated gradients are ordinary public Tensor formulas captured under
  the same graph profile. Reconcile current explanations and affected Model/config Javadocs.
  Change no executable behavior, enum shape, capability, lowering, or route.

### Normative recursive contract

1. `STRICT_IEEE` remains the current per-operation allowed-result set. `ACCELERATOR` is that set
   union results constructed by these `FLOAT32` rules. Other data types remain strict.
2. **Exact/discrete floor.** Preserve kind/attributes, arity/descriptors, shape/layout/axis mapping,
   contributor membership, mask/index/state/RNG transition, traversal, selected stored payload,
   cast/conversion, ordering, stability, tie, empty-domain, identity, divisor, and publication.
   Predicates gain no epsilon. DAZ may affect a comparison or extrema tie, but selection remains
   among original candidates under the Model tie rule; approximation cannot choose an index,
   mask, winner, ordering, or state transition.
3. **Primitive `FLOAT32` floor.** At each arithmetic site in the Model formula, a subnormal
   operand may be read as same-signed zero (DAZ), and a finite subnormal site result may become
   either signed zero (FTZ); stored inputs are unchanged. `ADD/SUB/MUL/DIV` still perform one
   round-to-nearest-even operation. A corresponding multiply/add with no observable intermediate
   may use one FMA, never across graph nodes. An elementary site (`POW`, logarithmic, exponential,
   error-function, root/reciprocal-root, sigmoid/tanh, or formula-named activation) may return an
   ordinary finite value at most five monotonically ordered `FLOAT32` representations from the
   correctly rounded exact result after DAZ and before FTZ. Five ULP is the smallest single ceiling
   covering current direct-route evidence: certified CPU primitives are one/two ULP except
   `TANH` at five; retained MPSGraph `EXP`/`SIGMOID` are one/two. It does not copy the unrelated
   four-ULP softmax oracle or loose composite ERF/GELU tolerance. A strict realization remains valid.
4. Domain and special classes remain formula-derived: NaN cannot become ordinary; finite cannot
   become NaN/infinity except through genuine overflow/domain after DAZ; required infinity/zero
   signs remain except explicit DAZ/FTZ, current extrema ties, or current final-zero freedom. Five
   ULP applies only to ordinary finite non-subnormal references; a subnormal reference is exact or
   FTZ. This is a primitive construction rule, not a final-output `allclose` envelope.
5. **Aggregate floor.** After exact mapping, masking, padding, bounds, and selection, use every
   declared contributor exactly once. Any binary tree may reassociate with per-step `FLOAT32`
   rounding, primitive DAZ/FTZ, and corresponding FMA while retaining exact empty/point identities
   and mandatory divisors. Never drop, duplicate, invent, pretruncate, or replace a term. This
   covers reductions, scans, overlap/scatter reductions, statistics/norms/losses, contractions,
   recurrent cells, and average pooling; extrema/winners remain exact/discrete.
6. **Composite inheritance.** Normalization, activation, loss, attention, pooling, convolution,
   random/dropout, recurrent, convenience composition, and generated gradients get no envelope.
   Evaluate exact guards and the current Model formula through the floors. An opaque selector is
   eligible only when its complete output set is proved inside the recursive set.

Still excluded: reduced precision, reciprocal substitution for division, algebraic identities not
in the Model formula, cross-node contraction, term loss, hidden-state changes, tolerance-based
predicate/selection changes, and backend-specific semantics.

### Complete Model-family inventory

| Family | Required floor assignment |
|---|---|
| `BinaryArithmeticKind`, `ScalarElementwiseKind` | `ADD/SUB/MUL/DIV/POW`, `MIN/MAX`, and `CLAMP`: primitive arithmetic plus exact selection/ties. |
| `UnaryElementwiseKind` | `ABS/NEG/RECIPROCAL`, log/exp/erf/root, floor/ceil/sign/ReLU, sigmoid/tanh/GELU/SiLU: primitive sites; formula-defined activations recurse. |
| `BinaryComparisonKind`, `BooleanLogicalKind`, `FloatingClassificationKind`, `WhereSelectionKind`, `CastKind` | Comparisons may DAZ operands; Boolean logic, classification, `WHERE`, and all 36 `CAST` pairs remain exact/discrete. |
| `AggregateReductionKind`, `CumulativeScanKind` | Ordinary/masked/sum-to-Shape reductions, extrema/arg-extrema, statistics/norms, Boolean folds, and cumulative sum/product use exact membership plus the applicable aggregate or selection floor. |
| `ContiguousKind`, `ShapeTransformKind`, `AxisTransformKind`, `SliceKind`, `PadKind`, `TileKind`, `TensorCompositionKind`, `WindowTransformKind` | Preserve mapping/payload; overlap folds use the aggregate floor. |
| `SelectKind`, `AxisGatherKind`, `GatherNdKind`, `OneHotKind`, `AxisScatterKind`, `ScatterNdKind` | Preserve indices/bounds/payload; only declared duplicate arithmetic aggregates. |
| `MatmulKind`, `Conv2dKind`, `Conv3dKind`, `Pool2dKind`, `Pool3dKind` | Contractions aggregate; max pooling selects exactly; average pooling aggregates; Conv1d/Pool1d remain visible compositions. |
| `SoftmaxKind`, `LayerNormKind`, `RmsNormKind`, `BatchNormKind`, `LossKind`, `ScaledDotProductAttentionKind` | Retain current guards/formulas and recurse through primitive/aggregate sites. |
| `OrderingKind`, `TopKKind`, `GraphRngKind`, `DropoutKind`, `RecurrentScanKind` | Ordering/indices and RNG mask/state are exact; dropout scaling and recurrent cell arithmetic recurse while traversal/valid lengths/outputs/state remain exact. |

### Generated-gradient consequences

| Owner/family | Consequence under the selected graph profile |
|---|---|
| Elementwise | Generated arithmetic, elementary sites, comparisons, `WHERE`, casts, and `SUM_TO_SHAPE` use their ordinary floor; saved forward outputs are exact stored values. |
| Reduction/normalization/loss | Generated shape edits, reductions/scans, arithmetic, extrema routing, `EXP/RSQRT`, softmax/log-softmax, one-hot, classification, and `WHERE` recurse. BatchNorm saved statistics remain exact auxiliaries. |
| MATMUL/attention/Conv2d | Generated MATMUL/transposes/sum-to-Shape, attention formulas, and Conv2d unfold/fold/MATMUL/sums require capability for the complete generated topology, not only the forward kind. |
| Pooling/index/layout/order/dropout | Saved/recomputed winners, indices, and masks remain exact; inverse movement, fold/scatter accumulation, and dropout scaling use assigned floors. |
| Unsupported derivatives | Documented nondifferentiable outputs remain so; Conv3d and recurrent differentiation stay fail-closed. |

### Metal reachability and migration ledger

- This task removes semantic incompleteness, not implementation blockers. The recursive set can
  admit retained `EXP`/`SIGMOID` FTZ plus one/two-ULP results, comparison DAZ, extrema ties, scan
  FTZ, softmax/BatchNorm FTZ, and contractions only after complete route proof.
- Direct-route numerical failures still outside it include ordinary finite/infinity scalar
  division, `RELU(NaN)`, `TANH(NaN)` and tanh negative-zero loss, max-pool NaN/winner loss, and
  BFLOAT16 Gather payload canonicalization. They need conforming composition/custom kernels.
- Structural blockers include the one-output/FLOAT32-INT32-BOOL schema, no zero-input/general
  multi-output/state or INT64-local route, limited wires/ranks/layouts/transfers, hidden vendor
  RNG, recurrent valid-length/atomic validation, and absent Conv3d/recurrent gradient owners.
  Full Model-op Metal coverage is specifiable, but not one current task or direct-MPSGraph-only.
- Public enum/wire vocabulary, cold transport, Runtime/Trace, CPU schema 67/tuning schema 2, and
  current Metal versions do not change: existing realizations remain valid strict subsets. The
  first new Metal route must bump node schema 11→12, append unary wires 20/21, bump workload/
  default-route/candidate/compatibility/route-policy/codec 12→13, reject stale values, and retain
  native ABI 4 and generic complete-plan wrapper schema 1.

## Non-goals

No executable Java/native change, new API/type, enum value, Config/Engine method, Runtime/Trace
field, backend capability, cache/tuning behavior, device probe, operation decomposition, generic
policy registry, full Metal implementation, FLOAT16 work, Conv3d gradient, or recurrent gradient.
Historical completed/blocked task evidence is not rewritten.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — graph-wide cold identity, strict-subset monotonicity, and fail-closed capability.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles) — sole allowed-result owner and the section replaced here.
- [Compiler/autograd — Numerical-profile compile identity](../../../../architecture/contracts/compiler-autograd.md#numerical-profile-compile-identity) — generated Tensor expressions share one transported profile.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules) — backends prove/realize a subset, not define floors.

If any operation lacks a complete formula, exact guard, special-class result, or floor assignment,
stop and report that named gap; do not add a backend-local envelope.

## Dependencies and integration

- Depends on: Model 0027–0029; Config 0006; Engine 0018; CPU 0017; Metal 0015/0019–0025 and retained blocker evidence through 0040; explicit approval of this minimal recursive redesign.
- Conflicts with: every concurrent edit to root/scoped architecture, numerical-profile ADRs, affected Model/config Javadocs, Compiler gradient wording, current API/glossary/backend guides, Model/Metal planning frontiers, or roadmap.
- Parallel group: None.
- Common base revision: `6535c0ffdebeb38945050497bbe1c0fe261950ca`
- Integration order: first; before any Metal 0051 implementation or reinterpretation of historical blocked tasks.
- Integration validation: total-recursive-numerical-floor architecture checkpoint.
- Shared-document integration owner: Main planner.

## Files and symbols

- Root/scoped architecture contracts — authority, floors, gradient consequence, backend proof.
- ADR `0021-total-recursive-accelerator-numerical-floor.md` and ADR 0019 status/link — clean supersession, evidence, alternatives, migration.
- Every concrete `model.operation/**/**Kind.java`, affected attrs, Tensor/composition Javadocs, and `NumericalProfile` — formula/floor reconciliation without executable changes.
- Current Tensor/compile/public API, module/lifecycle/boundary, capability/CPU/Metal/preparer guides, and glossary — explanatory reconciliation.
- Model/Metal masters and roadmap — status and serial successor only.

## Acceptance criteria

- The two-value public profile remains unchanged; `STRICT_IEEE` is unchanged; `ACCELERATOR` is a
  total recursive `FLOAT32` superset with no unassigned current family and no new policy layer.
- The five-ULP elementary-site rule is defined exactly and justified by current CPU/Metal evidence;
  special classes, domain, exact/discrete state, all contributors, and non-FLOAT32 behavior cannot
  be relaxed by that bound.
- Every current operation family and generated-gradient consequence is covered; Model formulas,
  guards, identities, divisors, selections, states, and special values are sufficiently explicit
  for backend conformance without per-backend invention.
- Existing CPU/Metal capabilities and identities do not change. Documentation clearly separates
  semantic reachability from implemented coverage and preserves historical blocker records.
- This task is the sole Ready frontier; Metal 0051 remains Draft until this task is Complete and
  independently approved.

## Validation

Worker validation:

```bash
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
git diff --check
```

Inspect generated Javadocs for every operation family and `NumericalProfile`; validate changed
Markdown links/anchors/fences/newlines, exhaustive family names, and absence of executable,
enum-shape, backend-capability, or version changes.

Integration/repository validation: architecture/documentation checkpoint after independent Class C
review; no backend/device execution because capability is unchanged.

## Follow-up

- Metal 0051: accelerator-only FLOAT32 `EXP`/`SIGMOID`, seeded-gradient topologies, and one fresh
  device oracle; update provider, analysis/preflight, graph/native switch, candidates/tuning/codec,
  and their unit/native/conformance plus `EngineExplicitCompositionMetalIntegrationTest`; no other unary or policy.
- Later serial Metal tasks requalify comparisons/extrema/scans, then cohesive composites/custom
  kernels; structural schema/type/multi-output/state work remains separate.
- CPU needs no migration: exact routes, artifacts, and session tuning remain valid strict members.

## Documentation and review impact

- Coordinated architecture, ADR, public numerical contract, Javadoc, API, glossary, backend-guide,
  and planning updates are mandatory; completed historical briefs remain unchanged.
- Independent Class C review must verify family coverage, recursion, five-ULP evidence, exact/
  aggregate invariants, gradients, identity conclusions, Metal blocker split, and no drift.

## Result

Empty until execution.
