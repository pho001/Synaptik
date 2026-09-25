# Task 0030: Total Recursive ACCELERATOR Numerical Floor

## Status

Complete

## Change class

Class C — this replaces the cross-module numerical meaning of `ACCELERATOR`, supersedes an
accepted architecture decision, and closes every current Model operation family under one
recursive contract without changing the public profile vocabulary or backend capability.

## Goal

Keep public `NumericalProfile` exactly `STRICT_IEEE` and `ACCELERATOR`, rest on completed Model
0031's backend-independent strict unary baseline for BFLOAT16/FLOAT32/FLOAT64, and make
`ACCELERATOR` a total `FLOAT32` superset through
three recursive floors: exact/discrete behavior, primitive floating evaluation, and
all-terms-once aggregation. Composite operations inherit their existing Model formula. Add no
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

1. `STRICT_IEEE` is the explicit current per-operation allowed-result set, including completed
   Model 0031's exact-reference primitive bounds and recursive native/one-wider unary sets for
   every accepted floating type. `ACCELERATOR` is that set union results constructed by these
   `FLOAT32` rules. Other data types remain strict.
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
   may use one FMA, never across graph nodes.
4. Only an irreducible `POW`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, or `TANH` site has
   an elementary allowance. For raw binary32 word `b`, the monotonic unsigned key is `~b` when the
   sign bit is set and `b ^ 0x80000000` otherwise. Distance is the unsigned absolute key
   difference; reference distance is zero, `-0` and `+0` are adjacent, and every ceiling is
   inclusive. After DAZ and before FTZ, an ordinary finite non-subnormal correctly rounded exact
   reference `r` admits only `distance(actual,r) <= 5`; a subnormal `r` is exact or FTZ.
5. An ordered-representation distance ceiling of five is conservative, not five ULP and not a
   minimality claim. Java scalar and Vector `TANH` inherit the at-most-2.5-ULP exact-result
   contract, and FLOAT32 lanes use the documented widen/evaluate/narrow adaptation. At a
   power-of-two binade boundary, smaller-side spacing is half the Java ULP, so that guarantee can
   span an ordered distance of five adjacent-representation steps; elsewhere the distance is at
   most four. Java one-ULP log/exp contracts and retained Metal `EXP`/`SIGMOID` evidence fit within
   ordered distance two and do not raise the ceiling.
6. Domain and special classes remain formula-derived: NaN cannot become ordinary; finite cannot
   become NaN/infinity except through genuine overflow/domain after DAZ; required infinity/zero
   signs remain except explicit DAZ/FTZ, current extrema ties, or current final-zero freedom.
   Ordered distance never substitutes for class, domain, or sign checks and is a primitive-site
   construction rule, not a final-output `allclose` envelope.
7. **Aggregate floor.** After exact mapping, masking, padding, bounds, and selection, use every
   declared contributor exactly once. Any binary tree may reassociate with per-step `FLOAT32`
   rounding, primitive DAZ/FTZ, and corresponding FMA while retaining exact empty/point identities
   and mandatory divisors. Never drop, duplicate, invent, pretruncate, or replace a term. This
   covers reductions, scans, overlap/scatter reductions, statistics/norms/losses, contractions,
   recurrent cells, and average pooling; extrema/winners remain exact/discrete.
8. **Composite inheritance.** Normalization, activation, loss, attention, pooling, convolution,
   random/dropout, recurrent, convenience composition, and generated gradients get no envelope.
   Evaluate exact guards and every site in the current Model formula through the floors. An opaque
   or first-class selector is eligible only when its complete output set over its complete
   advertised domain is proved inside the recursive set; a sample or operation name is insufficient.

### Closed primitive-site ledger

| First-class composite | Primitive and aggregate sites |
|---|---|
| Unary | `RSQRT` is `SQRT` then division. Sigmoid uses the exact sign guard and its `EXP`/negation/add/divide branch. Exact GELU names typed `0.5`, `1`, `2`, `SQRT`, division, `ERF`, addition, and two multiplications. Tanh GELU names typed `0.5`, `1`, `2`, `pi`, `0.044715`, `x*x`, `x^2*x`, division/root, additions, remaining multiplications, and `TANH`. SiLU uses the exact sign guard, stable sigmoid branch, and final multiplication. None is one elementary site. |
| Reductions/statistics | Mean is sum then exact divisor. Variance is mean, subtraction, `x*x`, sum, and `N-correction` division; standard deviation adds `SQRT`. L1 is `ABS` plus sum; L2 is `x*x`, sum, and `SQRT`; log-sum-exp is `EXP` per contributor, sum, then `LOG`. Scans and overlap/scatter reductions retain each declared contributor. |
| Softmax/normalization | Softmax is per-value `EXP`, sum, then division; log-softmax is per-value `EXP`, sum, `LOG`, then subtraction. Layer/RMS/BatchNorm enumerate their documented sums/divisors, centered or uncentered `x*x`, epsilon/momentum constants, additions/subtractions, `SQRT`, typed-one division, affine multiply/adds, saved outputs, and state transitions. |
| Loss | MSE is subtraction then `delta*delta`, reduction, and optional divisor. Categorical losses use exact max/target/ignore guards, score subtraction, `EXP`, sum, `LOG`, max addition, logit subtraction, target multiplication where declared, loss sum/negation, and optional divisor. |
| Linear/spatial/attention | MATMUL and convolution use mapped multiplies plus all-terms-once contractions; convolution then applies its optional bias addition. Average pooling sums every fixed-divisor position then divides; max pooling is exact selection. Attention has query/key multiply-contraction, explicit or default typed-one/`SQRT(E)` scale, exact guards, literal softmax sites, and value-weight multiply-contraction. |
| Dropout/recurrent | Kept dropout uses typed-one-minus-probability, typed-one division, and input multiplication; mask/state/drop result are exact. RNN/GRU/LSTM use their documented contractions, optional biases, gate additions, sigmoid branches, tanh sites, state multiplies/adds, exact traversal/valid-length guards, and exact published outputs/states. |

Formula-named constants are rounded once to nearest-even `FLOAT32` before first use; exact typed
attributes and stored inputs remain exact leaves. Visible compositions and generated gradients
recurse through the actual Tensor operations they contain.

Still excluded: reduced precision, reciprocal substitution for division, algebraic identities not
in the Model formula, cross-node contraction, term loss, hidden-state changes, tolerance-based
predicate/selection changes, and backend-specific semantics.

### Complete Model-family inventory

| Family | Required floor assignment |
|---|---|
| `BinaryArithmeticKind`, `ScalarElementwiseKind` | `ADD/SUB/MUL/DIV` use the primitive arithmetic floor, `POW` is an irreducible elementary site, and `MIN/MAX/CLAMP` retain exact candidate selection and ties. |
| `UnaryElementwiseKind` | Exact/discrete rows plus irreducible log/exp/erf/sqrt/tanh sites; `RSQRT`, sigmoid, GELU, tanh GELU, and SiLU recurse through the complete ledger above. |
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
  reach retained `EXP`/`SIGMOID` FTZ results, comparison DAZ, extrema ties, scan FTZ,
  softmax/BatchNorm FTZ, composite MSE/attention/recurrent arithmetic, and contractions only after
  proof for the complete route and advertised domain.
- Direct-route numerical failures still outside it include ordinary finite/infinity scalar
  division, `RELU(NaN)`, `TANH(NaN)` and tanh negative-zero loss, max-pool NaN/winner loss, and
  BFLOAT16 Gather payload canonicalization. Such selectors remain unusable; conforming
  custom/composed routes still need structural and complete-subset proof.
- Structural blockers include the one-output/FLOAT32-INT32-BOOL schema, no zero-input/general
  multi-output/state or INT64-local route, limited wires/ranks/layouts/transfers, hidden vendor
  RNG, recurrent valid-length/atomic validation, and absent Conv3d/recurrent gradient owners.
  Full Model-op Metal coverage is semantically specifiable, but not one current task or a
  direct-MPSGraph-only assumption.
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

- Depends on: completed Model 0031 strict unary baseline; Model 0027–0029; Config 0006; Engine
  0018; CPU 0017; Metal 0015/0019–0025 and retained blocker evidence through 0040; explicit
  approval of this minimal recursive redesign.
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

- The two-value public profile remains unchanged; completed Model 0031 owns backend-independent
  strict unary semantics for every accepted floating type; `ACCELERATOR` is a total recursive
  `FLOAT32` superset with no unassigned current family and no new policy layer.
- The inclusive ordered-binary32 distance ceiling is defined exactly and conservatively derived
  from current Java scalar/Vector and retained Metal evidence without a minimality claim; special
  classes, domain, exact/discrete state, all contributors, and non-FLOAT32 behavior cannot be
  relaxed by that bound.
- Every current operation family and generated-gradient consequence is covered; Model formulas,
  guards, identities, divisors, selections, states, and special values are sufficiently explicit
  for backend conformance without per-backend invention.
- Existing CPU/Metal capabilities and identities do not change. Documentation clearly separates
  semantic reachability from implemented coverage and preserves historical blocker records.
- No other task becomes Ready; Metal 0051 remains Draft until this task is Complete and
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

- Metal 0051: accelerator-only FLOAT32 `EXP`/`SIGMOID`, seeded-gradient topologies, and fresh
  recursive-floor certification for every candidate. Before capability becomes true, record the
  direct-MPSGraph versus fused-custom versus certified-composition choice using hot-run time,
  dispatch count, and temporary-memory evidence. Update provider, analysis/preflight, graph/native
  switch, candidates/tuning/codec, and focused tests only after that adjudication; add no runtime
  checks, benchmark matrices, other unary, or policy.
- Later serial Metal tasks requalify comparisons/extrema/scans, then cohesive composites/custom
  kernels; structural schema/type/multi-output/state work remains separate.
- CPU needs no migration: exact routes, artifacts, and session tuning remain valid strict members.

## Documentation and review impact

- Coordinated architecture, ADR, public numerical contract, Javadoc, API, glossary, backend-guide,
  and planning updates are mandatory; completed historical briefs remain unchanged.
- Independent Class C rereview verified completed Model 0031's exact-reference
  BFLOAT16/FLOAT32/FLOAT64 bounds and recursive native/one-wider sets, family/site coverage,
  accelerator ordered-distance derivation, exact/aggregate invariants, gradients, identity
  conclusions, Metal reachability/blocker split, and no executable drift.

## Result

Implemented the documentation-only clean cutover to the total recursive `ACCELERATOR` contract,
based on completed Model 0031's backend-independent strict unary allowed-result baseline for every
accepted floating type. That baseline now defines exact mathematical references, per-format
ordered distance, explicit `ERF` absolute/relative bounds, special/domain rules, and recursive
native/one-wider composites without implementation authority. The foundational authority and
accepted ADR 0021 then define exact/discrete, primitive FLOAT32, all-contributors-once aggregate,
and composite-inheritance accelerator floors. Accelerator ordered binary32 distance uses a
monotonic raw-bit key and an inclusive `distance <= 5` exact-reference ceiling; retained Java
2.5-ULP `TANH` evidence supplies the conservative binade-boundary rationale, without calling the
bound five ULP or claiming empirical minimality.

All forty concrete Model kinds, public Tensor profile/composition wording, `NumericalProfile`,
Compiler gradient transport, Engine package status, current API/architecture/glossary/backend
guides, and planning frontiers remain reconciled without changing executable statements, enum
shape, capability, schema, ABI, native export, cache, tuning, or Runtime/Trace behavior. The
closed ledger now enumerates constants, `x*x`/`x^2*x`, irreducible elementary sites, arithmetic,
aggregates, guards, selections, and state for every first-class composite formula. Opaque routes
require complete-domain recursive-subset proof.

The focused Model test/Javadoc/architecture command is `BUILD SUCCESSFUL` with 23 actionable tasks
(three executed, twenty up-to-date); validation of ten changed Markdown documents and 481 local
links/anchors, fences, final newlines, and `git diff --check` also pass. Historical Metal tasks
retain their recorded statuses/evidence; current wording distinguishes recursively reachable
arithmetic from remaining exact-selection, structural, schema, gradient, and route-proof blockers.
The recursive-contract remediation is `2d95ab71683698753c9ac64d9373fde8301c9404`; strict-baseline
remediation is `97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`. Independent Class C rereview after
`97cb9d11` returned `APPROVE` with zero findings. Status is `Complete`; Metal 0051 is the first
authorized executable successor.
