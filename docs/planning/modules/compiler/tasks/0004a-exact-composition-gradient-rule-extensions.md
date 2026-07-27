# Task 0004A: Exact-composition Gradient-rule Extensions

## Status

Complete

## Goal

Extend Compiler 0004's proved compiler-owned pre-capture automatic-differentiation pipeline with
one bounded set of first-order derivative rules whose formulas are exact compositions of current
public `Tensor` operations.

This task adds no derivative convention. Every selected row must be valid without choosing a
tie, endpoint, discontinuity, singularity, empty-domain, NaN/infinity, stochastic, saved-value,
mixed-floating cotangent, or numerical-kernel policy. Unsupported variants continue to fail in
`AutogradPreflight` before any derivative Tensor identity is allocated.

The task reuses the existing objective, target, implicit seed, object-identity accumulation,
logical-splat, one combined capture, graph phase, validation, optimization, and
`GraphCompilation` contracts unchanged.

## Scope

### Pipeline reuse

Do not redesign Compiler 0004. Extend only its closed preflight and named rule dispatch:

```text
original forward Tensor DAG
  -> complete fail-closed preflight
  -> ordinary public-Tensor derivative formulas
  -> existing deterministic identity-keyed accumulation
  -> existing one phase-aware combined capture
  -> existing validation and exact combined optimization
```

`GraphCompiler`, `GraphCapture`, `GraphCompilation`, the request shape, the implicit scalar unit
seed, result roles, graph-output ordering, and optimization pass order remain behaviorally
unchanged. A successful 0004A formula is an ordinary differentiable Tensor expression; this task
does not claim that Compiler 0006's future higher-order request contract is complete.

### Closed `SUPPORTED_0004A` matrix

`SUPPORTED_0004A` is additive to `SUPPORTED_0004`. No existing row is weakened. In the formulas
below:

- `g` is the selected output cotangent;
- `x`, `left`, and `right` are exact original inputs;
- `Z_x` is the existing explicit typed positive-zero logical splat expanded to `x.shape()`;
- `B_x(v)` is `v.sumToShape(x.shape())`;
- `swapLastTwo(v)` is `v.permute(...)` with only its final two axes exchanged; and
- all operation kinds, attribute objects, input positions, output positions, descriptors, and
  Shapes are the exact facts recorded by preflight.

| Family and exact variant | Additional preflight guard | Formula for selected differentiable roles |
|---|---|---|
| `UnaryElementwiseKind.ERF` with `NoOperationAttrs.INSTANCE` | Input and output have one exact equal floating type and Shape. | `g.mul(x.mul(x).neg().exp()).mul(C_erf)`, where `C_erf` is the exact typed representation of `2 / sqrt(pi)` specified below. |
| `AggregateReductionKind.SUM` with `MaskedReductionAttrs(axis)` | Data input and output have one exact equal floating type; mask input is exact BOOL; the current mask-broadcast and normalized-axis descriptor contract holds. | Restore the removed axis in `g`, expand to `x.shape()`, then `Tensor.where(mask, restored, Z_x)`. The mask role is non-differentiable. |
| `AggregateReductionKind.SUM` with `SumToShapeAttrs(targetShape)` | Input and output have one exact equal floating type. For every right-aligned pair, the target Dimension is structurally equal to the input Dimension or is static one. Leading input axes are permitted. Binding-dependent unresolved singleton-or-equal pairs are rejected. | `g.expand(x.shape())`. |
| `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE` | Both inputs and output are floating; the selected target input has the output's exact floating type; all current rank, contraction, promotion, and batch-broadcast descriptor facts hold. | The complete rank-case formulas below, followed by `B_left` or `B_right` where batch broadcasting may have occurred. |
| `SliceKind.SLICE` with `SliceAttrs` | Input and output have one exact equal floating type; current normalized finite coordinate and Shape facts hold. | `Z_x.sliceUpdate(g, starts, axes, steps)`. |
| `SliceKind.SLICE_UPDATE` with `SliceAttrs`, base role | Base, update, and output have one exact equal floating type and the current normalized replacement contract holds. | `g.sliceUpdate(Z_update, starts, axes, steps)`. |
| `SliceKind.SLICE_UPDATE` with `SliceAttrs`, update role | Same guard as the base role, and every selected base Dimension is static so the normalized finite coordinate sequences can be expressed by public `slice`. | Extract the exact normalized coordinate sequences from `g` with public `slice`, using the deterministic raw-end reconstruction below. |
| `SelectKind.SELECT` with `SelectAttrs(axis, index)` | Input and output have one exact equal floating type and the current normalized select facts hold. | `Z_x.sliceUpdate(g.expandDims(axis), [index], [axis], [1])`. |
| `PadKind.PAD` with `PadAttrs` | Input and output have one exact equal floating type; the typed padding value is non-differentiable metadata. | `g.cropToShape(x.shape(), Shape.of(beforeWidths))`. |
| `TileKind.TILE` with `TileAttrs` | Input and output have one exact equal floating type and every repeat is positive. | Reshape `g` to the interleaved Shape `[repeat0, x0, repeat1, x1, ...]`, sum the even repeat axes, then reshape to the exact `x.shape()`. Scalar tile returns `g`. |
| `TensorCompositionKind.CONCAT` with `CompositionAxisAttrs(axis)` | Every input and output has one exact equal floating type and the current same-rank/non-axis-Shape contract holds. | For input `i`, `g.cropToShape(input_i.shape(), prefix_i)`, where `prefix_i` is zero on other axes and the exact ordered symbolic sum of prior input extents on `axis`. |
| `TensorCompositionKind.STACK` with `CompositionAxisAttrs(axis)` | Every input and output has one exact equal floating type and the current exact input-Shape contract holds. | For input `i`, `g.select(axis, i)`. |

The `MATMUL` rank cases are exact:

| Left rank | Right rank | Left contribution | Right contribution |
|---|---|---|---|
| 1 | 1 | `g.mul(right)` | `g.mul(left)` |
| 1 | at least 2 | `B_left(g.expandDims(g.rank() - 1).matmul(swapLastTwo(right)).squeeze(g.rank() - 1))` | `B_right(left.expandDims(1).matmul(g.expandDims(g.rank() - 1)))` |
| at least 2 | 1 | `B_left(g.expandDims(g.rank()).matmul(right.expandDims(0)))` | `B_right(swapLastTwo(left).matmul(g))` |
| at least 2 | at least 2 | `B_left(g.matmul(swapLastTwo(right)))` | `B_right(swapLastTwo(left).matmul(g))` |

Preflight is role-aware. A `MATMUL` occurrence may therefore propagate to a selected same-type
input even when the unselected floating operand was promoted. It rejects a selected operand whose
cotangent would require a cross-floating conversion. Integral `MATMUL` remains
non-differentiable.

For `SLICE_UPDATE`'s update role, reconstruct each public `slice` entry from normalized
`SliceAttrs` without changing its coordinate sequence:

1. zero length uses raw start zero and raw end zero;
2. positive step uses the first coordinate after the finite sequence when representable and no
   greater than the static base extent, otherwise the base extent;
3. negative step uses the first coordinate after the finite sequence when non-negative, otherwise
   raw `-baseExtent - 1`, which normalizes to the clamped conceptual end `-1`; and
4. checked arithmetic is mandatory and any contradiction is a preflight failure, not an
   alternative formula.

This conversion is internal formula construction from already normalized immutable attributes.
It does not add a model operation or bypass the public `Tensor.slice` contract.

### Typed `ERF` coefficient

`C_erf` is a scalar `ScalarValue` attribute used by public scalar `mul`; it is not a new Tensor
leaf, constant-sidecar binding, eager value, or numerical implementation choice. Use these fixed
correctly rounded representations of `2 / sqrt(pi)`:

| Data type | Exact representation |
|---|---|
| `BFLOAT16` | raw bits `0x3F90` |
| `FLOAT32` | raw bits `0x3F906EBB` |
| `FLOAT64` | raw bits `0x3FF20DD750429B6D` |

Do not compute the coefficient from host transcendental functions during compilation. Preserve
the existing generated zero/one logical-splat rules unchanged; no additional splat source is
needed for this scalar operation attribute.

### Rule ownership

Keep the package-private named owners narrow:

- `ElementwiseGradientRules` gains only `ERF`;
- `ReductionGradientRules` gains masked `SUM` and the guarded `SUM_TO_SHAPE` variant;
- `LayoutGradientRules` gains `SLICE`, role-aware `SLICE_UPDATE`, `SELECT`, `PAD`, `TILE`,
  `CONCAT`, and `STACK`; and
- new `LinearAlgebraGradientRules` owns only `MATMUL` rank dispatch and last-two-axis
  permutation.

`FirstOrderAutograd` remains the single reverse-traversal and contribution-accumulation owner. It
may dispatch to the new linear-algebra owner but must not absorb formula implementations.
`AutogradPreflight` remains the single complete fail-closed selector.

Rules call only current public `Tensor` methods and public immutable model value types. They do
not construct `Operation`, `TensorProducer`, captured nodes, graph IDs, or descriptors directly.
Do not add a registry, map-based kind dispatch, service loader, reflection, string dispatch,
generic symbolic algebra, second graph, or public facade.

### Complete current-operation audit

The following audit is exhaustive over the current model `OperationKind` inventory. “0004”
means already supported and unchanged. “0004A” means exactly the guarded rows above. “0004B”
means a derivative convention must be selected first. “Later cohesive task” means policy-free
formula work that is too large or structurally distinct for this bounded change; progressive
planning deliberately assigns no detailed task ID here. “ND” means no cotangent propagates
through that role.

| Current operation inventory | Classification and reason |
|---|---|
| Binary `ADD`, `SUB`, `MUL`; scalar `ADD`, `SUB`, `MUL` | 0004 for its exact same-floating variants. Mixed-floating cotangent conversion remains 0004B. |
| Binary/scalar `DIV`, `POW`; `RECIPROCAL`, `LOG`, `LOG1P`, `SQRT`, `RSQRT` | 0004B: zero, invalid-domain, singularity, infinity, and NaN derivative behavior. |
| Binary/scalar `MIN`, `MAX`; scalar `CLAMP`; `ABS`, `FLOOR`, `CEIL`, `SIGN`, `RELU` | 0004B: ties, endpoints, discontinuities, signed zero, NaN, or subgradient selection. |
| `NEG`, `EXP`, `EXPM1`, `SIGMOID`, `TANH` | 0004. |
| `ERF` | 0004A with fixed typed coefficient bits. |
| `GELU`, `GELU_TANH_APPROXIMATION`, `SILU` | 0004B: otherwise regular formulas contain special-value products such as infinity times zero and require an explicit exceptional-value contract. |
| `WHERE` | 0004 for same-floating branches; condition is ND. Mixed-floating branch cotangents remain 0004B. |
| `CAST` | 0004 only for same-type floating identity. Cross-floating conversion remains 0004B; integral and BOOL roles are ND. |
| Comparisons `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | ND BOOL results. |
| BOOL `AND`, `OR`, `NOT`; `IS_FINITE`, `IS_NAN`, `IS_INF` | ND BOOL results. |
| Ordinary `SUM` | 0004. |
| Masked `SUM` | 0004A; mask is ND. |
| `SUM_TO_SHAPE` | 0004A only for the locally provable expansion subset. Binding-dependent unresolved singleton-or-equal inversion remains a later cohesive shape-binding task. |
| Ordinary or masked `MEAN`; `PROD`; reduction `MIN`, `MAX`; `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | 0004B: empty count, zero products/norms, extrema ties, singularities, or exceptional values. |
| `ALL`, `ANY`, `ARG_MAX`, `ARG_MIN` | ND outputs; extrema index/tie policy is not a floating cotangent route. |
| `CUM_SUM` | 0004. |
| `CUM_PROD` | 0004B: zero and exceptional-value product behavior. |
| `SOFTMAX`, `LOG_SOFTMAX` | 0004B: the current contract deliberately leaves empty slices, NaNs, and infinities open. |
| `CONTIGUOUS`, `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE` | 0004. |
| `SLICE` with `SliceAttrs`; `SLICE_UPDATE`; `SELECT`; `PAD`; `TILE`; `CONCAT`; `STACK` | 0004A exactly as guarded above. `PAD`'s constant and `SELECT`'s index are ND metadata. |
| `SLICE` with `CropToShapeAttrs` | Later cohesive shape-binding task: dynamic target-relative prefixes cannot in general be inverted through a current public update/pad operation. |
| `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, `SCATTER_ELEMENTS`, `SCATTER_ADD`, `SCATTER_ND` | Later cohesive indexing task using current scatter-add/gather primitives, with exact reduction-variant and duplicate-index validity guards. Index inputs are ND. |
| `ONE_HOT` | ND index/depth roles and output for current gradient purposes. |
| `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D` | Later cohesive window-adjoint task. Current public inverse/overlap-sum primitives make policy-free formulas plausible, but their complete dynamic geometry matrix is deliberately not added here. |
| `MATMUL` | 0004A for all current floating rank cases, role-aware exact-type targets, and current batch broadcasting. Integral routes are ND. |
| `CONV2D` | Later cohesive structured-linear task; grouped/batched window and parameter-role formulas require their own bounded matrix. |
| `MAX_POOL2D` | 0004B: maxima ties, NaNs, and saved-selection policy. |
| `AVERAGE_POOL2D` | Later cohesive pooling/window task after its divisor and padding-edge matrix is specified; no policy is selected here. |
| `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | Later cohesive loss task for regular variants; mean-empty and logits exceptional cases remain 0004B. Index labels are ND. |
| `SCALED_DOT_PRODUCT_ATTENTION` | Later cohesive multi-output structured task after 0004B selects all-masked-row, positive-infinity tie, and NaN behavior. Mask and state roles are ND. |
| `LAYER_NORM`, `RMS_NORM`, `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | Later cohesive normalization/multi-output task after 0004B selects zero-variance and exceptional-value behavior. Saved statistics require explicit logical-role handling, not a physical lifetime policy in this task. |
| `SORT`, `ARGSORT`, `TOP_K` | 0004B for selected-value ties, NaNs, discontinuities, and index routing; index outputs are ND. |
| `DROPOUT` | Later stochastic/multi-output task after explicit probability-edge and saved-mask rules; RNG state and mask outputs are ND. |
| graph RNG `INITIAL_STATE` | ND opaque state. |
| Unknown/custom kind, wrong attributes class, wrong cardinality, missing canonical output, or descriptor contradiction | Deterministic preflight rejection. |

`linear` has no independent operation kind: its current expression is
`PERMUTE -> MATMUL -> ADD`. It becomes differentiable through the selected `MATMUL` row plus
existing rules and receives no special compiler rule.

## Out of scope

- changing Compiler 0004's request, seed, identity, accumulation, capture, phase, result,
  validation, or optimization contracts
- any derivative row or attributes variant outside `SUPPORTED_0004` plus `SUPPORTED_0004A`
- selecting any 0004B derivative convention
- binding-dependent `SUM_TO_SHAPE` or target-relative crop inversion
- gather/scatter, windows, convolution, pooling, loss, attention, normalization, ordering,
  top-k, dropout, or RNG derivative implementation
- public compiler, gradient, registry, compile-request, result, artifact, or publication APIs
- explicit seeds, non-scalar objectives, disconnected-target zeros, Jacobians, create-graph, or
  higher derivative order
- changing model operations, Tensor public APIs, descriptors, producer output identity, or result
  carriers
- a second gradient algebra, captured-value-to-Tensor conversion, direct IR construction, mutable
  gradient state, tape, global cache, or backend-owned autograd
- runtime, prepare, engine, planning orchestration, lowering, backend, execution, trace, physical
  saved-value lifetime, or numerical kernel work
- new graph rewrites, constant-fold rows, cross-phase CSE, pass iteration, or approximate algebra
- architecture, ADR, module dependency, Gradle/build, Java-version, architecture-test,
  backend-conformance, or integration-test changes
- creating a detailed specification for 0004B or any later cohesive rule task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Planning guide](../../../planning-guide.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model contract-closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Training API](../../../../api/training-api.md)
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative; this task implements the already accepted
  compiler-owned pre-capture design and changes no architecture decision.
- Compiler owns preflight, rule selection, reverse accumulation, formula construction, combined
  capture, phases, and validation.
- Model owns only the ordinary public Tensor vocabulary and immutable operation metadata.
- Formulas exist as Tensor expressions before graph capture; graph-local IDs are assigned exactly
  once by the existing combined capture.
- Exact Tensor object identity remains request-local bookkeeping, never graph identity or
  persistent state.
- No dependency direction or module boundary changes. Architecture tests do not change.
- If a selected formula needs a new model API, derivative policy, public compiler contract,
  second algebra, or path outside maximum scope, stop and report the conflict.

## Package impact

Existing package:

- `io.github.pho001.synaptik.compiler` remains the single cohesive package-private compiler
  front-end boundary.

Types:

- `AutogradPreflight` extends its closed typed, role-aware occurrence selection.
- `ElementwiseGradientRules`, `ReductionGradientRules`, and `LayoutGradientRules` gain only their
  assigned rows.
- `LinearAlgebraGradientRules` is one new package-private stateless owner for `MATMUL`.
- `FirstOrderAutograd` changes only enough to dispatch the additional selected plan entries.

No package or public type is added. Tests remain in the mirrored compiler package.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and planning files:

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

No Tensor API, Training API, architecture, ADR, model, Gradle, or test-infrastructure file is
expected to change. Review those surfaces and record the reasoned no-change conclusion.

## Maximum scope

This task may create or modify at most:

- 6 compiler production files,
- 4 compiler test files, and
- 7 documentation/planning files,

for a strict ceiling of 17 touched paths.

If implementation needs another path, a model API, or a broader rule matrix, stop and propose a
follow-up instead of widening this task.

## Acceptance criteria

### Preflight and matrix

- Existing `SUPPORTED_0004` success and rejection behavior remains unchanged.
- Preflight accepts every `SUPPORTED_0004A` row only with its exact kind, attributes class,
  output/input role, descriptor, type, Shape, and local-constructibility guard.
- Preflight remains role-aware for promoted `MATMUL` and for the two `SLICE_UPDATE` inputs.
- Every unsupported occurrence on a selected route fails before a new Tensor ID is consumed, with
  the existing deterministic occurrence/output/input/kind/attributes/reason context.
- The complete current-operation classification above is reflected in code tests: no generic
  fallback or accidental kind admission exists.

### Formula behavior

- `ERF` uses only the fixed raw coefficient bits and public Tensor operations.
- Masked `SUM` restores the selected axis, broadcasts to the data Shape, masks with exact zero,
  and never propagates a cotangent to the BOOL mask.
- `SUM_TO_SHAPE` accepts exact-equal/static-one aligned Dimensions, including leading reductions,
  and rejects binding-dependent aligned pairs.
- Every floating `MATMUL` rank pairing has exact left/right formulas, batch unbroadcasting, vector
  insertion/removal, and selected-role type guards.
- Slice extraction, signed slice replacement, select, pad, tile, concat, and stack formulas
  preserve exact normalized geometry, input order, symbolic Shape references where specified,
  zero extents, and repeated-input contribution multiplicity.
- Every formula is built through public Tensor operations; no operation, producer, captured node,
  graph ID, or descriptor is manufactured directly.

### Existing pipeline

- Contribution accumulation remains left-associated and deterministic by original occurrence and
  input-position order.
- Repeated exact operands and repeated concat/stack inputs receive every positional contribution.
- Generated formulas join the existing combined Tensor DAG and are captured once with correct
  `FORWARD`/`BACKWARD` phases.
- Forward outputs, gradient result roles, output de-duplication, constant ingress, validation, and
  exact optimization behavior remain unchanged.
- `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` retain the Compiler 0004 mode
  contract.

### Documentation and completion

- Every changed Java type/method has meaningful complete Javadoc, including parameters, results,
  failures, nullability, and ownership where applicable.
- Compile API, autograd strategy, user guide, glossary impact, compiler master plan, roadmap, and
  this task agree on the exact supported/deferred matrix.
- Tensor API and Training API are reviewed and left unchanged because no public Tensor or
  training lifecycle contract changes; the completion summary records this.
- A separate clean-context documentation-focused agent finalizes affected Javadoc and prose after
  implementation, reusing successful Java test evidence unless it changes executable behavior.
- Final status changes to `Complete` only after implementation, documentation, all validation,
  exact-scope inspection, and the completion summary are complete.

## Tests / validation

Implementation tests must cover at least:

- positive `ERF` formula structure and exact coefficient bits for BFLOAT16/FLOAT32/FLOAT64;
- masked `SUM` full Shape restoration, broadcast mask, masked false branch, zero extent, mask ND,
  and wrong-type/attrs rejection;
- `SUM_TO_SHAPE` leading axes, exact aligned dimensions, static-one expansion, scalar target,
  zero extent, and binding-dependent rejection;
- all four `MATMUL` rank pairings, batched broadcast on either operand, vector promotion,
  repeated operand, selected promoted operand success/failure, zero contraction, and integral
  rejection;
- positive, negative, strided, empty, and identity `SLICE`;
- `SLICE_UPDATE` base/update roles, signed steps, empty replacement, repeated exact base/update,
  dynamic selected-base rejection only for the update role, and no partial allocation;
- scalar-result and dynamic-extent `SELECT`, scalar `PAD`, symbolic/zero-extent padding,
  scalar/dynamic `TILE`, multi-input symbolic `CONCAT`, `STACK`, and repeated exact composition
  inputs;
- unsupported 0004B and later-cohesive categories still failing preflight before allocation;
- end-to-end combined capture, phase facts, gradient roles, logical constants, optional
  optimization, and mode behavior for representative new rows; and
- existing Compiler 0004 regression coverage.

Run once after executable code stabilizes:

```bash
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
```

The documentation-focused pass validates links, anchors, fences, terminology, public-surface
claims, and exact support-matrix consistency without rerunning successful Java tests unless it
changes executable Java behavior.

Final repository checks:

```bash
git diff --check
git status --short
git diff --name-only
git ls-files --others --exclude-standard
```

The union of modified and untracked paths must equal only the affected-file list and remain at or
below 17 paths. No architecture-test, backend-conformance, integration-test, Gradle, model,
Tensor API, or Training API change is expected.

## Dependencies

- Compiler 0004 — Complete
- Model 0025 and Compiler 0001–0003B — Complete through Compiler 0004
- current public Tensor operations and operation metadata — Complete
- accepted compiler-owned pre-capture architecture and ADR 0009 — Complete

No 0004B policy decision is a dependency because every selected row is policy-free.

## Follow-up tasks

- Compiler 0004B selects explicit tie, endpoint, discontinuity, singularity, empty-domain,
  exceptional-value, and cross-floating cotangent policies before adding dependent formulas.
- Later cohesive tasks may address binding-dependent crop/shape inversion, indexing/scatter,
  windows, structured linear operations, pooling, losses, normalization, multi-output attention,
  ordering, and stochastic operations. Progressive planning assigns detailed specifications only
  when each becomes the current frontier.
- Compiler 0005 owns publication, planning orchestration, and immutable compile artifacts.
- Compiler 0006 owns public functional gradient requests, explicit seeds, derivative order, and
  the higher-order formula-coverage contract.

## Architecture impact

None. This task fills the already accepted Compiler 0004A extension point inside the compiler
package. It changes no module boundary, dependency direction, lifecycle, graph phase, capture,
public API, or authoritative architecture rule. No ADR or architecture test is required.

## Implementation prompt

Implement Compiler task 0004A exactly as specified in this file.

Work in a separate clean implementation context. First read `AGENTS.md`, `ARCHITECTURE.md`, the
focused architecture documents and ADR 0009, documentation rules and relevant profiles, planning
guide and roadmap, compiler master plan, Compiler 0001–0004, Model 0025, both model audits, Compile
API, Tensor API, Training API, autograd strategy/user guide, glossary, and the current compiler
and model operation sources/tests needed to verify every selected attributes variant.

Extend the existing package-private preflight and named rules only. Implement the exact additive
matrix, fixed `ERF` coefficient bits, role-aware `MATMUL` and `SLICE_UPDATE` guards, deterministic
formula construction, and tests. Preserve all Compiler 0004 request, identity, accumulation,
constant, capture, phase, result, validation, and optimization contracts. Do not add a public
surface, registry, second algebra, direct IR construction, derivative policy, model API, or
unlisted operation.

After Java behavior stabilizes, run the specified compiler tests and Javadoc once. Do not repeat
the repository-wide capability checkpoint reserved by the master plan for 0004B. Then use a
distinct clean documentation-focused context to independently finalize Javadoc, Compile API,
autograd strategy, user guide, glossary impact, task, master plan, and roadmap. Reuse successful
test evidence unless documentation changes executable Java behavior. Run the
documentation/scope/whitespace checks and record exact evidence.

If any selected row cannot be expressed through current public Tensor operations, if a derivative
policy is required, or if more than 17 paths are needed, stop and report the exact conflict rather
than widening scope. Do not commit or push unless explicitly requested.

## Local decisions

- The extension is deliberately vertical across four existing formula families, but remains
  bounded to formulas with no derivative convention.
- `ERF` coefficient representation is fixed at planning time so compilation never depends on host
  transcendental evaluation.
- `MATMUL` is role-aware under floating promotion; same-type selected cotangents are supported
  without forcing the unrelated operand to have that type.
- `SUM_TO_SHAPE` admits only locally constructible inverse expansion. Deferred binding
  obligations are not guessed.
- `SLICE_UPDATE` base and update roles have different constructibility guards.
- Data-movement formulas preserve repeated positional contributions and symbolic Shapes rather
  than collapsing equal Tensor occurrences.
- Plausible policy-free indexing, window, and structured formulas remain later cohesive work
  because including their complete variant matrices would exceed this task's risk and path budget.

## Known limitations

- The request remains one scalar objective, implicit unit seed, and ordered connected targets.
- Mixed-floating cotangents and all 0004B conventions remain unsupported.
- Binding-dependent `SUM_TO_SHAPE` and crop inversion remain unsupported.
- The operation audit classifies later work but implements only `SUPPORTED_0004A`.
- Higher-order request semantics and complete derivative closure remain Compiler 0006 work.

## Validation evidence

Planning and implementation evidence:

- architecture, focused design, planning, documentation, API, completed Compiler 0001–0004,
  Model 0025, model audits, compiler source/tests, and complete current operation-kind inventory
  reviewed
- no architecture or dependency conflict found
- selected rules verified as compositions of current public Tensor operations
- selected/deferred/non-differentiable classification recorded exhaustively
- task path budget fixed at 17
- implementation context `/root/implement_compiler_0004a_gradient_extensions` completed the six
  production-file and four test-file implementation without changing architecture, dependencies,
  public APIs, build configuration, or test infrastructure
- `./gradlew :modules:compiler:compileJava` and
  `./gradlew :modules:compiler:compileTestJava` passed
- the focused `AutogradPreflightTest`, `GradientRulesTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest` run first exposed one assertion-message expectation, then passed all 28
  tests after that test-only correction
- the final `./gradlew :modules:compiler:test` run passed 131 tests with zero failures, errors, or
  skips; `git diff --check` also passed in the implementation context
- documentation context `/root/document_compiler_0004a_gradient_extensions` changed no executable
  Java tokens and deliberately did not repeat the successful Java suite; comment-stripped SHA-256
  hashes for all six authorized production paths matched the hashes recorded before the Javadoc
  pass
- `./gradlew :modules:compiler:javadoc` passed; its one warning is the pre-existing undocumented
  default constructor of `FirstOrderAutograd.DerivativeConstants`, and generated pages for all
  six affected owners were inspected for the finalized contracts
- `node /tmp/validate_synaptik_markdown.js` passed 240 Markdown files, 4,402 local links, 271
  anchors, and 2,950 fence markers
- final whitespace, status, terminology, and exact 17-path scope checks passed
- repository-wide validation remains deferred to the Compiler 0004B capability checkpoint and CI
  under the planning guide's validation tiers

## Completion summary

- Completed the additive fail-closed `SUPPORTED_0004A` preflight matrix and rule dispatch for
  typed ERF, masked SUM, locally invertible SUM_TO_SHAPE, every floating MATMUL rank pairing, and
  the guarded exact layout/selection/composition families.
- Added and finalized the narrow `LinearAlgebraGradientRules` owner while preserving the existing
  request, seed, identity accumulation, deterministic repeated-position handling, one-capture,
  phase, result, validation, and optimization contracts.
- Added focused preflight, formula-structure, repeated-role, and end-to-end graph-compiler
  regression coverage across the four authorized test files.
- Finalized Javadocs, Compile API, autograd strategy, user guide, glossary, task, compiler master
  plan, and roadmap within the exact 17-path ceiling.
- Reviewed Tensor API and Training API and left them unchanged because the implementation changes
  no public Tensor surface, Tensor result/gradient lifecycle, training lifecycle, or executable
  contract.
- Reviewed `ARCHITECTURE.md`, focused architecture documents, ADR 0009, related operation
  contracts, architecture tests, backend conformance, integration tests, Gradle configuration,
  and other modules. No architecture decision, dependency boundary, backend behavior, end-to-end
  behavior, build contract, or cross-module public contract changed, so no update was required.
- No unresolved issue or task-local follow-up remains. Compiler 0004B retains derivative-policy
  selection and remains Draft without a detailed specification.

Status: Complete
