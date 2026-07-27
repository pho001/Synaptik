# Task 0004B: Shared-algebra Cotangent Normalization and Local Derivative Rules

## Status

Complete

## Goal

Extend the completed Compiler 0004/0004A first-order pre-capture automatic-differentiation
pipeline with one closed matrix for:

```text
mixed-floating cotangent Shape/DataType normalization
  + binary and scalar division differentiation rules
  + direct-zero FLOOR/CEIL/SIGN local derivative conventions
  + ordinary and masked mean differentiation rules
  -> ordinary public Tensor expressions
  -> the existing shared validation and exact optimization pipeline
```

The combined forward/backward graph has one shared algebra and one shared inference, validation,
numerical-semantics, and optimization contract. A generated gradient expression is an ordinary
Tensor expression. There is no gradient-only arithmetic, comparison, cast, NaN/infinity,
rewriting, folding, common-subexpression-elimination, or dead-code-elimination policy or pass.

Autograd adds only the local differentiation transformation: for one selected producer occurrence
and incoming cotangent, it constructs ordinary Tensor expressions, applies any genuinely required
local derivative or subgradient convention, and normalizes each contribution to its selected
input's Shape and DataType before deterministic accumulation.

## Scope

### Existing pipeline remains the owner

Extend only the existing package-private preflight, reverse-dispatch and derivative-constant
helper, and named gradient rule owners:

```text
complete original Tensor inventory
  -> complete fail-closed differentiation-rule preflight
  -> local formulas through ordinary public Tensor operations
  -> cotangent Shape/DataType normalization
  -> identity-keyed ordered contribution accumulation
  -> one phase-aware combined capture
  -> Compiler 0002 inference and validation
  -> existing exact shared optimization
```

Preserve the `GraphCompiler.compile(...)` parameter list, `AutogradPreflight.FirstOrderRequest`
record shape, implicit scalar unit seed, `GraphCompilation` result, result-role ordering,
one-capture semantics, graph phases, constant ingress, validation boundaries, and optimization
pass order.

`FirstOrderAutograd.apply()` currently routes only `AggregateReductionKind.SUM` and
`CumulativeScanKind.CUM_SUM` to `ReductionGradientRules`. Extend that existing closed dispatch
only enough to route preflight-approved ordinary and masked `AggregateReductionKind.MEAN`
occurrences to the same named reduction-rule owner. This is required plumbing for the selected
MEAN rows, not a new formula owner, rule family, policy, public API, or algebra.

### One shared algebra

Forward expressions and compiler-generated gradient expressions use the same model-owned
operation vocabulary and contracts:

- the model's public Tensor methods construct every arithmetic, cast, comparison, selection,
  reduction, and Shape expression;
- Compiler 0002 derives and validates descriptors and deferred constraints without a
  forward/backward semantic split;
- each generated node inherits the operation kind's current mathematical and numerical meaning,
  including its ordinary NaN, infinity, signed-zero, overflow, underflow, rounding, and
  empty-domain contract where one exists;
- the exact Compiler 0003A rewrite matrix and Compiler 0003B fold matrix apply under the same
  existing guards in either graph phase;
- dead-code elimination remains whole-graph and common-subexpression elimination remains
  phase-local; and
- no formula in this task authorizes reassociation, reciprocal/division substitution, floating
  constant evaluation, approximate algebra, a new rewrite, or a new pass.

`GraphPhase.BACKWARD` classifies generated work. It does not select a different algebra,
validation rule, or optimization pipeline.

### Shared notation and cotangent normalization

For each selected output:

- `g` is its accumulated cotangent;
- `x`, `left`, and `right` are exact original input Tensors;
- `y` is the producer's canonical exact selected output wrapper;
- `Z_t` and `O_t` are the existing explicit exact typed positive-zero and positive-one logical
  splats expanded through public `Tensor.expand` to Tensor `t`'s Shape;
- `R_x(v)` restores any reduction axes removed from `v` and expands through public Tensor
  operations to `x.shape()`; and
- `C_x(v)` first applies ordinary `v.sumToShape(x.shape())` when broadcast reversal is required,
  then applies ordinary `cast(x.dataType())` exactly when the result type differs from `x`'s
  floating type.

`C_x` is cotangent Shape/DataType normalization, not an alternative arithmetic or conversion
policy. `sumToShape` and `cast` retain their current model semantics and enter the same combined
capture, inference, validation, and optimization pipeline as any forward occurrence. Same-type
rows add no redundant cast.

For role-aware `MATMUL`, construct the existing rank-specific formula, reverse batch broadcasting
with `sumToShape`, and then cast once to the selected operand type if required. Every contribution
must have its selected input's exact Shape and floating DataType before it is appended for
accumulation.

### Exact typed generated constants

Retain the current request-local exact positive-zero and positive-one logical-splat behavior.
This task adds no half, bound, count, NaN, infinity, or other scalar leaf. Forward-ingress bindings
remain first, generated derivative bindings retain deterministic first-use order, and no value is
inferred from storage, labels, descriptors, factory history, Shape, or missing provenance.

Logical-one expansion is valid for static, dynamic, and expression Shapes. A logical splat has one
typed scalar fact for every eventual logical coordinate without storing, enumerating, or binding
those coordinates.

### Mixed-floating request and accumulation

Relax only the current exact objective-type restriction on requested targets and selected
intermediate inputs:

- the objective remains scalar, floating, gradient-eligible, and implicitly seeded in its exact
  type;
- every target remains floating, gradient-eligible, identity-unique, in objective ancestry, and
  connected through a selected differentiable route, but may have a different floating type;
- every selected output and selected differentiable input is floating;
- operation-specific promotion and Shape facts must match the current model contract; and
- every contribution passes through `C_x`, or the corresponding rank-specific normalization,
  before entering the selected input's contribution list.

The explicit `CAST` occurrences are ordinary model conversion requests. This task adds no
gradient-specific rounding, NaN-payload, conversion, or backend-instruction promise.

### Closed `SUPPORTED_0004B` matrix

The rows below are additive to `SUPPORTED_0004` and `SUPPORTED_0004A`. Existing same-type
construction remains unchanged except where dispatch is naturally shared with the normalized
path.

#### Mixed-floating normalization and division rules

| Family and exact variant | Preflight guard | Local formula and normalization |
|---|---|---|
| Binary `ADD` | Exact `NoOperationAttrs`, two floating inputs, current promotion and broadcast facts. | Left `C_left(g)`; right `C_right(g)`. |
| Binary `SUB` | Same as binary `ADD`. | Left `C_left(g)`; right `C_right(g.neg())`. |
| Binary `MUL` | Same as binary `ADD`. | Left `C_left(g.mul(right))`; right `C_right(g.mul(left))`. |
| Binary `DIV` | Exact `NoOperationAttrs`, two floating inputs, current promotion and broadcast facts. | Left `C_left(g.div(right))`; right `C_right(g.mul(left).neg().div(right.mul(right)))`. |
| Scalar `DIV` | Exact `ScalarValueAttrs`; input, output, and scalar have one exact floating type. | `g.div(attrs.value())`. |
| `WHERE` branch roles | Exact `NoOperationAttrs`; BOOL condition; current three-way broadcast; selected branch and output are floating. | Existing zero-routed true/false branch formula, followed by `C_branch`; condition is non-differentiable. |
| Floating-to-floating `CAST` | Exact `CastAttrs`; source and target are floating; target attribute equals the output DataType. | Same type returns `g`; otherwise `g.cast(sourceType)`. |
| Floating `MATMUL` | Existing complete rank, contraction, promotion, and batch-broadcast facts. | Existing four rank-case formulas; restore selected operand Shape, then cast once to that operand DataType when needed. |

Binary `DIV` is an ordinary local derivative formula expressed in the public Tensor vocabulary.
Construct the right contribution in the exact visible order shown above; do not replace it with
an unproved reassociation. Every generated `MUL`, `NEG`, and `DIV` node inherits the current model
semantics, and the existing global exact optimization rules apply under their normal guards.
There is no separate gradient singularity, NaN, infinity, signed-zero, overflow, underflow, or
rounding policy and no exemption from shared optimization.

#### Direct-zero local derivative conventions

| Family and exact variant | Local derivative convention |
|---|---|
| `FLOOR`, `CEIL`, `SIGN` | Return `Z_x` directly for the represented input, independent of `g`; do not construct `g.mul(0)` and do not construct a floating comparison. |

These are genuine first-order local derivative conventions at discontinuous or otherwise
nondifferentiable inputs. They choose a direct zero cotangent for every represented input,
including signed zeros and non-finite inputs, without defining or changing the forward operation's
model semantics.

#### Ordinary `MEAN`

Support current floating full, single-axis, and multi-axis `MEAN` variants for static, dynamic,
and expression selected extents. An empty multi-axis list remains a point domain. Do not compute
the domain count in host code, inspect `Shape.knownElementCount()`, compare the count with
DataType-specific exact-integer thresholds, or introduce a scalar count attribute.

Preflight requires one exact floating input/output type and the current family-owned descriptor
contract:

- full form uses `NoOperationAttrs.INSTANCE` and scalar output;
- single-axis form uses exact `AxisReductionAttrs` with its normalized axis and recorded
  keep-dimensions result; and
- multi-axis form uses exact `MultiAxisReductionAttrs` with ordered distinct normalized axes and
  the recorded keep-dimensions result.

Construct the denominator through ordinary Tensor operations:

```text
ones  = O_x
count = ones.sum()                  // full reduction
      | ones.sum(axis, true)        // single axis
      | ones.sum(axes, true)        // ordered multi-axis set, including []
dx    = R_x(g).div(count.expand(x.shape()))
```

The expanded logical-one expression and the reduction retain dynamic and expression Dimensions,
so the existing expression vocabulary represents a count whose concrete extent is bound later.
The `SUM`, `DIV`, and `EXPAND` occurrences have exactly their ordinary model numerical and Shape
semantics. A static or later-bound zero reduced domain produces an empty input cotangent Shape;
the generated denominator remains an ordinary zero-domain `SUM` result and receives no special
gradient execution rule.

#### Masked `MEAN`

Support the current masked, axis-removing floating `MEAN` variant with exact
`MaskedReductionAttrs`, ordered inputs `[input, mask]`, exact BOOL mask, current proof that the mask
broadcasts exactly to the input Shape, and the current output descriptor.

Use only ordinary public expressions:

```text
selected = Tensor.where(mask, O_x, Z_x)
count    = selected.sum(axis, true)
restored = R_x(g)
dx       = Tensor.where(
             mask,
             restored.div(count.expand(x.shape())),
             Z_x)
```

The mask role is non-differentiable. For a slice with zero selected values, the forward masked
mean is NaN under the current model contract; this task's explicit local derivative convention
returns zero for every input coordinate in that slice. The final `WHERE` expresses that choice
through ordinary conditional-selection semantics. It does not prescribe eager or lazy branch
evaluation, suppress an intermediate calculation, or add special compiler/backend execution
behavior.

### Complete deferred and non-differentiable classification

Every current family not in `SUPPORTED_0004`, `SUPPORTED_0004A`, or the exact rows above remains
rejected on a selected route.

| Current family | Classification after 0004B |
|---|---|
| Binary/scalar `POW` | **Blocked by general model semantics.** The current model leaves real-domain and numerical-edge behavior open. Autograd cannot repair that missing shared algebra contract. |
| `RECIPROCAL`, `LOG`, `SQRT` | **Blocked by incomplete general model semantics.** Zero/domain/signed-zero/non-finite meaning is not yet complete enough for a shared forward/backward graph contract. |
| `LOG1P`, `RSQRT` | **Later cohesive unary differentiation task.** Their current model semantics are sufficient for ordinary formulas, but this bounded task does not add that independent rule family. |
| `GELU`, `GELU_TANH_APPROXIMATION`, `SILU` | **Later cohesive activation differentiation task.** Generated operations would retain ordinary shared semantics; no special gradient exceptional-value algebra is implied. |
| Binary/scalar `MIN`/`MAX`, `CLAMP`, `ABS`, `RELU`, and reduction `MIN`/`MAX` | **Blocked by missing general floating-comparison semantics.** The public comparison contract still leaves NaN, signed-zero, and equality/tolerance behavior open, so the compiler cannot construct a model-owned predicate indirectly. |
| `PROD`, `CUM_PROD` | **Later cohesive product differentiation task.** Zero-safe prefix/suffix geometry and complete full/axis/multi-axis coverage are independently substantial; generated arithmetic would use the shared algebra. |
| `LOG_SUM_EXP` | **Later cohesive reduction differentiation task.** Its current forward semantics are not a new gradient algebra; the complete local formula and selected family coverage remain outside this bounded matrix. |
| `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | **Later cohesive statistical/norm differentiation task.** Correction and local derivative/subgradient conventions must be selected with the complete family. |
| `SOFTMAX`, `LOG_SOFTMAX` | **Blocked by incomplete general model numerical-edge semantics.** Empty slices, NaNs, infinities, and finite-precision behavior remain outside the current shared model contract. |
| `LAYER_NORM`, `RMS_NORM` | **Later cohesive normalization task**, after every required shared model semantic dependency is complete. |
| `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | **Later cohesive normalization/multi-output task.** Saved output slots are available, but role coverage and complete local formulas require their own matrix. |
| `MAX_POOL2D` | **Later cohesive pooling task.** The current first-sample selection semantics are model-owned; complete window routing remains outside 0004B. |
| `AVERAGE_POOL2D` | **Later cohesive pooling/window task.** Fixed divisor semantics are known, but complete dynamic window/fold geometry is separate. |
| Mean-squared-error and categorical-cross-entropy losses | **Later cohesive loss task.** Their complete prediction/target/reduction/ignored-role matrices are not added here. Index targets remain non-differentiable. |
| Scaled dot-product attention | **Later cohesive multi-output structured task.** Weights and mask roles, causal behavior, and complete local formulas must be selected together. |
| `SORT`, values output of `TOP_K` | **Later cohesive ordering task.** Stable order is fixed, but equal-key/cutoff and selected-value routing need a complete local convention. `ARGSORT` and top-K indices are non-differentiable. |
| `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, replacement/ADD scatter | **Later cohesive indexing/scatter task.** Variant, duplicate-target, and dynamic-Shape coverage is too large for 0004B. Index roles are non-differentiable. |
| MUL/MIN/MAX scatter reductions | **Later cohesive scatter-reduction task**, dependent on complete shared reduction and, for extrema, floating-comparison semantics. |
| `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `CONV2D` | **Later cohesive window/structured-linear tasks.** Their geometry matrices are independent of this task. |
| `DROPOUT` | **Later cohesive stochastic/multi-output task.** Probability endpoints, state roles, scaling formula, and canonical saved-mask use must be specified together. State and mask outputs are non-differentiable. |
| Comparisons, BOOL logic/classification, `ALL`, `ANY`, `ARG_MIN`, `ARG_MAX`, one-hot indices, graph RNG state | **Non-differentiable output/roles.** They may be fixed formula conditions but receive no cotangent. |
| Existing 0004A binding-dependent `SUM_TO_SHAPE`, target-relative crop, and unselected layout variants | **Unchanged later Shape/layout work.** |
| Unknown/custom kind, wrong attributes class/cardinality, missing canonical output, or descriptor contradiction | **Deterministic preflight failure.** |

This classification does not add all remaining gradient families and does not create detailed
follow-up task specifications.

### Preflight and failure ordering

Preserve the existing order:

1. validate `GraphCompiler` arguments and mode/request agreement without creating a Tensor;
2. inventory the complete original forward DAG;
3. validate objective/target membership, floating eligibility, and selected reachability;
4. validate every selected occurrence in deterministic producer/input order, including exact
   kind, attributes, output role, input role, descriptors, promotion, Shape, cotangent
   normalization, and any explicit local derivative convention;
5. reject the first unsupported fact with existing occurrence/output/input/kind/attributes/reason
   context; and
6. only after the complete selected slice succeeds, create the seed, generated zero/one splats, or
   a formula Tensor.

Known preflight rejection must consume no new `TensorId`, including an unsupported operation after
an otherwise valid mixed-floating, division, direct-zero, or mean occurrence. Later construction,
capture, inference, validation, or optimization failures retain the existing monotonic-ID
behavior.

### Determinism, capture, validation, and shared optimization

- Contribution insertion remains producer- and input-position-ordered, including repeated exact
  operands.
- Shape/DataType normalization finishes before contribution insertion.
- Scalar-splat generation and ingress binding follow deterministic first use.
- Forward and generated expressions are captured together once with the existing original-
  producer identity phase classification.
- Compiler 0002 inference and validation remain authoritative after capture and after every
  changed optimization candidate.
- The existing order remains mandatory canonicalization/validation, then exact rewrite, exact
  fold, whole-graph DCE, phase-local CSE, and whole-graph cleanup DCE once each when optional
  optimization is enabled.
- Existing rewrite/fold matrices and guards do not widen. Generated casts, divisions, `WHERE`
  nodes, and mean denominator expressions are ordinary candidates; they receive neither special
  protection nor special transformation.
- No cross-phase CSE, fixed point, new algebraic identity, reassociation, or pass is introduced.

### First-order and higher-order implications

This task implements first-order rules only. Generated formulas remain ordinary differentiable
Tensor expressions. Compiler 0006 must still define create-graph/derivative order, disconnected
higher-order zero behavior, phase-plus-order representation, and complete formula-operation
coverage.

0004B does not claim second derivatives, differentiate a local convention, or retrofit Tensor
gradient state. No selected row needs a hidden auxiliary output, but preflight and capture
continue inventorying every canonical producer output slot exactly as Compiler 0004 requires.

## Out of scope

- any row outside the exact `SUPPORTED_0004B` matrix
- changing model operation semantics, public Tensor methods, operation kinds/attributes, producer
  outputs, descriptors, cast conversion semantics, comparison semantics, or Shape contracts
- a gradient-specific arithmetic, comparison, cast, reduction, NaN/infinity, folding, rewriting,
  validation, or execution contract
- softmax, statistical/norm, product, activation, normalization, pooling, loss, attention,
  ordering, indexing/scatter, window, convolution, dropout, or batch-normalization rules
- a public derivative-policy enum, registry, facade, request, result, compile API, or artifact
- explicit seeds, non-scalar objectives, disconnected-target public policy, Jacobians, vector-
  Jacobian products, create-graph, or higher derivative order
- model-owned derivative dispatch, Tensor gradient/backward state, a second gradient algebra,
  captured-value-to-Tensor conversion, direct node construction, tape, or global cache
- new graph rewrite/fold rows, cross-phase CSE, approximate algebra, reassociation, or pass
  iteration
- runtime, prepare, engine, planning orchestration, trace emission, backend lowering, kernels,
  execution, physical saved-value lifetime, or numerical conformance
- architecture, ADR, dependency, Gradle/build, Java-version, architecture-test source,
  backend-conformance source, or integration-test source changes
- creating a detailed Compiler 0005/0006 or later gradient-family task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Training API](../../../../api/training-api.md)
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns the one shared operation algebra, inference inputs, public Tensor
  vocabulary, and numerical semantics for forward and generated expressions.
- `modules/compiler` owns local differentiation-rule selection, preflight, reverse accumulation,
  cotangent normalization, formula construction, combined capture, and validation orchestration.
- Formula construction uses only current public Tensor operations and immutable model values.
- Compiler identity maps and generated constant caches are request-local ephemeral bookkeeping,
  not graph IR, public state, or registries.
- No derivative dispatch moves into model and no gradient lifecycle state moves into Tensor.
- Graph-local IDs are assigned exactly once during the existing combined capture.
- No dependency or module boundary changes.
- If implementation requires a model/public API, a new algebra or optimization rule, an
  architecture change, or a seventeenth path, stop and report the conflict.

## Package impact

Existing package:

- `io.github.pho001.synaptik.compiler` remains the single package-private compiler front-end
  boundary.

Type placement:

- `AutogradPreflight` extends the closed selected-role, mixed-floating normalization, division,
  direct-zero convention, and MEAN guards.
- `ElementwiseGradientRules` owns mixed arithmetic/cast/WHERE normalization, division formulas,
  and the three direct-zero conventions.
- `ReductionGradientRules` owns ordinary and masked MEAN local formulas.
- `LinearAlgebraGradientRules` extends only selected-role cotangent normalization after its
  existing rank/Shape restoration.
- `FirstOrderAutograd` extends only its existing closed typed dispatch so preflight-approved
  ordinary and masked MEAN occurrences reach `ReductionGradientRules`.

No package or public type is added. Tests remain in the mirrored compiler package.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and planning files during implementation:

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification and record reasoned no-change conclusions for the Tensor API, Training
API, `ARCHITECTURE.md`, focused architecture documents, ADRs, architecture tests, model/config/
planning/trace/runtime/prepare/engine/training/backend sources and tests, backend conformance,
integration tests, and Gradle/build files.

## Maximum scope

This task may create or modify at most:

- 5 compiler production files,
- 4 compiler test files, and
- 7 documentation/planning files,

for a strict ceiling of 16 touched paths.

The fifth production path is mechanically necessary because the current
`FirstOrderAutograd.apply()` dispatch excludes `AggregateReductionKind.MEAN`; implementing the
selected ordinary/masked MEAN formulas only in `ReductionGradientRules` would leave them
unreachable. The user's standing automatic approval for necessary higher path counts authorizes
this exact one-path increase. It does not authorize another operation family, formula, test,
documentation surface, or architectural change.

If another path, model API, public compiler type, test-infrastructure change, new optimization
rule, or broader differentiation matrix is required, stop and propose a follow-up or task
revision.

## Acceptance criteria

### Preflight and request contract

- Existing 0004/0004A success, failure, and zero-ID behavior remains unchanged.
- A target may differ from the objective's floating type only when every selected role has a
  complete ordinary `sumToShape`/`cast` normalization path under this task.
- Every selected contribution has the exact target input Shape and floating DataType before
  accumulation.
- Preflight admits only the exact new kind/attributes/output/input/descriptor/Shape/local-
  convention rows.
- Every unsupported row in the classification fails before any derivative Tensor ID is consumed.
- Diagnostics retain deterministic occurrence, output-role, input-role, kind, attributes, and
  reason context.

### Formula and local-convention behavior

- Mixed `ADD`/`SUB`/`MUL`, `WHERE`, and every selected `MATMUL` rank/role unbroadcast in the
  promoted type and cast once afterward; same-type formulas add no cast.
- Floating cross-type `CAST` returns exactly one ordinary reverse cast to the source type.
- Binary/scalar `DIV` constructs the exact local formulas specified above across BFLOAT16,
  FLOAT32, and FLOAT64; generated nodes retain ordinary model semantics and shared optimization
  eligibility without a singularity-specific policy.
- `FLOOR`, `CEIL`, and `SIGN` return direct exact zeros without an incoming-cotangent
  multiplication or floating comparison.
- Ordinary full/single-axis/multi-axis `MEAN` constructs its denominator by reducing `O_x` for
  static, dynamic, expression, zero, and empty-axis-list geometry; no host count or exact-integer
  threshold exists.
- Masked `MEAN` constructs its true-count through `WHERE` plus `SUM`, routes its quotient through
  a final ordinary `WHERE`, and applies the explicit all-false-slice zero-cotangent convention
  without claiming execution suppression.
- `FirstOrderAutograd` routes only the newly preflight-approved ordinary and masked `MEAN`
  occurrences to `ReductionGradientRules`; existing dispatch and formula ownership remain
  otherwise unchanged.
- Repeated operands retain repeated positional contributions in deterministic order.

### Shared algebra, capture, validation, and optimization

- Forward and generated expressions obey the same model algebra, inference, validation, and
  numerical contracts.
- Existing generated zero/one constants remain storage-free, non-gradient, explicitly bound
  logical splats with unchanged exact bits and deterministic order.
- Forward ingress and all generated derivative bindings retain deterministic order and fixed
  sources remain absent from bindable inputs.
- One combined capture preserves every original/generated phase, canonical output identity,
  output de-duplication, and target role.
- Compiler 0002 validation runs at the existing boundaries.
- Existing exact rewrite/fold matrices, phase-local CSE, whole-graph DCE, and pass order remain
  unchanged and apply under the same guards to both phases.
- No public compiler declaration, registry, second algebra, direct IR formula, Tensor gradient
  state, gradient-only pass, or new dependency is introduced.

### Documentation and completion

- Every changed Java declaration has meaningful complete Javadoc for local differentiation,
  cotangent normalization, constants, ordering, types, Shape, failures, and ownership.
- Compile API, autograd strategy, user guide, glossary, task, master plan, and roadmap agree that
  forward/backward share one algebra and on the exact selected/deferred matrix.
- Tensor API and Training API are reviewed unchanged because no public Tensor or training
  lifecycle contract changes.
- The independent clean-context documentation pass reuses successful Java evidence unless it
  changes executable behavior or records a concrete stale-evidence risk.
- Exact 16-path scope, statuses, links, anchors, fences, final newlines, terminology, and
  whitespace pass.
- The compiler transformation/autograd capability checkpoint passes before this task is marked
  `Complete`.

## Tests / validation

Implementation-focused tests must cover:

- all three unordered mixed-floating type pairs, with the narrower selected input in every
  applicable operand or branch position;
- broadcast-before-cast ordering for binary/WHERE and every `MATMUL` rank pairing;
- cross-floating cast reversal, shared/repeated mixed inputs, and mixed contribution accumulation;
- exact binary/scalar division formula structure with optional optimization disabled, plus proof
  that enabled optimization uses only the existing shared pass matrix and order;
- FLOOR/CEIL/SIGN direct-zero structure with no comparison or `g * 0` occurrence;
- full, single-axis, multi-axis, point-domain, static-empty, dynamic, expression-extent, and
  masked-mean variants, including all-false masks and absence of exact-count thresholds;
- existing zero/one exact bits, explicit ingress, deterministic reuse/order, and no storage
  inference;
- zero-ID preflight rejection for every deferred/blocked classification; and
- end-to-end modes, one capture, phases, result roles, constants, shared optional optimization,
  and 0004/0004A regressions.

After executable code stabilizes, run once:

```bash
./gradlew :modules:compiler:test
git diff --check
```

The implementation context must hand the exact test evidence and unchanged executable state to a
separate clean-context documentation-focused agent. That pass runs:

```bash
./gradlew :modules:compiler:javadoc
node /tmp/validate_synaptik_markdown.js
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the established Node validator is unavailable, use or recreate the equivalent repository-wide
check for local Markdown links and anchors, balanced fences, final newlines, and trailing
whitespace, and record the exact command and counts.

After implementation and documentation validation are stable, run the compiler
transformation/autograd capability checkpoint:

```bash
./gradlew test :testing:architecture-tests:test
```

Record XML test counts and Gradle outcomes. Do not rerun successful module tests in the
documentation pass without executable changes. Repository-wide validation after this checkpoint
is deferred to CI unless executable code changes or a concrete repository-wide risk appears.

Planning this task does not run Java tests or the checkpoint.

## Dependencies

- Compiler 0004A — Complete
- Compiler 0004, 0001–0003B — Complete
- Model 0025 — Complete
- Current model operation semantics and public Tensor operations used by the selected formulas —
  Complete
- Config 0002 and ADR 0009 — Complete

No architecture or model change blocks the selected matrix. Current dynamic/expression Shape,
logical-splat expansion, ordinary reduction, binary arithmetic, `WHERE`, `sumToShape`, and cast
contracts can represent every selected formula.

## Follow-up tasks

- Later cohesive differentiation tasks may address the explicitly classified unary, activation,
  product, statistical/norm, softmax, normalization, pooling, loss, attention, ordering,
  indexing/scatter, window/structured-linear, stochastic, and batch-normalization families when
  each reaches the progressive-planning frontier. Do not create those detailed specifications
  now.
- General model semantic gaps, including floating comparisons and incomplete softmax/power/unary
  edge contracts, must be resolved by their model owner before dependent differentiation rules.
- Compiler 0005 remains the next ordered compiler row after successful 0004B and the capability
  checkpoint. It owns publication, planning orchestration, diagnostics, and immutable compile
  artifacts.
- Compiler 0006 remains Draft and owns public functional requests, explicit seeds, create-graph/
  derivative order, and higher-order representation.

## Architecture impact

Expected impact: None.

This task fills the accepted compiler-owned local differentiation extension point while preserving
one model algebra for the combined graph. It changes no architecture rule, dependency direction,
public lifecycle, graph phase vocabulary, model semantics, or backend/runtime responsibility. If
implementation requires any such change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation context:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused compiler/autograd/training/lifecycle/dependency
architecture documents, ADR 0009, documentation rules and applicable profiles, the planning guide
and roadmap, the compiler master plan, completed Compiler 0001–0004A, Model 0025 and the model
audits, current Compile/Tensor/Training APIs, autograd strategy/user guide/glossary, this task
specification, and the complete current compiler/model source and tests needed to verify the
selected and deferred operation matrix.

Implement Compiler 0004B exactly as specified. Preserve the one shared forward/backward Tensor
algebra and the existing package-private request, one-capture, identity, accumulation, phase,
validation, and exact optimization contracts. Add only the closed mixed-floating cotangent
normalization, division, direct-zero local-convention, and ordinary/masked mean matrix. Do not add
a public surface, registry, second algebra, gradient-only numerical or optimization policy, model
API, floating-comparison-dependent rule, later operation family, runtime/backend behavior, build
change, or seventeenth path. Extend `FirstOrderAutograd` only enough to route the selected ordinary
and masked MEAN occurrences to `ReductionGradientRules`. Stop on an architecture, shared-semantic,
formula-expressibility, or maximum-scope conflict.

After executable code stabilizes, run the compiler module tests once and hand the exact diff and
evidence to a distinct clean-context documentation-focused agent. That pass must independently
finalize affected Javadocs, Compile API, autograd strategy, user guide, glossary impact, planning
status/evidence, links, and no-change conclusions in the same overall change. It must reuse the
successful Java evidence unless executable behavior changes or a concrete rerun risk is recorded.
Run final Javadoc/documentation/scope checks, then run the recorded compiler
transformation/autograd capability checkpoint before marking the task Complete.

Update this task with local decisions, limitations, exact validation evidence, implementation
notes, completion summary, and final status. Do not create a later task specification, commit, or
push.
```

## Local decisions

- The combined graph has one shared model algebra; Compiler 0004B adds local differentiation
  formulas, explicit local conventions, and cotangent normalization only.
- Mixed cotangents use ordinary `sumToShape` followed by ordinary `cast`; the operations inherit
  the same semantics and optimization guards as forward occurrences.
- Binary/scalar DIV uses its ordinary public expression formula and receives no separate gradient
  singularity or exceptional-value policy.
- FLOOR, CEIL, and SIGN use the genuine local first-order convention of a direct exact-zero
  cotangent.
- Ordinary MEAN derives its denominator by reducing generated logical ones. Dynamic and expression
  selected extents are representable, so no static-count gate or DataType-specific exact-count
  threshold remains.
- Masked MEAN uses an ordinary final `WHERE` plus an explicit all-false-slice zero-cotangent local
  convention and makes no branch-evaluation claim.
- `FirstOrderAutograd.java` is the necessary fifth production path because its current closed
  dispatch sends only SUM and CUM_SUM to `ReductionGradientRules`; the authorized change adds only
  ordinary/masked MEAN routing and leaves formula ownership in `ReductionGradientRules`.
- A clean implementation attempt stopped before edits after verifying that the original
  four-production-file scope could not make the selected MEAN rules reachable. Under the user's
  standing automatic approval for necessary higher path counts, the exact production count is
  therefore five and the exact total ceiling is sixteen.
- Floating-comparison-dependent formulas remain blocked by the general model comparison contract;
  independently cohesive families remain deferred rather than widening 0004B.

## Known limitations

- The request remains one scalar objective with one implicit unit seed and connected targets.
- Cotangent conversion is expressed by current public CAST and therefore promises no conversion
  behavior beyond the shared model contract.
- `SUPPORTED_0004B` is not a complete operation inventory for autograd; every classified later or
  blocked family remains fail-closed.
- The task implements only first-order formulas. Compiler 0006 still owns derivative order and
  higher-order lifecycle.

## Validation evidence

Planning evidence:

- required architecture, ADR, documentation profiles, planning contracts, completed compiler
  tasks, model closure/audit artifacts, APIs/guides/glossary, compiler production/tests, and model
  operation/expression semantics were audited;
- the current logical-splat, `expand`, reduction, and symbolic Shape contracts can express
  ordinary and masked MEAN denominators for static, dynamic, and expression extents;
- no architecture or dependency conflict was found;
- every selected formula uses the shared public Tensor algebra and every selected contribution
  has an ordinary Shape/DataType normalization path;
- general model-semantics gaps and later cohesive families are classified explicitly;
- a clean implementation attempt made no edits and stopped after verifying from
  `FirstOrderAutograd.apply()` that ordinary/masked MEAN would otherwise fall through the closed
  dispatch instead of reaching `ReductionGradientRules`;
- the user's standing automatic approval for necessary higher path counts adds only
  `FirstOrderAutograd.java`, fixing the implementation ceiling at 16 paths: five compiler
  production files, four compiler test files, and seven documentation/planning files; and
- Java tests, Javadoc, and the capability checkpoint were not run during the planning-only pass;
  their final implementation/documentation outcomes are recorded below.

Implementation evidence:

- implementation context
  `/root/implement_compiler_0004b_shared_algebra_retry` changed exactly the five authorized
  compiler production paths and four authorized compiler test paths;
- its final `./gradlew :modules:compiler:test` run passed with `BUILD SUCCESSFUL in 1s`,
  13 actionable tasks (`1 executed`, `12 up-to-date`), and 18 XML suites containing 136 tests,
  0 skipped, 0 failures, and 0 errors;
- its focused development run of `AutogradPreflightTest`, `GradientRulesTest`,
  `FirstOrderAutogradTest`, and `GraphCompilerTest` also passed after an earlier failure exposed
  and the implementation fixed unreachable generated-seed ingress for a direct-zero target
  gradient; and
- no executable Java or test changed after that final module evidence. The documentation pass
  therefore reused it rather than repeating the successful module suite.

Documentation evidence:

- clean documentation-focused context
  `/root/implement_compiler_0004b_shared_algebra_retry/compiler_0004b_docs` independently read
  the architecture and ADR contracts, General/API-Javadoc/Planning/User-guide/Example profiles,
  planning guide, task/master/roadmap, actual five-production/four-test diff, Compile/Tensor/
  Training APIs, autograd strategy/user guide/glossary, and directly relevant model/compiler
  contracts;
- it finalized affected Javadocs plus exactly `docs/api/compile-api.md`,
  `docs/design/notes/autograd-strategy.md`, `docs/user-guide/autograd.md`,
  `docs/glossary.md`, this task, the compiler master plan, and the roadmap without changing
  executable Java or tests;
- `./gradlew :modules:compiler:javadoc` passed after final Javadoc edits with
  `BUILD SUCCESSFUL in 1s`, 7 actionable tasks (`2 executed`, `5 up-to-date`), and no warning;
- `node /tmp/validate_synaptik_markdown.js` passed repository-wide: 241 Markdown files,
  4,429 local links, 273 anchors, and 2,966 fence markers;
- `git diff --check` passed;
- the sorted union of tracked modifications and untracked files contains exactly 16 paths:
  five compiler production files, four compiler tests, and seven documentation/planning files;
- manual review confirmed balanced fences, valid local links/anchors, final newlines, no trailing
  whitespace, shared-algebra wording, exact selected/deferred classification, and no unexpected
  public, architecture, dependency, build, backend, runtime, or execution claim;
- Tensor API remained unchanged because 0004B adds no public Tensor method or model semantic;
  Training API remained unchanged because it adds no public gradient/training request,
  publication, optimizer, session, prepare, run, or lifecycle behavior;
- `ARCHITECTURE.md`, focused architecture pages, and ADR 0009 remained unchanged because the
  implementation fills the accepted compiler-owned pre-capture extension point without changing
  ownership, dependency direction, phase/capture structure, or the one-algebra contract;
- architecture tests, model/config/planning/trace/runtime/prepare/engine/training/backend sources
  and tests, backend conformance, integration tests, and Gradle/build files remained unchanged
  because there is no boundary, public model, cross-module, execution, or build behavior change;
  and
- after the finalized documentation handoff, implementation context
  `/root/implement_compiler_0004b_shared_algebra_retry` ran
  `./gradlew test :testing:architecture-tests:test`; it passed with `BUILD SUCCESSFUL in 2s`,
  48 actionable tasks (`6 executed`, `42 up-to-date`), and a stored configuration cache;
- the checkpoint XML audit found 167 suites and 1,275 tests with 0 skipped, 0 failures, and
  0 errors: backend-contract 4 suites/22 tests, config 4/17, model 127/1,018, planning 8/63,
  compiler 18/136, trace 3/16, and architecture-tests 3/3; and
- no executable Java changed before or after that checkpoint. Task, master-plan, and roadmap
  status are synchronized to Complete; Compiler 0005 and 0006 remain Draft, and neither has a
  detailed task specification.

## Implementation notes

- `AutogradPreflight` now accepts floating targets independently of objective type and validates
  only the exact selected 0004B binary, scalar DIV, WHERE, floating CAST, direct-zero unary,
  ordinary/masked MEAN, and mixed-floating MATMUL rows before derivative allocation.
- `ElementwiseGradientRules` normalizes selected mixed contributions with ordinary
  `sumToShape` followed by ordinary `cast` when required, constructs the exact binary/scalar DIV
  formulas, and returns direct exact zeros for FLOOR/CEIL/SIGN.
- `LinearAlgebraGradientRules` retains all four MATMUL rank formulas, reverses batch broadcasting,
  and casts once to the selected operand type when promotion requires it.
- `ReductionGradientRules` derives ordinary MEAN counts by reducing logical ones and masked MEAN
  true counts with ordinary WHERE/SUM, then uses the final ordinary WHERE for the selected
  all-false-slice zero convention.
- `FirstOrderAutograd` routes preflight-approved ordinary and masked MEAN occurrences to the
  reduction owner. It also retains only generated constant bindings reachable from returned
  gradient expressions, so direct-zero target gradients do not expose an unreachable seed input.
- Focused tests extend preflight, formula structure, mixed-floating normalization, direct-zero,
  static/dynamic/expression/empty MEAN, dispatch, and both optimization-mode coverage while
  preserving existing 0004/0004A behavior.
- The independent documentation pass finalized all five affected implementation contracts,
  synchronized the four explanatory/public-status surfaces, added the reusable cotangent-
  normalization glossary distinction, and recorded reasoned no-change conclusions for every
  reviewed out-of-scope surface.

## Completion summary

- Completed changes: added the closed 0004B mixed-floating cotangent-normalization, binary/scalar
  DIV, direct-zero FLOOR/CEIL/SIGN, ordinary/masked MEAN, and closed MEAN-dispatch behavior while
  preserving one shared Tensor algebra, fail-closed preflight, deterministic accumulation, one
  combined capture, and the existing validation/optimization contracts.
- Files changed or created: exactly five compiler production files, four compiler test files, and
  seven documentation/planning files.
- Tests and validation: compiler module tests passed with 18 suites/136 tests; the required
  repository/architecture checkpoint passed with 167 suites/1,275 tests and no skipped tests,
  failures, or errors; compiler Javadoc, repository Markdown links/anchors/fences, exact scope,
  statuses, final newlines, whitespace, and `git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_compiler_0004b_shared_algebra_retry/compiler_0004b_docs` independently
  finalized all affected implementation Javadocs, Compile API, autograd strategy, user guide,
  glossary, task evidence, compiler master plan, and roadmap.
- Documentation impact: current status and the exact selected/deferred matrix now consistently
  describe one shared forward/backward algebra and the bounded 0004B local-rule extension.
- Javadoc review: all five affected implementation contracts were reviewed; four required final
  documentation edits, while `ElementwiseGradientRules` already contained the complete 0004B
  normalization/formula contract drafted with implementation.
- Glossary impact: added the reusable cotangent-normalization distinction and synchronized
  autograd, cast, logical-splat, masked-reduction, MATMUL, and implementation-status entries.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005 and 0006 remain their ordered Draft rows without
  detailed specifications.

Status: Complete
