# Task 0005D: Attention, Convolution, Pooling, and Loss Gradient Completion

## Status

Complete.

## Goal

Complete the remaining structured machine-learning part of the current first-order gradient
inventory inside `modules/compiler`. Preserve and verify the implemented `MATMUL` rules and public
`linear(...)` composition, then add fail-closed compiler-owned reverse formulas for scaled
dot-product attention, grouped NCHW convolution, NCHW max/average pooling, mean-squared error, and
dense- and index-target categorical cross-entropy with logits.

Every derivative remains an ordinary public `Tensor` expression. Use canonical same-occurrence
attention weights where they exist, current public window transforms for convolution/pooling,
retained forward obligations for symbolic formulas, and exact Shape/data-type normalization for
every selected floating input. Unsupported signatures fail during preflight before any generated
Tensor or derivative constant is allocated.

This task adds no model operations, hidden outputs, second gradient algebra, physical saved values,
runtime tape, Tensor gradient state, backend behavior, execution, or higher-order requests.

## Scope

### Source-backed operation and role inventory

Inventory current source at implementation time and keep this matrix exact. `D` is a floating
input role with an implemented contribution, `ND` is intentionally non-differentiable, and `FC`
is structurally gradient-eligible but deliberately fail-closed because the exact rule is not
expressible from the signature's canonical outputs and current public Tensor algebra.

| Occurrence | Output slot | Ordered roles | Disposition |
|---|---:|---|---|
| `MATMUL` | 0 | left, right | floating `D`, `D`; signed-integral `ND`, `ND` |
| public `linear(weight[, bias])` | visible composition | input, weight, optional bias | verify `PERMUTE -> MATMUL -> optional ADD`; no `LINEAR` kind/rule |
| one-output attention | 0 output | query, key, value, optional BOOL mask | query/key/value `FC`; mask `ND` |
| two-output attention | 0 output | query, key, value, optional BOOL mask | query/key/value `D`; mask `ND` |
| two-output attention | 1 weights | query, key, value, optional BOOL mask | query/key `D`; value/mask `ND` |
| `CONV2D` without/with bias | 0 | input, weight, optional bias | every floating data role `D` |
| `MAX_POOL2D` | 0 | input | input `D`; attrs are configuration |
| `AVERAGE_POOL2D` | 0 | input | input `D`; attrs are configuration |
| `MEAN_SQUARED_ERROR` | 0 | prediction, target | both `D` for `NONE`, `SUM`, `MEAN` |
| dense categorical cross-entropy with logits | 0 | logits, dense target | both `D` for `NONE`, `SUM`, `MEAN` |
| index categorical cross-entropy with logits | 0 | logits, INT32/INT64 target | logits `D`, target `ND` for all reductions, subject to static class depth |

One-output attention producers literally have no canonical weights slot. Reconstructing a sibling,
re-running attention, or rebuilding masked softmax would violate canonical-output and exceptional-
value contracts. Any selected path through one-output attention therefore fails closed. The
explicit `scaledDotProductAttentionWithWeights(...)` forms are the differentiable signatures.

Index loss is representable only when preflight observes a statically known positive class extent.
`Tensor.oneHot(long)` requires that depth. Dynamic class extent, and static zero class extent
accepted by forward empty/all-ignored semantics, fail closed before construction. Do not add a
range Tensor, materialized index constant, or integral selection operation.

### Notation and shared construction

For input `x`, occurrence output `y`, and incoming cotangent `g`:

- `R_x(t)` is existing exact cotangent normalization: `sumToShape(x.shape())` where required,
  followed by cast to `x.dataType()` where required.
- `Z_x`/`O_x` are request-local exact typed positive-zero/positive-one logical splats.
- Floating operands are cast to the occurrence output type before formula arithmetic.
- `T(t)` swaps the final two axes; `Σ_A(t, keep)` is ordinary sum over axes `A`.
- `P(a,b) = where(a == +0, +0, a*b)` is the selection-safe product used where exact-zero `a`
  semantically excludes `b` before arithmetic.
- A logical count is formed by expanding one exact typed one over the exact domain Shape and
  summing it, never by host `long` multiplication/conversion.
- Reshapes/permutations use ordinary public operations and exact symbolic `Dimension` expressions
  selected or inferred from the forward occurrence.

Keep helpers family-owned and package-private. Add no registry, derivative DSL, broad builder,
generic manager/service/processor, utility package, or public compiler surface.

### Existing `MATMUL` and public `linear(...)`

Preserve the exact task-0004A matrix:

| Left rank | Right rank | Left contribution before normalization | Right contribution before normalization |
|---|---|---|---|
| 1 | 1 | `g * right` | `g * left` |
| 1 | at least 2 | squeeze temporary row of `expandDims(g) @ T(right)` | `expandDims(left,1) @ expandDims(g,-2)` |
| at least 2 | 1 | `expandDims(g,-1) @ expandDims(right,0)` | `T(left) @ g` |
| at least 2 | at least 2 | `g @ T(right)` | `T(left) @ g` |

Verify vector/vector, vector/matrix, matrix/vector, batched/broadcast forms, mixed floating types,
exact restoration, and signed-integral rejection.

Verify `input.linear(weight, bias)` through its actual weight transpose, `MATMUL`, and `ADD`, into
input, weight, and bias. Do not add a `LINEAR` row, kind, facade, or special rule.

### Scaled dot-product attention

For a two-output occurrence:

```text
Q=query; K=key; V=value
W=producer.output(1)                 [...batch,L,S]
O=producer.output(0)                 [...batch,L,Ev]
```

For absent scale, expand one typed logical one over the exact query embedding `E`, sum to its
scalar logical count, and apply `rsqrt`, yielding `1/sqrt(E)` without reading a binding. A present
scale retains its exact typed `ScalarValue`.

Define:

```text
softmaxPullback(W,U)
  = P(W, U - sum(P(W,U), axis=-1, keepDimensions=true))
```

For output slot zero with cotangent `gO`:

```text
U  = gO @ T(V)
dS = softmaxPullback(W,U)
dV = R_V(selectionSafeContractOverL(W,gO))
dQ = R_Q(scale * selectionSafeContractOverS(dS,K))
dK = R_K(scale * selectionSafeContractOverL(dS,Q))
```

For weights slot one with cotangent `gW`:

```text
dS = softmaxPullback(W,gW)
dQ = R_Q(scale * selectionSafeContractOverS(dS,K))
dK = R_K(scale * selectionSafeContractOverL(dS,Q))
```

Selection-safe contractions are ordinary expand/`P`/sum/reshape compositions equivalent to the
named matrix contractions. Do not use raw `0*NaN` products there. `U` may be ordinary `MATMUL`;
excluded entries are removed by the subsequent saved-weight pullback.

If both outputs contribute, traverse canonical wrappers in stable output-slot order and use
existing identity accumulation. Slot zero alone contributes to value. Policy:

- explicit/causal masks are `ND`; exact saved `W` alone records eligibility;
- excluded, all-masked, and all-negative-infinity rows contribute positive zero;
- positive-infinity ties use exact saved split weights and the saved-output Jacobian;
- eligible NaN weights follow raw saved-output arithmetic and propagate NaN;
- exact-zero saved weight selects zero even when its score was eligible negative infinity;
- finite rounding/reassociation remains that of emitted public operations.

### Grouped NCHW convolution

For input `X:[N,Cin,H,W]`, weight `F:[Cout,CinPerGroup,kH,kW]`, cotangent
`Gout:[N,Cout,Hout,Wout]`, positive static `groups`, and exact convolution-matching
`Window2dAttrs` with `ceilMode=false`:

```text
Xcol = unfold2d(X,window)                       [N,Cin*kH*kW,L]
Og   = Cout/groups
Kg   = CinPerGroup*kH*kW
Xg   = reshape/permute(Xcol)                    [groups,N,Kg,L]
Fg   = reshape(F)                               [groups,Og,Kg]
Gg   = reshape/permute(Gout)                    [groups,N,Og,L]

dXg  = expandDims(T(Fg),batch=N) @ Gg           [groups,N,Kg,L]
dX   = R_X(fold2d(inverseReshapePermute(dXg),X.shape(),window))

dFg  = sum(Gg @ T(Xg),axis=N)                   [groups,Og,Kg]
dF   = R_F(reshape(dFg,F.shape()))

dBias = R_bias(sum(Gout,axes=[N,Hout,Wout],keepDimensions=false))
```

Equivalent public reshape/permute/expand/`MATMUL` ordering is allowed, but group isolation is
mandatory. Do not create a dense cross-group matrix, backward kind, element enumeration, or
physical columns.

Inference/preflight retain/prove positive static kernels, input/output channel divisibility,
`Cin == groups*CinPerGroup`, bias/output-channel equality, NCHW rank/type, effective-kernel fit,
and exact output geometry. Dynamic/expression channel/spatial dimensions remain supported when
typed constraints prove these relations; do not impose new static requirements.

Empty domains and exceptional values follow ordinary unfold/`MATMUL`/sum/fold semantics. Select no
extra NaN/infinity/signed-zero repair.

### Average pooling

With matching `Window2dAttrs`, flatten `g` to `[N,C,L]` and form divisor `D` by expanding logical
ones over `[kH,kW]` and summing:

```text
perPosition = restored(g)/D
columns     = expand/reshape(perPosition)        [N,C*kH*kW,L]
dInput      = R_input(fold2d(columns,input.shape(),window))
```

This is fixed count-padding: every logical position receives `g/D`, while fold contributes only
in-bounds positions. Dilation and literal ceil mode use the exact forward grid. All-padding
windows contribute to no input. Never materialize host `kH*kW`, use valid-sample counting, or add
divisor configuration. Exceptional/empty/overlap behavior is ordinary division/fold behavior.

### Max pooling

`MAX_POOL2D` has no indices output, but its exact first logical winner is reconstructable without
re-running max pooling:

1. Unfold input through matching window with exact typed negative-infinity padding; reshape to
   `[N,C,L,K]`, `K=kH*kW`.
2. Unfold `O_input` through the same window with positive-zero padding. Nonzero entries mark
   in-bounds samples, distinguishing real negative infinity from padding.
3. Align the exact same-occurrence output to `[N,C,L,1]`.
4. Candidate means in-bounds and either both values are NaN or numerically equal with signed-zero
   agreement. Since equality equates signed zeros, compare `O/candidate` with `O/output` in the
   both-zero case; the reciprocals are signed infinities.
5. Convert candidates to zero/one, use `argMax(-1,false,FIRST_INDEX)`, one-hot its static positive
   `K`, and intersect with candidates. The intersection makes all-padding select nothing.
6. Route aligned `g` with `where(selected,g,+0)`, reverse the column layout, and fold to input.

Policy:

- equal finite values, infinities, signed zeros, and multiple NaNs route only to the first
  increasing kernel-height/kernel-width in-bounds sample;
- positive zero wins over negative zero;
- selected NaN routes the incoming cotangent to that exact first NaN coordinate;
- padding never wins, including a tie with real negative infinity;
- all-padding and zero batch/channel domains contribute zero;
- overlapping winners add through `fold2d`;
- no hidden index, unpool kind, storage, or physical saved state is created.

### Loss cotangent restoration

For `NONE` result domain Shape `S`:

```text
restore(NONE,g,S) = g
restore(SUM,g,S)  = expand(g,S)
restore(MEAN,g,S) = expand(g,S)/logicalCount(S)
```

Categorical losses insert a singleton at the normalized class axis afterward. MSE uses the full
prediction Shape. Empty domain yields an empty restored Tensor even though scalar forward mean is
NaN. Never use a static host element count.

### Mean-squared error

After output-type casts:

```text
D  = prediction-target
dP = R_prediction( 2*D*restore(reduction,g,prediction.shape()))
dT = R_target(     -2*D*restore(reduction,g,prediction.shape()))
```

Two and negative two are exact in result type. Preserve this unexpanded difference formula.
NaN/infinity/signed-zero/overflow/underflow and empty behavior follows it for all reductions,
mixed floating types, and dynamic Shapes.

### Dense-target categorical cross-entropy

Let class axis `c`, `P=softmax(logits,c)`, `LP=logSoftmax(logits,c)`, target `T` cast to result
type, restored cotangent `R` with singleton class dimension, and `Tsum=sum(T,c,true)`:

```text
dLogits = R_logits((P*Tsum-T)*R)

rawTarget = (-LP)*R
dTarget = R_target(where(
    (T == +0) AND NOT isFinite(LP),
    +0,
    rawTarget))
```

Retain `Tsum`; do not silently normalize targets. Existing finite/non-negative/normalized target
requirements remain caller/execution obligations, and violating inputs gain no gradient
guarantee. The target rule selects zero only at the forward zero-weight/non-finite-log-probability
discontinuity; zero target with finite `LP` retains ordinary `-LP`. Other exceptional logits
follow current softmax/log-softmax and raw arithmetic.

### Index-target categorical cross-entropy

INT32/INT64 target is always `ND`. For static positive depth `C`, let `upper` be `C-1`
when representable in the target type and that type's maximum value otherwise:

```text
safeTarget = minimum(
    maximum(target,exactIntegral(0)),
    exactIntegral(upper))
hotBool    = oneHot(safeTarget,C)
hot        = where(hotBool,O_logits,Z_logits)
```

No-ignore:

```text
dLogits = R_logits((softmax(logits,c)-hot)*restoredScale)
```

Present exact typed `ignoreIndex`:

```text
delta       = target-ignoreIndex
ignored     = equalTo(delta,delta-delta)
activeFloat = where(ignored,Z_sample,O_sample)
activeCount = sum(activeFloat)

restoredScale(NONE/SUM) = ordinary restored cotangent
restoredScale(MEAN)     = ordinary restored cotangent/activeCount

dLogits = R_logits(where(
    expandDims(ignored,c),
    Z_logits,
    (softmax(logits,c)-hot)*restoredScale))
```

Integral subtraction is exact modulo target width, so `delta == delta-delta` exactly detects the
configured ignore value without an integral Tensor constant. Scalar `maximum`/`minimum` before
one-hot makes any ignored value safe while staying in the exact integral category; it is identity
for obligation-satisfying non-ignored targets. Out-of-range non-ignored targets already have no
forward execution meaning.

Final `where` excludes ignored logits before arithmetic, so ignored rows remain zero with
NaN/infinite logits and all-ignored `MEAN`. `MEAN` divides by non-ignored count; `NONE`/`SUM` do
not. Dynamic/zero class depth fails closed.

### Inference and preflight

Review `StructuredOperationInference`; extend only for a missing typed constraint. It must retain:

- attention embedding positivity/equality, key/value sequence equality, three-way batch and mask
  broadcast, exact output count/descriptors, and scale type;
- convolution channel divisibility/equality, bias length, effective-kernel fit, and NCHW geometry;
- pooling effective-kernel/grid obligations for floor and literal ceil modes;
- MSE positional Shape equality;
- dense/index loss target Shape and positive-class-or-empty/all-ignored forward conditions.

Preflight inspects canonical producers/attrs, normalizes axes once, selects requested roles, and
proves formula prerequisites before construction:

- exactly two canonical attention wrappers with exact descriptors;
- convolution group reshapes preserve exact symbolic element counts;
- pool window Shapes match attrs and output descriptor;
- reduction restoration matches exact `NONE` domain;
- index class extent is static positive before `oneHot`.

One-output attention, dynamic/zero-depth index loss, malformed manual operations, unexpected
attrs/counts/slots, unknown structured kinds, and unproved obligations fail with deterministic
occurrence/role detail.

### Failure order, determinism, producers, and constants

Preserve request-wide order:

1. validate request/objective/targets and complete preflight;
2. reject every unsupported signature/slot/role/constraint;
3. only then allocate seed, splats, and derivative expressions;
4. reverse-traverse exact original producers/canonical wrappers;
5. accumulate by Tensor identity in deterministic encounter order;
6. capture combined forward/backward graph once with exact phase provenance.

Failure consumes no Tensor IDs and leaves no partial state. Generated floating constants use the
request-local exact-bit cache in first-use order: positive zero, positive one, and other exact
typed splats as reached. Exact two and negative two remain scalar-operation coefficients; pooling
padding uses exact positive-zero or negative-infinity scalar attributes where required. Integral
zero/`upper` are same-type `ScalarValue` attrs, not Tensor leaves. Reuse exact type/bit matches.

Attention weights and max-pool outputs are exact original Tensor identities. Max winner-selection
expressions are new backward-phase public occurrences, not physical saved state.

### Transitive emitted-operation boundary

Formulas may emit only current public operations already covered by Compiler 0004–0005C or this
task: cast; scalar/binary arithmetic; comparison/logical/classification/`where`; `rsqrt`;
softmax/log-softmax; sum/arg-max; Shape normalization; expand/reshape/permute/squeeze/unsqueeze;
`MATMUL`; `oneHot`; `unfold2d`; and `fold2d`.

Lock the exact emitted-kind inventory in tests. Task 0005E owns the source-wide transitive audit
before higher-order requests.

## Out of scope

- changes under `modules/model` or any public Tensor/operation/result-carrier signature
- new hidden outputs, range/integral-selection operations, or backward kinds
- differentiable one-output attention or dynamic/zero-depth index loss
- derivatives for masks, integral targets/indices, axes, attrs, reductions, scales, groups, or
  window/configuration values
- attention dropout, transposed convolution, broader bias rules, pool indices/unpool, configurable
  average divisor, extra losses, label smoothing, class weights, or sparse conversions
- execution, numerical evaluation, backend/conformance, kernels, runtime, prepare, engine,
  training, optimizers, storage, buffers, or tape state
- public compile expansion, explicit seeds/objectives/targets, disconnected policy, `createGraph`,
  higher order, or task 0005E's full closure checkpoint
- architecture/dependency/module/Gradle changes or unrelated refactors

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Model 0023 adjoint audit](../../model/tasks/0023-adjoint-expressibility-audit.md)
- [Model 0023D window transforms](../../model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
- [Model 0023F attention weights](../../model/tasks/0023f-scaled-dot-product-attention-weights-output.md)
- [Model 0025 canonical outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative.
- Preflight, formulas, accumulation, phase assignment, and combined capture remain compiler-owned.
- Forward/backward expressions share one Tensor algebra, inference, validation, numerical, and
  optimization contract.
- Only canonical original-producer Tensors are same-occurrence auxiliaries.
- Identity maps are request-local implementation state, not graph/artifact representation.
- No runtime/prepare/engine/training/backend dependency, Tensor gradient state, mutable gradient
  storage, second algebra, tape, physical buffer, schedule, or execution unit is added.
- Architectural uncertainty stops implementation for clarification.

## Package impact

The cohesive `io.github.pho001.synaptik.compiler` package gains four package-private final family
owners:

- `AttentionGradientRules`
- `ConvolutionGradientRules`
- `PoolingGradientRules`
- `LossGradientRules`

`AutogradPreflight` owns role/prerequisite selection; `FirstOrderAutograd` owns dispatch and
request-local constants; `StructuredOperationInference` changes only if source review finds a
missing forward constraint. Add no subpackage, public type, registry, or generic facade.

## Affected files

Production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AttentionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ConvolutionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PoolingGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LossGradientRules.java`

Tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AttentionGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionAndPoolingGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LossGradientRulesTest.java`

Documentation/planning:

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless contradiction requires stopping: architecture/ADR files; Training API;
model capabilities/master/tasks and current structured/window/canonical-output source/tests;
`LinearAlgebraGradientRules`; completed Compiler 0004A/0004B/0005B/0005C; architecture,
backend-conformance, integration, Gradle, and other modules.

## Maximum scope

At most 20 paths: seven production, seven tests, and six documentation/planning paths above.
A separate documentation-focused clean context finalizes Javadocs/docs/status in those paths
without repeating successful Java suites unless it changes Java behavior.

If a model/API operation, twenty-first path, another module, architecture/dependency/build change, or
different derivative policy is needed, stop for clarification.

## Acceptance criteria

### Inventory and preflight

- Every assigned signature, output slot, reduction, and role appears exactly once.
- Existing floating `MATMUL` and visible linear composition are revalidated; integral roles remain
  fail-closed.
- One-output attention and dynamic/zero-depth index loss fail before allocation.
- Two-output attention uses exact canonical weights; masks/indices/configuration remain `ND`.
- Malformed/unknown/unproved occurrences fail with deterministic occurrence/slot/role context.

### Formulas

- Attention covers both slots, scale forms, mixed types, broadcasting, masks, all-masked/all
  negative-infinity, positive-infinity ties, NaNs, and stable accumulation as specified.
- Grouped/ungrouped convolution restores exact symbolic group/window geometry and input types.
- Average pool uses count-padding fold; max pool reconstructs first in-bounds winner including
  NaN/infinity/signed-zero/real-negative-infinity/padding/overlap/all-padding cases.
- MSE and dense/index loss cover every reduction, mixed type, dynamic/empty domain,
  ignore/all-ignored, and stated exceptional policy.
- Tests inspect exact kinds, attrs, ordered provenance, canonical identities/slots, constants, and
  absence of storage/hidden backward kinds—not only names/counts.

### Pipeline and boundaries

- One deterministic combined capture, exact phases/publication, and logical-splat ingress remain.
- Failed preflight consumes no ID and creates no partial formula.
- Emitted kinds stay inside the stated shared-algebra boundary.
- No model/runtime/prepare/engine/training/backend/architecture/dependency/Gradle/public compile
  behavior changes.
- Java contracts/Javadocs and affected docs are finalized in an independent documentation context;
  reviewed no-change conclusions are recorded.

## Tests / validation

Focused:

```text
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest \
  --tests io.github.pho001.synaptik.compiler.AttentionGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.ConvolutionAndPoolingGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.LossGradientRulesTest
```

Required once:

```text
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
git diff --check
```

Also record source-signature/role inventory; emitted kind/attrs/provenance and canonical-output
inspection; ID-delta failures; combined phase/publication/constant/determinism checks; scans proving
no `LINEAR`, hidden/backward/tape/registry/public API; exact 20-path ceiling inventory; Markdown
links/headings/fences/whitespace/status; and General/API-Javadoc/Planning/Example documentation
review.

Repository-wide validation is deferred to Compiler 0005E's first-order checkpoint and CI. No
architecture/backend-conformance/integration test is required because no such behavior changes.

## Dependencies

- Models 0023, 0023D, 0023F, and 0025 — Complete.
- Compilers 0004A, 0004B, 0005B, and 0005C — Complete.
- Stable current structured operations, window transforms, canonical outputs, inference,
  preflight, one-capture, and publication contracts.

## Follow-up tasks

- Compiler 0005E performs source-wide first-order/fail-closed and transitive-formula audit, records
  intentional limitations, and runs the capability checkpoint.
- Compiler 0006 may add explicit functional gradient requests only after 0005E.
- A future focused model/compiler task may reconsider one-output attention or dynamic/zero-depth
  index loss only if a generally useful public capability is independently justified.

## Architecture impact

None intended. This implements compiler-owned reverse formulas through the existing Tensor algebra
and one combined capture under ADR 0009. Architecture documents/tests, dependencies, and build
remain unchanged.

## Implementation prompt

Work in a clean implementation context. Read root `AGENTS.md`, authoritative/focused architecture
and ADR 0009, documentation rules/profiles, planning guide/roadmap, compiler/model master plans and
capabilities, this task, Model 0023/0023D/0023F/0025, Compiler 0004A/0004B/0005B/0005C, and every
affected/review-only source/test/document above.

Implement only this exact matrix. Preserve `MATMUL`; verify visible linear; add four narrow
package-private rule owners; extend inference/preflight only as required; keep derivatives public
Tensor expressions. Use canonical two-output attention weights, grouped unfold/`MATMUL`/fold,
count-padding average fold, exact max winner reconstruction, and exact loss policies.

Fail before allocation for one-output attention, dynamic/zero-depth index loss, malformed
occurrences, unexpected slots/attrs/counts, and unproved constraints. Never invent a hidden output,
recompute attention, add backward kinds, read storage/bindings, materialize window/index data, add
Tensor gradient state, or touch another module.

Run focused tests, the full compiler tests once, Javadoc, scope/status/link/manual checks, and
`git diff --check`. Then use a separate clean documentation context to finalize Javadocs, Compile
API, glossary, task evidence/summary, master plan, and roadmap without repeating Java tests unless
behavior changes. Return both context IDs, exact paths/evidence/no-change conclusions/issues and:

```text
Status: Complete
```

only when every criterion passes; otherwise use the repository's exact incomplete format.

## Local decisions

- Exact same-occurrence weights take precedence over one-output attention coverage.
- Attention uses saved-output Jacobian, zero all-masked/all-negative-infinity policy, exact saved
  positive-infinity split, and selection-safe contractions.
- Convolution uses symbolic grouped window composition; average pool uses count-padding.
- Max pool routes first logical in-bounds winner, distinguishes signed zeros, and routes selected
  NaN.
- MSE remains unexpanded and applies exact `2` or `-2` before the restored cotangent so target
  signed-zero/NaN behavior is not changed by a final negation; dense target is zero only at
  zero-weight/non-finite-log-probability; index loss sanitizes before one-hot and excludes ignored
  rows after scale.
- No Model prerequisite is needed for the representable matrix.

## Known limitations

- One-output attention intentionally fails because its producer has no canonical weights. Gradient
  callers must construct `WithWeights`, even if weights are not a final published output.
- Index loss requires statically positive class depth; dynamic and zero-class empty/all-ignored
  forward cases fail closed.
- Current scalar-objective, implicit-unit-seed, requested-target first-order contract remains;
  task 0006 owns broader requests.
- Formulas are symbolic; execution/conformance remain later lifecycle work.

## Validation evidence

Planning evidence:

- Read repository instructions, architecture/focused docs/ADR, documentation profiles, planning
  guide/roadmap, compiler/model master/capabilities, completed tasks through Compiler 0005C, model
  adjoint/prerequisite tasks, and current structured source/tests.
- Confirmed one- and two-output attention, grouped convolution, separate pools, every loss/reduction,
  public dynamic window/fold operations, and canonical producer wrappers.
- Confirmed static-depth ignored index loss is expressible by clamp-before-one-hot and exact
  integral masking. Only one-output attention and dynamic/zero class depth remain fail-closed.
- Confirmed a clean repository and no prior detailed 0005D file before planning.

Implementation validation:

- Implementation context: `/root/implement_compiler_0005d`.
- The required focused seven-suite command passed after the final executable correction.
- The replacement `./gradlew :modules:compiler:test` run passed after correcting the MSE target
  coefficient order: 28 XML suites, 189 tests, zero skipped tests, zero failures, and zero errors.
- The exact BFLOAT16/FLOAT32/FLOAT64 negative-two bit patterns and MSE expression order are covered
  directly. No executable Java or test changed after the replacement full-module evidence.
- Final source/test inspection confirmed the exact role/output matrix, canonical attention/max-
  pool identities, grouped convolution geometry, pooling policies, loss reductions, stable
  accumulation, preflight-before-allocation failures, combined phase/publication behavior, and
  emitted ordinary-operation boundary.

Documentation validation:

- Documentation context: `/root/implement_compiler_0005d/compiler_0005d_docs`.
- Applied the General, API/Javadoc, Planning, and Example profiles; no new standalone example was
  necessary because the existing Tensor API examples remain accurate and the formula tables are
  the task-appropriate evidence.
- Finalized Javadocs for both changed owners and all four new family owners; synchronized Compile
  API, Tensor API, glossary, task, compiler master plan, and roadmap.
- Compiler Javadoc generation, Markdown links/headings/anchors/fences/final-newline/trailing-
  whitespace checks, exact 20-path-ceiling inventory, lifecycle-status checks, forbidden-surface
  scans, and `git diff --check` passed after the final documentation edits.

## Implementation notes

- Added package-private `AttentionGradientRules`, `ConvolutionGradientRules`,
  `PoolingGradientRules`, and `LossGradientRules`; extended `AutogradPreflight` and
  `FirstOrderAutograd` without changing public compiler APIs.
- Added three new suites covering the four family owners and extended
  `StructuredOperationInferenceTest`, `AutogradPreflightTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest`.
- `StructuredOperationInference` was reviewed and remained unchanged because its existing typed
  constraints already prove every required formula prerequisite. `LinearAlgebraGradientRules`
  was reviewed unchanged; its MATMUL matrix and the public PERMUTE/MATMUL/ADD linear composition
  remain sufficient.
- Tensor API required a narrowly authorized sixth documentation path because its attention,
  convolution, pooling, and loss sections explicitly described the new compiler gradients as
  future work. The task ceiling therefore increased from 19 to 20 paths; the final change uses
  19.
- Training API remained accurate because 0005D adds no public gradient publication, optimizer,
  prepared execution, or training-session contract. Model capabilities/master/tasks,
  architecture/ADR/tests, backend conformance, integration, Gradle, and other modules required no
  change.

## Completion summary

- Completed the exact representable structured-neural first-order matrix and preserved MATMUL/
  linear behavior, one combined capture, compiler ownership, and exact cotangent normalization.
- Added or changed 19 paths: six production files, seven test files, and six documentation/
  planning files, within the authorized 20-path ceiling.
- Focused and full compiler validation passed with 189 final module tests; Javadoc, Markdown,
  scope, status, surface, and whitespace validation also passed.
- Unresolved implementation issues: none. Intentional limitations remain one-output attention and
  index loss with dynamic or zero class depth; Compiler 0005E owns the closure checkpoint.
- Required follow-up: Compiler 0005E, which remains Draft without a detailed specification.

Status: Complete
