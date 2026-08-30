# Task 0025J: First-class NCDHW max/average Pool3d semantics

## Status

Complete

## Goal

Add first-class, rank-specific maximum and fixed-count average pooling for floating NCDHW
Tensors. Each successful receiver call creates one fresh one-input, one-output semantic occurrence:

```text
[N, C, D, H, W] -> MAX_POOL3D or AVERAGE_POOL3D -> [N, C, D_out, H_out, W_out]
```

Pool3d is first-class because enumerating depth windows through current operations would make the
producer graph depend on the input depth and cannot give one bounded, Shape-independent
composition that preserves depth padding, dilation, and literal ceil windows. This task owns Model
metadata and provenance only. Compiler adoption, gradients, lowering, backend capability,
generated code, storage, Runtime behavior, and execution remain downstream work.

## Scope

- Add immutable public `MaxPool3dAttrs` and `AveragePool3dAttrs` records with depth, height, and
  width kernel, stride, symmetric padding, and dilation components plus literal `ceilMode`.
- Add one `Pool3dKind` family with exactly `MAX_POOL3D` and `AVERAGE_POOL3D`, each paired only with
  its own attributes class and an exact one-input, one-output signature.
- Add `Tensor.maxPool3d(MaxPool3dAttrs)` and
  `Tensor.averagePool3d(AveragePool3dAttrs)`.
- Add separate package-private, final, field-free `TensorMaxPool3dExpressions` and
  `TensorAveragePool3dExpressions` construction owners.
- Accept only BFLOAT16, FLOAT32, and FLOAT64 rank-five NCDHW input, retain its exact type, batch and
  channel Dimension references, and gradient-request metadata, and derive exact static or
  canonical-symbolic depth, height, and width output Dimensions.
- Preserve the selected maximum, average, exceptional-value, rounding, failure-order, canonical-
  output, and provenance contracts below.
- Finalize affected Javadocs and explanatory documentation in a distinct clean documentation-
  focused context before marking the task Complete.

## Out of scope

- `unfold3d`, `fold3d`, Pool3d adjoints, saved maximum indices, pooling-specific gradient
  primitives, backward state, or any other Model gradient behavior.
- Compiler forward adoption or descriptor validation, graph inventory changes, derivative
  allocation, lowering, Planning capability, Prepare behavior, CPU generated code, Runtime
  interpretation, execution, or performance claims.
- A public or private `PoolNd`, dynamic-rank API, geometry arrays, adaptive/global pooling,
  asymmetric intrinsic padding, valid-sample average division, divisor override, or NDHWC layout.
- Changes to Pool1d composition, Pool2d semantics, Conv3d, Shape/Dimension contracts, factory
  seams, storage, dependencies, Gradle, architecture contracts, or architecture tests.
- Detailed specifications or implementation for Model 0025K, Compiler 0006B1/0006B2, CPU
  0008G1, or Model 0026.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [NCHW Max Pool2d semantics](0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
- [NCHW Average Pool2d semantics](0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
- [NCDHW Conv3d precedent](0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)
- [NCW Pool1d composition](0025i-ncw-max-average-pool1d-composition.md)
- [Compiler Pool2d gradients](../../compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [CPU Pool2d generated execution](../../../backends/cpu/tasks/0008g-portable-max-average-pool2d-execution.md)

## Architecture constraints

- Work stays in Model and directly affected documentation/planning. Model owns immutable
  backend-independent operation semantics, Tensor descriptors, expression provenance, and
  construction-time validation; it does not capture graphs, create gradients, select algorithms,
  advertise backend support, allocate result storage, or execute.
- `Tensor` remains the fluent facade and delegates to package-private construction owners. The two
  numerical families remain separate; no broad pooling manager, helper, public facade, or shared
  mutable state is added.
- Every successful call creates one `Operation`, one `TensorProducer`, and its one canonical output
  atomically through the existing factory seam. The returned Tensor is exactly producer
  `output(0)`.
- Each kind owns an immutable stable singleton signature list. `MAX_POOL3D` accepts exactly
  `MaxPool3dAttrs`; `AVERAGE_POOL3D` accepts exactly `AveragePool3dAttrs`; each signature accepts
  exactly one input and one output.
- `requiresGrad` remains passive Tensor metadata. This task defines no derivative and makes no
  claim that Compiler can adopt, validate, differentiate, lower, prepare, or execute either kind.
- Compiler 0006B1 remains the sole forward-adoption and fail-closed-backward owner; Model 0025K and
  Compiler 0006B2 remain the ordered prerequisites for exact gradients; CPU 0008G1 remains the
  generated execution owner.
- No architecture, dependency, module-boundary, lifecycle, build, conformance, or integration-test
  rule changes.

## Package impact

- Existing packages used:
  - `io.github.pho001.synaptik.model.operation` for `OperationAttrs`, `OperationKind`,
    `OperationSignature`, and `Operation`;
  - `io.github.pho001.synaptik.model.operation.pooling` for the rank-specific pooling family;
  - `io.github.pho001.synaptik.model.tensor` for Tensor receivers, descriptors, factory creation,
    producer ownership, and provenance;
  - `io.github.pho001.synaptik.model.shape` for static and canonical symbolic output geometry; and
  - `io.github.pho001.synaptik.model.datatype` for the existing floating-type boundary.
- Packages added or changed:
  - no package is added;
  - `io.github.pho001.synaptik.model.operation.pooling` gains two public attrs records and one
    public semantic-kind enum; and
  - `io.github.pho001.synaptik.model.tensor` gains two package-private construction owners and two
    public Tensor receiver methods.
- Type placement:
  - rank-specific immutable intrinsic geometry belongs beside existing Pool2d attrs;
  - operation identity and exact signatures belong to `Pool3dKind`; and
  - input-aware rank/type/geometry validation plus descriptor/provenance construction belongs in
    the two family-specific tensor helpers, not in attrs, `Tensor`, or a generic utility.

## Exact public surface

`MaxPool3dAttrs` and `AveragePool3dAttrs` are public records implementing `OperationAttrs` with
exactly these components in declaration and validation order:

```text
long kernelDepth
long kernelHeight
long kernelWidth
long strideDepth
long strideHeight
long strideWidth
long paddingDepth
long paddingHeight
long paddingWidth
long dilationDepth
long dilationHeight
long dilationWidth
boolean ceilMode
```

Kernel, stride, and dilation components must be positive. Padding components must be non-negative.
Canonical constructors validate in exact declaration order and use the established messages
`<name> must be positive: <value>` and `<name> must be non-negative: <value>`. The records expose
no defaults, builder, nested type, array geometry, normalization, divisor option, or mutable state.

`Pool3dKind.values()` is exactly `[MAX_POOL3D, AVERAGE_POOL3D]`. Its two signatures are respectively
`OperationSignature.fixed(MaxPool3dAttrs.class, 1, 1)` and
`OperationSignature.fixed(AveragePool3dAttrs.class, 1, 1)`. Cross-pairing either kind with the
other attrs class fails through the existing exact-class signature contract.

The only new Tensor declarations are:

```java
public Tensor maxPool3d(MaxPool3dAttrs attrs)
public Tensor averagePool3d(AveragePool3dAttrs attrs)
```

There is no alias such as `avgPool3d`, no overload, no defaults receiver, and no PoolNd spelling.
These methods increase the exact public `Tensor` declared-method count from 208 to 210 and add only
the names `maxPool3d` and `averagePool3d` to the locked public-name set.

## Output geometry

The input Shape is `[N, C, D, H, W]`; the output Shape is
`[N, C, D_out, H_out, W_out]`. Output batch and channel positions retain the exact input Dimension
references. For each spatial input extent `X` and its matching kernel count `k`, symmetric padding
per side `p`, dilation `d`, and stride `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

`ceilMode == true` selects the literal ceiling formula for depth, height, and width.
It does not decrement a terminal output whose window begins wholly in trailing padding; an
all-padding terminal window remains part of the output grid.

All literal geometry constants and static calculations use checked signed-`long` arithmetic.
For a static input extent, a negative numerator fails before producer construction. Zero spatial
input is valid when symmetric padding makes the numerator non-negative; zero batch or channel is
always valid and contains no values. For an unresolved spatial Dimension, construction retains
the canonical `addConstant`, matching floor-or-ceiling divide, then `addConstant(1)` expression.
The exact obligation that a later concrete binding make each numerator non-negative remains
visible for Compiler or binding validation. Spatial derivation and failure order are depth, then
height, then width.

## Maximum numerical semantics

- Each output window enumerates logical kernel coordinates in increasing depth, then height, then
  width order. The corresponding input coordinate is
  `output * stride - padding + kernelCoordinate * dilation` on each axis.
- Out-of-bounds padding positions are excluded from selection and are never eligible winners. A
  window with no in-bounds sampled coordinate has exact negative infinity in the input/result
  floating type.
- Any in-bounds NaN dominates every non-NaN. Among multiple NaNs, the first eligible logical
  depth-height-width occurrence wins. NaN payload, sign, quiet/signaling form, and preservation are
  unspecified.
- Otherwise ordinary numerical maximum applies: positive zero ranks above negative zero,
  infinities use ordinary order, and numerically equal eligible candidates retain the first
  depth-height-width occurrence.
- Maximum pooling retains the selected represented non-NaN value in the exact input type; it has
  no average accumulator, mixed-type promotion, saved-index output, hidden winner buffer, or
  algorithm/traversal promise beyond the semantic first-winner order.

## Average numerical semantics

- Every logical kernel position counts in the fixed positive divisor, whose mathematical value is
  `kernelDepth * kernelHeight * kernelWidth`. Dilation changes sampled coordinates, not the
  divisor. There is no valid-sample count, divisor override, count-padding option, or zero-divisor
  case.
- Each in-bounds position contributes its input value once. Each out-of-bounds position
  contributes conceptual exact positive zero and still counts once in the divisor.
- BFLOAT16 and FLOAT32 accumulate and divide in FLOAT32. FLOAT64 accumulates and divides in
  FLOAT64. The sum is divided once by the fixed divisor, and BFLOAT16 narrows only once after that
  division. Construction must not materialize the three-factor divisor in `long`; its mathematical
  product may exceed `Long.MAX_VALUE` while every per-axis geometry calculation remains valid.
- Any in-bounds NaN produces NaN. Positive and negative infinity together produce NaN; otherwise a
  present infinity retains its sign. Padding introduces no NaN or infinity.
- An exact-zero finite mean is negative zero only when every divisor position is an in-bounds
  negative zero. Cancellation, any positive zero, or any padding contribution produces positive
  zero. An all-padding window therefore has exact positive zero.
- Finite accumulation may be reassociated, so no fixed traversal, bitwise result, NaN payload/sign
  preservation, or cross-backend rounding identity is promised. This permission does not weaken
  the specified accumulator type, one final division, BFLOAT16 narrowing, or exceptional-value
  classes.

## Construction and failure contract

Each family helper validates and constructs in this order:

1. require non-null `input`, then non-null `attrs`, with messages `input` and `attrs`;
2. require the input type to be BFLOAT16, FLOAT32, or FLOAT64 and retain it exactly;
3. require rank five;
4. derive and validate depth, then height, then width geometry using checked arithmetic;
5. create the exact result descriptor with unresolved layout and unchanged `requiresGrad`;
6. create the paired `Operation` with the exact attrs reference; and
7. call the existing one-output derived factory seam with ordered inputs `[input]`.

The established family messages are `maxPool3d ...` and `averagePool3d ...`. Geometry failures use
`<family> effective kernel does not fit padded <axis>: input=<dimension>,`
`effectiveKernel=<value>, padding=<value>`.
Every locally decidable null, type, rank, geometry, arithmetic, descriptor, operation/signature, or
input-list failure occurs before Tensor identifier allocation. A successful invocation consumes
exactly one fresh ID. Identifier exhaustion remains the existing factory failure and is not
reinterpreted.

Each result is a fresh, unlabeled, storage-free canonical Tensor with unresolved layout. Its
provenance has a fresh producer, the paired Pool3d kind, the exact supplied attrs reference,
ordered exact inputs `[input]`, output index zero, and the exact output descriptor; the returned
Tensor is the exact object from producer `output(0)`. Repeated calls with equal inputs and attrs
create identity-distinct Operations, producers, descriptors, IDs, and canonical outputs.

## Affected files

Production/Javadoc, exactly six paths:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/MaxPool3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/AveragePool3dAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool3dExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Focused tests, exactly three new paths:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMaxPool3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAveragePool3dExpressionTest.java`

Existing exact Tensor public-method-count owners, exactly seventeen paths, change only from 208 to
210 plus the central `TensorTest` name-set additions:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/RecurrentScanExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorConv3dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPool1dExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation/planning, exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only with an explicit no-change conclusion unless a contradiction requires stopping:
`ARCHITECTURE.md`, current architecture plan, ADRs, Training API, Pool1d/Pool2d/Conv3d contracts,
Compiler/CPU master plans and production/tests, backend conformance, integration, architecture
tests, Gradle/build files, and every other module.

## Maximum scope

- Hard ceiling: 33 paths total — six production/Javadoc, twenty test, and seven
  documentation/planning paths listed above.
- The seventeen existing count-owner test files receive only the mechanical `208 -> 210` update,
  except `TensorTest` also adds the two exact method names. Any other change in those owners is out
  of scope.
- No compatibility alias, default overload, shared geometry abstraction, unrelated refactor, or
  downstream implementation is permitted within the ceiling.
- Stop and request clarification before exceeding 33 paths or changing architecture, Shape/
  Dimension contracts, factory seams, operation-signature infrastructure, dependencies, build
  configuration, Compiler/CPU code, or tests outside the exact list.

## Acceptance criteria

- The two attrs records have the exact thirteen-component surface, declaration-order validation,
  equality/hash behavior, and family-specific Javadocs specified above.
- `Pool3dKind` has exactly two constants in the specified order, stable immutable exact signatures,
  accepted own-attrs Operations, and rejected cross-family attrs.
- Tensor exposes exactly the two new receiver methods, increasing its public method count from 208
  to 210 without another public name or overload.
- Both helpers accept every current floating type, reject integral/BOOL and wrong-rank input, and
  derive exact NCDHW static floor, literal-ceil, terminal all-padding, and canonical symbolic
  geometry while retaining exact N/C references.
- Static and symbolic geometry uses checked arithmetic, validates depth/height/width in order,
  accepts empty axes where the formula is valid, and never materializes the average divisor.
- Max semantics are documented exactly as excluded padding, all-padding negative infinity, NaN
  dominance, positive-over-negative zero, ordinary infinities, and first eligible
  depth-height-width winner.
- Average semantics are documented exactly as fixed three-factor count-padding, specified
  accumulator/division/narrowing domains, permitted reassociation, and the Pool2d-compatible NaN,
  infinity, signed-zero, and all-padding policies.
- Null/type/rank/geometry/arithmetic failures precede factory allocation; every success creates
  exactly one fresh canonical output and one exact one-input Pool3d producer occurrence.
- Javadocs cover every public type/method, every parameter, result, constraint, nullability,
  failure, metadata/provenance effect, and current-versus-planned boundary without restating code.
- Tensor API, Compile API, glossary, Model capabilities, task, master plan, and roadmap agree that
  Model construction is current while Compiler 0006B1, Model 0025K, Compiler 0006B2, and CPU
  0008G1 remain Draft downstream owners. Training API and architecture contracts remain accurate
  without edits.
- 0025J is the sole detailed Ready Model task before implementation. 0025K and 0026 remain Draft
  without task files; Compiler 0006B1/0006B2 and CPU 0008G1 remain Draft without task files.

## Tests / validation

Focused implementation tests:

```text
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Required once after executable Java stabilizes:

```text
./gradlew :modules:model:test
```

Required in the final clean documentation pass, without repeating the successful Java suite when
only documentation/Javadocs change:

```text
./gradlew :modules:model:javadoc
git diff --check
```

Also record:

- reflection and `javap` inspection of the exact records, kind/signatures, Tensor declarations,
  helper visibility/field-free shape, and public-method count 210;
- exact static/symbolic depth-height-width formulas, terminal all-padding grids, checked overflow,
  unmaterialized three-factor average divisor, failure ordering, ID deltas, producer freshness,
  exact attrs/input/output-index provenance, and canonical `producer.output(0)` identity;
- source/Javadoc inspection for max and average exceptional-value/rounding contracts, because
  Model construction evaluates no tensor values;
- Markdown headings, links, anchors, fences, and status/order/dependency checks;
- scans proving 0025J is the sole detailed Ready Model task and that no task files exist for Model
  0025K/0026, Compiler 0006B1/0006B2, or CPU 0008G1;
- exact 33-path ceiling, exact package placement, no forbidden downstream/build/architecture
  changes, and whitespace validation.

Repository-wide validation is deferred to the next recorded capability checkpoint and CI. No
architecture, backend-conformance, integration, Compiler, CPU, or Gradle suite is required because
this task changes one module without changing a shared boundary or executable backend behavior.

## Dependencies

- Complete Model 0020A and 0020A1 define the exact maximum and fixed-count average Pool2d
  numerical and geometry policies extended by one spatial axis.
- Complete Model 0025H supplies the first-class NCDHW rank-specific attrs, helper, symbolic
  geometry, validation-order, and canonical-provenance precedent.
- Complete Model 0025I closes the rank-specific pooling naming and the reason Pool3d, unlike
  Pool1d, must be first-class.
- Complete Model 0018K–0018N and 0018V supply current immutable operation signatures, floating
  type boundaries, Tensor expression ownership, and capability-reset constraints.
- Current canonical TensorProducer/factory and Shape/Dimension expression contracts are stable.

## Follow-up tasks

- Model 0025K remains Draft and will add the smallest public NCDHW `unfold3d`/`fold3d` algebra
  needed to express exact Pool3d adjoints. It must not add pooling-specific gradient primitives.
- Compiler 0006B1 remains Draft and will independently infer/final-validate both Pool3d forward
  occurrences as ordinary flat nodes while rejecting any backward-capable request containing one
  before derivative allocation.
- Compiler 0006B2 remains Draft after 0025K/0006B1 and will close exact gradients: average divides
  by logical `kernelDepth * kernelHeight * kernelWidth`; maximum reconstructs the first eligible
  depth-height-width winner with exact NaN and signed-zero occurrence matching.
- CPU 0008G1 remains Draft after the Model and Compiler chain. Its direct NCDHW Pool3d generated
  scalar/parallel-scalar code must support only proved static resolved current-floating cases,
  report truthful kind capability, allocate zero workspace/materialization, and match a specialized
  optimal clean Java oracle in semantic algorithm, hot-loop/dataflow shape, avoidable-overhead
  profile, Class-File/decompilation/forbidden-structure checks, and five-fork generated/direct
  `<= 1.15x` evidence.
- CPU 0008H remains after that complete inserted pooling chain.

## Architecture impact

None. The task adds one rank-specific operation family and two Tensor expression constructors
inside existing Model ownership. `ARCHITECTURE.md`, the current architecture plan, ADRs,
dependency rules, module descriptors, and architecture tests remain unchanged.

## Implementation prompt

### Clean implementation task prompt

Implement Model task 0025J exactly within this specification in a separate clean implementation
context. Read root `AGENTS.md`, architecture contracts, planning guide/roadmap, this task, Model
master/capabilities, completed 0020A/0020A1/0025H/0025I, and directly relevant Pool2d/Conv3d
source and tests in full. Do not use a GSD workflow, commit, or push.

Add only the six production paths, three focused suites, and mechanical changes to the seventeen
exact public-method-count owners listed here. Preserve exact attrs ordering, kind/signature pairing,
NCDHW geometry, max/average numerical contracts, validation and ID ordering, fresh canonical
provenance, and absence of execution/gradient claims. Run the focused command and one full Model
test command. Inspect reflection/`javap`, exact paths, and downstream absence. Do not edit the
seven documentation/planning paths; production Javadocs may remain provisional for the required
clean documentation review. Hand the stabilized diff and exact test evidence to that distinct
documentation-focused context. If an
architecture contradiction or need beyond the 33-path ceiling appears, stop and request
clarification. Return the required completion summary and do not mark Complete until the clean
documentation pass finishes.

### Clean documentation task prompt

In a distinct clean documentation-focused context, review the stabilized 0025J diff and recorded
implementation evidence. Read root `AGENTS.md`, documentation rules, General/API-Javadoc/
Planning/Example profiles, architecture contracts, this task, Model master/capabilities,
Tensor/Compile/Training APIs, glossary, directly affected production Javadocs/tests, and downstream
Compiler/CPU rows in full. Do not repeat a successful Java test suite unless executable Java
changes or concrete evidence requires it; do not commit or push.

Independently finalize the six affected production Javadocs and exactly the seven documentation/
planning paths. Clearly distinguish current Model metadata from Draft Compiler 0006B1, Model
0025K, Compiler 0006B2, and CPU 0008G1 behavior; record reasoned no-change conclusions for
Training API, architecture/ADRs/tests, conformance/integration, Gradle, downstream production/tests,
and other modules. Run Model Javadoc, Markdown links/anchors/fences, exact status/dependency/order,
task-file absence, exact 33-path, and `git diff --check` gates. Mark 0025J Complete only when all
criteria pass; otherwise return the exact incomplete follow-up.

## Local decisions

- Pool3d is first-class rather than a repeated Pool2d composition because the number of depth
  slices would depend on Shape. One operation occurrence is the bounded metadata representation.
- Max and average have separate attrs and construction owners because excluded-padding extrema and
  fixed-divisor accumulation are distinct semantic contracts; sharing coordinates does not make
  their numerical policies interchangeable.
- Public geometry is rank-specific primitive fields, matching Conv3d and Pool2d precedent. There
  is no dynamic-rank array contract or generic PoolNd abstraction.
- Symmetric intrinsic padding is the current contract. Asymmetric behavior remains explicit PAD
  composition or later separately justified semantics.
- Literal ceil mode is preserved without framework-style terminal-window suppression.
- Average's divisor is a positive mathematical product and is intentionally not calculated in
  Model construction, avoiding a false `long` representability restriction.
- Max first-winner order extends Pool2d height-width order by making depth the outer logical kernel
  coordinate. Average finite accumulation remains reassociable and does not acquire that traversal
  promise.
- Separate family helpers are deliberately small and field-free; no generic geometry helper is
  introduced for only two implementations.
- Documentation uses General, API/Javadoc, Planning, and Example profiles. The glossary adds the
  reusable NCDHW Pool3d distinction; architecture documents remain unchanged because ownership and
  boundaries do not change.

## Known limitations

- Only BFLOAT16, FLOAT32, and FLOAT64 NCDHW input with symmetric intrinsic padding is included.
- Average pooling is fixed count-padding only; valid-sample division and divisor override are
  absent.
- No saved maximum indices, adaptive/global pooling, asymmetric intrinsic padding, NDHWC, PoolNd,
  or dynamic-rank geometry exists.
- Model does not evaluate values. Compiler adoption, exact gradients, CPU generated execution,
  other backends, NN layers, and performance remain downstream.

## Validation evidence

- The stabilized implementation evidence was reused without rerunning Java tests because this
  clean documentation context changed no executable Java. The final Model XML reports 138 suites
  and 1,098 tests with zero skips, failures, or errors. The four focused owners named by this task
  account for 31 passing tests: 4 Pool3d semantics, 6 maximum-expression, 6 average-expression,
  and 15 Tensor surface tests.
- Clean documentation context `/root` independently reviewed the complete 33-path diff, all six
  affected production source/Javadocs, all three new focused suites, the seventeen mechanical
  public-count owners, completed Pool2d/Pool1d/Conv3d precedents, Tensor/Compile/Training APIs,
  glossary, Model capabilities/master plan, and downstream Compiler/CPU planning boundaries. It
  changed no executable Java.
- `./gradlew :modules:model:javadoc` passed after the Javadoc review with two tasks up to date.
  Rendered `MaxPool3dAttrs.html`, `AveragePool3dAttrs.html`, `Pool3dKind.html`, and both Pool3d
  sections in `Tensor.html` contain the reviewed geometry, numerical, provenance, parameter,
  result, failure, and current-versus-planned contracts. Package-private helper Javadocs were
  reviewed directly in source.
- The documented Java 26 metadata example compiled and ran against Model classes. It printed
  `Shape[1, 2, 3, 3, 3]` twice, `MAX_POOL3D`, `AVERAGE_POOL3D`, and `true`, confirming exact output
  Shape metadata, kinds, and one-input provenance without evaluating values.
- Reflection reported exactly 210 public Tensor methods, exactly two Pool3d receivers, thirteen
  record components in each attrs type, kind order `[MAX_POOL3D, AVERAGE_POOL3D]`, and exact
  singleton one-input/one-output signature pairings. `javap -public` confirmed the two receiver
  declarations and exact record/kind surfaces. `javap -private` confirmed both helpers are
  package-private final, field-free, privately constructed, and expose only package-private
  `apply` plus a private geometry method.
- A runtime inventory scan over compiled Model operation enums reported exactly 40 families, 113
  constants, and 134 signatures. Source scans found no Pool3d references in Compiler or CPU
  production/tests, architecture tests, backend conformance, or integration tests, confirming
  that current support stops at Model metadata.
- The targeted Markdown validator passed all seven documentation/planning files: 1,060 local
  links, 792 target anchors, balanced fences, final newlines, no carriage returns, and no trailing
  whitespace. The exact-scope check passed 33 authorized paths: six production/Javadoc, twenty
  Model tests, and seven documentation/planning files. The staging area is empty.
- Status/dependency/order checks set 0025J Complete in this task, Model master plan, and roadmap;
  preserve Model 0025K, Compiler 0006B1/0006B2, and CPU 0008G1 as Draft; preserve the ordered
  `0025I -> 0025J -> 0025K -> 0006B1 -> 0006B2 -> 0008G1 -> 0008H` chain; and find no detailed
  task files for Model 0025K/0026, Compiler 0006B1/0006B2, or CPU 0008G1.
- `git diff --check` passed on the final combined change.
- Training API remains unchanged because Pool3d adds no Training-owned optimizer, session,
  orchestration, gradient, or execution surface. `ARCHITECTURE.md`, focused architecture pages,
  ADRs, and architecture tests remain unchanged because module ownership, dependencies, and
  lifecycle rules did not change. Backend conformance and integration tests remain unchanged
  because no backend or end-to-end behavior exists. Gradle/build files remain unchanged because
  no dependency or build contract changed. Downstream Compiler/CPU production and tests remain
  unchanged because Draft 0006B1/0006B2/0008G1 own adoption, gradients, and execution. Other
  modules remain unchanged because this task is confined to Model semantics and directly affected
  documentation/planning.

## Implementation notes

- Added public immutable `MaxPool3dAttrs` and `AveragePool3dAttrs`, exact `Pool3dKind` semantics,
  separate field-free construction owners, and exactly two Tensor receivers. Construction retains
  exact floating type, N/C Dimensions, `requiresGrad`, checked static or canonical-symbolic NCDHW
  geometry, and one fresh canonical output with exact provenance.
- Added three focused suites and mechanically changed the seventeen existing Tensor public-count
  owners from 208 to 210; only `TensorTest` additionally adds the two exact method names.
- The documentation pass finalized all six affected Javadocs and updated Tensor/Compile APIs,
  glossary, Model capabilities, task, Model master plan, and roadmap. It keeps current Model
  metadata distinct from Draft Compiler 0006B1, Model 0025K, Compiler 0006B2, and CPU 0008G1.

## Completion summary

- Completed changes: Added first-class NCDHW maximum and fixed-count average Pool3d Model
  semantics, exact rank-specific geometry, numerical policies, canonical provenance, focused
  tests/public locks, finalized Javadocs, and synchronized explanatory/planning documentation.
- Files changed or created: Exactly 33 authorized paths: six production/Javadoc, twenty Model
  tests, and seven documentation/planning files.
- Tests and validation: Reused the stabilized 138-suite/1,098-test Model result with 31 passing
  focused-owner tests; passed Model Javadoc, rendered inspection, Java 26 example,
  reflection/`javap`, 40/113/134 inventory and downstream-absence scans, 1,060-link/792-anchor
  Markdown validation, exact scope/status/order/task-file/staging checks, and `git diff --check`.
- Documentation-agent review: Clean context `/root` independently finalized affected Javadocs,
  APIs, glossary/capability impact, planning evidence, and status without executable Java edits.
- Documentation impact: Tensor and Compile APIs, glossary, Model capabilities, task, Model master
  plan, and roadmap now describe current Model Pool3d metadata and exact downstream ownership.
- Javadoc review: Both attrs records and constructors, both kind constants/signatures, both
  package-private construction owners, both Tensor methods, and the affected Tensor overview have
  complete contracts; no additional production Javadoc change was necessary after review.
- Glossary impact: Added the reusable Pool3d/NCDHW distinction and synchronized Model-versus-
  Compiler inventory counts and downstream boundaries.
- Unresolved issues: None within task 0025J.
- Follow-up required: Model 0025K remains the next Draft Model frontier; Compiler 0006B1/0006B2
  and CPU 0008G1 remain the ordered Draft consumers.

Status: Complete
