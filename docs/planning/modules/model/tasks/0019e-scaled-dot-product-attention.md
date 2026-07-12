# Task 0019E: Scaled Dot-Product Attention

## Status

Complete

## Goal

Add one backend-independent, one-output scaled dot-product attention semantic operation and its
public `Tensor` construction. The receiver is query. Key, value, an optional BOOL mask, causal
selection, and default or explicit scale are represented without dropout, hidden random state,
public attention weights, gradients, decomposition, execution, or backend policy.

This is first-class rather than literal MATMUL/mask/softmax/MATMUL composition so the selected
all-masked-row, masked-special-value, positive-infinity-tie, and stability contracts survive
compiler inspection. A future compiler may decompose it only while preserving the whole meaning;
backend prepare may choose a conforming fused or unfused implementation.

## Rationale and mental model

```text
query  [..., L, E]
key    [..., S, E]  -> scores/weights [..., L, S]
value  [..., S, Ev] -> output         [..., L, Ev]

scores  = (query @ transpose(key)) * scale
weights = maskedSoftmax(scores, final axis S)
output  = weights @ value
```

Leading batch/head prefixes broadcast right-aligned across query, key, and value. A mask
broadcasts exactly to the score Shape. `true` includes and `false` excludes a position before its
score or value special values participate. Causal mode additionally requires `j <= i` for
zero-based query position `i` and key position `j`, including rectangular `L`/`S`.

### Conceptual metadata example

The documentation pass must include a clearly labeled conceptual example equivalent to:

```java
Tensor output = query.scaledDotProductAttention(
        key,
        value,
        mask,
        new ScaledDotProductAttentionAttrs(Optional.empty(), true));
```

For `query=[2,4,8]`, `key=[1,6,8]`, `value=[2,6,10]`, and `mask=[4,6]`, the
score Shape is `[2,4,6]` and output Shape is `[2,4,10]`. Scale is `1 / sqrt(8)`, mask broadcasts
across batch, and causal mode also excludes `j > i`. This shows expression metadata, not numeric
execution, compiler capture, gradients, or backend support.

## Scope

- Add one public `SCALED_DOT_PRODUCT_ATTENTION` kind in a focused attention package.
- Add public immutable `ScaledDotProductAttentionAttrs(Optional<ScalarValue> scale,
  boolean causal)`.
- Give the kind one three-to-four-input, exactly-one-output signature.
- Add exactly four unambiguous receiver methods: key/value defaults; key/value/attrs;
  key/value/mask defaults; and key/value/mask/attrs.
- Add one package-private field-free helper owning all local validation, Shape inference,
  descriptor/operation construction, and one factory delegation.
- Define exact Shape, type, mask, scale, numerical, metadata, provenance, validation-order,
  freshness, side-effect, and deferred-constraint contracts.
- Add semantic/expression tests and update every global public-`Tensor` inventory/count from 171
  to 175.
- Finalize Javadocs, Tensor/Compile API, glossary, capabilities, and planning in a mandatory
  separate clean-context documentation pass.

## Out of scope

- dropout, training mode, `GraphRngState`, hidden/global RNG, random draws, dropout masks, or state
  output
- public attention weights, auxiliary/saved outputs, multi-output carrier, or saved statistics
- gradients, adjoints, backward operations, trainable parameters, compiler capture, graph-wide
  solving, canonicalization, decomposition, fusion, or optimization
- algorithms, online softmax, tiling, materialization, fixed accumulation order, kernels, backend
  capability/lowering, prepare, runtime, execution, conformance, or integration
- additive masks/bias, offsets, causal variants, grouped-query/sparse attention, caches, packed
  sequences, quantized/complex/unsigned types, or FLOAT16
- runtime scalar Tensor scale, nullable mask, null-as-default, NaN sentinel, cast insertion,
  primitive scale overloads, broad options framework, or `Optional<Tensor>`
- changes to MATMUL, softmax, where, masked reductions, dropout/RNG, `ScalarValue`, promotion,
  `ShapeBroadcast`, Dimension, provenance, signatures, or factory contracts
- Gradle, another module, architecture/ADR/test, focused architecture, backend-conformance, or
  integration changes

## Exact public API and attributes

The receiver is query. Add exactly:

```java
public Tensor scaledDotProductAttention(Tensor key, Tensor value)
public Tensor scaledDotProductAttention(
        Tensor key, Tensor value, ScaledDotProductAttentionAttrs attrs)
public Tensor scaledDotProductAttention(Tensor key, Tensor value, Tensor mask)
public Tensor scaledDotProductAttention(
        Tensor key, Tensor value, Tensor mask, ScaledDotProductAttentionAttrs attrs)
```

The short forms use `new ScaledDotProductAttentionAttrs(Optional.empty(), false)`. Causal mode or
explicit scale uses attrs. Tensor-versus-attrs third parameters avoid ambiguity. Add no boolean,
primitive-scale, nullable-mask, varargs, builder, or general-options overload.

The attrs record is public because public provenance exposes exact typed semantic attributes. It
is operation-specific, not a broad options framework. `Optional.empty()` has one genuine meaning:
compute `1 / sqrt(E)` after embedding extent binding. A present exact `ScalarValue` is the scale.

The record constructor null-checks `scale`; when present it requires FLOAT64/FLOAT32/BFLOAT16,
decodes the exact value, and requires it finite and strictly positive. Retain exact bits. Use:

```text
scale must have a floating data type, but was <dataType>
scale must be finite and positive: <value>
```

Zero and negative values erase/reverse similarity order and are not selected attention semantics;
NaN/infinity are invalid configuration rather than score data.

## Operation model

Add public types under `io.github.pho001.synaptik.model.operation.attention`:

- `ScaledDotProductAttentionKind`, with exactly `SCALED_DOT_PRODUCT_ATTENTION`;
- `ScaledDotProductAttentionAttrs`, the record above.

The kind's exact stable signature is:

```java
OperationSignature.inputRange(ScaledDotProductAttentionAttrs.class, 3, 4, 1)
```

Unmasked inputs are `[query,key,value]`; masked inputs are `[query,key,value,mask]`. Both create
one descriptor/output, one wrapper, one producer, one ID, and provenance output index zero. There
is no attention-weight/state output or dropout parameter. The structural range does not replace
public/helper role validation or later compiler validation.

## Shape and deferred-constraint contract

### Axes and ranks

Require query, key, and value ranks at least two:

```text
query = [...queryBatch, L, E]
key   = [...keyBatch,   S, E]
value = [...valueBatch, S, Ev]
```

Final two axes are matrix axes; no vector promotion exists. Static-zero query `E` fails. Dynamic
`E` defers positivity. Unequal static query/key `E` fails; any unresolved pair defers equality
because `E` is absent from output. Unequal static key/value `S` fails; unresolved pairs defer
equality. Name these obligations `attention embedding positivity`, `attention embedding
equality`, and `attention key/value sequence equality`. No model constraint object is added;
future compiler capture re-derives and proves them.

### Three-way batch/head broadcasting

Right-align all three prefixes together. Missing axes behave as singleton. Per result batch axis:

1. reject more than one distinct static non-singleton extent;
2. with one static non-singleton, retain its exact first reference in query/key/value order and
   defer each unequal unresolved participant to singleton-or-that-extent;
3. otherwise ignore static singletons; remaining unresolved Dimensions must be structurally equal
   or reject because no exact output is derivable;
4. retain the first remaining unresolved reference in query/key/value order, or if all singleton/
   missing retain the first present input singleton reference.

Unpaired axes retain exact references. Named/expression equality is structural; constrained
unknown equality follows current identity semantics. Deferred obligations are `attention batch
broadcast` constraints. Do not change/use pairwise `ShapeBroadcast`, synthesize unknown output,
or bind dimensions.

### Score, output, and mask

```text
score  = [...exact broadcastBatch, exact query L, exact key S]
output = [...exact broadcastBatch, exact query L, exact value Ev]
```

New outer Shapes retain all selected exact Dimension references.

A mask must be exact BOOL and rank no greater than score rank. Scalar mask is valid: true admits
all otherwise eligible positions; false masks all. Accept every rank zero through score rank that
right-broadcasts exactly to the preselected score Shape. Per aligned axis:

- static mask singleton or structurally equal mask Dimension is accepted, retaining score ref;
- unresolved mask against static non-singleton score defers singleton-or-score equality;
- every other unequal pair rejects, including static non-singleton mask against unresolved score
  and two unequal unresolved Dimensions.

Unpaired leading score axes see implicit mask singletons. Deferred obligations are `attention mask
broadcast` constraints. Mask never changes score/output refs or uses `ShapeBroadcast`.

Causal eligibility is exactly `j <= i`, top-left aligned with no offset/window/right alignment.
Explicit and causal masks combine by logical AND.

## Types, scale, and metadata

Require query, key, value floating in that order. Result type is:

```java
DataType queryKeyType = DataTypePromotion.promoteFloating(queryType, keyType);
DataType resultType = DataTypePromotion.promoteFloating(queryKeyType, valueType);
```

No casts. Explicit scale must exactly match `resultType`:

```text
scale data type must match promoted attention data type: scale=<scaleType>, promoted=<resultType>
```

Absent scale remains absent even for static `E` and means exact semantic `1 / sqrt(E)` after
positive binding; model does not precompute/round it. No scalar Tensor is needed because scale is
occurrence configuration, not graph data.

Result descriptor: promoted type; exact output Shape; unresolved layout; requires-grad OR of
query/key/value only; empty label/storage. BOOL mask is non-gradient. One successful call creates
fresh descriptor, operation, producer, wrapper, ID. Attrs-bearing overloads retain the exact attrs
reference; inputs are exact ordered refs. Repeated calls are identity-distinct and inputs/attrs
remain unchanged.

## Numerical semantics

For each batch/query row, compute eligible ideal dot-product scores times scale, softmax only over
eligible final-axis `S` positions, then weighted value sum. Excluded positions have positive-zero
weight and are excluded before score and value arithmetic; masked key/value NaN/infinity cannot
contaminate the row.

Per row:

- no eligible positions: all weights/output components positive zero;
- any eligible NaN score: eligible weights and non-empty output components NaN, excluded weights
  positive zero;
- one or more eligible positive infinities: split unit weight equally among them and give all
  other eligible positions positive-zero weight; weighted value arithmetic still follows the
  general eligible-position rule below;
- all eligible scores negative infinity: weights/output positive zero, not NaN;
- otherwise finite scores use ideal exponential ratios, eligible negative infinities get zero,
  and ideal weights total one.

Thus explicit/causal all-masked rows and `S==0` rows are zero. Eligible query/key NaN follows dot
propagation. Eligible value NaN follows floating multiply/add even if its non-masked weight is zero
from a negative-infinity score; only eligibility exclusion guarantees ignoring it.

FLOAT64 accumulates scores and output sums in FLOAT64. FLOAT32/BFLOAT16 accumulate both in
FLOAT32; BFLOAT16 converts the final output. Scale participates in that domain. Reassociation,
fused multiply-add, and any stable softmax satisfying special rules are allowed. Later conformance
sets tolerances. No traversal order, bitwise result, cross-backend identical rounding, or algorithm
is promised. Ordinary IEEE multiplication/addition governs eligible infinities/signed zero subject
to reassociation; only mandated empty/all-masked/all-negative-infinity zero is positive zero.

Empty decisions: `L==0` is valid with no rows; `S==0` is valid with positive-zero output for every
existing query row; static `E==0` is invalid and dynamic `E` defers positivity; `Ev==0` is valid
with an empty output axis; zero batch/head axes are valid. Determinism fixes mathematical
eligibility/ties/results for one conforming prepared implementation, not bitwise portability.

## Validation order, failures, and ID effects

Complete every local check before descriptor/operation/producer/ID creation:

1. null-check query, key, value, mask when present, attrs;
2. validate query/key/value floating and promote in order;
3. validate query/key/value ranks in order;
4. reject static-zero query embedding;
5. reject static query/key embedding mismatch;
6. reject static key/value sequence mismatch;
7. derive batch axes leading to trailing;
8. build local score/output Shapes;
9. validate present mask BOOL, rank, then axes leading to trailing;
10. validate present scale exact promoted type (attrs already validates floating/finite/positive);
11. construct descriptor/operation;
12. delegate once to `TensorFactory.createDerived` with exact inputs.

Use exact messages:

```text
query must have a floating data type, but was <dataType>
key must have a floating data type, but was <dataType>
value must have a floating data type, but was <dataType>
query rank must be at least 2: <rank>
key rank must be at least 2: <rank>
value rank must be at least 2: <rank>
attention embedding dimension must be positive: <dimension>
attention query/key embedding dimensions must match: query=<dimension>, key=<dimension>
attention key/value sequence dimensions must match: key=<dimension>, value=<dimension>
cannot broadcast attention batch dimensions at result batch axis <axis>: query=<dimension>, key=<dimension>, value=<dimension>
cannot derive exact attention batch dimension at result batch axis <axis>: query=<dimension>, key=<dimension>, value=<dimension>
mask must have BOOL data type, but was <dataType>
mask rank must not exceed attention score rank: mask=<maskRank>, score=<scoreRank>
mask cannot broadcast exactly to attention score shape at axis <axis>: mask=<dimension>, score=<dimension>
scale data type must match promoted attention data type: scale=<scaleType>, promoted=<resultType>
```

The first batch message is conflicting static non-singletons; the second is unequal unresolved
candidates without a selecting static extent. Embedding/sequence mismatch messages apply only to
unequal statics. Every failure through operation construction consumes no ID/producer/wrapper;
successful calls consume exactly one ID. Factory exhaustion retains current no-rollback behavior.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)
- [Shared provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar](0018n-typed-scalar-value-contract.md)
- [Mask redesign](0018q-masked-reduction-redesign.md)
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)
- [Softmax expressions](0016j-softmax-tensor-expressions.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. Tensor remains public mutable state, not IR.
- Attention types record backend-independent meaning and no support/route/storage/runtime behavior.
- Direction is tensor -> attention operation/datatype/shape; operation attention imports no Tensor,
  graph, compiler, runtime, prepare, or backend type.
- Compiler owns capture, deferred proofs, legal decomposition, gradients, adjoints/saved values.
- Backend prepare owns conforming fused/unfused lowering, algorithms, materialization, kernels;
  runtime executes prepared work without original operations on its hot path.
- Attention has no technical dropout dependency. Future attention dropout must consume explicit
  `GraphRngState`, never hidden/global RNG.
- No architecture, dependency, lifecycle, focused-architecture, Gradle, or cross-module change.

## Package impact

Existing: datatype, shape, operation, tensor. Add
`io.github.pho001.synaptik.model.operation.attention` for public attention identity and attrs only.

Type placement:

- `...attention.ScaledDotProductAttentionKind` — identity/signature owner.
- `...attention.ScaledDotProductAttentionAttrs` — public inspectable immutable semantics.
- `...tensor.TensorScaledDotProductAttentionExpressions` — package-private construction owner.
- `...tensor.Tensor` — established public fluent facade.

Tests mirror production packages for package-private inspection.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (6):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — four exact
  signatures and count 171 -> 175.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  count only, 171 -> 175.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java` —
  count only, 171 -> 175.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java` —
  count only, 171 -> 175.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime/Training APIs; related operation/type/shape/provenance
contracts; architecture/ADRs/tests; conformance/integration; Gradle; other modules.

## Maximum scope

Exactly 17 paths maximum: four production, six test, seven documentation/planning. `Tensor.java`
changes only imports, four methods/Javadocs, class operation inventory. Four existing tests change
only stated global inventories/counts. Stop for path 18, another type/test, existing helper change,
architecture, Gradle, or cross-module work.

This is cohesive and inside the planning guide's 12–18 path guardrail. Kind, attrs, public
construction, API locks, and docs must agree in one compilable capability; splitting creates an
unusable intermediate and duplicates one tightly coupled numerical/mask review. No 0019E1 split.

## Javadoc and documentation requirements

- Fully document kind, attrs, helper, and four Tensor methods: API, Shape/type/mask/scale/causal,
  numerical/empty, metadata/provenance, validation/failures, freshness, and layer boundaries.
- Every parameter/return/failure has `@param`/`@return`/`@throws` as applicable.
- Tensor API gets mental model, overload/Shape tables, conceptual metadata and small eligibility
  examples, special/empty rules, and current-model versus planned execution boundary.
- Compile API says current expression metadata and future compiler-owned capture/constraints/
  decomposition/gradients; it does not claim compiler support.
- Review glossary terms scaled dot-product attention, attention score, causal mask, all-masked row.
- Synchronize capabilities/task/master/roadmap: 0019–0019E Complete, no model task Ready, and
  0020+ Draft without detailed specs.
- Record reasoned no-change conclusions for Runtime/Training API, related contracts, architecture,
  conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact kind/attrs/input range/one output and exact four receiver overloads exist; count is 175.
- All rank/axis/batch/embedding/sequence/score/output/Dimension-reference rules pass.
- BOOL mask ranks/scalar/dynamic contracts and exact fourth input pass.
- Default dynamic scale and exact typed finite-positive explicit scale pass; no casts/scalar input.
- Promotion/accumulation/gradient/layout/label/storage/freshness/provenance/ID contracts pass.
- Masked special values, eligible NaN, +infinity ties, all -infinity/all-masked rows, signed-zero,
  empty extents, reassociation/tolerance/determinism are documented/tested without evaluation.
- Exact validation order/messages and no-ID failures pass; success allocates one fresh ID.
- No dropout/weights/multi-output/gradient/compiler/algorithm/backend/runtime/architecture work.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 17 paths, packages, public surface, and statuses pass.
- Separate clean documentation pass reuses Java evidence and records no-change conclusions.
- 0019–0019E are Complete; no model task is Ready; 0020+ remain Draft without specs.

## Tests / validation

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover semantic/helper/public surfaces, overloads, all static/dynamic Shape cases,
floating combinations/scales, mask ranks/scalar, causal rectangles, validation/ID effects,
metadata/provenance/freshness, and numerical contracts as metadata/Javadoc without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also check local Markdown links/anchors, fences, terminology, conceptual labels, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/signatures, exactly one Ready frontier,
0019–0019D Complete, 0020+ Draft, and no later specs. Repository validation is deferred to the
selected-modern-operations checkpoint after 0022 or CI; no repository-wide contract changes.

## Dependencies

- 0001–0002, 0018M–0018M1: types, Dimensions, Shapes, symbolic extents.
- 0005–0007, 0011–0013, 0018K–0018L: operations/signatures, Tensor/factory/provenance.
- 0016I–0016J: softmax semantics/terminology.
- 0018N: exact typed scalar.
- 0018Q: BOOL mask exclude-before-value precedent.
- 0019: MATMUL axes/promotion/accumulation/deferred-contraction precedent.

0019B/0019B1 is table-order history, not a technical dependency.

## Follow-up tasks

- 0020 stays Draft/unchanged.
- Compiler later owns capture/constraints/decomposition/gradients/adjoints/saved values.
- Backend/conformance/runtime/integration later own algorithms/tolerances/lowering/kernels/storage/
  prepared execution/numeric evidence.
- Future attention dropout is separate and must consume explicit `GraphRngState`.

Do not create another detailed spec during implementation.

## Architecture impact

Expected impact: None. Stop and report any architecture, dependency, lifecycle, focused-doc,
cross-module, or scope need.

## Implementation prompt

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, focused architecture,
documentation/planning rules, roadmap, model capabilities/master, completed operation/signature/
Tensor/provenance/shared-output/shape/broadcast/typed-scalar/MATMUL/softmax/mask/dropout/linear
tasks, current related source/tests/APIs/glossary, and task 0019E.

Implement task 0019E exactly inside 17 paths. Update every global public Tensor inventory/count
171 -> 175 up front. Preserve contracts. Stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand actual diff
and Java evidence to a separate clean documentation agent in the same change; it finalizes
Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks while reusing Java
evidence. Complete only after all criteria; keep 0019–0019D Complete and 0020+ Draft/no specs.
```

## Documentation-agent handoff

Provide task, complete diff, exact focused/final evidence and post-test Java-change state,
API/Shape/type/mask/scale/numerical/provenance policies, seven docs paths, and validation. The clean
agent reads AGENTS/architecture/rules/profiles, task, source/tests/generated Javadoc, all four APIs,
glossary/planning, and related contracts; finalizes docs and records reasoned no-change conclusions.
It does not repeat successful Java tests absent executable change/stale evidence/concrete risk.

## Local decisions

- Kept scaled dot-product attention first-class so eligibility and special-value rules remain
  inspectable rather than becoming an implicit primitive chain.
- Used one public operation-specific attrs record and exactly four overloads; absent scale remains
  semantic state rather than a precomputed static value.
- Kept every unresolved equality, positivity, and singleton-or-equal fact deferred without adding
  a model constraint object or changing shared broadcasting.

## Known limitations

- Deferred attention constraints are recorded semantically, not solved in model.
- FLOAT16, dropout, bias/additive masks, weights, gradients, compiler/backend/execution are absent.
- Reassociation/stable algorithm freedom means no cross-backend bitwise promise.

## Validation evidence

Implementation context: `/root/task_0019e_implementation`.

- The prescribed seven-suite focused command passed 43 tests with zero failures, errors, or skips.
- After executable Java stabilized, exactly one actual final `./gradlew :modules:model:test` run
  passed 848 tests across 107 XML suites with zero failures, errors, or skips. An earlier attempted
  chained final command did not reach Gradle because wrapper-cache lock access was denied; it is
  not counted as a test run. No executable Java changed after the successful final suite.
- `git diff --check` passed after Java stabilization. The separate documentation context changed
  only Javadocs and the seven authorized documentation/planning paths, so it did not repeat the
  successful Java suites.

Documentation context:
`/root/task_0019e_implementation/task_0019e_docs`, applying the General, API/Javadoc, Planning,
and Example profiles.

- Independently reviewed repository instructions; the architecture and focused module boundary;
  documentation/planning rules; roadmap, capabilities, master plan, and this task; all four
  production and six test paths; Tensor/Compile/Runtime/Training APIs; glossary; and directly
  related operation, signature, Tensor, provenance, shared-output, Shape/broadcast, typed-scalar,
  MATMUL, softmax, mask, dropout, and linear contracts.
- Finalized Javadocs for the attention kind, attrs, package-private helper, Tensor class inventory,
  and all four public receiver methods. Parameter, return, failure, Shape/type/mask/scale/causal,
  numerical, provenance, freshness, side-effect, and lifecycle boundaries agree with source and
  focused tests. No executable statement changed.
- Tensor API now includes the Shape/overload mental model, eligibility and numerical rules, and
  conceptual `[2,4,8]` query, `[1,6,8]` key, `[2,6,10]` value, `[4,6]` mask example deriving score
  `[2,4,6]` and output `[2,4,10]`. Compile API records current model metadata and future
  compiler-owned proof, decomposition, and gradient work without claiming compiler support.
- Glossary defines scaled dot-product attention, attention score, causal mask, and all-masked row.
  Capabilities, master plan, roadmap, and this task consistently mark 0019 through 0019E Complete;
  task 0020 and later remain Draft without detailed specifications and no model task is Ready.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  Javadoc contains the public attrs/kind pages and exactly four public Tensor attention overloads.
  The package-private helper is absent from the public generated surface.
- `javap` confirmed 175 public Tensor methods excluding the constructor and exactly the four
  required overload descriptors. Package/import scans found the two public attention types in the
  focused operation package, the helper in the Tensor package, and no compiler, runtime, prepare,
  backend, training, or cross-module import.
- Targeted Markdown validation resolved 575 local links and 156 heading fragments across the seven
  changed Markdown files. Fences are balanced; every changed/new path is non-empty, ends with a
  newline, and has no trailing whitespace.
- Exact-scope validation found only the authorized 17 paths: four production, six tests, and seven
  documentation/planning paths. No task 0020-or-later specification exists. Final status,
  terminology, package, public-surface, generated-Javadoc, and `git diff --check` audits passed.
- Runtime API needs no change because this task creates no prepared execution, schedule, resource,
  binding, run-state, residency, or executable contract. Training API needs no change because it
  adds no parameter ownership, dropout state, gradient formula, backward operation, optimizer, or
  training workflow.
- Related operation/signature, Tensor/provenance/shared-output, Shape/broadcast, typed-scalar,
  MATMUL, softmax, mask, dropout, and linear contracts remain accurate because attention adds a
  separate focused family without changing them. Architecture/ADRs/tests, conformance/integration,
  Gradle, and other modules need no change because module ownership, dependency direction, build
  structure, and executable backend/runtime behavior are unchanged.

## Implementation notes

Implemented immutable attention attrs and one exact three-to-four-input/one-output kind, then
delegated the four public overloads to one package-private stateless helper. The helper completes
all local validation and exact Shape selection before one derived-Tensor factory call. The clean
documentation pass finalized Javadocs and the seven authorized documentation/planning paths
without changing executable Java.

## Completion summary

- Completed changes: added first-class one-output scaled dot-product attention semantics, immutable
  scale/causal attrs, exact Shape/type/mask/deferred-constraint policy, and four public receiver
  overloads without dropout or attention-weight output.
- Files changed or created: exactly four production Java, six model-test, and seven
  documentation/planning paths authorized by this task.
- Tests and validation: reused the passing 43-test focused selection and the sole successful final
  848-test/107-suite model run; model Javadoc, generated-page, `javap`, import/package, Markdown
  link/anchor/fence, terminology, exact-scope, status/no-later-spec, newline/whitespace, and
  `git diff --check` validations passed.
- Documentation-agent review: completed independently in the clean context named above.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap now describe current attention metadata and future lifecycle boundaries. Runtime and
  Training APIs were reviewed and remain accurate without changes.
- Javadoc review: all four affected production paths were reviewed and finalized; executable Java
  did not change after the recorded model suite.
- Glossary impact: added focused definitions for scaled dot-product attention, attention score,
  causal mask, and all-masked row.
- Unresolved issues: None.
- Follow-up required: None for task 0019E. Task 0020 remains Draft.

Status: Complete
