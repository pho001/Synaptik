# Task 0023E: Cumulative Scan Normalization and Product

## Status

Complete

## Goal

Add the generally useful one-axis cumulative-product expression required for zero-safe product
adjoints, while atomically normalizing the existing sum-only scan type names into one coherent
cumulative-scan family.

The completed public surface must retain both existing `cumSum` overloads and add exactly
`cumProd(int axis)` plus `cumProd(int axis, boolean exclusive, boolean reverse)`. Both operations
must preserve Shape positions, exact input data type, gradient-eligibility metadata, and
one-input producer/provenance construction without reading values or selecting execution.

## Scope

- Replace `CumulativeSumKind` with public `CumulativeScanKind` containing exactly `CUM_SUM` and
  `CUM_PROD`, in that order.
- Replace `CumulativeSumAttrs` with public `CumulativeScanAttrs` containing exactly normalized
  non-negative `axis`, `exclusive`, and `reverse` components in the existing order.
- Replace package-private `TensorCumulativeSumExpressions` with one shared
  `TensorCumulativeScanExpressions` construction boundary.
- Preserve the two existing public `Tensor.cumSum` signatures and their complete observable
  construction behavior while migrating their producer metadata to the normalized scan types.
- Add exactly two public methods:

  ```java
  Tensor cumProd(int axis)
  Tensor cumProd(int axis, boolean exclusive, boolean reverse)
  ```

- Make the short `cumProd` overload select inclusive forward mode explicitly.
- Accept exactly FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 input; reject BOOL through the same
  numeric-input boundary as cumulative sum.
- Normalize one positive or negative caller axis exactly once through the input Shape.
- Preserve the exact input Shape reference, data type, and `requiresGrad`; leave result layout
  unresolved and create a fresh unlabeled storage-free Tensor.
- Record one exact operation, ordered input `[input]`, one output descriptor, and producer output
  index zero.
- Define all four product modes and multiplicative-one exclusive boundaries without executing
  them.
- Preserve current cumulative-sum meanings, validation order, signatures, identity effects, and
  numerical-policy boundaries.
- Update every exact public-Tensor inventory from 194 to 196 atomically.
- Finalize affected Javadocs, Tensor and Compile API explanations, glossary terminology,
  capability status, task evidence, master plan, and roadmap through a separate clean-context
  documentation pass in the same overall change.

## Out of scope

- another scan kind such as cumulative minimum, maximum, logical scan, segmented scan, prefix
  count, rolling window, or multi-axis scan
- a no-axis, flattened, full-tensor, axis-list, named-axis, keep-dimensions, result-type, `out`, or
  destination form
- compatibility aliases for `CumulativeSumKind`, `CumulativeSumAttrs`, or
  `TensorCumulativeSumExpressions`; parallel product-only kind/attribute/helper types
- changing the public `cumSum` signatures, behavior, validation, result metadata, or ID effects
- BOOL truthiness, implicit conversion, promotion, widening, accumulation-type selection, or
  caller-selectable output type
- value or storage access, allocation, multiplication, prefix materialization, mutation,
  execution, parallel-scan algorithms, vectorization, fusion, or kernel choice
- a product-reduction change, another aggregate-reduction overload, or changes to
  `AggregateReductionKind.PROD`
- floating rounding, accuracy, payload, reproducibility, backend reassociation, or execution
  algorithm guarantees beyond the selected mathematical subset and existing product special-value
  contract
- gradient rules, backward kinds, autograd traversal, compiler adoption of the task-0023 product
  formula, saved values, accumulation, or optimization
- graph capture, compiled graph changes, planning, prepare, runtime, backend, engine, training,
  ONNX, conformance, or integration behavior
- Operation, OperationKind, OperationAttrs, OperationSignature, TensorDescriptor, Shape,
  DataType, TensorFactory, TensorProducer, TensorProvenance, or storage-foundation changes
- dependencies, Gradle/build changes, architecture or ADR changes, another module, unrelated
  refactors, task 0023F implementation, or a detailed 0023F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Task 0016H](0016h-cumulative-sum-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0018U1](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Adjoint expressibility result](../adjoint-expressibility-audit.md)
- [Task 0023D](0023d-public-fold-axis-and-dynamic-window-transforms.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent cumulative-scan meaning and public Tensor expression
  construction. It does not own scan execution, kernels, backend support, or autograd traversal.
- `Tensor` remains public mutable API state and is not an intermediate-representation node. Every
  successful call returns a fresh Tensor whose immutable producer metadata describes the request.
- One shared kind and attributes vocabulary is required because sum and product have the same
  one-input, one-output, shape-preserving scan structure. The kind selects the arithmetic identity;
  the attributes select only normalized axis, inclusion, and traversal direction.
- The atomic rename is a deliberate pre-stabilization contract normalization. No old public type
  or package-private helper alias remains, so no transitional state can represent the same scan
  through competing vocabularies.
- Each kind owns exactly one family signature accepting `CumulativeScanAttrs`, one ordered input,
  and one output.
- `exclusive == false` includes the current element. `exclusive == true` emits the identity before
  the first traversed element: additive zero for CUM_SUM and multiplicative positive one for
  CUM_PROD.
- `reverse == false` traverses increasing logical indices; `reverse == true` traverses decreasing
  logical indices. Returned positions remain in input index order.
- Integral CUM_PROD retains the exact INT32 or INT64 input type and means fixed-width two's-
  complement modular multiplication, consistently with ordinary integral product reduction.
- Floating CUM_PROD uses the selected product subset at each output: NaN propagates, zero times
  infinity is NaN, zero and infinity signs follow multiplication parity, and the empty/exclusive
  identity is positive one. No NaN payload, intermediate rounding, bitwise result, algorithm, or
  backend route is promised.
- A zero-length scan axis produces a zero-length result without an identity element because there
  is no output position. An exclusive non-empty scan emits the kind-specific identity at the first
  position in traversal order.
- Construction accepts dynamic and zero extents because it does not inspect values or require a
  known scan length. It retains the exact Shape reference.
- The result layout is unresolved even when the input layout is resolved. Scan construction does
  not promise a view or storage alias.
- `requiresGrad` is preserved as eligibility metadata. This task adds no gradient rule. Later
  compiler work may compose two exclusive product scans for zero-safe product adjoints.
- Validation is local and deterministic: input nullity, kind nullity at the helper boundary,
  numeric input eligibility, Shape access, one axis normalization, attributes construction, then
  descriptor/operation/producer/factory construction.
- Local failures occur before `TensorFactory.createDerived` and consume no Tensor identifier.
  Successful calls allocate exactly one fresh result identifier.
- Generic operation foundations remain unchanged; family-owned signatures enforce typed
  attributes and occurrence counts.
- Runtime hot paths never consume `Operation` or Tensor producer metadata. No compiler, prepare,
  runtime, or backend contract changes.
- Stop if implementation requires a third scan semantic type, another public method, another
  module, another dependency, or an architecture decision.

## Package impact

Existing package retained:

```text
io.github.pho001.synaptik.model.operation.scan
  Typed shape-preserving ordered scan meanings and immutable scan parameters.
```

Renamed public types:

- `CumulativeSumKind` -> `CumulativeScanKind`
- `CumulativeSumAttrs` -> `CumulativeScanAttrs`

Renamed package-private construction boundary:

- `TensorCumulativeSumExpressions` -> `TensorCumulativeScanExpressions`

No package is added. The scan package remains separate from aggregate reduction because scans
retain one output position for every input position.

## Required contract

### Shared semantic vocabulary

Create exactly:

```java
public enum CumulativeScanKind implements OperationKind {
    CUM_SUM,
    CUM_PROD
}
```

The enum has no additional project constant, alias, nested type, or operation-specific state.
Each constant returns the same immutable singleton signature list:

```java
OperationSignature.fixed(CumulativeScanAttrs.class, 1, 1)
```

`CUM_SUM` preserves the completed cumulative-addition meaning. `CUM_PROD` selects cumulative
multiplication. Enum order is part of the exact tested public surface.

### Shared attributes

Create exactly:

```java
public record CumulativeScanAttrs(
        int axis,
        boolean exclusive,
        boolean reverse) implements OperationAttrs
```

The record retains the existing validation and value semantics. It rejects a negative normalized
axis with `IllegalArgumentException` and exact message:

```text
axis must be non-negative: <axis>
```

It accepts zero through `Integer.MAX_VALUE`, preserves both booleans exactly, exposes explicit
documented accessors, and has no overload, factory, mode enum, sentinel, cache, or helper API.

### Product modes

For logical input `[2, 3, 4]`, CUM_PROD means:

| Exclusive | Reverse | Result |
|---|---|---|
| false | false | `[2, 6, 24]` |
| true | false | `[1, 2, 6]` |
| false | true | `[24, 12, 4]` |
| true | true | `[12, 4, 1]` |

The table defines logical meaning, not eager computation or a required execution algorithm.
Existing cumulative-sum examples and meanings remain unchanged.

### Public construction

`Tensor` retains:

```java
public Tensor cumSum(int axis)
public Tensor cumSum(int axis, boolean exclusive, boolean reverse)
```

and adds exactly:

```java
public Tensor cumProd(int axis)
public Tensor cumProd(int axis, boolean exclusive, boolean reverse)
```

Both short forms explicitly delegate with `exclusive=false` and `reverse=false`. Every method
delegates once to the shared helper with the exact matching kind.

The field-free final helper has one private zero-argument constructor and exactly these three
methods:

```java
static Tensor apply(
        Tensor input,
        CumulativeScanKind kind,
        int axis,
        boolean exclusive,
        boolean reverse)

private static void validateNumericInput(Tensor input)

private static Tensor create(
        Tensor input,
        Shape shape,
        CumulativeScanKind kind,
        CumulativeScanAttrs attrs)
```

`apply` null-checks `input` with message `input`, then `kind` with message `kind`, validates the
five numeric data types, retains the exact Shape, normalizes the axis once, constructs exact
attributes, and delegates to `create`. BOOL rejection retains exact message:

```text
input must have a numeric data type, but was BOOL
```

`create` builds an unresolved descriptor retaining exact type, Shape, and eligibility; an
`Operation(kind, attrs)`; exact ordered input `[input]`; one output descriptor; and exactly one
`TensorFactory.createDerived` result with no label or storage.

## Affected files

Production — exactly seven paths:

- delete `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumKind.java`
- delete `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumAttrs.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanKind.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanAttrs.java`
- delete `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressions.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeScanExpressions.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests — exactly nineteen paths:

- delete `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumSemanticsTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanSemanticsTest.java`
- delete `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeSumExpressionTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCumulativeScanExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxSemanticsTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation and planning — exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping: `ARCHITECTURE.md`, focused
architecture documentation, Training and Runtime APIs, aggregate-product contracts, TensorFactory,
producer/provenance foundations, architecture tests, Gradle, other modules, completed task
history, conformance, and integration tests.

## Maximum scope

This task may create, delete, rename, or modify exactly the 33 authorized paths above: seven
production paths, nineteen tests, and seven documentation/planning paths. The larger-than-normal
scope is deliberate because the two public methods activate thirteen independent exact Tensor
surface locks and the public scan-type rename must be atomic. It must not be used for unrelated
cleanup.

Stop if a thirty-fourth path, another type or test, another public method, a compatibility alias,
an existing foundation change beyond the named scan contracts, another module, dependency,
Gradle, architecture change, compiler adoption, execution behavior, or detailed task 0023F
specification is required.

## Acceptance criteria

- The old `CumulativeSumKind`, `CumulativeSumAttrs`, and `TensorCumulativeSumExpressions` types no
  longer exist in production or test imports, source, generated Javadoc, or current API prose.
- `CumulativeScanKind` contains exactly `CUM_SUM` and `CUM_PROD` in order and each exposes only the
  exact one-input/one-output `CumulativeScanAttrs` signature.
- `CumulativeScanAttrs` has exactly the three specified record components, validation order,
  message, accessors, and ordinary record value semantics.
- Existing `cumSum` overloads retain their public signatures and complete construction behavior.
- Exactly two `cumProd` overloads exist; no alias, static variant, or additional scan API appears.
- Default and all four explicit product modes construct exact normalized attributes and exact
  CUM_PROD metadata.
- All five numeric input types are accepted; BOOL is rejected before axis normalization and ID
  allocation.
- Positive, negative, scalar-invalid, dynamic, static, zero-extent, and extreme valid axes receive
  focused coverage with deterministic messages and no-ID failure evidence.
- Result descriptor, unresolved layout, exact Shape reference, type, eligibility, label, storage,
  freshness, one-input producer, output index, and ID exhaustion behavior match the specified
  contract.
- Resolved input layout, label, provenance, and host storage are not inspected or mutated.
- Integral modular product and floating special-value/identity semantics are documented without
  execution or backend claims.
- `OperationSignatureTest` and the scan semantic test prove both kinds' typed pairing and fixed
  occurrence counts.
- All thirteen exact public Tensor surface locks move from 194 to 196 and no unrelated method is
  added or removed.
- Focused validation and exactly one final model suite pass after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes all affected Javadocs, Tensor and
  Compile APIs, glossary, capability/task/master/roadmap status, links, examples, and formatting
  without repeating successful Java tests unless executable behavior changes or a concrete risk
  is recorded.
- Final inventory contains exactly the authorized 33 paths; `git diff --check` passes; 0023E is
  Complete only after both passes; 0023F and later tasks remain Draft without detailed specs.
- Architecture, dependencies, Gradle, other modules, compiler/runtime/backend behavior, and
  completed unrelated contracts remain unchanged.

## Tests / validation

Implementation-focused tests while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.scan.CumulativeScanSemanticsTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.normalization.SoftmaxSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorCumulativeScanExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Exactly one final model suite after executable Java stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation agent records exact suite/test counts and hands the diff plus evidence to the
documentation agent. The documentation pass does not rerun successful Java tests unless it
changes executable Java behavior or records a concrete reason.

Documentation pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation agent also validates current Java 26 API reflection and generated Javadoc for
the two kinds, shared record, four public Tensor methods, exact 196-method Tensor surface, absence
of old scan names, local Markdown links/anchors/fences/final newlines, exact 33-path scope,
synchronized status, and absence of a detailed 0023F specification.

Repository-wide tests are deferred to the next recorded model capability checkpoint or CI because
this task changes only one module and no dependency, build, architecture, or cross-module
contract.

## Dependencies

- Tasks 0016G and 0016H for the completed cumulative-sum semantics and public construction being
  normalized rather than behaviorally changed.
- Tasks 0018K and 0018L for family-owned signatures and exact producer/provenance construction.
- Tasks 0018U and 0018U1 for selected numeric input domains and integral modular product policy.
- Task 0023 and its completed audit matrix for the zero-safe product-adjoint proof.
- Task 0023D is the completed immediately preceding frontier.

## Follow-up tasks

- Task 0023F remains Draft for a same-occurrence attention weights output.
- Task 0024 remains Draft and depends on completion of every selected task-0023 follow-up.
- Later compiler work may construct prefix/suffix CUM_PROD expressions for product adjoints and
  owns traversal, saved-value lifetime, accumulation, optimization, and policy-deferred gradient
  boundaries.
- Later planning/prepare/backend work owns implementation algorithms, reassociation choices,
  kernels, execution, accuracy validation, and performance.

## Architecture impact

Expected impact: None.

This task normalizes and extends backend-independent model semantics inside `modules/model`. It
does not change ownership, dependencies, lifecycle stages, runtime hot-path inputs, or backend
selection. Stop and report if implementation reveals a required architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0001/0002/0005/0006/0013/0016G/0016H/0018K/0018L/0018U/0018U1/
0023/0023D, task 0023E, Tensor and Compile APIs, glossary, and every affected/review-only
source/test named by task 0023E in full.

Implement task 0023E exactly inside its 33 authorized paths. Atomically replace the sum-only scan
type/helper names with the shared CUM_SUM/CUM_PROD scan family, preserve both public cumSum
contracts, and add exactly two public cumProd overloads with the selected numeric, mode, Shape,
descriptor, producer, provenance, and ID behavior. Add no compatibility alias, other scan,
gradient/compiler adoption, value execution, dependency, Gradle, architecture change, or later
task. Stop on architecture, dependency, completed-contract, validation-order, affected-file, or
maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023E Complete until both passes and every acceptance criterion succeed. Leave 0023F
and every later task Draft without a detailed specification.
```

## Local decisions

- Use one normalized scan family rather than add duplicated product-only kind, attributes, and
  helper contracts.
- Perform the public semantic-type rename atomically without aliases because the API is still
  pre-stabilization and a transition would create two representations for one scan family.
- Keep the two public methods per arithmetic kind for symmetry and preserve the existing
  inclusive-forward short form.
- Reuse the current five-type numeric domain and exact-type result contract.
- Select integral modular multiplication and the existing floating product special-value meaning;
  leave algorithm, payload, rounding, and backend accuracy outside model construction.
- Retain shape-preserving unresolved-layout metadata and one fresh producer occurrence rather than
  introducing a view, eager scan, or output carrier.

## Known limitations

- This task constructs metadata and executes no cumulative sum or product.
- Floating bitwise results, NaN payloads, intermediate rounding, and cross-backend reproducibility
  are not selected here.
- Product adjoints at NaN, infinity, and other policy-deferred derivative boundaries remain later
  compiler-policy work even though zero-safe regular-domain composition becomes expressible.
- Only one normalized axis is supported. Multi-axis product reduction remains composition through
  reshape/permute and one scan axis.
- No compatibility aliases preserve the old public semantic type names.

## Validation evidence

- Implementation context `/root/implement_0023e` ran the focused command in this task after
  executable Java stabilized. It passed 44 tests across the five selected suites: 8 scan-semantic,
  5 operation-signature, 9 softmax-semantic, 7 cumulative-scan-expression, and 15 Tensor tests,
  with zero failures, errors, or skips. That context then ran exactly one final
  `./gradlew :modules:model:test`; it passed 1,008 tests across 126 suites with zero failures,
  errors, or skips.
- Documentation context `/root/implement_0023e/docs_0023e` reused both successful Java-test runs.
  It changed only Javadocs and Markdown after that evidence, introduced no executable Java
  behavior, and therefore did not repeat the tests. It applied the General, API/Javadoc, Planning,
  and Example documentation profiles. It independently reviewed the architecture and planning
  contracts, actual combined diff, final scan source/tests, public Tensor integration, exact
  surface locks, Tensor/Compile/Training/Runtime APIs, glossary, capabilities, master plan, and
  roadmap.
- The documentation pass finalized Javadocs for `CumulativeScanKind`, `CumulativeScanAttrs`,
  `TensorCumulativeScanExpressions`, and the four affected `Tensor` methods. It finalized all seven
  authorized documentation/planning paths. The Tensor API documents all four product modes for
  `[2, 3, 4]`, multiplicative positive-one exclusive boundaries, zero-length axes, exact-width
  integral modular multiplication, and selected floating special values without claiming an
  algorithm, rounding, payload, gradient, compiler, runtime, backend, or execution behavior. The
  Compile API distinguishes current model expressibility from planned compiler adoption.
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits with `BUILD SUCCESSFUL`; both
  `compileJava` and `javadoc` executed. Generated pages exist for `CumulativeScanKind`,
  `CumulativeScanAttrs`, and `Tensor`; targeted `rg` checks found `CUM_PROD`, both `cumProd`
  overloads, zero-length-axis, modular-multiplication, and zero-times-infinity wording. A targeted
  old-name scan found no `CumulativeSumKind`, `CumulativeSumAttrs`, or
  `TensorCumulativeSumExpressions` in current production/test source, generated Javadoc, or current
  API prose.
- `javac --release 26 -cp modules/model/build/classes/java/main -d
  /tmp/cumulative-scan-api-check /tmp/CumulativeScanApiCheck.java` and
  `java -cp /tmp/cumulative-scan-api-check:modules/model/build/classes/java/main
  CumulativeScanApiCheck` passed. The Java 26 reflection check confirmed enum order
  `[CUM_SUM, CUM_PROD]`; exact record components `axis:int`, `exclusive:boolean`, and
  `reverse:boolean`; one exact `CumulativeScanAttrs` one-input/one-output signature for each kind;
  exactly 196 declared public Tensor methods; exactly the four public `cumSum`/`cumProd` overloads;
  and absence of all three old classes. A matching `javap -public` check showed only the two enum
  constants, shared record constructor, and four Tensor signatures.
- An initial interactive Java 26 JShell check printed the same enum, record, 196-method,
  four-overload, and old-class-absence results, but JShell failed while flushing its user-history
  preferences and did not terminate cleanly in the sandbox. The standalone `javac`/`java` check
  above replaced that environment-sensitive helper and exited successfully. An initial targeted
  generated-Javadoc `rg` also used an inapplicable module-qualified output prefix and failed with
  missing paths; the corrected scan used the actual generated paths and passed. Neither helper
  failure changed a repository file or invalidated the successful checks.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 209 Markdown files, 3,495 local links, 205
  local anchors, 2,634 fence markers, final newlines, and trailing whitespace. The checker covered
  tracked and untracked Markdown together. Targeted status checks confirmed 0023E is Complete in
  this task, the master plan, and roadmap; 0023F and 0024 remain Draft; and no detailed 0023F task
  specification exists.
- The combined output of `git diff --name-only` and
  `git ls-files --others --exclude-standard`, deduplicated and sorted, contains exactly the 33
  authorized paths: seven production paths, nineteen test paths, and seven documentation/planning
  paths. `git diff --check` passed.
- No-change review concluded that Training and Runtime APIs remain accurate because they expose no
  current scan, gradient, compilation, or execution promise. `ARCHITECTURE.md`, focused
  architecture/ADR/tests, aggregate-product contracts, `TensorFactory`, producer/provenance
  foundations, Gradle/dependencies, other modules, backend conformance, and integration tests also
  remain unchanged and accurate. The change is model-owned semantic vocabulary and storage-free
  expression construction only; it changes no module boundary, dependency, graph capture,
  numerical algorithm, gradient rule, runtime state, backend support, or execution contract.

## Implementation notes

- Atomically replaced `CumulativeSumKind`, `CumulativeSumAttrs`, and
  `TensorCumulativeSumExpressions` with the shared cumulative-scan names and no compatibility
  aliases. `CumulativeScanKind` contains exactly `CUM_SUM` and `CUM_PROD`, in order, and both use
  exact `CumulativeScanAttrs` one-input/one-output signatures.
- Preserved the two public `cumSum` overloads and their construction behavior. Added exactly two
  public `cumProd` overloads, with the short form explicitly selecting inclusive forward mode.
- Retained the five-type numeric domain, exact Shape/type/eligibility metadata, unresolved result
  layout, fresh unlabeled storage-free identity, one-input producer/provenance, output index zero,
  validation order, deterministic failures, and ID behavior. No executable Java changed after the
  recorded final model suite.
- Updated every exact public Tensor surface lock from 194 to 196. No task-0023F specification or
  later implementation was created.

## Completion summary

- Completed changes: normalized the cumulative-scan semantic/helper family, preserved cumulative
  sum, and added cumulative product with four modes and selected integral/floating boundaries.
- Files changed or created: exactly the authorized seven production, nineteen test, and seven
  documentation/planning paths; no additional repository path changed.
- Tests and validation: focused 44 tests and one final 1,008-test/126-suite model run passed in the
  implementation context; model Javadoc, Java 26 reflection/`javap`, generated-Javadoc scans,
  Markdown validation, old-name absence, 196-method/four-overload surface checks, exact 33-path
  inventory, status checks, and `git diff --check` passed in the documentation context.
- Documentation-agent review: `/root/implement_0023e/docs_0023e` completed the required independent
  General, API/Javadoc, Planning, and Example profile pass without duplicating Java tests.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, this task, master plan,
  and roadmap now describe current cumulative-scan construction and its deliberate non-claims.
- Javadoc review: all affected scan-family and Tensor contracts are complete and generated cleanly.
- Glossary impact: normalized cumulative-scan terminology replaces the removed sum-only type names.
- Unresolved issues: None.
- Follow-up required: None. Task 0023F remains the next concise Draft frontier.

Status: Complete
