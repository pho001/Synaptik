# Task 0019A2: One-Hot Encoding

## Status

Complete

## Goal

Add one first-class, one-input model semantic and exactly one public `Tensor` expression:

```java
public Tensor oneHot(long depth)
```

The receiver is an INT32 or INT64 Tensor of logical indices. A successful call appends one
positive static trailing axis of extent `depth` and produces a non-differentiable BOOL Tensor.
For each input index `i`, position `i` on the corresponding trailing result axis is `true` and
every other position is `false`.

One-hot is a first-class semantic because the current metadata-only expression vocabulary cannot
derive it without inspecting index values. An eager range/comparison composition would also
allocate a depth-sized host Tensor through `TensorFactory` and obscure the single rank-changing
logical operation that compiler capture must eventually see.

## Rationale and mental model

One input position becomes one trailing logical row:

```text
indices Shape [d0, ..., dn] + static depth D
    -> BOOL result Shape [d0, ..., dn, D]
```

For conceptual INT64 indices `[2, 0, 1]` and depth `3`:

```text
input:  [2, 0, 1]
result: [[false, false, true],
         [true,  false, false],
         [false, true,  false]]
```

The example defines exact logical values but model construction creates only metadata. It does
not read the three input values or allocate result storage. A scalar index `2` with depth `4`
produces Shape `[4]` and conceptually `[false, false, true, false]`.

## Scope

- Add public `OneHotKind.ONE_HOT` in the existing operation-index package.
- Add public immutable `OneHotAttrs(long depth)` with intrinsic positive-depth validation.
- Give the kind exactly one `OneHotAttrs`, one-input, one-output signature.
- Add one field-free package-private `TensorOneHotExpressions` construction helper.
- Add exactly `public Tensor oneHot(long depth)` to `Tensor`; the receiver is the sole ordered
  producer input and supplies indices rather than values to encode.
- Accept exactly INT32 and INT64 receiver metadata.
- Append `new StaticDimension(depth)` after every exact receiver Dimension reference.
- Produce BOOL, `requiresGrad=false`, unresolved layout, no label, and no host storage.
- Record one fresh producer, output index zero provenance, and one fresh Tensor ID with no
  intermediate Tensor.
- Define exact false/true logical values and execution-time index validity while preserving the
  construction-time no-value-inspection boundary.
- Add focused semantic/expression tests and update every current global kind/signature/public API
  inventory affected by the new kind or method.
- Finalize Javadocs, Tensor API, Compile API, glossary impact, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.

## Out of scope

- an alias, static Tensor form, overload, primitive-array convenience, initializer, factory
  method, or eager result
- configurable axis, dynamic/Tensor depth, depth zero, on/off values, output data type, ignore
  index, negative wrapping, clamping, default row, all-false invalid row, or sparse representation
- composition through `TensorFactory`, range, comparison, broadcast, cast, scatter, or any
  materialized input/result values
- labels, storage allocation, resolved layout, input mutation, or reuse of the receiver ID
- gradient rules, straight-through estimation, autograd construction, or training behavior
- compiler capture, constant folding, bounds proof, backend support/lowering/kernels, preparation,
  runtime checks, execution, conformance, or integration implementation
- changes to `DataType`, `ScalarValue`, `Shape`, `Dimension`, `StaticDimension`,
  `DimensionExpressions`, `TensorDescriptor`, `TensorFactory/createDerived`, `TensorProducer`,
  `TensorProvenance`, `Operation`, `OperationKind`, `OperationAttrs`, `OperationSignature`, existing
  indexing kinds/attributes/helpers, Gradle, another module, `ARCHITECTURE.md`, or focused
  architecture documentation
- dropout/RNG, sorting/top-K, linear, attention, or a detailed specification for 0019B or later

## Exact semantic and API contract

### Package, kind, attributes, and signature

Add public types with these exact declaration shapes (their bodies supply the validation and
signature behavior specified below):

```java
package io.github.pho001.synaptik.model.operation.index;

public enum OneHotKind implements OperationKind {
    ONE_HOT
}

public record OneHotAttrs(long depth) implements OperationAttrs {
    // Compact constructor specified below.
}
```

`OneHotKind.ONE_HOT` owns one stable immutable signature list containing exactly:

```java
OperationSignature.fixed(OneHotAttrs.class, 1, 1)
```

There is no parameterless or alternate signature. The ordered input is exactly `[indices]`; the
sole output is the dense BOOL encoding. `OneHotAttrs` retains the supplied primitive `long`
unchanged after validation and has no Tensor, Shape, axis, on/off value, output type, policy, or
execution state.

The canonical `OneHotAttrs` constructor rejects every `depth <= 0` with exactly:

```text
depth must be positive: <depth>
```

Positive values through `Long.MAX_VALUE` are structurally valid. `long` is selected because Shape
axis extents and `StaticDimension` already use `long`; using `int` would introduce an artificial
public range mismatch, while a Tensor depth would make the result rank known but its final extent
non-static and broaden the operation into dynamic configuration.

Place both semantic values in `io.github.pho001.synaptik.model.operation.index`. One-hot consumes
logical indices and defines their coordinate-to-indicator meaning, so that existing cohesive
package owns it. Do not add a generic encoding package for two types or place public semantics in
the Tensor package.

### Public receiver role

Add exactly:

```java
public Tensor oneHot(long depth)
```

The receiver is named `indices` in the helper, is never mutated, and is the exact sole producer
input. Add no alias such as `onehot`, `toOneHot`, or `encodeOneHot`, no overload, and no static or
factory form. `Tensor.oneHot(depth)` delegates exactly once to
`TensorOneHotExpressions.apply(this, depth)`.

### Value semantics and index validity

For every logical receiver coordinate `p` whose eventual integral value is `i`, and every
trailing result coordinate `j` in `[0, depth)`, exact ideal meaning is:

```text
result[p..., j] = (i == j)
```

The off value is exact BOOL `false`; the selected on value is exact BOOL `true`. This is logical
equality, not a numerical approximation, probability, count, or floating encoding. There is no
rounding, accumulation, overflow, NaN, infinity, or signed-zero policy because accepted indices
are signed integral and the result is BOOL.

A valid executed operation requires `0 <= i < depth` for every index value. Negative and
out-of-range indices are invalid at execution: they do not wrap, clamp, select a default, or
produce an all-false row. Construction must not inspect host storage even when the receiver is an
eager constant. A later compiler may reject provably invalid captured constants; after dynamic
inputs are bound, backend preparation or its prepared executable must prove or check validity and
fail safely. The exact future failure status or exception belongs to the execution contract and
is not invented here. Runtime must execute prepared behavior without inspecting the original
`Operation`.

### Shape and empty cases

For receiver Shape `[d0, ..., dn]`, construct a new Shape whose ordered Dimensions are:

```text
[exact d0 reference, ..., exact dn reference, new StaticDimension(depth)]
```

- Preserve every existing Dimension object by reference, not merely by structural equality.
- Append exactly one fresh `StaticDimension(depth)`; do not rewrite or bind existing static,
  named dynamic, or expression Dimensions.
- Scalar receiver Shape `[]` produces rank-one Shape `[depth]`.
- A receiver with any zero static extent is valid and produces zero logical rows, for example
  `[0, N] -> [0, N, depth]`. There are no index values to validate in a zero-element input.
- Positive `depth` is required even for a zero-element receiver; depth zero is never an empty-row
  spelling.
- `Long.MAX_VALUE` depth is accepted structurally. Construction does not call
  `Shape.knownElementCount()` or reject a potential total-element-count overflow; later storage or
  execution layers own representability of actual materialization.

### Result metadata and provenance

Construct exactly one result descriptor:

```text
data type:     BOOL
shape:         input Dimensions plus StaticDimension(depth)
layout:        unresolved (Optional.empty())
requiresGrad:  false
```

The result has no label and no host storage. It does not inherit receiver type, gradient request,
layout, label, storage, or existing provenance. The descriptor's `requiresGrad=false` follows the
current BOOL non-differentiability contract and does not define a gradient rule.

Create exactly one operation `new Operation(OneHotKind.ONE_HOT, attrs)` and delegate exactly once
to `TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(indices))`.
Every successful call creates one fresh single-output `TensorProducer`, one
`TensorProvenance(producer, 0)`, and one newly allocated Tensor ID. The producer retains exact
ordered input `[indices]`, exact operation/attributes, and the sole exact descriptor. There is no
range Tensor, comparison Tensor, cast, broadcast, intermediate producer, grouped output, or
second ID. Repeated valid calls return identity-distinct results and producers while leaving the
receiver unchanged.

### Validation order, exact messages, and ID effects

`TensorOneHotExpressions.apply(indices, depth)` performs exactly this order:

1. `Objects.requireNonNull(indices, "indices")`;
2. read the receiver descriptor and require exact INT32 or INT64 data type;
3. construct `new OneHotAttrs(depth)`, thereby requiring positive depth;
4. build the appended result Shape and BOOL descriptor;
5. construct the exact operation;
6. delegate once to `TensorFactory.createDerived`.

Use these exact task-owned failures:

```text
oneHot indices data type must be INT32 or INT64: <type>
depth must be positive: <depth>
```

Null helper input fails with exact message `indices`. Type rejection precedes depth validation;
tests must cover a non-index receiver combined with non-positive depth to lock that order. Depth
validation precedes Shape, descriptor, operation, producer, provenance, and ID construction.

Every null, type, or depth failure consumes no Tensor ID. All locally controlled validation and
metadata construction completes before the sole factory delegation. Successful construction
consumes exactly one ID. Identifier exhaustion retains the existing exact message
`tensor identifier space exhausted`; there is no rollback or alternative allocation path.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains wholly inside model-owned operation semantics, public Tensor metadata
  construction, and their documentation/planning records.
- `Tensor` remains public mutable API state and is not graph IR. The producer remains pre-capture
  occurrence identity and does not add graph-local IDs to Tensor.
- `Operation` expresses backend-independent one-hot meaning and gains no backend support, route,
  kernel, storage, device, runtime, or prepared-execution state.
- Model construction may validate local type/depth/Shape metadata but must not read index values.
- Compiler owns capture, graph-wide validation, constant analysis, canonicalization, and autograd.
  Backend prepare owns lowering, bounds strategy, specialization, and kernel choice. Runtime
  executes prepared schedules only and does not consume `Operation` on the hot path.
- Package direction remains `model.tensor -> model.operation.index`, datatype, shape, and layout.
- No architecture, dependency, lifecycle, module-boundary, build, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.index`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`

Packages added or changed:

- No package is added. The existing operation-index package gains the one-hot kind and attributes;
  the existing Tensor package gains one field-free helper and one public facade method.

Type placement:

- `io.github.pho001.synaptik.model.operation.index.OneHotKind` — public backend-independent
  coordinate-to-indicator semantic identity and family-owned signature.
- `io.github.pho001.synaptik.model.operation.index.OneHotAttrs` — public immutable positive static
  depth parameter for that semantic.
- `io.github.pho001.synaptik.model.tensor.TensorOneHotExpressions` — package-private owner of local
  receiver/depth validation, exact Shape/descriptor construction, and derived provenance.
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent API owner.

Tests mirror production packages because they inspect exact package-private helper and semantic
surfaces.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/OneHotAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/OneHotKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorOneHotExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/index/OneHotSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorOneHotExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
  — add `OneHotKind`/`OneHotAttrs` to the existing exhaustive production-kind signature matrix.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — add the exact
  public name/signature and change the public method count from 161 to 162.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — change only the shared total public Tensor method count from 161 to 162.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — change only the shared total public Tensor method count from 161 to 162.

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless final implementation makes a current statement inaccurate: Training API;
DataType/ScalarValue, Shape/Dimension/StaticDimension/DimensionExpressions, TensorDescriptor,
TensorFactory/createDerived, producer/provenance, existing indexing kinds/helpers, public factory
inventories, architecture/ADRs/tests, conformance/integration, Gradle, and other modules.

The existing exhaustive inventories discovered before planning are the production-kind signature
matrix in `OperationSignatureTest` and the public Tensor inventories/counts in `TensorTest`,
`TensorBinaryArithmeticTest`, and `TensorMatmulExpressionTest`. `OperationKindTest` tests only the
interface contract, `OperationAttrsTest` tests only the marker surface, and `TensorFactoryTest`
tests the unchanged factory surface; none requires a change.

## Maximum scope

At most four production, six test, and seven documentation/planning files: exactly 17 paths.
`Tensor.java` changes only for the one import/reference needed by its class Javadoc if applicable,
the exact method, and complete Javadoc. `OperationSignatureTest` changes only for the exhaustive
one-hot family entry/imports. The three historical public API inventory tests change only for the
new exact name/signature/count as described above.

The cohesive 17-path exception is justified because the semantic kind, intrinsic attributes,
public expression, focused tests, and already-global inventories must land together in one
compilable state; splitting would temporarily make the exhaustive signature or public API
contracts false. Stop before an eighteenth path, another production/test type, an existing
indexing helper edit, a factory edit, cross-module work, or any architecture/build change.

## Javadoc and explanatory documentation requirements

- Give `OneHotKind`, `OneHotAttrs`, `TensorOneHotExpressions`, and `Tensor.oneHot` meaningful,
  detailed Javadocs covering receiver role, exact value formula, type/depth/Shape contract,
  Dimension identity, scalar/empty/maximum-depth cases, result metadata, validation order/messages,
  producer/provenance/ID effects, index-value boundary, and unsupported layers/options.
- Every constructor/method input receives `@param`, every non-void method receives `@return`, and
  expected null/argument/exhaustion failures receive `@throws` with exact relevant conditions.
- Add the `[2, 0, 1]`, depth-three newcomer example and scalar example to Tensor API. Label values
  conceptual and make clear that construction does not inspect storage or execute encoding.
- Update Compile API only to add one-hot to the current model-expression inventory and preserve
  planned capture/constant-validation/execution boundaries; do not claim compiler support.
- Review Training API and record no change: this task adds fixed `requiresGrad=false` BOOL
  metadata but no gradient rule, parameter role, optimizer, training graph, or execution contract.
- Add or revise a glossary entry for one-hot encoding because it is a reusable public semantic;
  distinguish its trailing class/depth axis from indexing selection and state invalid index
  behavior without implying current execution.
- Update capabilities to replace the provisional all-false-invalid-row statement with the final
  invalid-at-execution policy and to describe the implemented contract only after completion.
- Keep task, master plan, and roadmap synchronized. Record reasoned no-change conclusions for all
  reviewed unchanged contracts.

## Acceptance criteria

- Exactly one `ONE_HOT` kind and one `OneHotAttrs(long depth)` record exist in the operation-index
  package with exact one-input/one-output signature.
- Attributes accept every positive `long`, including `Long.MAX_VALUE`, reject zero/negative depth
  with the exact message, retain the value, and expose no extra instance state or API.
- Exactly one public `oneHot(long)` exists and the public Tensor method count is 162.
- Only INT32 and INT64 receiver metadata succeeds; every other current DataType fails with the
  exact message and before depth validation or ID allocation.
- Result Shape preserves every input Dimension reference in order and appends one new
  `StaticDimension(depth)`, covering scalar, static, zero, named dynamic, and expression cases.
- Result metadata is exact BOOL, `requiresGrad=false`, unresolved layout, no label, and no storage;
  the receiver is unchanged.
- Every success has one exact `ONE_HOT` operation, one producer with exact input `[indices]`,
  provenance output index zero, and one fresh ID. There are no intermediate tensors or additional
  IDs.
- Construction reads no values. Tests establish semantic metadata without allocating storage or
  pretending to execute the conceptual value example.
- Execution requires every index in `[0, depth)`; negative/out-of-range values are documented as
  invalid with no wrap/clamp/default/all-false behavior, while exact future enforcement remains
  outside this task.
- Scalar, zero-element, `Long.MAX_VALUE` depth, repeated calls, input non-mutation, validation
  order/messages, no-ID failures, identifier exhaustion, exact semantic surfaces, and immutable
  signature/attributes behavior are covered.
- No factory initializer/composition, configurable option, sparse form, gradient/compiler/backend/
  runtime implementation, dependency, Gradle, architecture, focused-architecture, or other-module
  work lands.
- Every existing exhaustive global signature/public API inventory is updated now; unrelated
  marker/interface/factory inventories remain unchanged for the recorded reasons.
- Focused tests, exactly one final model test after Java stability, final model Javadoc,
  documentation/link/scope/status/formatting checks, and `git diff --check` pass.
- A separate clean-context documentation pass finalizes all authorized Javadocs/documentation,
  reuses the final Java evidence, and records its context and no-change conclusions.
- Tasks 0019, 0019A, and 0019A1 remain Complete; 0019A2 becomes Complete only after all evidence.
  Tasks 0019B–0019E and every later task remain Draft without detailed specifications.

## Tests / validation

Required focused command during implementation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.index.OneHotSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorOneHotExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Focused tests cover exact kind/record/helper/public surfaces; signature immutability and global
coverage; all accepted/rejected data types; all depth boundaries; validation order/messages and ID
effects; scalar/static/zero/dynamic/expression Shapes with exact Dimension identity; exact BOOL
descriptor/provenance/freshness; one-ID success; receiver non-mutation; conceptual value contract
without eager execution; and absence of options, composition, or storage inspection.

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

The mandatory separate clean-context documentation pass receives and reuses that successful final
Java evidence. It does not rerun Java tests unless it changes executable Java behavior, evidence
is stale/missing, or it records a concrete cross-check risk. After final Javadoc edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminology, examples,
generated Javadoc, final newlines, trailing whitespace, exact 17-path scope, package placement,
the 162-public-method inventory, the exhaustive operation-signature matrix, one Ready/active model
frontier, coherent dependencies, synchronized task/master/roadmap status, and absence of a
0019B-or-later detailed specification.

Repository-wide validation is deferred to the selected-modern-operations capability checkpoint
after task 0022 and to CI. This is task-tier, single-module semantic metadata with no dependency,
architecture, shared-build, or cross-module change.

## Dependencies

- Completed tasks 0001 and 0002 supply exact INT32/INT64/BOOL metadata, `long` Shape extents,
  `StaticDimension`, scalar/empty Shapes, and exact Dimension retention.
- Completed tasks 0005–0007 and 0018K supply operation kinds/attributes, exact family-owned
  signatures, occurrence validation, and descriptor eligibility.
- Completed tasks 0011–0013 and 0018L supply public Tensor state, central derived construction,
  fresh identity, one-producer/output-index provenance, and storage-free result contracts.
- Completed tasks 0018N and 0018O clarify that this operation neither uses `ScalarValue` nor
  changes canonical indexing/bounds policy.
- Completed tasks 0019, 0019A, and 0019A1 are the preceding table frontiers and remain unchanged.

## Follow-up tasks

- 0019B remains Draft and solely owns explicit graph RNG and dropout.
- 0019C remains Draft for sorting and top-K; 0019D remains Draft for `linear`; 0019E remains Draft
  for scaled dot-product attention.
- Task 0023 may later define compiler-generated backward semantics if an explicit one-hot gradient
  need is selected; this task itself defines no gradient.
- Compiler, backend conformance, backend/runtime, and integration work later owns capture,
  constant analysis, safe bounds enforcement, lowering, materialization, and execution.

Do not create another detailed task specification during 0019A2 implementation.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, dependency, lifecycle, focused-architecture,
cross-module, factory, or scope change, stop and report the conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed DataType, Shape,
Tensor, factory/createDerived, provenance, signature-hardening, ScalarValue, indexing/Gather,
reset, 0019, 0019A, and 0019A1 tasks, current affected source/tests/API inventories, and task
0019A2.

Implement docs/planning/modules/model/tasks/0019a2-one-hot-encoding.md exactly inside its 17
authorized paths. Preserve current contracts. Stop on architecture uncertainty, scope overflow,
another type/test/document need, any factory/existing-indexing-helper change, or cross-module work.
Do not commit or push.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same overall
change. That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and
documentation checks while reusing successful Java evidence. Synchronize status only after all
criteria pass; keep 0019, 0019A, and 0019A1 Complete and 0019B/later Draft without specs.
```

## Documentation-agent handoff

Give the separate clean-context documentation agent this task, the complete implementation diff,
exact focused/final model evidence and whether Java changed afterward, the selected value/Shape/
type/depth/provenance/bounds policies, the seven authorized documentation paths, and required
Javadoc, Markdown, scope, and status validation.

The documentation agent independently reads AGENTS, architecture, documentation rules and the
General/API-Javadoc/Planning/Example profiles, this task, actual production/tests/generated
Javadoc, Tensor/Compile/Training APIs, glossary, capabilities/master/roadmap, and directly related
DataType/Shape/operation/signature/descriptor/factory/producer/provenance/indexing contracts. It
finalizes Javadocs and explanatory documentation, checks conceptual examples, and records reasoned
no-change conclusions for Training API, foundational and existing indexing contracts,
architecture/ADRs/tests, conformance/integration, Gradle, other modules, and Draft follow-ups.

It does not repeat successful Java tests unless executable Java changes, evidence is stale or
missing, or a concrete recorded risk requires a rerun. It records its clean-context identifier,
reused evidence, files/topics reviewed, commands/results, glossary impact, limitations, and
unresolved issues.

## Local decisions

- Kept one-hot as one first-class `ONE_HOT` occurrence rather than composing range, comparison,
  broadcast, cast, or factory allocation. This preserves the exact model meaning and creates no
  intermediate Tensor.
- Chose strict eventual bounds: every executed index must satisfy `0 <= i < depth`. Negative and
  out-of-range values are invalid and do not wrap, clamp, select a default, or produce an
  all-false row. Model construction remains value-blind.
- Retained `long` depth through `Long.MAX_VALUE` and avoided element-count calculation during
  construction. Storage and execution layers remain responsible for materializability.
- Preserved every input Dimension object by reference and appended one fresh
  `StaticDimension(depth)`, including for scalar, zero-element, dynamic, and expression Shapes.
- Fixed BOOL and `requiresGrad=false` as result metadata. This records non-differentiable model
  metadata and does not introduce a gradient rule.

## Known limitations

- Model construction cannot validate receiver values because it deliberately does not inspect
  storage; safe invalid-index rejection remains an execution-layer obligation.
- A structurally valid positive depth and Shape need not be materializable by a particular host or
  backend. This task records logical metadata only.

## Validation evidence

Planning context `/root/plan_0019a2` read the required architecture, documentation/planning,
capability, completed-task, API, source, and test contracts; inspected the current global
signature/public API inventories; and refined the invalid-index policy before implementation.

Implementation context `/root/task_0019a2_implementation` recorded these final Java results before
the documentation pass, and executable Java did not change afterward:

- The required focused Gradle command selecting `OneHotSemanticsTest`,
  `TensorOneHotExpressionTest`, `OperationSignatureTest`, and `TensorTest` passed 35 tests across
  the four suites: 6, 9, 5, and 15 tests respectively, with zero failures, errors, or skips. The
  exact command remains recorded under Tests / validation.
- Exactly one final `./gradlew :modules:model:test` passed 780 tests across 96 suites with zero
  failures, errors, or skips.
- Targeted bytecode inspection confirmed one `long` field/constructor/accessor for `OneHotAttrs`,
  the exact `OneHotKind` enum and stable signature list, and a field-free
  `TensorOneHotExpressions` with a private constructor and package-private
  `apply(Tensor, long)`.

Documentation context `/root/task_0019a2_implementation/docs_finalize_0019a2` independently read
the architecture contract; documentation and planning rules; General, API/Javadoc, Planning, and
Example profiles; task, master plan, roadmap, capabilities, APIs, glossary, final source/tests,
generated Javadoc, and directly related model contracts. It finalized the four production
Javadocs, Tensor API, Compile API, glossary, capabilities, and planning records. It reused the
successful Java evidence because it changed comments and documentation only. Final documentation
validation passed:

- `./gradlew :modules:model:javadoc` generated model Javadoc successfully after the final edits.
- Generated pages for `OneHotAttrs`, `OneHotKind`, and `Tensor.oneHot(long)` were inspected for the
  formula, Shape/type/depth/provenance boundaries, parameters, returns, and failures.
- Local Markdown targets and heading anchors, balanced fences, terminology and conceptual
  examples, final newlines, trailing whitespace, and `git diff --check` passed.
- The first three invocations of the temporary local Markdown checker did not complete a valid
  documentation check: the first exposed unavailable `Array#filter_map` in the installed Ruby,
  and the next two exposed an inaccurate slash-to-anchor slug rule. The checker was corrected for
  compatibility and GitHub-style slugs; its final invocation passed all seven authorized
  documentation files with no missing target or anchor.
- The final diff contains exactly 17 authorized paths: four production, six test, and seven
  documentation/planning files. Package placement is exact, Tensor retains 162 public methods,
  and the exhaustive signature matrix includes `OneHotKind`/`OneHotAttrs`.
- Task/master/roadmap statuses and dependencies are synchronized: 0019, 0019A, 0019A1, and 0019A2
  are Complete; 0019B and every later task remain Draft; no 0019B-or-later detailed specification
  exists, and no model task is currently Ready.

Training API required no change because this task adds fixed non-differentiable BOOL metadata but
no gradient rule, parameter role, optimizer, training graph, or execution behavior. DataType,
ScalarValue, Shape/Dimension/StaticDimension/DimensionExpressions, TensorDescriptor,
TensorFactory/createDerived, TensorProducer/TensorProvenance, Operation/OperationKind/
OperationAttrs/OperationSignature, and existing indexing/Gather contracts required no change
because implementation reused their existing contracts without modifying them. Architecture,
ADRs, architecture tests, conformance/integration tests, Gradle, and other modules required no
change because the task adds model-owned semantic metadata and public construction only, with no
dependency, lifecycle, backend, runtime, build, or cross-module behavior.

## Implementation notes

- Added exact public `OneHotAttrs`, `OneHotKind.ONE_HOT`, field-free construction helper, and
  `Tensor.oneHot(long)` within existing operation-index and Tensor packages.
- Added focused semantic/expression coverage and updated the exhaustive production-kind signature
  matrix and all three current public Tensor inventories to 162 methods.
- Final documentation clearly distinguishes conceptual `[2, 0, 1]`/depth-three and scalar values
  from current metadata-only construction. Compile API lists the expression without claiming
  compiler support, and the glossary distinguishes one-hot encoding from Gather selection.
- No executable Java changed during the documentation pass, so the final model test evidence
  remains current and was not duplicated.

## Completion summary

- Completed changes: Added first-class one-hot semantic metadata and exactly
  `Tensor.oneHot(long)`, with exact validation, Shape, BOOL metadata, provenance, identity, and
  future bounds contracts.
- Files changed or created: Exactly four production, six test, and seven documentation/planning
  files listed under Affected files.
- Tests and validation: Focused 35-test command and final 780-test model suite passed in the
  implementation context; final model Javadoc, documentation, scope, status, and whitespace
  checks passed in the documentation context.
- Documentation-agent review: Completed in clean context
  `/root/task_0019a2_implementation/docs_finalize_0019a2` using the required profiles and reused
  Java evidence.
- Documentation impact: Tensor API, Compile API inventory, capabilities, task, master plan, and
  roadmap are synchronized with the implemented metadata-only boundary.
- Javadoc review: Finalized for `OneHotAttrs`, `OneHotKind`, `TensorOneHotExpressions`, and
  `Tensor.oneHot(long)`; generated Javadoc passed inspection.
- Glossary impact: Added reusable one-hot encoding terminology and its distinction from Gather.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
