# Task 0016G: Cumulative-Sum Semantic Kind and Attributes

**Editorial reading key (later context, not a status update).**

A cumulative sum preserves every axis position and reports partial totals, unlike a reduction that removes a domain. The axis, inclusion flag and traversal flag are independent attributes. Exclusive omits the current position, and reverse changes visit order rather than reversing output coordinates. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current cumulative-scan modes](../../../../reference/tensor/fixed-scans.md#cumulative-scan-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [parameters](../../../../glossary.md#term-parameter) for a
single-[axis](../../../../glossary.md#term-axis) cumulative sum. The [model](../../../../glossary.md#term-model) must represent the normalized [scan](../../../../glossary.md#term-cumulative-scan) axis, whether the current
[element](../../../../glossary.md#term-element) is excluded, and whether traversal proceeds from the end of the axis, without retaining a
[Tensor](../../../../glossary.md#tensor), deriving a [descriptor](../../../../glossary.md#term-descriptor), reading values, or reporting backend support.

This task creates the complete semantic foundation consumed by task 0016H. It does not add public
`Tensor.cumSum` expression methods.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `CumulativeSumKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly one enum [constant](../../../../glossary.md#term-constant), `CUM_SUM`.
- Add one public `CumulativeSumAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly normalized
  non-negative `int axis`, `boolean exclusive`, and `boolean reverse` components, in that order.
- Reject negative stored [axes](../../../../glossary.md#term-axis) with the exact common normalized-axis failure contract.
- Preserve both booleans exactly and support all four exclusive/reverse combinations.
- Define typed composition of `CUM_SUM` with `CumulativeSumAttrs` without changing generic
  [`Operation`](../../../../glossary.md#operation) [validation](../../../../glossary.md#term-validation).
- Document inclusive/exclusive and forward/reverse meaning, including additive-zero boundary
  behavior for exclusive scans, without executing it.
- Document one logical input and shape-preserving ordered scan semantics as family context without
  storing arity, Shape, DataType, descriptor, or result state.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [scan glossary](../../../../glossary.md#term-cumulative-scan); [arity glossary](../../../../glossary.md#term-arity); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Add one focused same-package semantic-contract test.
- Add the cohesive `model.operation.scan` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize Javadocs, Tensor API semantic reference, glossary, task evidence, master plan, and
  roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.cumSum`, another Tensor method, overload, factory, builder, expression helper, or
  task-0016H implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- another scan such as cumulative product, minimum, maximum, logical scan, prefix count, segmented
  scan, rolling window, or parallel scan primitive
- input Tensor, Shape, axis normalization against rank, shape preservation implementation, result
  descriptor, label, identity, layout, storage, provenance, or <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived`
- input eligibility, output DataType, promotion, overflow, accumulation precision, BFLOAT16
  conversion, integer width, BOOL rejection, or local/graph-wide inference<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [precision glossary](../../../../glossary.md#term-precision); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- negative caller axis, an all-axes sentinel, multiple axes, axis collection, named axis, optional
  axis, or no-axis form
- value/storage access, allocation, copying, [materialization](../../../../glossary.md#materialization), mutation, [aliasing](../../../../glossary.md#term-aliasing), or execution
- parallel prefix algorithm, work partitioning, associativity optimization, numerical
  reproducibility, vectorization, fusion, cost, route, or kernel selection<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel))</span>
- empty-[axis](../../../../glossary.md#term-axis) execution, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), or numerical error policy
- [gradients](../../../../glossary.md#term-gradient), backward [scan](../../../../glossary.md#term-cumulative-scan) semantics, gradient rules, [autograd](../../../../glossary.md#autograd), [optimizer](../../../../glossary.md#term-optimizer), or training behavior
- operation factory, compatibility validator, registry, parser, visitor, string dispatch, arity or
  result-kind metadata, backend support, lowering, or executable behavior<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- changes to Operation foundations, aggregate/masked reduction contracts, DataType, Shape, Tensor,
  graph records, existing Java tests, dependencies, Gradle, architecture, or another module<span data-glossary-navigation="true"> ([masked reduction glossary](../../../../glossary.md#masked-reduction); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, conformance, integration,
  or a detailed task-0016H specification<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Task 0016F1](0016f1-masked-sum-and-mean-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.cumSum(axis)` and
`Tensor.cumSum(axis, exclusive, reverse)`. The short form means inclusive forward traversal. The
complete form represents all four direction/inclusion combinations. Public [axes](../../../../glossary.md#term-axis) may be negative,
output [shape](../../../../glossary.md#shape) and [data type](../../../../glossary.md#data-type--datatype) match the input, [BOOL](../../../../glossary.md#data-type--datatype) is rejected, and legacy execution evidence covers
floating, [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), non-[contiguous](../../../../glossary.md#term-contiguous) input, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu), and [Metal](../../../../glossary.md#term-metal) paths.

For one axis [vector](../../../../glossary.md#term-vector) `[1, 2, 3]`, the selected meanings are:

| Exclusive | Reverse | Semantic result |
|---|---|---|
| false | false | `[1, 3, 6]` |
| true | false | `[0, 1, 3]` |
| false | true | `[6, 5, 3]` |
| true | true | `[5, 3, 0]` |

These values explain [operation](../../../../glossary.md#operation) meaning only; this task computes none of them. Legacy [arity](../../../../glossary.md#term-arity),
family, cost, [fusion](../../../../glossary.md#term-fusion), result-kind, expression text, [backend](../../../../glossary.md#backend) facts, storage loops, [lowering](../../../../glossary.md#lowering), and
[kernel](../../../../glossary.md#kernel) routes are not copied. Task 0016H owns public [Tensor](../../../../glossary.md#tensor) [validation](../../../../glossary.md#term-validation), negative-axis
[normalization](../../../../glossary.md#term-normalization), [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance) construction, type eligibility, and freshness. [Compiler](../../../../glossary.md#term-compiler) and
backend work later own [graph](../../../../glossary.md#graph) behavior, [gradients](../../../../glossary.md#term-gradient), numerical execution, storage, and [kernels](../../../../glossary.md#kernel).

## Architecture constraints

- [Operation](../../../../glossary.md#operation) kinds and attributes are [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics owned by
  `modules/model`.
- `CumulativeSumKind.CUM_SUM` identifies one ordered cumulative-addition scan. It stores no axis,
  input, output, graph occurrence, executable behavior, or backend information.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [axis glossary](../../../../glossary.md#term-axis); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `CumulativeSumAttrs` is the complete attributes type. Its axis is already normalized and
  non-negative; a later Tensor expression validates and normalizes a caller axis against Shape.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape))</span>
- `exclusive == false` includes the current [element](../../../../glossary.md#term-element) in its output prefix. `exclusive == true`
  emits the accumulated prior elements and therefore emits additive zero at the first traversed
  position.
- `reverse == false` traverses from lower to higher indices. `reverse == true` traverses from
  higher to lower indices. Reverse changes traversal direction, not output indexing or Shape.
- Every exclusive/reverse combination is valid. There is no separate default attributes singleton,
  negative sentinel, mode enum, or kind per combination.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Cumulative sum has one logical input and preserves logical Shape positions as semantic family
  context. Those facts are not stored in the kind or attributes and are not implemented here.
- Generic `Operation` continues to validate only non-null kind and attributes. It must not discover
  families, enforce typed pairing, arity, axis bounds, DataType, result Shape, or backend support.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- Stable enum/record text is diagnostic, not serialization, ONNX, reflective dispatch, backend
  route, or kernel identity.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [dispatch glossary](../../../../glossary.md#term-dispatch); [backend route glossary](../../../../glossary.md#backend-route); [kernel glossary](../../../../glossary.md#kernel); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Package direction is `model.operation.scan -> model.operation` only. It must not depend on
  Tensor, datatype, shape, layout, storage, graph, compiler, planning, runtime, prepare, backend,
  or training packages.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation requires another semantic type/component, Tensor/Shape/DataType behavior,
  compatibility validation, dependency, or architecture change.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  generic immutable <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`Operation`](../../../../glossary.md#operation) composition.

Package added:

```text
io.github.pho001.synaptik.model.operation.scan
  Typed shape-preserving ordered scan meanings and immutable scan parameters.
```

The package is separate from `model.operation.reduction` because an [aggregate reduction](../../../../glossary.md#aggregate-reduction) removes
or contracts logical positions, while cumulative sum preserves one output position for every input
position and changes only each position's [dependency](../../../../glossary.md#term-dependency) prefix.

Type placement:

- `io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind` — public cumulative-sum
  semantic identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.model.operation.scan.CumulativeSumAttrs` — public immutable normalized
  axis, exclusion, and direction parameters.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `CumulativeSumSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum CumulativeSumKind implements OperationKind {
    CUM_SUM
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `CUM_SUM`.

The constant means one-input cumulative addition along the [normalized axis](../../../../glossary.md#normalized-axis) carried by
`CumulativeSumAttrs`. It does not define input eligibility, result [descriptor](../../../../glossary.md#term-descriptor), numerical policy,
[gradient](../../../../glossary.md#term-gradient) behavior, execution, or [backend](../../../../glossary.md#backend) availability.

### [Scan](../../../../glossary.md#term-cumulative-scan) attributes

Create exactly:

```java
public record CumulativeSumAttrs(
        int axis,
        boolean exclusive,
        boolean reverse) implements OperationAttrs
```

The record has exactly three components in that order, one public [canonical](../../../../glossary.md#term-canonical-form) constructor, explicit
documented `axis()`, `exclusive()`, and `reverse()` accessors, and record-generated `equals`,
[`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`. Add no overload, factory, builder, mode enum, field, nested type,
optional, sentinel, [cache](../../../../glossary.md#term-cache), or helper [API](../../../../glossary.md#term-application-programming-interface-api).

The canonical constructor rejects every negative [`axis`](../../../../glossary.md#term-axis) before ordinary record assignment with
`IllegalArgumentException` and exact message:

```text
axis must be non-negative: <axis>
```

Accept zero, positive values, and `Integer.MAX_VALUE` structurally. The record cannot prove that
the axis exists for an eventual input [Shape](../../../../glossary.md#shape). Both booleans are primitive, require no [validation](../../../../glossary.md#term-validation),
and are retained exactly.

### Typed composition and meaning

Document this valid pairing without adding generic compatibility [validation](../../../../glossary.md#term-validation):

```java
new Operation(CumulativeSumKind.CUM_SUM, attrs)
```

The four attribute combinations mean:

- `(exclusive=false, reverse=false)` — inclusive forward prefix;
- `(exclusive=true, reverse=false)` — exclusive forward prefix with zero at the lowest index;
- `(exclusive=false, reverse=true)` — inclusive reverse suffix-style accumulation;
- `(exclusive=true, reverse=true)` — exclusive reverse accumulation with zero at the highest
  index.

All forms retain output positions in input order. `reverse` controls accumulation traversal only;
it does not reverse the returned [Tensor](../../../../glossary.md#tensor). `exclusive` excludes only the current [element](../../../../glossary.md#term-element), not the
entire prefix. Additive zero is semantic [identity](../../../../glossary.md#term-identity-meanings), not an allocated Tensor or stored attribute.

Generic [`Operation`](../../../../glossary.md#operation) remains unchanged and does not enforce pairing, input count, [axis](../../../../glossary.md#term-axis) bounds,
[Shape](../../../../glossary.md#shape) preservation, type eligibility, numerical behavior, [gradients](../../../../glossary.md#term-gradient), or [backend](../../../../glossary.md#backend) support.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumAttrs.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/CumulativeSumSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), Training API, capabilities,
[Operation](../../../../glossary.md#operation) foundations, aggregate/[masked reduction](../../../../glossary.md#masked-reduction) contracts, focused architecture, ADRs/tests,
[conformance](../../../../glossary.md#term-conformance)/integration tests, and [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one test, and five documentation/[planning](../../../../glossary.md#term-planning) files: eight paths total.
Do not modify an existing Java contract or test. Stop beyond this scope or if [Tensor](../../../../glossary.md#tensor)/[Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype)
behavior, another semantic type/component, a compatibility validator, [dependency](../../../../glossary.md#term-dependency), or architecture
change is needed. Do not create task 0016H.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the kind, its sole [constant](../../../../glossary.md#term-constant), the record, [canonical](../../../../glossary.md#term-canonical-form) constructor, and every accessor.
- Explain cumulative sum, one logical input, shape preservation as semantic context, normalized
  axis, inclusive/exclusive behavior, traversal direction, output indexing, and all four modes.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [normalized axis glossary](../../../../glossary.md#normalized-axis))</span>
- Include the concrete `[1, 2, 3]` examples and explain why each boundary position is zero or
  contains a suffix/prefix total.
- Document every parameter, structural axis validation, accepted boundaries, return meaning, and
  exact failure.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- Clearly defer DataType eligibility, descriptor/provenance construction, values, numerical policy,
  gradients, compiler behavior, storage, and execution.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Review Operation foundations and reduction-family Javadocs and record reasoned no-change
  conclusions or stop.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation))</span>

## Acceptance criteria

- Exactly one public one-constant enum, one public three-component record, and one focused test are
  added under the planned scan package; no other Java file or API changes.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Exact enum [constant](../../../../glossary.md#term-constant)/name/order and absence of project [metadata](../../../../glossary.md#term-metadata) are verified.
- Exact record components/order/types, interface, constructor/accessors, and absence of extra
  project API/state are verified.
- Negative [axes](../../../../glossary.md#term-axis) fail with exact type/message; zero, positive, and maximum axes succeed.
- All four [boolean](../../../../glossary.md#data-type--datatype) combinations retain exact values and documented meanings.
- [`Operation`](../../../../glossary.md#operation) composes the exact kind and attributes reference without generic family [validation](../../../../glossary.md#term-validation).
- Record equality, hashing, diagnostic text, and distinct typed identity from aggregate reduction
  kinds/attributes are verified.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [aggregate reduction glossary](../../../../glossary.md#aggregate-reduction))</span>
- No Tensor, DataType, Shape, storage, graph, compiler, runtime, backend, dependency, build, or
  architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/scope and documentation
  validation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation-focused agent finalizes permitted Javadocs/API/glossary/
  planning and records reasoned no-change conclusions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- 0016G becomes Complete only after both passes; 0016H remains Draft without a specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.scan.CumulativeSumSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers enum/package/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); record component order; explicit accessors;
negative/boundary [axes](../../../../glossary.md#term-axis); all four [boolean](../../../../glossary.md#data-type--datatype) combinations; generated value semantics and exact text;
typed [Operation](../../../../glossary.md#operation) composition and attributes [identity](../../../../glossary.md#term-identity-meanings); and absence of fields, nested types,
[Tensor](../../../../glossary.md#tensor)/Shape/[DataType](../../../../glossary.md#data-type--datatype)/execution state, [aliases](../../../../glossary.md#aliasing-and-arena), registries, or [metadata](../../../../glossary.md#term-metadata).

Manually inspect reflection, `javap -p -c -s`, source, and imports for the exact enum/record shape,
constructor validation before assignment, explicit accessors, no extra project state/API, and no
cross-layer type or behavior. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor API/glossary, links/anchors/
fences/whitespace, exact eight paths, synchronized statuses, package map, and absence of a
task-0016H specification.

## Dependencies

- 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition.
- 0016A establishes normalized-axis attribute conventions and the distinct aggregate-reduction
  family.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

## Follow-up tasks

- 0016H remains Draft for public [shape](../../../../glossary.md#shape)-preserving `Tensor.cumSum` expressions, negative-axis
  normalization, numeric eligibility, descriptor/provenance construction, and freshness.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0016I–0016J remain Draft for [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax) semantics/expressions.
- Compiler tasks own capture/canonicalization/autograd; backend/config/conformance tasks own scan
  algorithms, numerical behavior, storage, kernels, routes, and cross-backend parity.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [scan glossary](../../../../glossary.md#term-cumulative-scan); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task adds [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model)-owned [operation](../../../../glossary.md#operation) semantics in a cohesive [scan](../../../../glossary.md#term-cumulative-scan)
package without changing [module](../../../../glossary.md#term-module) boundaries, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings). Stop if
an architecture change is required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0016A/0016F/0016F1/0016G, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/Operation and aggregate/masked
reduction contracts/tests, and Java 26 Gradle configuration.

Implement task 0016G exactly. Add only CumulativeSumKind.java, CumulativeSumAttrs.java, and
CumulativeSumSemanticsTest.java under io.github.pho001.synaptik.model.operation.scan for Java
code/tests.

The public enum implements OperationKind and contains exactly CUM_SUM with no project fields,
methods, nested types, aliases, arity, or metadata. The public record implements OperationAttrs
and contains exactly int axis, boolean exclusive, and boolean reverse in order, explicit documented
accessors, exact negative-axis validation/message, and no extra API/state. Document all four modes,
one logical input, shape-preserving/output-order meaning, and explicit CUM_SUM/attributes pairing.

Do not add Tensor methods, DataType/Shape/result/provenance rules, value execution, scan algorithm,
gradients, graph/compiler/planning/runtime/backend behavior, factories/registries, dependencies,
build/architecture changes, existing Java edits, or later specs. Stop beyond eight paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0016G, model master plan, and roadmap only for planning status/evidence. Do not mark
0016G Complete until both passes succeed. Leave 0016H Draft without a specification. Do not commit
or push.
```

## Local decisions

- Use `CumulativeSumKind.CUM_SUM` rather than legacy class/name casing. The underscore matches
  typed semantic enum naming such as `ARG_MAX` while public fluent naming remains a later Tensor
  concern.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Use one attributes record for all four modes rather than four kinds or a mode enum; axis,
  inclusion, and direction are independent semantic parameters.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Add a separate `operation.scan` package because scans preserve logical positions and differ from
  aggregate reductions that contract positions.
- Store only the normalized axis and two primitive flags. One input, Shape preservation, additive
  identity, and output order are documented meaning, not duplicated state.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Treat exclusive boundary zero as fixed cumulative-sum semantics while deferring numeric
  representation and execution.
- Keep public Tensor construction and DataType eligibility in 0016H so this task remains a small
  semantic foundation.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>

## Known limitations

- No public Tensor expression, Shape normalization/descriptor, provenance, value execution, or
  gradients yet.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [normalization glossary](../../../../glossary.md#term-normalization); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Single-[axis](../../../../glossary.md#term-axis) cumulative sum only; no segmented, multi-axis, or other cumulative family.
- No numerical accuracy, [overflow](../../../../glossary.md#term-overflow), [backend](../../../../glossary.md#backend), [ONNX](../../../../glossary.md#term-onnx), [compiler](../../../../glossary.md#term-compiler), or [kernel](../../../../glossary.md#kernel) behavior.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused architecture docs; documentation/planning
rules; roadmap; [model](../../../../glossary.md#term-model) capabilities/master plan; tasks 0005, 0006, 0016A, 0016F, and 0016F1;
current [Operation](../../../../glossary.md#operation) and reduction contracts/tests; [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs; glossary; and Java
26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The legacy branch was read directly. It confirms exactly `cumSum(axis)` and
`cumSum(axis, exclusive, reverse)`, negative public [axes](../../../../glossary.md#term-axis), four inclusion/direction modes,
[shape](../../../../glossary.md#shape)/type preservation, [BOOL](../../../../glossary.md#data-type--datatype) rejection, floating/[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype), non-[contiguous](../../../../glossary.md#term-contiguous) input, [ONNX](../../../../glossary.md#term-onnx), [CPU](../../../../glossary.md#term-cpu),
and [Metal](../../../../glossary.md#term-metal) evidence. Legacy operation traits, callbacks, storage, execution loops, [lowering](../../../../glossary.md#lowering),
[backend](../../../../glossary.md#backend) support, and [kernels](../../../../glossary.md#kernel) are excluded or reassigned.

Planning selected one single-[constant](../../../../glossary.md#term-constant) kind, one three-component attributes record, one focused
test, and the new [scan](../../../../glossary.md#term-cumulative-scan) package. Existing Operation foundations suffice; no [dependency](../../../../glossary.md#term-dependency),
foundational contract, or architecture change is required.

Pre-implementation planning validation after synchronizing this task, the model master plan, and
roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three planning files.
- All 167 relative Markdown links across the three planning files resolve locally.
- Markdown fence counts are balanced: fourteen in this task, two in the master plan, and zero in
  the roadmap.
- All 20 task-template headings are present, together with the focused Capability origin,
  Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task, model master plan, and roadmap consistently identify 0016G as [Ready](../../../../glossary.md#term-ready) and 0016H as Draft.
- No detailed task-0016H specification exists.
- Repository scope is exactly this task, the model master plan, and the roadmap; no Java, API,
  architecture, Gradle, or other file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation and independent documentation validation:

- The implementation pass added exactly `CumulativeSumKind`, `CumulativeSumAttrs`, and the
  same-package `CumulativeSumSemanticsTest`. It changed no existing Java source or test.
- Clean documentation-focused Codex context
  `019f3258-da07-7f00-9b6b-6895d9e94e0d` applied General style, API and Javadoc style, Planning
  style, and Example format. It independently inspected the complete implementation diff, final
  scan and related operation/reduction source and tests, generated model Javadoc, bytecode,
  imports, APIs, glossary, planning state, architecture documents, and Java 26 build configuration.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Both new production Javadocs are complete without revision. They document one logical input,
  shape preservation, normalized-axis ownership, explicit <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`CUM_SUM`/`CumulativeSumAttrs` pairing,
  inclusive/exclusive and forward/reverse behavior, output order, all four `[1, 2, 3]` examples,
  the exclusive additive-zero boundary, every constructor parameter and accessor result, the
  exact negative-axis failure, immutable record semantics, and all deferred behavior.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- `docs/api/tensor-api.md` now classifies the cumulative-sum semantic pair as current, gives the
  exact composition and four-mode example, and keeps public `Tensor.cumSum` explicitly planned
  for task 0016H. `docs/glossary.md` now defines cumulative sum/scan and synchronizes operation
  kind/attribute status; its adjacent review also corrected two stale statements that still
  described already-completed public masked reduction construction as planned.<span data-glossary-navigation="true"> ([masked reduction glossary](../../../../glossary.md#masked-reduction))</span>
- The first post-review focused-test attempt failed before Gradle task execution because the
  restricted sandbox could not open
  `/Users/phujka/.gradle/wrapper/dists/gradle-9.6.1-bin/4ticwg1pgcbps2hj28r8so764/gradle-9.6.1-bin.zip.lck`
  (`FileNotFoundException`, `Operation not permitted`). This was environment-only; the approved
  rerun of the exact command passed from cache with <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`BUILD SUCCESSFUL` and 3 actionable tasks:
  1 from cache and 2 up-to-date.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.scan.CumulativeSumSemanticsTest` — `BUILD SUCCESSFUL`;
  the XML report contains 8 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 55 XML suites contain 425 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`; generated pages contain both scan types,
  `CUM_SUM`, the canonical constructor, all three explicit accessors, the four concrete modes,
  return/parameter/failure documentation, and the semantic-versus-executable boundary.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `./gradlew test` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36 actionable tasks
  with no failing task.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `git diff --check` passed with no output.
- `javap -p -c -s` confirmed one enum constant and no project fields/methods/nested types, plus
  exactly the three record fields/components/accessors, generated record value methods, and the
  negative-axis validation before all record field assignments. Focused reflection tests confirm
  the same public/final shapes, exact interfaces and constructors, typed distinction from
  aggregate reduction, valid boundary axes and all four flag combinations, exact diagnostic text,
  and exact-reference <span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../../glossary.md#aggregate-reduction))</span>`Operation` composition.
- Production import inspection found only [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs). No Tensor,
  DataType, Shape, layout, storage, graph, compiler, planning, runtime, prepare, backend, trace, or
  training dependency or state was introduced. Package placement and direction are exactly<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace))</span>
  `model.operation.scan -> model.operation`, matching the master-plan package map.
- Local Markdown target/anchor validation passed for all 252 relative links in the five changed
  documentation/planning files. Markdown fences are balanced, targeted trailing-whitespace and
  terminology checks passed, the four numerical examples were recalculated, and generated
  Javadoc rendering was inspected directly.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Final repository scope is exactly the authorized eight paths: two new production files, one new
  focused test, Tensor API, glossary, this task, model master plan, and roadmap. Task 0016G is
  synchronized as Complete. Task 0016H remains Draft, and no `0016h-*` task specification exists.
- `OperationKind`, `OperationAttrs`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and `Operation` remain accurate unchanged:
  the new family implements their open typed contracts and does not change generic compatibility
  validation. Aggregate and masked reduction contracts remain accurate unchanged because a scan
  preserves positions and uses a separate typed family without modifying any reduction API,
  attributes, expression construction, or test.
- `capabilities.md` remains accurate unchanged because it already inventories `cumSum` with
  exclusive/reverse options and distinguishes semantic representation from public and executable
  support. Compile API remains accurate unchanged because no public expression, capture,
  inference, canonicalization, artifact, or execution behavior was added. Training API remains
  accurate unchanged because no gradient, autograd, optimizer, publication, or session behavior
  changed.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because this task
  changes no module boundary, dependency direction, lifecycle, backend behavior, or end-to-end
  execution. Java 26 root/model Gradle configuration remains accurate unchanged because no
  dependency, source set, language level, preview/incubator feature, task, or module changed.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [feature glossary](../../../../glossary.md#term-software-feature))</span>

## Implementation notes

- Added the exact one-[constant](../../../../glossary.md#term-constant) `CumulativeSumKind` semantic enum and exact three-component
  `CumulativeSumAttrs` record in the new cohesive [scan](../../../../glossary.md#term-cumulative-scan) package.
- Added the focused eight-test contract suite covering exact API shape, structural axis validation,
  all four modes, generated record semantics, typed composition, and cross-layer exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis); [validation glossary](../../../../glossary.md#term-validation))</span>
- The documentation-focused pass finalized Tensor API, glossary, task evidence, model master plan,
  and roadmap. It changed no Java source because the submitted Javadocs were already complete.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model))</span>
- No public Tensor method, result descriptor, provenance, value execution, scan algorithm,
  gradient, compiler, runtime, backend, dependency, build, or architecture behavior was added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: Implemented and documented the backend-independent cumulative-sum semantic
  identity and immutable normalized-axis, exclusive, and reverse attributes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [axis glossary](../../../../glossary.md#term-axis))</span>
- Files changed or created: Exactly two production Java files, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused scan semantics 8/8, all 425 model tests across 55 suites, generated
  model Javadoc, root tests, bytecode/reflection/import checks, numerical-example and rendered-doc
  review, Markdown link/anchor/fence/whitespace checks, exact scope/status checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scan glossary](../../../../glossary.md#term-cumulative-scan); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean Codex context
  `019f3258-da07-7f00-9b6b-6895d9e94e0d` completed the mandatory independent pass using the
  API/Javadoc and Planning profiles plus General style and Example format.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now distinguish current cumulative-sum semantic
  representation from planned public Tensor/executable behavior; task, master plan, and roadmap
  are synchronized.
- Javadoc review: Both new public contracts and all required members are complete unchanged;
  operation foundations and aggregate/masked reduction Javadocs remain accurate.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [masked reduction glossary](../../../../glossary.md#masked-reduction))</span>
- Glossary impact: Added cumulative sum/scan terminology and corrected stale masked-reduction
  implementation-status wording found during adjacent-contract review.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016G. Task 0016H remains Draft without a detailed
  specification.

Status: Complete
