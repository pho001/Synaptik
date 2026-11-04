# Task 0015G: Cast Semantic Kind and Attributes

**Editorial reading key (later context, not a status update).**

A cast requests conversion of each value to a target type; it is not merely changing the label on existing bytes. The target type is intrinsic attributes state, while the source type comes from the eventual operand descriptor. Numerical conversion policy remains unselected at this checkpoint. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current cast and conversion-policy reference](../../../../reference/tensor/selection-casts.md#cast-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [target](../../../../glossary.md#term-target)-data-type [parameter](../../../../glossary.md#term-parameter)
for an explicit [elementwise](../../../../glossary.md#term-elementwise) cast. The [operation](../../../../glossary.md#operation) family must represent one `CAST` request paired
with one non-null target [`DataType`](../../../../glossary.md#data-type--datatype), without retaining an input [Tensor](../../../../glossary.md#tensor), constructing a result
[descriptor](../../../../glossary.md#term-descriptor), converting a value, choosing numerical conversion policy, or reporting backend
support.

This task creates the complete semantic foundation consumed by task 0015H. It does not add the
public `Tensor.cast(DataType)` expression method.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `CastKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly one [constant](../../../../glossary.md#term-constant), `CAST`.
- Add one public `CastAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one
  `DataType targetDataType` component.
- Require a non-null [target](../../../../glossary.md#term-target) with exact `NullPointerException("targetDataType")` behavior.
- Accept and retain every current [`DataType`](../../../../glossary.md#data-type--datatype): [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype),
  and [`BOOL`](../../../../glossary.md#data-type--datatype).
- Define `CastKind.CAST` plus `CastAttrs` as the valid family composition through typed
  documentation without changing generic [`Operation`](../../../../glossary.md#operation) [validation](../../../../glossary.md#term-validation).
- Document one logical input and elementwise target-type conversion as family context without
  storing arity, source type, input identity, result descriptor, or conversion rules.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [arity glossary](../../../../glossary.md#term-arity); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Add one focused same-package test covering exact vocabulary, attributes, composition, record
  value semantics, typed identity, and exclusions.
- Add the cohesive `model.operation.elementwise.cast` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.cast(DataType)`, another Tensor method, overload, factory, builder, or expression
  helper<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- input Tensor, source data type, shape, layout, result descriptor, label, identity, storage,
  provenance, expression chaining, or <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived`
- source/target compatibility rules, same-type identity elimination, fresh-result policy, shape
  preservation implementation, result `requiresGrad`, or local/[graph](../../../../glossary.md#graph)-wide [inference](../../../../glossary.md#term-inference)
- numerical conversion behavior, rounding, truncation, saturation, overflow, underflow, precision,
  NaN, infinity, signed zero, BFLOAT16 encoding, BOOL zero/non-zero semantics, or invalid-value
  policy<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [saturation glossary](../../../../glossary.md#term-saturation); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [precision glossary](../../../../glossary.md#term-precision); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- eager value conversion, storage reads or writes, allocation, copy, materialization, aliasing,
  constant folding, canonicalization, or cast-chain simplification<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- gradient eligibility, floating-to-floating gradient rules, non-floating gradient barriers,
  backward graph construction, autograd, optimizer, or training behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- an attributes hierarchy, source-and-target pair record, conversion-mode enum, rounding-mode enum,
  optional target, boxed/string target, registry, parser, compatibility matrix, visitor, or service<span data-glossary-navigation="true"> ([compatibility matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- [aliases](../../../../glossary.md#aliasing-and-arena) such as `CONVERT`, `AS_TYPE`, `TO_DTYPE`, or one enum [constant](../../../../glossary.md#term-constant) per source/[target](../../../../glossary.md#term-target) pair
- fields on `CastKind` for arity, category, source/target type, cost, fusion, differentiability,
  backend support, lowering, route, kernel, or execution metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changes to [`DataType`](../../../../glossary.md#data-type--datatype), [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), existing
  operation families, Tensor, graph records, or existing Java tests
- compiler, planning, prepare, runtime, backend, engine, tracing, ONNX, dependency, Gradle,
  architecture, or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0015H specification

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
- [Task 0001](0001-data-type-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0015E](0015e-where-selection-semantic-kind.md)
- [Task 0015F](0015f-where-selection-tensor-expression.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes fluent `Tensor.cast(DataType targetType)`. Its
semantic [descriptor](../../../../glossary.md#term-descriptor) retains one non-null [target](../../../../glossary.md#term-target) [data type](../../../../glossary.md#data-type--datatype), while the input [Tensor descriptor](../../../../glossary.md#tensor-descriptor)
provides the source type. Legacy execution evidence covers floating [precision](../../../../glossary.md#term-precision) changes, [BFLOAT16](../../../../glossary.md#term-bfloat16),
integral/floating conversion, [BOOL](../../../../glossary.md#data-type--datatype)/floating conversion, strided inputs, expression chains, [ONNX](../../../../glossary.md#term-onnx),
[CPU](../../../../glossary.md#term-cpu), and [Metal](../../../../glossary.md#term-metal) routes. The legacy public builder also treats a same-type cast as an [identity](../../../../glossary.md#term-identity-meanings) and
installs a [gradient](../../../../glossary.md#term-gradient) [callback](../../../../glossary.md#term-callback) only for floating-to-floating conversions.

The legacy descriptor additionally exposes family, [arity](../../../../glossary.md#term-arity), cost, [fusion](../../../../glossary.md#term-fusion), result-kind,
expression-string, and [backend](../../../../glossary.md#backend)-facing facts. Those mechanisms are not copied. This task retains
only the explicit `CAST` meaning and target [`DataType`](../../../../glossary.md#data-type--datatype). Task 0015H owns public [Tensor](../../../../glossary.md#tensor) construction,
result descriptor facts, same-type behavior, and gradient-eligibility policy. [Compiler](../../../../glossary.md#term-compiler), backend,
ONNX, and [conformance](../../../../glossary.md#term-conformance) work later own [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), differentiation, mapping, conversion
semantics, [lowering](../../../../glossary.md#lowering), storage access, and execution.

## Architecture constraints

- [Operation](../../../../glossary.md#operation) kinds and attributes are [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics owned by
  `modules/model`.
- `CastKind` identifies the requested conversion family. It stores no parameter, input, source
  type, graph occurrence, result fact, executable behavior, or backend information.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `CastAttrs` is the complete attributes value for `CastKind.CAST`; its sole semantic parameter is
  the exact non-null target <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>[`DataType`](../../../../glossary.md#data-type--datatype).
- The source data type belongs to the later input Tensor or graph value descriptor. Duplicating it
  in attributes would create two sources of truth and is forbidden.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [graph value glossary](../../../../glossary.md#graph-value); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Family documentation defines the valid `CAST`/`CastAttrs` pairing. Generic `Operation` continues
  to validate only non-null components and must not gain family discovery or compatibility checks.
- All six current `DataType` values are valid targets. This representability does not promise that
  every backend implements every source/target conversion route.
- A target equal to a future input source type remains a valid semantic parameter. Whether public
  expression construction returns its input, creates an explicit node, or leaves identity removal
  to compiler optimization is not decided here.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- One logical input and elementwise conversion are family context rather than stored arity or
  shape metadata. No source value or conversion is performed.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [arity glossary](../../../../glossary.md#term-arity); [shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Conversion rounding, overflow, truth, special-value, and gradient policies are not attributes
  fields and are not implied by <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [overflow glossary](../../../../glossary.md#term-overflow); [gradient glossary](../../../../glossary.md#term-gradient))</span>`DataType` retention.
- Stable enum names and record text are diagnostic, not serialization, ONNX, registry, reflection,
  backend-dispatch, or kernel identifiers.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [kernel glossary](../../../../glossary.md#kernel))</span>
- Package direction is `model.operation.elementwise.cast -> model.operation, model.datatype`. It
  must not depend on Tensor, shape, layout, storage, graph, compiler, planning, runtime, prepare,
  backend, or training packages.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation requires another attributes component/type, a Tensor or result contract,
  conversion policy, compatibility validator, dependency, or architecture change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and
  generic [`Operation`](../../../../glossary.md#operation) composition.
- `io.github.pho001.synaptik.model.datatype` — supplies the [immutable](../../../../glossary.md#term-immutable) [target](../../../../glossary.md#term-target) [`DataType`](../../../../glossary.md#data-type--datatype) vocabulary.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.cast
  Typed explicit data-type conversion semantics and immutable target-type attributes.
```

The package is below [`elementwise`](../../../../glossary.md#term-elementwise) because one cast request converts corresponding logical
[elements](../../../../glossary.md#term-element) independently and has one logical input. The `cast` [leaf](../../../../glossary.md#term-leaf) separates conversion [parameters](../../../../glossary.md#term-parameter)
from arithmetic, comparison, logical, and selection families without introducing a broad [dtype](../../../../glossary.md#data-type--datatype)
utility package.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind` — public semantic family
  enum.
- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs` — public immutable target
  data-type parameter.
- `CastSemanticsTest` — same-package focused test for both cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum CastKind implements OperationKind {
    CAST
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text `CAST`.

`CAST` means explicit [elementwise](../../../../glossary.md#term-elementwise) conversion of one logical input to the [target](../../../../glossary.md#term-target) [data type](../../../../glossary.md#data-type--datatype) carried
by `CastAttrs`. It does not define the source type, result [descriptor](../../../../glossary.md#term-descriptor), numerical conversion,
[gradient](../../../../glossary.md#term-gradient) behavior, execution, or [backend](../../../../glossary.md#backend) availability.

### [Target](../../../../glossary.md#term-target) attributes

Create exactly:

```java
public record CastAttrs(DataType targetDataType) implements OperationAttrs {
    public CastAttrs {
        targetDataType = Objects.requireNonNull(targetDataType, "targetDataType");
    }

    @Override
    public DataType targetDataType() {
        return targetDataType;
    }
}
```

The record has exactly one component and no additional instance state, static state, nested type,
overload, factory, or helper. Construction validates only non-null presence and then retains the
exact enum reference through ordinary record assignment. Null fails with
`NullPointerException("targetDataType")`.

The explicit accessor exists for complete [Javadoc](../../../../glossary.md#term-javadoc) and returns the exact stored non-null reference.
All six current targets are accepted without source knowledge, [normalization](../../../../glossary.md#term-normalization), defaulting, or
capability [lookup](../../../../glossary.md#term-lookup).

Record-generated equality and hashing use the target enum value. Equal targets produce equal
records and equal [hashes](../../../../glossary.md#term-hash); different targets produce unequal records. Generated diagnostic text is
not a [serialization](../../../../glossary.md#term-serialization), parser, [ONNX](../../../../glossary.md#term-onnx), [backend](../../../../glossary.md#backend), or [dispatch](../../../../glossary.md#term-dispatch) contract.

### [Operation](../../../../glossary.md#operation) composition

Every [target](../../../../glossary.md#term-target) composes explicitly as:

```java
CastAttrs attrs = new CastAttrs(DataType.FLOAT32);
Operation operation = new Operation(CastKind.CAST, attrs);
```

`Operation` retains the exact kind and attributes references. Do not use
`NoOperationAttrs.INSTANCE`: target [data type](../../../../glossary.md#data-type--datatype) is an intrinsic semantic [parameter](../../../../glossary.md#term-parameter). Do not add a
family factory, enum `operation()` method, compatibility map, or generic-operation [validation](../../../../glossary.md#term-validation).

### Naming and typed [identity](../../../../glossary.md#term-identity-meanings)

- `values()` returns exactly one [constant](../../../../glossary.md#term-constant), `CAST`.
- Equality and hashing of `CastKind` remain standard Java enum behavior.
- A private test-local or future family constant named `CAST` remains a different typed value.
- No source/[target](../../../../glossary.md#term-target)-pair constants or [aliases](../../../../glossary.md#aliasing-and-arena) are introduced.
- Enum and record text remain diagnostics and must not be parsed or used for [dispatch](../../../../glossary.md#term-dispatch).

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastAttrs.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [`DataType`](../../../../glossary.md#data-type--datatype), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`Operation`](../../../../glossary.md#operation), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and
  concrete kind/attributes family Javadocs/tests.
- Tensor and Tensor expression contracts, focused architecture documents, ADRs, architecture
  tests, backend-conformance tests, integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: eight paths
total.

No existing Java source or test may change. Do not modify [DataType](../../../../glossary.md#data-type--datatype), [Operation](../../../../glossary.md#operation) foundations, [Tensor](../../../../glossary.md#tensor),
capabilities, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api), [Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module),
or unrelated documentation. Stop if implementation requires another production/test concept,
source-type state, conversion policy, factory/validator, another documentation file, or a ninth
path. Do not create task 0015H.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `CastKind` as backend-independent parameterized elementwise conversion vocabulary and
  distinguish it from input, source type, target attributes, result descriptor, graph occurrence,
  numerical conversion, and executable support.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [descriptor glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Document `CAST` with its one-input [elementwise](../../../../glossary.md#term-elementwise) meaning, required `CastAttrs`, and explicit
  deferral of source/target compatibility, same-type behavior, result inference, numerical policy,
  gradients, execution, and backend availability.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Document the valid kind/attributes pairing and state that generic [`Operation`](../../../../glossary.md#operation) does not enforce it.
- Document `CastAttrs` with its exact [target](../../../../glossary.md#term-target) role, [immutable](../../../../glossary.md#term-immutable) [ownership](../../../../glossary.md#term-ownership-meanings), all accepted current
  [`DataType`](../../../../glossary.md#data-type--datatype) values, null [validation](../../../../glossary.md#term-validation), record equality/hashing, and diagnostic-only text.
- Document the [canonical](../../../../glossary.md#term-canonical-form) constructor with `@param targetDataType`, exact non-null constraint,
  reference retention, and `@throws NullPointerException` message.
- Document the explicit accessor with a non-null `@return` describing exact stored-reference
  retention and absence of source knowledge, conversion, or capability validation.
- Explain why source data type is not duplicated and why target representability does not promise
  backend support or define conversion details.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public `CastKind` enum and one public `CastAttrs` record are added in the planned
  cast package.
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind), declares exactly `CAST`, and adds no project field, method,
  nested type, alias, constant body, category, arity, cost, fusion, result, backend, or execution
  metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [constant glossary](../../../../glossary.md#term-constant); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `CastAttrs` implements [`OperationAttrs`](../../../../glossary.md#operationattrs) and has exactly one `DataType targetDataType` component,
  one canonical constructor, one explicit component accessor, and no additional state or API.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Null [target](../../../../glossary.md#term-target) fails with exact `NullPointerException` message `targetDataType`.
- Every current [`DataType`](../../../../glossary.md#data-type--datatype) is accepted, retained by exact reference, and returned unchanged.
- Equal targets produce equal records/equal hash codes; different targets produce unequal records;
  diagnostic text is not treated as serialization or dispatch.<span data-glossary-navigation="true"> ([hash codes glossary](../../../../glossary.md#hash-codes-and-content-digests))</span><span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `CAST` constructs a valid [`Operation`](../../../../glossary.md#operation) with every target attributes value, retaining exact kind
  and attributes references. No factory or family compatibility validator is added.
- `CAST` remains typed-distinct from an equally named kind in another enum.
- No source type, Tensor, shape, layout, descriptor, provenance, storage, numerical conversion,
  same-type policy, gradient, graph, compiler, planning, runtime, backend, dependency, or
  architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/scope
  checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate clean-context documentation-focused agent finalizes new Javadocs, Tensor API,
  glossary, task evidence, model master plan, and roadmap in the same change and records reasoned
  no-change conclusions for Compile API, Training API, capabilities, architecture, and related
  contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0015G becomes Complete only after both passes. Task 0015H remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact package, public/final enum and record shapes, interfaces, record component order/type,
  constructor parameters, fields, methods, and absence of nested types;<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- exact sole enum constant, inherited stable name, standard enum identity, and typed distinction
  from a private test-local equal-name kind;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- null [target](../../../../glossary.md#term-target) with exact exception type/message before record construction;
- exact reference retention and accessor results for all six current [`DataType`](../../../../glossary.md#data-type--datatype) targets;
- generated equality, hashing, and diagnostic text for equal/different targets without treating
  text as serialization;<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- explicit [`Operation`](../../../../glossary.md#operation) composition for every target with exact kind/attributes references;
- target-only attributes with no source type, Tensor input, descriptor, conversion mode, factory,
  registry, compatibility map, or dependency; and<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- distinction from `NoOperationAttrs.INSTANCE` and parameterless families.

Manually inspect `javap -p -c -s` and reflection for exact enum/record shape, component order,
constructor null validation, explicit accessor, and absence of extra project [API](../../../../glossary.md#term-application-programming-interface-api)/state. [Scan](../../../../glossary.md#term-cumulative-scan)
production imports and [Gradle](../../../../glossary.md#term-gradle) dependencies: `CastKind` may import only [`OperationKind`](../../../../glossary.md#operationkind); `CastAttrs`
may import only `DataType`, [`OperationAttrs`](../../../../glossary.md#operationattrs), and `Objects`. Confirm no [Tensor](../../../../glossary.md#tensor), [shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout),
storage, [provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), training, [gradient](../../../../glossary.md#term-gradient),
cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, reflection, or service type appears. Validate generated
[Javadoc](../../../../glossary.md#term-javadoc), Tensor API status, glossary terminology, links/anchors/fences/whitespace, exact eight-path
[scope](../../../../glossary.md#term-scope), synchronized statuses, package-map placement, and absence of a task-0015H specification.

## Dependencies

- Task 0001 supplies the complete current [`DataType`](../../../../glossary.md#data-type--datatype) vocabulary.
- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and [`OperationAttrs`](../../../../glossary.md#operationattrs).
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and exact reference retention.
- Completed parameterized scalar attributes establish record/Javadoc/test conventions but are not
  Java dependencies of this family.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Follow-up tasks

- 0015H remains Draft for public `Tensor.cast(DataType)`, source/target handling, shape/result
  descriptor construction, same-type policy, gradient eligibility, provenance, and storage-free
  derived Tensor creation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Compiler tasks later own cast canonicalization, redundant-chain elimination, graph capture,
  autograd expansion, and optimization legality.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [autograd glossary](../../../../glossary.md#autograd))</span>
- Backend, ONNX, and conformance tasks later own mapping, conversion matrices, rounding,
  saturation/truncation, BOOL behavior, special values, lowering, storage access, and execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [BOOL glossary](../../../../glossary.md#data-type--datatype); [lowering glossary](../../../../glossary.md#lowering))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [Operation](../../../../glossary.md#operation) semantics,
typed attributes, and [DataType](../../../../glossary.md#data-type--datatype) vocabulary to `modules/model`. The new package refines that
[ownership](../../../../glossary.md#term-ownership-meanings) without [Tensor](../../../../glossary.md#tensor) behavior, dependencies, [inference](../../../../glossary.md#term-inference), storage, or executable state.

If implementation requires Tensor behavior, result inference, numerical conversion policy,
backend [metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0005/0006/0014E/0015E/0015F/0015G, Tensor API,
Compile API, Training API, glossary, current DataType/OperationKind/OperationAttrs/
NoOperationAttrs/Operation and concrete kind/attributes contracts/tests, and Java 26 Gradle
configuration.

Implement task 0015G exactly. Add only CastKind.java, CastAttrs.java, and CastSemanticsTest.java
under io.github.pho001.synaptik.model.operation.elementwise.cast for Java code/tests.

CastKind is a public enum implementing OperationKind with exactly CAST and no project fields,
methods, nested types, aliases, arity, or metadata. CastAttrs is a public record implementing
OperationAttrs with exactly non-null DataType targetDataType, exact NullPointerException message
targetDataType, explicit documented accessor, and no other state/API. Accept and retain every
current DataType. Compose CAST explicitly with CastAttrs; do not use NoOperationAttrs or add a
compatibility validator.

Do not add Tensor.cast, source type, result descriptor, shape/layout/provenance, same-type policy,
conversion mode or numerical rules, gradients, graph/compiler/planning/runtime/backend/ONNX
behavior, factories/registries, dependencies/build/architecture changes, existing Java edits, or
later specs. Stop beyond eight paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff/evidence to a separate
clean-context documentation agent in the same change. It must inspect source/tests/generated
Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-contract/
capability/Compile API/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015G, model master plan, and roadmap only for planning status/evidence. Do not mark
0015G Complete until both passes succeed. Leave 0015H Draft without a specification. Do not commit
or push.
```

## Local decisions

- The family uses `CastKind.CAST` and `CastAttrs(targetDataType)`. The kind identifies conversion;
  attributes carry the one intrinsic parameter.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- The package is `operation.elementwise.cast`: casts operate independently at corresponding
  logical positions, while the leaf name keeps conversion separate from arithmetic families.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Only the target type is stored. The future input descriptor is the authoritative source type,
  so duplicating source type in attributes would risk inconsistency.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- All current DataType values are representable targets. Backend availability and exact
  source/target numerical behavior are separate later contracts.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Backend glossary](../../../../glossary.md#backend))</span>
- Same-type target attributes are valid. Identity elimination versus an explicit cast expression
  is deliberately deferred to task 0015H and compiler optimization policy.<span data-glossary-navigation="true"> ([Identity glossary](../../../../glossary.md#term-identity-meanings); [cast expression glossary](../../../../glossary.md#cast-expression); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- `CastAttrs` is not parameterless; `NoOperationAttrs.INSTANCE` would lose the target type and is
  therefore invalid for the documented family composition.

## Known limitations

- No public cast [Tensor](../../../../glossary.md#tensor) expression exists until task 0015H.
- Source eligibility, same-type behavior, result descriptor, shape/layout, gradient eligibility,
  and provenance are undefined here.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- The [model](../../../../glossary.md#term-model) stores no [rounding](../../../../glossary.md#term-rounding), [saturation](../../../../glossary.md#term-saturation), [overflow](../../../../glossary.md#term-overflow), special-value, or [BOOL](../../../../glossary.md#data-type--datatype) conversion policy.
- Generic [`Operation`](../../../../glossary.md#operation) does not enforce the documented `CAST`/`CastAttrs` pairing.
- No compiler capture, optimization, ONNX mapping, backend support, storage interpretation, or
  execution is implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0001, 0005,
0006, 0014E, 0015E, and 0015F; current [DataType](../../../../glossary.md#data-type--datatype), [Operation](../../../../glossary.md#operation) foundations, concrete kind/attributes
source/tests, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms public fluent
`Tensor.cast(DataType)`, one non-null [target](../../../../glossary.md#term-target)-type semantic [parameter](../../../../glossary.md#term-parameter), all current target values,
floating/[BFLOAT16](../../../../glossary.md#term-bfloat16)/integral/[BOOL](../../../../glossary.md#data-type--datatype) conversions, same-type [identity](../../../../glossary.md#term-identity-meanings) behavior, strided inputs,
expression chaining, [gradients](../../../../glossary.md#term-gradient) for floating conversions, [ONNX](../../../../glossary.md#term-onnx) mapping, and [backend](../../../../glossary.md#backend) execution
evidence. Legacy input/result coupling, [arity](../../../../glossary.md#term-arity)/[fusion](../../../../glossary.md#term-fusion)/cost/result traits, same-type [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form),
gradient callbacks, conversion loops, storage access, [lowering](../../../../glossary.md#lowering), and backend policy are excluded
from this semantic-only task.

Planning selected one enum and one target-only attributes record. The source type remains external
to Operation attributes, all target values remain representable, and numerical/executable policy
is deferred. No existing dependency, foundational contract, or architecture rule changes.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact enum/record shapes, bounded scope, validation, implementation handoff, decisions,
  limitations, and completion-evidence sections.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown file and heading anchor linked from this task, the model master plan, and
  the roadmap resolves. Markdown fence counts are balanced.
- Initial planning status inspection found 0015G [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked
  model-master row, and its linked roadmap row/current-frontier text before implementation. Final
  implementation status is recorded below. Task 0015H remains `Draft` in both queues.
- Package-map inspection found exactly one new planned package,
  `model.operation.elementwise.cast`, with direction only to `model.operation` and
  `model.datatype`.
- Pre-implementation planning scope inspection found exactly this new task, the model master plan,
  and the roadmap changed. No Java, test, API, glossary, Gradle, architecture, AGENTS, or
  other-module path had changed at that stage.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>
- No task-0015H specification exists.

Implementation and independent documentation validation:

- Added the exact public `CastKind` enum and `CastAttrs` record in the planned cast package, plus
  one same-package focused test. No existing Java source or test changed.
- The focused `CastSemanticsTest` passed all 6 tests. The aggregate model suite passed all 353
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` and `./gradlew test` passed. Generated Javadoc contains both
  new public types, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` passed.
- `javap -p -c -s` and reflection-based tests confirmed the exact enum/record shapes, sole enum
  constant, one record component, constructor null check and message, explicit accessor, generated
  record methods, and absence of extra project state or API.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Production-import inspection found only [`OperationKind`](../../../../glossary.md#operationkind) for `CastKind`, and only `DataType`,
  [`OperationAttrs`](../../../../glossary.md#operationattrs), and `Objects` for `CastAttrs`. No Tensor, descriptor, shape, layout, storage,
  graph, compiler, runtime, backend, training, gradient, conversion-policy, registry, or service
  dependency was added.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry))</span>
- The clean-context documentation-focused pass in
  `/root/implement_model_0015d/review_model_0015d_docs` applied the General, API reference,
  example, and planning profiles. It found the two new Javadocs complete without revision,
  finalized the Tensor API and glossary, and synchronized this task, the model master plan, and
  the roadmap.
- Compile API remains accurate unchanged because no public Tensor expression or compiler capture
  was added. Training API remains accurate unchanged because cast gradient eligibility and rules
  are deferred. The model capability baseline already inventories explicit cast and therefore
  needs no status-level change.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Architecture documents, ADRs, architecture tests, backend-conformance tests, and integration
  tests remain accurate unchanged because ownership, dependencies, module boundaries, and
  executable behavior did not change. <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`DataType`, `OperationKind`, `OperationAttrs`, `Operation`,
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), existing concrete family contracts, Tensor contracts, and Java 26 build
  configuration also remain accurate unchanged.
- Markdown review found balanced fences, resolved local links and heading anchors, consistent cast
  terminology, and no trailing whitespace. Scope inspection found exactly the authorized eight
  paths. Task 0015H remains Draft and no task-0015H specification exists.
- Final implementation-context reruns passed the 6 focused tests, all 353 model tests, model
  Javadoc, root tests, and `git diff --check`. A generated-Javadoc scan first used a nonexistent
  module-qualified output directory and failed without changing files; the corrected scan of
  `modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/operation/elementwise/cast`
  found both rendered public types, the target component, constructor failure, and accessor
  documentation.

## Implementation notes

- `CastKind` contains exactly `CAST` and relies only on inherited enum identity, names, and value
  methods.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `CastAttrs` retains one exact non-null [target](../../../../glossary.md#term-target) [`DataType`](../../../../glossary.md#data-type--datatype); it neither stores nor derives a source
  type and performs no compatibility or backend-support lookup.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- The documented family composition is explicit `Operation(CastKind.CAST, CastAttrs)`. Generic
  [`Operation`](../../../../glossary.md#operation) remains unchanged and does not validate family compatibility.
- No source/result inference, numerical conversion, same-type policy, gradients, public Tensor
  expression, execution, backend route, dependency, or architecture behavior was introduced.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient); [Tensor glossary](../../../../glossary.md#tensor); [backend route glossary](../../../../glossary.md#backend-route); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed changes:

- Added [backend](../../../../glossary.md#backend)-independent cast semantic [identity](../../../../glossary.md#term-identity-meanings) and [immutable](../../../../glossary.md#term-immutable) [target](../../../../glossary.md#term-target)-data-type attributes.
- Added focused contract tests for vocabulary, shape, null behavior, all targets, value semantics,
  typed identity, explicit operation composition, and exclusions.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation))</span>
- Finalized the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), glossary, task evidence, [model](../../../../glossary.md#term-model) master plan, and roadmap.

Files changed or created: exactly the two production files, one focused test, and five authorized
documentation/[planning](../../../../glossary.md#term-planning) files listed under Affected files.

[Validation](../../../../glossary.md#term-validation) performed: focused and aggregate model tests, model [Javadoc](../../../../glossary.md#term-javadoc), root tests,
`git diff --check`, generated-Javadoc inspection, `javap`, reflection, import/[dependency](../../../../glossary.md#term-dependency) review,
Markdown link/anchor/fence/whitespace review, exact-[scope](../../../../glossary.md#term-scope) inspection, and status/spec checks.

Unresolved issues: none within task 0015G. Public expression construction and all conversion,
[gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), backend, and execution behavior remain deliberately assigned to 0015H or
later owning layers.

Required follow-up: plan task 0015H separately when it becomes the active planning frontier.

Status: Complete
