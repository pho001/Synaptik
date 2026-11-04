# Task 0015B: [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) Comparison [Tensor](../../../../glossary.md#tensor) Expressions

**Editorial reading key (later context, not a status update).**

An elementwise comparison produces a truth value for each corresponding pair, not a numerical minimum or an ordering of the whole tensor. The receiver and argument retain ordered roles, and right-aligned broadcasting determines corresponding positions. The output is BOOL with no gradient eligibility; promotion concerns comparison operands, not the output type. The floating-only historical admission is narrower than the linked current reference. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

Focused current reading: [Current comparison reference](../../../../reference/tensor/arithmetic.md#binary-comparison-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Expose the six implemented [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) comparison semantics as public, [backend](../../../../glossary.md#backend)-independent [Tensor](../../../../glossary.md#tensor)
expression methods. Each successful call must validate floating input types, apply the existing
floating compatibility hierarchy, derive a locally provable right-aligned broadcast [shape](../../../../glossary.md#shape), and
return a fresh storage-free [`BOOL`](../../../../glossary.md#data-type--datatype) Tensor whose [provenance](../../../../glossary.md#provenance) records the exact comparison and ordered
input Tensor [identities](../../../../glossary.md#term-identity-meanings).

This task makes comparisons capturable through the public [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api). It does not compare numeric
values, allocate result storage, [capture](../../../../glossary.md#term-capture-meanings) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), define numerical edge behavior, create
[gradient](../../../../glossary.md#term-gradient) rules, or report backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add public [`Tensor`](../../../../glossary.md#tensor) methods `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`,
  `equalTo`, and `notEqualTo`.
- Add one package-private `TensorComparisonExpressions` helper that owns the shared local
  validation and derived-Tensor construction path.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Accept every pair formed from [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), and [`FLOAT64`](../../../../glossary.md#term-float64).
- Validate the pair and its common floating comparison domain through the existing
  `DataTypePromotion.promoteFloating` contract without storing a promoted result type.
- Compute the result [shape](../../../../glossary.md#shape) through the existing `ShapeBroadcast.broadcast` contract.
- Create an unresolved-[layout](../../../../glossary.md#layout) [`BOOL`](../../../../glossary.md#data-type--datatype) result [descriptor](../../../../glossary.md#term-descriptor) with `requiresGrad=false` regardless of
  either input descriptor's gradient eligibility.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Construct exactly one [`Operation`](../../../../glossary.md#operation) from the selected `BinaryComparisonKind` and
  `NoOperationAttrs.INSTANCE`.
- Construct exactly one `TensorProvenance` with ordered inputs `[left, right]`.
- Delegate final [identity](../../../../glossary.md#term-identity-meanings)-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Update the exact Tensor public-[API](../../../../glossary.md#term-application-programming-interface-api) reflection test and add one focused [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-comparison test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- eager numeric comparison, reading input values, constant folding, canonicalization, interning,
  or returning a cached result<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- defining [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [tolerance](../../../../glossary.md#term-tolerance), total-order, or approximate-equality behavior
- integral or boolean comparison inputs, numeric truthiness, implicit cross-category conversion,
  explicit cast insertion, or extension beyond the selected floating baseline<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- scalar-number overloads, reverse comparisons, static factory forms, aliases, generic predicates,
  comparator objects, or an expression-builder public type<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- output labels, caller-supplied labels, expression strings, symbols, or [serialization](../../../../glossary.md#term-serialization) names
- resolved output layouts, broadcast stride plans, zero-stride views, aliases, materialization
  policy, or layout preservation from either operand<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [views glossary](../../../../glossary.md#term-view-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- propagation of input `requiresGrad`, gradient values, comparison gradients, straight-through
  estimators, backward graph construction, training-root rules, or autograd execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd))</span>
- [operation](../../../../glossary.md#operation)-family attributes, factories, registries, parsers, reflection discovery, or changes to
  `BinaryComparisonKind`, `Operation`, [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), or [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs)
- graph traversal, cycle checks, node/value IDs, graph capture, compiled graph construction,
  common-subexpression elimination, compiler inference, or publication binding<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>
- planning ownership, capability providers, backend support, fusion, cost, lowering, kernels,
  runtime residency, prepare, execution, tracing, or engine behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle changes, architecture changes, another module, another operation family, or
  a detailed task-0015C specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes `Tensor.greaterThan(Tensor)`,
`greaterOrEqual(Tensor)`, `lessThan(Tensor)`, `lessOrEqual(Tensor)`, `equalTo(Tensor)`, and
`notEqualTo(Tensor)`. Its public construction path accepts floating inputs, rejects integral and
[boolean](../../../../glossary.md#data-type--datatype) inputs, applies [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [broadcasting](../../../../glossary.md#broadcasting), returns a boolean result, disables result [gradient](../../../../glossary.md#term-gradient)
eligibility, and retains left/right operand order. Legacy execution tests cover all six relations,
[FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) values, broadcast [masks](../../../../glossary.md#term-boolean-mask), boolean chaining, and use as a `where` condition.

The legacy implementation also stored [mutable](../../../../glossary.md#term-mutable) broadcast plans inside [operation](../../../../glossary.md#operation) objects, attached
[runtime](../../../../glossary.md#term-runtime)-oriented behavior, defined executable numerical behavior, and coupled comparison results
to [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and [backends](../../../../glossary.md#backend). Those mechanisms are not copied. The new [model](../../../../glossary.md#term-model) retains only the
selected public capability and represents it through [immutable](../../../../glossary.md#term-immutable) [descriptors](../../../../glossary.md#term-descriptor) and [provenance](../../../../glossary.md#provenance).
Numerical comparison policy belongs to backend-[conformance](../../../../glossary.md#term-conformance) work, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) and [inference](../../../../glossary.md#term-inference) belong
to the [compiler](../../../../glossary.md#term-compiler), and execution belongs to prepared backends.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and must not become an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- `TensorComparisonExpressions` performs deterministic local model validation only. It must not
  traverse provenance, capture a graph, evaluate data, or inspect backend capability.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- [`Operation`](../../../../glossary.md#operation) owns only backend-independent semantic kind and attributes. It contains no operands,
  broadcast result, backend support, numerical policy, or executable behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- `TensorProvenance` owns the exact ordered `[left, right]` input identities. All six comparisons
  preserve caller order; even `EQUAL` and `NOT_EQUAL` must not reorder or canonicalize operands.
- Result [identity](../../../../glossary.md#term-identity-meanings) comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- The result is storage-free. Public expression construction must not allocate physical buffers,
  inspect input storage, or attach either operand's storage.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- The result [descriptor](../../../../glossary.md#term-descriptor) has `DataType.BOOL`, `Optional.empty()` [layout](../../../../glossary.md#layout), and
  `requiresGrad=false`. This remains true for static [shapes](../../../../glossary.md#shape) and for inputs requesting [gradients](../../../../glossary.md#term-gradient).
- Dynamic shapes are accepted only where `ShapeBroadcast` can prove compatibility locally: equal
  symbolic dimensions and static singleton expansion. The helper must not create constraints or
  defer a known local incompatibility.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Input compatibility uses `DataTypePromotion.promoteFloating` exactly once. Its returned common
  floating type validates the semantic comparison domain but is not the output type and is not
  stored in the BOOL result descriptor. Integral and boolean inputs are rejected; no cast is
  inserted.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- A BOOL comparison result is non-differentiable model metadata. This task does not define whether
  or how a later compiler treats comparisons inside a training graph.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [training graph glossary](../../../../glossary.md#training-graph))</span>
- Package direction is `model.tensor -> model.operation.elementwise.comparison`, plus existing
  `model.tensor -> model.operation`, `model.datatype`, and `model.shape`. The operation package must
  not import Tensor and no package cycle may be introduced.
- Stop if implementation requires a changed foundational contract, resolved layout, storage
  access, graph capture, gradient rule, numerical policy, dependency, or architecture decision.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public Tensor comparison methods, local result
  construction, descriptors, provenance, and the derived Tensor factory seam.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- `io.github.pho001.synaptik.model.operation` — supplies [`Operation`](../../../../glossary.md#operation) and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — supplies the six exact
  `BinaryComparisonKind` values.
- `io.github.pho001.synaptik.model.datatype` — supplies floating compatibility validation and the
  fixed <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>[`BOOL`](../../../../glossary.md#data-type--datatype) result [data type](../../../../glossary.md#data-type--datatype).
- `io.github.pho001.synaptik.model.shape` — supplies local right-aligned [broadcasting](../../../../glossary.md#broadcasting).

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the six one-argument methods and delegates shared behavior.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- `io.github.pho001.synaptik.model.tensor.TensorComparisonExpressions` — package-private, stateless
  construction boundary colocated with Tensor, descriptors, provenance, and the package-private
  factory seam it must use.
- `TensorBinaryComparisonTest` — same-package focused test so it can verify the package-private
  helper without widening production visibility.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly these public methods to `Tensor`:

```java
public Tensor greaterThan(Tensor right)
public Tensor greaterOrEqual(Tensor right)
public Tensor lessThan(Tensor right)
public Tensor lessOrEqual(Tensor right)
public Tensor equalTo(Tensor right)
public Tensor notEqualTo(Tensor right)
```

Each method delegates exactly once to
`TensorComparisonExpressions.apply(this, right, <KIND>)` and returns that exact result:

| Tensor method | Exact kind |
|---|---|
| `greaterThan` | `GREATER_THAN` |
| `greaterOrEqual` | `GREATER_OR_EQUAL` |
| `lessThan` | `LESS_THAN` |
| `lessOrEqual` | `LESS_OR_EQUAL` |
| `equalTo` | `EQUAL` |
| `notEqualTo` | `NOT_EQUAL` |

The public methods perform no separate [validation](../../../../glossary.md#term-validation), promotion, [broadcasting](../../../../glossary.md#broadcasting), [descriptor](../../../../glossary.md#term-descriptor) creation,
allocation, [provenance](../../../../glossary.md#provenance) construction, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), or storage access. There are no overloads
in this task. The receiver is always the ordered left operand and the [argument](../../../../glossary.md#term-argument) is always the
ordered right operand.

### Package-private helper [shape](../../../../glossary.md#shape)

Create exactly one package-private final non-record class:

```java
final class TensorComparisonExpressions {
    private TensorComparisonExpressions() {
    }

    static Tensor apply(Tensor left, Tensor right, BinaryComparisonKind kind) {
        // exact construction contract below
    }
}
```

The helper has no fields, nested types, public/protected members, overloads, caches, registries, or
[operation](../../../../glossary.md#operation)-specific branches. Its constructor prevents instantiation. `apply` is package-private
and static so [Tensor](../../../../glossary.md#tensor) can delegate without exposing an independent public expression service.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs these steps in exact order:

1. require non-null `left`, `right`, and `kind`, in that order, with messages `left`, `right`, and
   `kind`;
2. call `DataTypePromotion.promoteFloating(left.descriptor().dataType(),
   right.descriptor().dataType())` exactly once to validate and determine the common comparison
   domain; do not use its returned type as the result type;
3. call `ShapeBroadcast.broadcast(left.descriptor().shape(), right.descriptor().shape())` exactly
   once;
4. create exactly one `TensorDescriptor` from `DataType.BOOL`, the broadcast [shape](../../../../glossary.md#shape),
   `Optional.empty()` [layout](../../../../glossary.md#layout), and `false` [gradient](../../../../glossary.md#term-gradient) eligibility;
5. create exactly one `Operation(kind, NoOperationAttrs.INSTANCE)`;
6. create exactly one `TensorProvenance(operation, List.of(left, right))`;
7. call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

Null and semantic validation complete before ID allocation. Existing delegated exceptions and
messages remain unchanged:

- a public method given null fails with `NullPointerException("right")`;
- a non-floating left or right [data type](../../../../glossary.md#data-type--datatype) fails through `DataTypePromotion`, identifying the first
  invalid side in left-to-right order;
- an incompatible or locally unprovable shape pair fails through `ShapeBroadcast` with its existing
  result-axis diagnostic; and<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- exhausted [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) space fails through `TensorFactory.createDerived` only after all local
  model values have been constructed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

Do not catch, translate, aggregate, or replace these failures. Validation failures before
`createDerived` consume no Tensor ID. Do not add a production ID-inspection hook to test ordering.

### Input compatibility and result [descriptor](../../../../glossary.md#term-descriptor)

All nine ordered floating type pairs are accepted. The existing hierarchy
`BFLOAT16 < FLOAT32 < FLOAT64` determines the common conceptual comparison domain. The helper does
not add a cast [Operation](../../../../glossary.md#operation), mutate either descriptor, or expose that common type in the [BOOL](../../../../glossary.md#data-type--datatype) output.

For every successful expression:

- `dataType` is exactly `DataType.BOOL`;
- [`shape`](../../../../glossary.md#shape) is the exact [immutable](../../../../glossary.md#term-immutable) result of existing right-aligned [broadcasting](../../../../glossary.md#broadcasting);
- [`layout`](../../../../glossary.md#layout) is empty for both static and dynamic results; and
- `requiresGrad` is always false, even when either or both operands request [gradients](../../../../glossary.md#term-gradient).

The descriptor records logical result facts only. It does not retain either input descriptor,
input layout, [host storage](../../../../glossary.md#host-storage), promoted type, broadcast [strides](../../../../glossary.md#term-stride), or a [materialization](../../../../glossary.md#materialization) decision.

### [Provenance](../../../../glossary.md#provenance) and [identity](../../../../glossary.md#term-identity-meanings)

The output is a fresh [Tensor](../../../../glossary.md#tensor) with:

- a new factory-assigned [opaque](../../../../glossary.md#term-opaque) [`TensorId`](../../../../glossary.md#tensorid);
- an empty label;
- no [host storage](../../../../glossary.md#host-storage);
- one exact `BinaryComparisonKind` paired with `NoOperationAttrs.INSTANCE`; and
- ordered [immutable](../../../../glossary.md#term-immutable) provenance inputs containing exact references `[left, right]`.

Repeated inputs are valid. `tensor.equalTo(tensor)` stores the same exact Tensor reference in both
ordered positions. Repeating an equal comparison creates another Tensor identity. Symmetric
mathematical meaning does not authorize operand sorting, interning, or [common-subexpression
elimination](../../../../glossary.md#term-common-subexpression-elimination-cse).

### No numerical or [gradient](../../../../glossary.md#term-gradient) behavior

Every valid call creates the requested semantic expression without reading storage. This task does
not decide how [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), or [signed zero](../../../../glossary.md#term-signed-zero) compare and does not implement [tolerance](../../../../glossary.md#term-tolerance)-based
equality. It also does not create gradient [provenance](../../../../glossary.md#provenance) from the [BOOL](../../../../glossary.md#data-type--datatype) result back to floating inputs.
[Compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and execution tasks own those later concerns within their
architecture boundaries.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorComparisonExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryComparisonTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, [`Shape`](../../../../glossary.md#shape), `ShapeBroadcast`, `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, [`Operation`](../../../../glossary.md#operation), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), and
  `BinaryComparisonKind` Javadocs and tests.
- Existing arithmetic/unary/scalar expression contracts, focused architecture documents, ADRs,
  architecture tests, backend-conformance tests, and integration tests.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [scalar glossary](../../../../glossary.md#term-scalar); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the six exact public comparison methods,
their Javadocs, exact [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) expectations, and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Do not change
existing fields, constructor, [metadata](../../../../glossary.md#term-metadata)/storage behavior, arithmetic/[unary](../../../../glossary.md#term-unary)/[scalar](../../../../glossary.md#term-scalar) methods,
equality, hashing, diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract,
result-[gradient](../../../../glossary.md#term-gradient) behavior, resolved-[layout](../../../../glossary.md#layout) policy, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler) behavior, a
numerical edge policy, another documentation file, or more than ten paths, stop and propose a
follow-up or architecture decision. Do not create task 0015C.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update Tensor type Javadoc only as needed to include comparison expressions while preserving the
  distinction between public Tensor state, provenance, graph IR, and executable values.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Every new public method must explain ordered operand roles, floating-only compatibility,
  broadcasting, fixed BOOL output, unresolved result layout, false gradient eligibility, fresh
  identity, storage absence, provenance, and deferral of numerical execution and gradient rules.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [BOOL glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Every method must document the non-null right operand with `@param`, the fresh derived BOOL Tensor
  with `@return`, and delegated null, data-type, [shape](../../../../glossary.md#shape), and [identity](../../../../glossary.md#term-identity-meanings)-exhaustion failures with
  `@throws`.
- Document the package-private helper and `apply` method with exact validation/construction order,
  input ownership, fixed result facts, side effects, and failure behavior. The private constructor
  needs meaningful documentation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Keep all six methods independently understandable while using links to shared contracts to avoid
  unexplained repetition in generated Javadoc.
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) declares exactly the six new public one-[argument](../../../../glossary.md#term-argument) methods with [parameter](../../../../glossary.md#term-parameter) and return type
  `Tensor`; no overload, [alias](../../../../glossary.md#aliasing-and-arena), or unrelated public [API](../../../../glossary.md#term-application-programming-interface-api) is added.
- Every method maps to the exact `BinaryComparisonKind` in the contract table and delegates once to
  the shared package-private helper.
- `TensorComparisonExpressions` has exactly the specified visibility, finality, constructor,
  method, and zero-field surface.
- Null [validation](../../../../glossary.md#term-validation) order and messages are exact, and public null operands fail as `right`.
- All six methods accept all nine floating type pairs with compatible shapes and validate the
  common comparison domain through the existing promotion contract.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>
- Integral and boolean inputs are rejected without implicit casting, ID allocation, or storage
  access.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Static scalar, zero-sized, rank-mismatched, singleton-expanded, and multi-axis broadcast results
  are represented correctly.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis))</span>
- Equal symbolic dimensions and symbolic/static-singleton pairs succeed; incompatible static or
  locally unprovable dynamic pairs fail through <span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>`ShapeBroadcast`.
- Every result descriptor has exact BOOL data type, empty layout, and false gradient eligibility
  regardless of input gradient flags.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [BOOL glossary](../../../../glossary.md#data-type--datatype); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Every result is a fresh, unlabeled, storage-free Tensor with exact kind,
  `NoOperationAttrs.INSTANCE`, and [immutable](../../../../glossary.md#term-immutable) ordered exact input references.
- Self-use preserves the same input reference twice; repeated and symmetric calls are not interned,
  reordered, or canonicalized.
- No input Tensor metadata, provenance, label, storage association, or storage contents are mutated
  or retained as output storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- No numeric comparison, tolerance, gradient rule, graph state, backend fact, dependency, or
  architecture change is added.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0015B becomes Complete only after both passes. Task 0015C remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and [shape](../../../../glossary.md#shape);
- all six public method-to-kind mappings;
- exact [Operation](../../../../glossary.md#operation) and [singleton](../../../../glossary.md#term-singleton) attributes references;
- ordered exact provenance inputs, including relation direction, symmetric kinds, and repeated
  self-input;<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- all nine floating type pairs and fixed [BOOL](../../../../glossary.md#data-type--datatype) result type;
- [scalar](../../../../glossary.md#term-scalar), zero-sized, different-[rank](../../../../glossary.md#term-rank), singleton, multi-[axis](../../../../glossary.md#term-axis), and dynamic [broadcasting](../../../../glossary.md#broadcasting);
- empty layout, always-false gradient eligibility, empty label, absent host storage, fresh identity,
  and absence of canonicalization;<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [identity glossary](../../../../glossary.md#term-identity-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- null helper/public operands, non-floating operands, incompatible static shapes, and unprovable
  dynamic shapes; and
- preservation of input metadata, provenance, labels, gradient eligibility, and storage
  associations.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>

Manually inspect `javap -p -c -s`, method [bytecode](../../../../glossary.md#term-bytecode), reflection, and imports for the exact [Tensor](../../../../glossary.md#tensor)
method [descriptors](../../../../glossary.md#term-descriptor), helper shape, one-delegation paths, validation order, fixed BOOL descriptor,
and no [synchronization](../../../../glossary.md#term-synchronization) on the new public methods. Confirm no numeric access, storage operation,
cast operation, resolved [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient) propagation/rule, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) type,
cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), service, [dependency](../../../../glossary.md#term-dependency), or build change appears. Validate generated
[Javadoc](../../../../glossary.md#term-javadoc), Tensor/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary, links/anchors/fences/whitespace, exact ten-path [scope](../../../../glossary.md#term-scope),
synchronized statuses, and absence of a task-0015C specification.

## Dependencies

- Task 0001 supplies [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, floating categories, and fixed [BOOL](../../../../glossary.md#data-type--datatype) [metadata](../../../../glossary.md#term-metadata).
- Task 0002 supplies [`Shape`](../../../../glossary.md#shape) and `ShapeBroadcast` local right-aligned [broadcasting](../../../../glossary.md#broadcasting).
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition.
- Task 0007 supplies `TensorDescriptor` and enforces non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) BOOL [descriptors](../../../../glossary.md#term-descriptor).
- Task 0011 supplies public [Tensor](../../../../glossary.md#tensor) state and the exact [API](../../../../glossary.md#term-application-programming-interface-api) surface to extend.
- Task 0012 supplies centralized Tensor [identity](../../../../glossary.md#term-identity-meanings) allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered [provenance](../../../../glossary.md#provenance) and `TensorFactory.createDerived`.
- Task 0015A supplies the exact six parameterless `BinaryComparisonKind` values.

## Follow-up tasks

- 0015C remains Draft for parameterless [BOOL](../../../../glossary.md#data-type--datatype) logical semantic kinds.
- 0015D remains Draft for BOOL-only logical [Tensor](../../../../glossary.md#tensor) expressions and [broadcasting](../../../../glossary.md#broadcasting).
- 0015E–0015F remain Draft for ternary `where` semantics and public expression construction.
- 0015G–0015H remain Draft for typed cast semantics and public expression construction.
- Compiler tasks later own provenance traversal, graph capture, inference, optimization, and
  training-graph treatment.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [provenance glossary](../../../../glossary.md#provenance); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph))</span>
- Backend and conformance tasks later own exact numerical comparison behavior, mixed-floating
  lowering, storage access, and execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [lowering glossary](../../../../glossary.md#lowering))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public [Tensor](../../../../glossary.md#tensor) state, [backend](../../../../glossary.md#backend)-independent
[Operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and minimal [provenance](../../../../glossary.md#provenance) to `modules/model`. The new helper remains
inside that boundary and adds no [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), backend, [device](../../../../glossary.md#term-device), or executable state.

If implementation requires [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), numerical comparison policy, [gradient](../../../../glossary.md#term-gradient) behavior, backend
[metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0014B/0015A/0015B,
Tensor API, Compile API, Training API, glossary, current DataTypePromotion/ShapeBroadcast/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/BinaryComparisonKind contracts and
tests, and Java 26 Gradle configuration.

Implement task 0015B exactly. Modify Tensor.java and add package-private final
TensorComparisonExpressions.java for production. Update TensorTest only for the exact six-method
API surface and add TensorBinaryComparisonTest. Add exactly greaterThan/greaterOrEqual/lessThan/
lessOrEqual/equalTo/notEqualTo(Tensor), each delegating once to the shared helper and exact kind.

The helper must null-check left/right/kind, validate all floating pairs exactly once through
DataTypePromotion, broadcast exactly once through ShapeBroadcast, create an unresolved BOOL
descriptor with requiresGrad=false, create Operation(kind, NoOperationAttrs.INSTANCE), preserve
exact ordered provenance [left, right], and delegate once to TensorFactory.createDerived with no
label/storage. Every valid call returns a fresh derived Tensor. Do not inspect values/storage,
execute comparisons, define NaN/signed-zero/tolerance behavior, canonicalize or reorder operands,
insert casts, propagate gradient eligibility, define gradient rules, capture a graph, add
overloads, change existing contracts, or introduce compiler/runtime/backend behavior.

Stop beyond ten paths or on architecture uncertainty. Run every specified focused/aggregate test,
Javadoc, javap/reflection/bytecode/import/manual, documentation/link/whitespace/scope/status check.
Then hand the actual diff and evidence to a separate clean-context documentation agent in the same
change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/Tensor API/
Compile API/glossary/planning, record related-contract/capability/Training API/architecture
no-change conclusions, and rerun validation.

Update task 0015B, model master plan, and roadmap only for planning status/evidence. Do not mark
0015B Complete until both passes succeed. Leave 0015C Draft without a specification. Do not commit
or push.
```

## Local decisions

- Public method names preserve the selected legacy capability while semantic enum names remain
  descriptive uppercase identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- All six comparisons are floating-only, including equality and inequality. Integral and BOOL
  comparisons require a future explicit capability decision rather than accidental expansion.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Mixed floating pairs are accepted because the legacy public validation allowed them and the
  current model has one explicit floating-promotion hierarchy. Promotion validates the comparison
  domain; the result descriptor remains BOOL and stores no promoted type.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Comparison output always has `requiresGrad=false`. Input gradient eligibility is neither mutated
  nor propagated through a non-differentiable BOOL descriptor.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Operand order is preserved for every relation. Symmetric equality does not authorize a different
  provenance contract or eager canonicalization.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [eager glossary](../../../../glossary.md#term-eager-evaluation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- No comparison attrs record is introduced because every kind is parameterless and
  `NoOperationAttrs.INSTANCE` is complete.

## Known limitations

- The task constructs semantic comparison expressions but does not calculate [boolean](../../../../glossary.md#data-type--datatype) values.
- NaN, infinity, signed-zero, equality tolerance, and cross-precision execution behavior remain
  unspecified until backend/conformance work.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [tolerance glossary](../../../../glossary.md#term-tolerance); [precision glossary](../../../../glossary.md#term-precision); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Integral and [BOOL](../../../../glossary.md#data-type--datatype) comparisons are not supported by this selected public baseline.
- No compiler capture, training-graph handling, gradient rule, backend support, or execution is
  implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, 0014B, and 0015A; current [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), promotion,
broadcast, [operation](../../../../glossary.md#operation), and comparison-kind source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs and glossary;
and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms six public Tensor
method names, floating-only local eligibility, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [broadcasting](../../../../glossary.md#broadcasting), [BOOL](../../../../glossary.md#data-type--datatype)/no-[gradient](../../../../glossary.md#term-gradient) results,
ordered operands, and execution uses as [masks](../../../../glossary.md#term-boolean-mask) and `where` conditions. [Mutable](../../../../glossary.md#term-mutable) legacy broadcast
plans, [runtime](../../../../glossary.md#term-runtime) coupling, traits, [kernels](../../../../glossary.md#kernel), execution, and gradient/[compiler](../../../../glossary.md#term-compiler) machinery are excluded.

Planning selected one helper and one focused test because all six methods share the same validation,
descriptor, provenance, and factory path. The task adds no package, dependency, foundation change,
or architecture decision. Task 0015C remains only the next Draft queue entry.

Planning validation:

- `git diff --check` passed, and targeted whitespace inspection found no trailing whitespace in
  the three changed planning paths.
- The required-section scan found every canonical task-specification section, including package
  impact, exact scope, validation, implementation handoff, decisions, limitations, and completion
  evidence sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The relative Markdown-[target](../../../../glossary.md#term-target) [scan](../../../../glossary.md#term-cumulative-scan) resolved every local `.md` link in this task, the model master
  plan, and the roadmap. Markdown fence counts are balanced and no changed link uses an unresolved
  heading anchor.
- Status inspection found 0015B [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0015C remains `Draft` in both queues.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other module path changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>
- No task-0015C specification exists.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0015b` added the six exact public Tensor methods,
  one package-private helper, the focused comparison test, and the exact Tensor API-shape test
  updates. Independent documentation context<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
  `/root/implement_model_0015b/review_model_0015b_docs` then reread the architecture contract,
  focused model/compiler/training architecture pages, documentation and planning rules, current
  APIs, glossary, capabilities/master plans, relevant completed tasks, actual diff/source/tests,
  test reports, bytecode, imports, and generated Javadoc. It applied General plus API/Javadoc
  style to API and Javadoc review, Planning style to planning files, and Example format to the new
  complete comparison-expression example.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest` passed during the final pass
  from the Gradle build cache. The final fresh aggregate execution contains all 9 focused tests
  with zero failures, errors, or skips.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed during the final pass from the Gradle
  build cache. The final fresh aggregate execution contains all 14 Tensor tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:test` passed, then
  `./gradlew :modules:model:test --rerun-tasks` freshly compiled and executed the suite. The 42 XML
  suites contain 317 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc` passed up to date, then
  `./gradlew :modules:model:javadoc --rerun-tasks` freshly regenerated Javadoc without warnings.
  Generated `Tensor.html` contains exactly one detail section for each of the six methods, with
  parameter, return, failure, fixed BOOL/false-gradient, provenance, and deferred-behavior text.
  The package-private helper Javadoc was reviewed in source because public Javadoc generation does
  not publish package-private types.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- `./gradlew test` passed for the repository with 36 actionable tasks up to date and no failure.
  Architecture, backend-conformance, and integration test modules currently have no test sources;
  the root run still compiled their existing module markers without error.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test); [run glossary](../../../../glossary.md#term-run))</span>
- The complete `ComparisonExpressionExample` was compiled with
  `javac --release 26 -cp modules/model/build/classes/java/main -d /private/tmp` and run with the
  model classes. Its ten output lines exactly matched the documented `BOOL`, `Shape[2, 3]`,
  unresolved-layout, false-gradient, unlabeled, storage-free, `LESS_OR_EQUAL`, parameterless,
  ordered-input, and fresh-identity results. The temporary source was removed and is not in scope.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `javap -p -c -s` confirms exactly six `(Tensor)Tensor` public methods, each loading the exact
  kind and invoking `TensorComparisonExpressions.apply` once. The helper is final and package-
  private with zero fields, one private zero-argument constructor, and one package-private static
  method; bytecode confirms left/right/kind null checks, one floating-promotion call, one broadcast
  call, fixed <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`DataType.BOOL`/empty-[layout](../../../../glossary.md#layout)/false-gradient descriptor, one Operation, ordered
  `List.of(left, right)` provenance, and one `createDerived` call. No monitor instruction or
  synchronized comparison method exists.
- Reflection tests confirm the exact helper and six-method API shapes, no overloads, and no
  synchronization. Import and bytecode scans found only model-owned data-type, shape, operation,
  Tensor, and JDK collection/null-check contracts; no graph/compiler/planning/prepare/runtime/
  backend type, storage access, cast, gradient rule, registry, service, numerical execution, or
  build dependency appears.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [registry glossary](../../../../glossary.md#term-registry))</span>
- The first two local-link validator attempts failed because the inline Ruby checker used an
  interpolated heading regex and then a `filter_map` method unavailable in the installed Ruby.
  After replacing those checker constructs, one run reported the existing
  `compiled-graph--compiledgraphmodel` anchor falsely because whitespace was collapsed unlike
  GitHub heading rules. The corrected validator preserves adjacent heading spaces as adjacent
  hyphens and then resolved every local file target and heading anchor across all six changed
  documentation/planning files. These were validation-script defects, not documentation defects.
- Markdown fence counts are balanced; targeted scans found no trailing whitespace in any of the
  ten paths; terminology and implementation-status scans found no stale claim that comparison
  construction is planned; and `git diff --check` passed.
- Final scope contains exactly the ten authorized paths: two production files, two test files,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0015B is
  synchronized as Complete in all three planning locations. Task 0015C remains Draft and no
  task-0015C specification exists.
- The independent documentation pass found the Tensor type, all six public method Javadocs, and
  helper type/constructor/apply Javadocs complete and accurate, so it changed no Java declaration,
  executable logic, or test. Tensor API now documents the implemented comparison surface and a
  complete runnable metadata example. Compile API now inventories the six current comparison
  methods while keeping compiler entry, traversal, capture, inference, optimization, artifacts,
  and engine behavior planned.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact); [engine glossary](../../../../glossary.md#term-engine))</span>
- Existing glossary terms were sufficient. Implementation status, [`OperationKind`](../../../../glossary.md#operationkind), Provenance,
  Tensor, and common distinctions were updated for comparison construction; no new reusable
  domain term was introduced.
- Training API remains accurate unchanged because false descriptor gradient eligibility adds no
  gradient object, rule, autograd, optimizer, session, or training-graph treatment. Model
  capabilities remain accurate unchanged because they already select all six public comparison
  names, floating-category rules, broadcasting, BOOL results, and layer separation.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture pages, and ADRs remain accurate unchanged because the
  change stays within model-owned Tensor semantics, descriptors, and provenance and adds no module
  boundary, dependency, lifecycle, backend, runtime, or training decision. Architecture tests need
  no update because dependency rules did not change. Backend-conformance and integration tests
  need no update because no numerical backend behavior or end-to-end execution exists yet.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Java 26/Gradle configuration remains accurate unchanged because the change uses existing stable
  language/library contracts and adds no dependency, preview/incubator feature, or build change.
  Existing <span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature))</span>`DataTypePromotion`, `ShapeBroadcast`, `TensorDescriptor`, `TensorProvenance`,
  `TensorFactory.createDerived`, `Operation`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), `BinaryComparisonKind`, and
  arithmetic/unary/scalar expression Javadocs and tests remain accurate because this task composes
  their existing contracts without changing them.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [scalar glossary](../../../../glossary.md#term-scalar))</span>

## Implementation notes

- Added exactly six public [Tensor](../../../../glossary.md#tensor) comparison methods and one package-private stateless helper.
- Every successful comparison creates a fresh, unlabeled, storage-free unresolved BOOL result with
  false gradient eligibility and exact ordered operation provenance after existing floating and
  local-broadcast validation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- The independent documentation pass finalized Tensor API, Compile API, glossary, and planning
  status/evidence. It preserved the implementation and already-complete Javadocs unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>
- No compiler, training, architecture, dependency, numerical execution, or backend behavior was
  added.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: Implemented and documented the six floating-only, broadcast-aware public
  binary comparison Tensor expressions with fixed BOOL results and ordered provenance.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Tensor glossary](../../../../glossary.md#tensor); [BOOL glossary](../../../../glossary.md#data-type--datatype); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the authorized two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused comparison tests 9/9, focused Tensor tests 14/14, all 317 model
  tests across 42 suites, regenerated model Javadoc, root tests, example compilation/execution,
  bytecode/reflection/import/absence checks, local links/anchors, fences, terminology, whitespace,
  exact scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean documentation context
  `/root/implement_model_0015b/review_model_0015b_docs` completed the independent pass using
  General, API/Javadoc, Planning, and Example-format guidance.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and Compile API now describe current comparison expression
  construction while preserving all compiler and execution lifecycle behavior as planned.
  Training API, capabilities, architecture/ADRs/tests, conformance/integration tests, build
  configuration, and related foundational/expression documentation remain accurate unchanged for
  the reasons recorded above.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [lifecycle glossary](../../../../glossary.md#lifecycle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Javadoc review: Tensor and helper Javadocs fully document ordered inputs, floating validation,
  broadcasting, fixed result facts, identity/storage/provenance, failures, and deferred behavior;
  no documentation-pass Java edit was needed.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Glossary impact: Existing implementation-status, [`OperationKind`](../../../../glossary.md#operationkind), Provenance, Tensor, and common-
  distinction text now includes comparison construction; no new reusable term was needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0015B. Task 0015C remains Draft without a specification.

Status: Complete
