# Task 0015H: Cast [Tensor](../../../../glossary.md#tensor) Expression

## Status

Complete

## Goal

Expose the implemented `CAST` semantic family through one public, [backend](../../../../glossary.md#backend)-independent
`Tensor.cast(DataType targetDataType)` expression method. Every non-null [target](../../../../glossary.md#term-target) [data type](../../../../glossary.md#data-type--datatype) must be
representable from every current source data type, including a target equal to the source. Every
successful call returns a fresh storage-free [Tensor](../../../../glossary.md#tensor) whose [descriptor](../../../../glossary.md#term-descriptor) retains the exact input
[Shape](../../../../glossary.md#shape), changes only the requested data type and derived [gradient](../../../../glossary.md#term-gradient) eligibility, and whose [provenance](../../../../glossary.md#provenance)
records the exact input plus `CastKind.CAST` and `CastAttrs(targetDataType)`.

This task constructs explicit [cast expression](../../../../glossary.md#cast-expression) [metadata](../../../../glossary.md#term-metadata) only. It does not read or convert values,
define numerical conversion rules, allocate result storage, create gradient rules, [capture](../../../../glossary.md#term-capture-meanings) a
[compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), remove redundant casts, or define backend support.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public instance method, `Tensor.cast(DataType targetDataType)`.
- Add one package-private final `TensorCastExpressions` helper with one package-private static
  `apply(Tensor, DataType)` entry and no other behavior.
- Require a non-null input and [target](../../../../glossary.md#term-target) [data type](../../../../glossary.md#data-type--datatype) with deterministic [parameter](../../../../glossary.md#term-parameter)-name messages.
- Accept every one of the 36 current source/target [`DataType`](../../../../glossary.md#data-type--datatype) pairs without compatibility lookup,
  implicit conversion, or backend-capability validation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [backend glossary](../../../../glossary.md#backend); [validation glossary](../../../../glossary.md#term-validation))</span>
- Retain the input [descriptor](../../../../glossary.md#term-descriptor)'s exact [immutable](../../../../glossary.md#term-immutable) [`Shape`](../../../../glossary.md#shape) reference.
- Leave result layout unresolved, even when the input layout is resolved or the target equals the
  source data type.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Derive result [gradient](../../../../glossary.md#term-gradient) eligibility as
  `input.requiresGrad && sourceDataType.isFloating() && targetDataType.isFloating()`.
- Construct exactly one [`Operation`](../../../../glossary.md#operation) from `CastKind.CAST` and one new
  `CastAttrs(targetDataType)`.
- Construct exactly one `TensorProvenance` with the exact ordered input list `[input]`.
- Delegate final [identity](../../../../glossary.md#term-identity-meanings)-bearing construction exactly once to `TensorFactory.createDerived` with
  no label and no storage.
- Return a fresh explicit cast expression for same-type requests; do not return the input or
  eliminate the operation in the model API.<span data-glossary-navigation="true"> ([cast expression glossary](../../../../glossary.md#cast-expression); [model glossary](../../../../glossary.md#term-model))</span>
- Update the exact [Tensor](../../../../glossary.md#tensor) public-[API](../../../../glossary.md#term-application-programming-interface-api) reflection test and add one focused cast-expression test.
- Finalize affected Javadocs, Tensor API, Compile API status, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- eager value conversion, host-storage reads or writes, allocation, copy, materialization,
  mutation, aliasing, or output storage attachment<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- numerical conversion policy, including rounding, truncation, saturation, overflow, underflow,
  precision loss, NaN, infinity, signed zero, BFLOAT16 bit conversion, BOOL zero/non-zero meaning,
  or invalid-value behavior<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [saturation glossary](../../../../glossary.md#term-saturation); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [precision loss glossary](../../../../glossary.md#term-precision-loss); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- returning the input for same-type requests, constant folding, identity elimination, cast-chain
  simplification, canonicalization, interning, common-subexpression elimination, or caching<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding); [identity glossary](../../../../glossary.md#term-identity-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- source/target compatibility tables, backend-support checks, conversion modes, rounding modes,
  attributes beyond the existing target data type, or operation registries<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [backend glossary](../../../../glossary.md#backend); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- resolved output layout, input-layout propagation, stride preservation, view preservation,
  materialization policy, or shape inference beyond exact reference retention<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference))</span>
- gradient values, gradient callbacks, backward operations, cast-back rules, autograd expansion,
  optimizer behavior, training-root policy, or training execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- changes to [`DataType`](../../../../glossary.md#data-type--datatype), `CastKind`, `CastAttrs`, [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`,
  `TensorProvenance`, `TensorFactory`, or another existing Java contract
- another public overload, static cast entry, alias, factory, builder, expression public type, or
  caller-supplied label<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- graph traversal, graph IDs, cycle checks, graph capture, compiled graph construction,
  publication, compiler inference, optimization, or artifacts<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [publication glossary](../../../../glossary.md#term-publication); [compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- planning ownership, capability providers, backend selection, fusion, costs, lowering, kernels,
  prepare, runtime residency, execution, tracing, ONNX mapping, or engine behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [ONNX glossary](../../../../glossary.md#term-onnx); [engine glossary](../../../../glossary.md#term-engine))</span>
- dependencies, Gradle changes, architecture changes, another module, unrelated refactors, or a
  detailed task-0016 specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0015G](0015g-cast-semantic-kind-and-attributes.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes fluent `Tensor.cast(DataType targetType)`. Its
public builder accepts conversions among floating, [BFLOAT16](../../../../glossary.md#term-bfloat16), integral, and [BOOL](../../../../glossary.md#data-type--datatype) data types,
preserves the input [shape](../../../../glossary.md#shape), disables [gradient](../../../../glossary.md#term-gradient) eligibility when either side is non-floating, and
supports floating-to-floating backward conversion. Legacy tests cover same-type requests, mixed
floating [precision](../../../../glossary.md#term-precision), BFLOAT16, integral/floating and BOOL/floating conversion, strided inputs,
expression chaining, gradients, [ONNX](../../../../glossary.md#term-onnx) mapping, and [CPU](../../../../glossary.md#term-cpu) and [Metal](../../../../glossary.md#term-metal) execution routes.

Legacy returns the original [Tensor](../../../../glossary.md#tensor) when source and [target](../../../../glossary.md#term-target) types are equal. The new [model](../../../../glossary.md#term-model) does not
copy that [eager](../../../../glossary.md#term-eager-evaluation) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form). A public call records the caller's explicit semantic request;
later [compiler](../../../../glossary.md#term-compiler) optimization owns removal of redundant casts and cast-chain simplification. This
preserves representability and keeps model construction deterministic and consistent with the
current expression APIs, while deliberately moving the optimization to its architectural [owner](../../../../glossary.md#term-owner).

Legacy code also installs Tensor-local gradient callbacks, performs storage conversion, carries
[operation](../../../../glossary.md#operation) traits, and couples construction to executable routes. Those mechanisms are not copied.
This task retains only public model expression construction. Compiler [autograd](../../../../glossary.md#autograd) later owns [backward
graph](../../../../glossary.md#backward-graph) expansion; compiler optimization owns canonicalization; concrete [backends](../../../../glossary.md#backend) and [conformance](../../../../glossary.md#term-conformance)
work own numerical conversion, [lowering](../../../../glossary.md#lowering), storage access, and execution.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and must not become an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node) or executable value.
- `TensorCastExpressions` performs deterministic local model construction only. It must not read
  values or storage, traverse provenance, capture a graph, query backend support, or execute a
  conversion.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [provenance glossary](../../../../glossary.md#provenance); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- [`Operation`](../../../../glossary.md#operation) owns only the exact [backend](../../../../glossary.md#backend)-independent `CAST` kind and target attributes. Source
  type belongs to the input descriptor and must not be duplicated in <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`CastAttrs`.
- `TensorProvenance` owns the exact one-input relationship. Its ordered list is exactly `[input]`.
- Result [identity](../../../../glossary.md#term-identity-meanings) comes only from the existing package-private `TensorFactory.createDerived` seam.
  No second allocator, caller-supplied ID, registry, cache, or service is introduced.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- Every successful call creates a fresh, unlabeled, storage-free Tensor, including same-type casts.
  Model construction does not perform compiler-owned canonicalization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Every current source/target pair is representable. This does not promise executable support from
  every backend or define conversion results for edge values.
- The result Shape is the input descriptor's exact immutable reference. Cast does not broadcast,
  reshape, infer, reconstruct, or normalize dimensions.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [immutable glossary](../../../../glossary.md#term-immutable); [dimensions glossary](../../../../glossary.md#dimension))</span>
- Result layout is always unresolved. A change of element width may invalidate input byte geometry,
  while even a same-type cast is a new logical value whose physical realization is not established
  by expression construction.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element))</span>
- Result gradient eligibility is true only when the input requested gradients and both source and
  target data types are floating. This is descriptor metadata, not a gradient-rule or backend
  differentiability guarantee.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Package direction is `model.tensor -> model.operation.elementwise.cast`, plus existing
  `model.tensor -> model.operation` and `model.datatype`. The cast package must not import Tensor,
  and no package cycle may be introduced.
- Stop if implementation requires a foundational contract change, conversion policy, storage
  access, graph capture, gradient rule, backend query, dependency, or architecture decision.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns the public cast method, local result-descriptor
  construction, provenance, and package-private derived Tensor factory seam.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- `io.github.pho001.synaptik.model.operation` — supplies [immutable](../../../../glossary.md#term-immutable) [`Operation`](../../../../glossary.md#operation) composition.
- `io.github.pho001.synaptik.model.operation.elementwise.cast` — supplies the exact `CAST` identity
  and immutable target-data-type attributes.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies the source and target type vocabulary and
  floating-category metadata.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public fluent expression surface; it receives
  only the exact `cast` method and delegates all construction behavior.
- `io.github.pho001.synaptik.model.tensor.TensorCastExpressions` — package-private, stateless local
  cast-expression boundary colocated with Tensor, descriptors, provenance, and the package-private
  factory seam it must use.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `TensorCastExpressionTest` — same-package focused test so it can verify helper shape and behavior
  without widening production visibility.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [widening glossary](../../../../glossary.md#term-widening))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly this public instance method to `Tensor`:

```java
public Tensor cast(DataType targetDataType)
```

It delegates exactly once and returns the exact result:

```java
return TensorCastExpressions.apply(this, targetDataType);
```

The method is public, final only through the final enclosing class, non-static, non-synchronized,
and has no overload or [alias](../../../../glossary.md#aliasing-and-arena). It performs no [validation](../../../../glossary.md#term-validation), [descriptor](../../../../glossary.md#term-descriptor) construction, factory call, or
other work outside the one helper delegation.

### Helper [shape](../../../../glossary.md#shape)

Create exactly one package-private final production helper:

```java
final class TensorCastExpressions {
    private TensorCastExpressions() {
    }

    static Tensor apply(Tensor input, DataType targetDataType) {
        // exact validation and construction contract below
    }
}
```

The helper has zero fields, zero nested types, one private zero-[argument](../../../../glossary.md#term-argument) constructor, and exactly
one package-private static `apply` method. It declares no public or protected [API](../../../../glossary.md#term-application-programming-interface-api), overload,
factory, [cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), compatibility table, or test hook.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly this order:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(targetDataType, "targetDataType")`.
3. Read `sourceDataType` from `input.descriptor().dataType()` and read the exact input [Shape](../../../../glossary.md#shape).
4. Derive `requiresGrad` as input eligibility AND floating source AND floating [target](../../../../glossary.md#term-target).
5. Create one `TensorDescriptor` from target type, exact input Shape reference,
   `Optional.empty()`, and derived eligibility.
6. Create one `CastAttrs` from the exact target reference.
7. Create one [`Operation`](../../../../glossary.md#operation) from `CastKind.CAST` and the exact attributes value.
8. Create one `TensorProvenance` from that operation and `List.of(input)`.
9. Delegate exactly once to
   `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` and return its exact
   result.

No source/target pair is rejected after the two null checks. A same-type request follows all nine
steps and creates a fresh explicit [cast expression](../../../../glossary.md#cast-expression); it does not return before [descriptor](../../../../glossary.md#term-descriptor) or
[identity](../../../../glossary.md#term-identity-meanings) construction. Failures before the factory call consume no [Tensor](../../../../glossary.md#tensor) ID. A valid request may
fail only if the central allocator is exhausted, after local [immutable](../../../../glossary.md#term-immutable) [model](../../../../glossary.md#term-model) values are built.

### Result [descriptor](../../../../glossary.md#term-descriptor) and [gradient](../../../../glossary.md#term-gradient) eligibility

The result descriptor has exactly:

- `dataType == targetDataType`;
- `shape == input.descriptor().shape()` by reference;
- `layout().isEmpty()`; and
- `requiresGrad == input.descriptor().requiresGrad()
  && sourceDataType.isFloating() && targetDataType.isFloating()`.

All 36 current source/[target](../../../../glossary.md#term-target) pairs are valid. `TensorDescriptor` already prevents true gradient
eligibility on non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) source [tensors](../../../../glossary.md#tensor), but the helper still expresses the complete
source-and-target rule explicitly. Floating-to-floating casts retain an existing true request.
Floating-to-integral, floating-to-[BOOL](../../../../glossary.md#data-type--datatype), integral-to-floating, BOOL-to-floating, integral-to-
integral, and BOOL-related casts produce false eligibility. No backward rule is created.

The exact input [Shape](../../../../glossary.md#shape) reference is valid for [scalar](../../../../glossary.md#term-scalar), zero-sized, static, and dynamic shapes.
Resolved input [layout](../../../../glossary.md#layout) is never copied. The helper does not construct dense layout, preserve [view](../../../../glossary.md#term-view-meanings)
geometry, or access [referenced element span](../../../../glossary.md#referenced-element-span).

### [Operation](../../../../glossary.md#operation) and [provenance](../../../../glossary.md#provenance)

Every result has present provenance containing:

```java
new Operation(CastKind.CAST, new CastAttrs(targetDataType))
```

and exact ordered inputs:

```java
List.of(input)
```

The attributes retain the exact [target](../../../../glossary.md#term-target) enum reference. The operation does not duplicate the source
type, input, result [descriptor](../../../../glossary.md#term-descriptor), conversion policy, or [backend](../../../../glossary.md#backend) fact. Repeated calls create fresh
[Tensor](../../../../glossary.md#tensor), attributes, operation, and provenance values. A cast chain records only its immediate
input at each link, so `input.cast(A).cast(B)` remains two explicit expressions until a [compiler](../../../../glossary.md#term-compiler)
chooses a legal rewrite.

### [Ownership](../../../../glossary.md#term-ownership-meanings) and side effects

The input [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor), [Shape](../../../../glossary.md#shape), label, [provenance](../../../../glossary.md#provenance), [host](../../../../glossary.md#term-host)-storage association, and storage
contents remain unchanged. The output has empty label and no [host storage](../../../../glossary.md#host-storage). The helper retains the
input only through [immutable](../../../../glossary.md#term-immutable) provenance and retains the input Shape through the result descriptor.
It never retains storage as output storage, reads memory, converts a value, or mutates either
Tensor.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressionTest.java`

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
- Existing [`DataType`](../../../../glossary.md#data-type--datatype), `TensorDescriptor`, `TensorProvenance`, `TensorFactory`, [`Operation`](../../../../glossary.md#operation),
  `CastKind`, and `CastAttrs` Javadocs/tests.
- Existing expression contracts, focused architecture documents, ADRs, architecture tests,
  backend-conformance tests, integration tests, and Gradle configuration.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, and six documentation/[planning](../../../../glossary.md#term-planning) files: ten paths
total.

`Tensor.java` and `TensorTest.java` may change only for the one exact public `cast` method, its
import/Javadocs, exact [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) expectations, and non-[synchronization](../../../../glossary.md#term-synchronization) assertions. Do not change
existing fields, constructor, [metadata](../../../../glossary.md#term-metadata)/storage behavior, expression methods, equality, hashing,
diagnostics, or unrelated tests.

If implementation needs another production/test concept, a changed foundational contract,
source/[target](../../../../glossary.md#term-target) [matrix](../../../../glossary.md#term-matrix-meanings), numerical policy, resolved-[layout](../../../../glossary.md#layout) policy, storage access, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)
behavior, [gradient](../../../../glossary.md#term-gradient) rule, another documentation file, or more than ten paths, stop and propose a
follow-up or architecture decision. Do not create task 0016.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update Tensor type Javadoc only as needed to include explicit cast expression construction while
  preserving the distinctions among public Tensor state, provenance, graph IR, gradient
  eligibility, and executable values.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [cast expression glossary](../../../../glossary.md#cast-expression); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `Tensor.cast` must explain the target role, acceptance of every current source/target pair,
  exact Shape retention, unresolved result layout, gradient-eligibility rule, fresh identity
  including same-type requests, absent label/storage, operation attributes, one-input provenance,
  and deferred numerical conversion, canonicalization, gradients, and execution.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Document `targetDataType` with `@param`, the fresh result with `@return`, and exact null and
  identifier-exhaustion failures with `@throws`.
- Document the package-private helper, private constructor, and `apply` method with exact
  validation/construction order, reference ownership, source/target roles, ID side effects, and
  failure behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Explain why the model does not return the receiver for a same-type request and that compiler
  optimization owns redundant-cast removal.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Explain that gradient eligibility is descriptor metadata only and does not create a cast-back
  gradient rule or promise backend differentiability.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [backend glossary](../../../../glossary.md#backend))</span>
- Review related foundational and expression Javadocs and record why they remain accurate or stop
  on an out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) declares exactly one new `public Tensor cast(DataType)` method; no static form, overload,
  alias, or unrelated public API is added.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The public method delegates once to `TensorCastExpressions.apply` and performs no other work.
- `TensorCastExpressions` has exactly the specified visibility, finality, constructor, method,
  fields, and nested-type shape.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Null validation, source/shape reads, eligibility derivation, descriptor, attributes, operation,
  provenance, and factory delegation occur in the exact specified order.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- All 36 current source/[target](../../../../glossary.md#term-target) pairs succeed without compatibility or [backend](../../../../glossary.md#backend) [lookup](../../../../glossary.md#term-lookup).
- Every result has exact target type, exact input Shape reference, empty layout, derived gradient
  eligibility, empty label, no host storage, fresh identity, exact operation/attributes, and exact
  one-input provenance.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [host storage glossary](../../../../glossary.md#host-storage); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Same-type requests return a fresh explicit CAST expression and consume one identity; they never
  return the input or disappear during model construction.<span data-glossary-navigation="true"> ([CAST expression glossary](../../../../glossary.md#cast-expression); [model glossary](../../../../glossary.md#term-model))</span>
- Floating-to-floating casts preserve a true input request. Every cast with a non-floating source
  or target has false result eligibility. False input eligibility remains false for all targets.
- Scalar, zero-sized, fully static, dynamic, resolved-layout, unresolved-layout, leaf, and derived
  inputs are accepted without Shape reconstruction or storage access.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Repeated calls and cast chains remain fresh and explicit, with each provenance value retaining
  only its exact immediate input.
- No input Tensor metadata, provenance, label, storage association, or storage contents are
  mutated or attached as output storage.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- No value conversion, numerical policy, gradient rule, graph state, backend fact, dependency,
  build option, or architecture change is added.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/bytecode/
  scope checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate documentation-focused agent finalizes Javadocs, Tensor API, Compile API, glossary,
  task evidence, master plan, and roadmap in the same change and records reasoned no-change
  conclusions for Training API, capabilities, architecture, and related contracts.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0015H becomes Complete only after both passes. Task 0016 remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact helper class/constructor/method/field visibility and [signature](../../../../glossary.md#term-signature);
- exact public `cast(DataType)` signature, visibility, instance form, non-synchronization, and one
  delegation;
- every one of the 36 current source/[target](../../../../glossary.md#term-target) pairs and exact result target type;
- exact [Shape](../../../../glossary.md#shape) reference retention for [scalar](../../../../glossary.md#term-scalar), zero-sized, static, and dynamic inputs;
- empty result [layout](../../../../glossary.md#layout) for resolved, unresolved, and same-type inputs;
- exact gradient-eligibility matrix, including true and false floating inputs and every
  non-floating source/target category;
- exact `CastKind.CAST`, exact `CastAttrs` type and target, and exact one-input [provenance](../../../../glossary.md#provenance);
- fresh same-type results, repeated calls, and immediate-input cast chains without
  canonicalization;<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- empty label, absent storage, fresh [identity](../../../../glossary.md#term-identity-meanings), and unchanged input [metadata](../../../../glossary.md#term-metadata)/storage/content;
- helper/public null validation and no identity allocation for null target; and
- identifier-exhaustion propagation only after valid local construction.

Manually inspect `javap -p -c -s`, method [bytecode](../../../../glossary.md#term-bytecode), reflection, and imports for the exact [Tensor](../../../../glossary.md#tensor)
method [descriptor](../../../../glossary.md#term-descriptor), helper shape, sole delegation, validation/construction order, complete
eligibility expression, exact Shape reference, empty layout/label, fixed `CAST` and typed target
attributes, one-input provenance, and absence of [synchronization](../../../../glossary.md#term-synchronization) on new public/helper entries.
Confirm no numeric or storage access, source/target [matrix](../../../../glossary.md#term-matrix-meanings), [backend](../../../../glossary.md#backend) [lookup](../../../../glossary.md#term-lookup), conversion loop,
same-type early return, resolved layout, [gradient](../../../../glossary.md#term-gradient) rule, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/backend type, cost,
[fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), service, [dependency](../../../../glossary.md#term-dependency), or build change appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc),
Tensor/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary, links/anchors/fences/whitespace, exact ten-path [scope](../../../../glossary.md#term-scope),
synchronized statuses, and absence of a task-0016 specification.

## Dependencies

- Task 0001 supplies the exact six-value [`DataType`](../../../../glossary.md#data-type--datatype) vocabulary and floating/differentiable
  metadata.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition.
- Task 0007 supplies `TensorDescriptor` and its differentiability [validation](../../../../glossary.md#term-validation).
- Task 0011 supplies public [Tensor](../../../../glossary.md#tensor) state and the exact [API](../../../../glossary.md#term-application-programming-interface-api) surface to extend.
- Task 0012 supplies centralized Tensor [identity](../../../../glossary.md#term-identity-meanings) allocation through `TensorFactory`.
- Task 0013 supplies immutable ordered [provenance](../../../../glossary.md#provenance) and `TensorFactory.createDerived`.
- Task 0015G supplies exact `CastKind.CAST` semantics and `CastAttrs(targetDataType)`.

## Follow-up tasks

- Task 0016 remains Draft for reduction and [scan](../../../../glossary.md#term-cumulative-scan) [operation](../../../../glossary.md#operation) semantics and [Tensor](../../../../glossary.md#tensor) expressions.
- Compiler tasks later own provenance traversal, graph capture, redundant-cast and cast-chain
  canonicalization, autograd expansion, optimization legality, and compile artifacts.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [provenance glossary](../../../../glossary.md#provenance); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Backend, ONNX, and conformance tasks later own mapping, supported conversion matrices, rounding,
  truncation/saturation, BOOL behavior, special values, lowering, storage access, and execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [rounding glossary](../../../../glossary.md#term-rounding); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [saturation glossary](../../../../glossary.md#term-saturation); [BOOL glossary](../../../../glossary.md#data-type--datatype); [lowering glossary](../../../../glossary.md#lowering))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public [Tensor](../../../../glossary.md#tensor) state, [backend](../../../../glossary.md#backend)-independent
[Operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and minimal [provenance](../../../../glossary.md#provenance) to `modules/model`. This task composes
those contracts and leaves [compiler](../../../../glossary.md#term-compiler)-owned [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) plus backend-owned conversion and
execution in their existing layers.

If implementation requires storage conversion, a [gradient](../../../../glossary.md#term-gradient) rule, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction), backend [metadata](../../../../glossary.md#term-metadata),
another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0006/0007/0011/0012/0013/0015G/0015H, Tensor API,
Compile API, Training API, glossary, current DataType/TensorDescriptor/Tensor/TensorFactory/
TensorProvenance/Operation/CastKind/CastAttrs contracts and tests, and Java 26 Gradle configuration.

Implement task 0015H exactly. Modify Tensor.java and add package-private final
TensorCastExpressions.java for production. Update TensorTest only for the exact one-method public
API surface and add TensorCastExpressionTest. Add exactly cast(DataType), delegating once to the
shared helper.

The helper has exactly one package-private static apply(input,targetDataType) method. Follow the
task's exact input/target/source/descriptor/attributes/operation/provenance/construction order and
messages. Accept all 36 current source/target pairs. Retain the exact input Shape, leave layout
unresolved, and set requiresGrad only when the input requested it and both types are floating.
Create exact CAST/CastAttrs semantics, exact one-input provenance, and delegate once to
TensorFactory.createDerived with no label/storage. Every valid call, including same-type cast,
returns a fresh explicit expression; compiler work later owns redundant-cast elimination.

Do not inspect or convert values/storage, return the input, add a compatibility matrix or numerical
policy, preserve resolved layout, define gradient rules, capture a graph, add overloads, change
existing contracts, or introduce compiler/runtime/backend behavior. Stop beyond ten paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0015H, model master plan, and roadmap only for planning status/evidence. Do not mark
0015H Complete until both passes succeed. Leave 0016 Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public method keeps the legacy-compatible fluent name and [target](../../../../glossary.md#term-target)-only [signature](../../../../glossary.md#term-signature):
  `input.cast(targetDataType)`. Source type remains an input-[descriptor](../../../../glossary.md#term-descriptor) fact.
- All 36 current source/target pairs are representable. Executable conversion support and exact
  numerical results remain backend and conformance responsibilities rather than model validation.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation))</span>
- Same-type calls create fresh explicit expressions. Legacy returned the input, but that eagerly
  erased the requested semantic operation. The current architecture assigns redundant-operation
  canonicalization to compiler optimization, and all existing public expression methods preserve
  explicit caller requests as fresh Tensor state.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The result retains the exact input Shape reference because cast is elementwise and changes no
  logical dimensions. It leaves layout unresolved because expression construction does not assert
  physical geometry or storage reuse, particularly across element-width changes.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [elementwise glossary](../../../../glossary.md#term-elementwise); [dimensions glossary](../../../../glossary.md#dimension); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element))</span>
- Gradient eligibility follows the legacy semantic boundary: it survives only an already-eligible
  floating-to-floating cast. The task creates no backward rule; compiler autograd later decides
  whether and how to cast gradients back.<span data-glossary-navigation="true"> ([Gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd))</span>
- One single-entry package-private helper is sufficient because the family has one semantic kind
  and one public arity. A generic conversion framework would add abstraction without another
  current use.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity))</span>

## Known limitations

- The result records cast semantics and [provenance](../../../../glossary.md#provenance) but contains no converted values or storage.
- Numerical behavior for precision loss, overflow, truncation, BFLOAT16, BOOL, NaN, infinity, and
  signed zero remains unspecified until backend/conformance work.<span data-glossary-navigation="true"> ([precision loss glossary](../../../../glossary.md#term-precision-loss); [overflow glossary](../../../../glossary.md#term-overflow); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [NaN, infinity, and signed zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Gradient](../../../../glossary.md#term-gradient) eligibility is [metadata](../../../../glossary.md#term-metadata) only; no [autograd](../../../../glossary.md#autograd) rule or cast-back expression is implemented.
- Same-type and redundant chained casts remain explicit until a future compiler optimization
  proves and performs their removal.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- No compiler capture, ONNX mapping, backend support, native storage conversion, or execution is
  implied.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [ONNX glossary](../../../../glossary.md#term-onnx))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the [architecture contract](../../../../glossary.md#architecture-contract) and focused [module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[lifecycle](../../../../glossary.md#lifecycle) explanations;
documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan; tasks 0001, 0006,
0007, 0011, 0012, 0013, and 0015G; current [DataType](../../../../glossary.md#data-type--datatype), TensorDescriptor, [Tensor](../../../../glossary.md#tensor), TensorFactory,
TensorProvenance, [Operation](../../../../glossary.md#operation), CastKind, CastAttrs, and expression-helper source/tests; Tensor,
[Compile](../../../../glossary.md#compile), and Training APIs plus glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms fluent public
`Tensor.cast(DataType)`, all current source/[target](../../../../glossary.md#term-target) categories, exact [shape](../../../../glossary.md#shape) preservation,
same-type input return, floating-only [gradient](../../../../glossary.md#term-gradient) propagation, strided inputs, expression chaining,
[ONNX](../../../../glossary.md#term-onnx) mapping, and [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) execution evidence. Legacy Tensor callbacks, storage conversion,
operation traits, [backend](../../../../glossary.md#backend) routes, and same-type [eager](../../../../glossary.md#term-eager-evaluation) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) are excluded from the model
expression task. The same-type difference is deliberate: explicit expression construction owns
requested semantics, while the [compiler](../../../../glossary.md#term-compiler) owns redundant-cast removal.

Planning selected one public method and one package-private single-entry helper. Existing data-type,
[descriptor](../../../../glossary.md#term-descriptor), [provenance](../../../../glossary.md#provenance), and derived-factory contracts are sufficient; no new package, dependency,
foundation contract, or architecture rule is required.

Planning validation:

- `git diff --check` passed, and targeted trailing-whitespace inspection found no matches in the
  three changed planning paths.
- The canonical section scan found every required task-specification section, including exact
  public/helper contracts, package impact, bounded scope, validation, implementation handoff,
  decisions, limitations, and completion-evidence sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scan glossary](../../../../glossary.md#term-cumulative-scan); [scope glossary](../../../../glossary.md#term-scope); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Every local Markdown target linked from this task, the model master plan, and the roadmap
  resolves. Markdown fences are balanced in all three changed paths.
- Status inspection found 0015H [`Ready`](../../../../glossary.md#term-ready) in this specification, its linked model-master row, and
  its linked roadmap row/current-frontier text. Task 0016 remains `Draft` in both queues, and no
  task-0016 specification exists.
- Package inspection found no new package. The planned direction remains from `model.tensor` to
  existing operation, cast, and datatype contracts without a reverse dependency.
- Scope inspection found exactly this new task, the model master plan, and the roadmap changed. No
  Java, test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module path glossary](../../../../glossary.md#term-module-path))</span>

Implementation and independent documentation validation:

- The implementation pass changed exactly four Java/test paths: `Tensor.java`,
  `TensorCastExpressions.java`, `TensorTest.java`, and `TensorCastExpressionTest.java`.
  Independent source, test, diff, reflection, and bytecode inspection confirmed the exact public
  method/helper shapes, one public delegation, input-then-target null checks, source and exact Shape
  reads, complete gradient-eligibility expression, one unresolved descriptor, one <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`CastAttrs`,
  one `CAST` Operation, exact one-input provenance, and one `createDerived` delegation.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest` passed 9 tests with zero
  failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed 14 tests with zero failures, errors,
  or skips.
- `./gradlew :modules:model:test` passed all 362 tests across 48 suites with zero failures,
  errors, or skips. The matrix tests cover all 36 source/target pairs, exact Shape-reference
  retention, resolved/unresolved and scalar/zero/static/dynamic inputs, fresh same-type/repeated/
  chained expressions, gradient eligibility, storage/content non-observation, null ordering, and
  allocator exhaustion.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `./gradlew :modules:model:javadoc` passed. Generated `Tensor.html` contains the public cast
  signature, fresh same-type contract, exact target/shape/result semantics, failure documentation,
  and compiler-owned redundant-cast boundary. The default public/protected Javadoc output omits the
  package-private helper, so its complete type, constructor, parameter, return, ownership, ordering,
  and failure Javadocs were reviewed directly in source.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Javadoc glossary](../../../../glossary.md#term-javadoc); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The independent documentation-focused context
  `/root/review_model_0015h_docs` applied General style, API/Javadoc style, Planning style, and
  Example format. It found the affected `Tensor` and `TensorCastExpressions` Javadocs complete
  without revision, then finalized the Tensor API, Compile API, glossary, this task, model master
  plan, and roadmap.
- The Tensor API now gives a newcomer-oriented public cast contract and complete compiled example,
  including all-pair representability, fresh same-type identity, exact Shape retention, unresolved
  layout, gradient eligibility, typed attributes, immediate-input provenance, storage ownership,
  and deferred numerical/compiler/backend behavior. <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout))</span>`javac -cp
  modules/model/build/classes/java/main -d /private/tmp/synaptik-cast-doc
  CastExpressionExample.java` passed for the transient extracted example, and `java -cp
  modules/model/build/classes/java/main:/private/tmp/synaptik-cast-doc
  CastExpressionExample` printed all 12 documented lines exactly. The transient source file was
  removed after validation.
- The Compile API now inventories cast expression construction as current and keeps compiler entry,
  traversal, graph capture, redundant-cast/cast-chain canonicalization, optimization, and artifacts
  planned. No compiler behavior or callable compiler contract was added.<span data-glossary-navigation="true"> ([cast expression glossary](../../../../glossary.md#cast-expression); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- The glossary now defines cast expression, updates implementation status, and aligns Cast,
  Tensor, provenance, operation-kind/attributes, and Tensor-versus-graph-value terminology without
  changing architecture authority.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- `javap -classpath modules/model/build/classes/java/main -p -c -s` for `Tensor` and
  `TensorCastExpressions` confirmed the exact method descriptors and modifiers, sole public
  delegation, zero-field helper, private constructor, validation/construction order, full
  short-circuit eligibility expression, exact Shape use, empty layout/label, typed operation and
  one-input provenance, one factory call, and absence of synchronization or same-type early return.
  Focused reflection tests independently confirm the same exact public/helper shapes.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Import/package scans found only the planned model/JDK dependencies. Source and bytecode contain no
  value or storage access, conversion loop, source/target compatibility map, resolved output
  layout, gradient rule, graph/compiler/runtime/backend type, cost/fusion/route state, registry,
  service, cache, dependency, or build change.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [runtime glossary](../../../../glossary.md#term-runtime); [fusion glossary](../../../../glossary.md#term-fusion); [registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- `./gradlew test` passed the complete repository lifecycle with all 36 actionable tasks
  successful. No architecture, backend-conformance, or integration test change is required because
  task 0015H changes no dependency rule, backend behavior, or end-to-end numerical execution.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- `git diff --check` passed. Targeted trailing-whitespace scans returned no matches. Fence counts
  are balanced in every changed Markdown file. The corrected local Markdown checker resolved 227
  file targets and heading anchors across the six changed documentation/planning files with zero
  errors. Two preliminary checker invocations stopped before repository validation because the
  first Ruby regular expression interpolated a heading quantifier and the installed Ruby lacks
  `Array#filter_map`; the compatible rerun produced the passing result.
- Exact scope inspection found only the authorized ten paths: two production files, two test files,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Status is
  synchronized to Complete in this task and both queues. Task 0016 remains Draft and no task-0016
  specification exists. No commit or push was performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- The implementation handoff accurately reported that one combined diagnostic shell command was
  terminated after producing no output; its report inspection and Javadoc portions were rerun
  separately and succeeded. This documentation pass did not rely on that combined attempt and
  independently reran every required focused/aggregate test, Javadoc, root, bytecode, import,
  documentation, scope, and status validation.
- `docs/api/training-api.md` remains accurate unchanged because this task adds only immutable
  gradient-eligibility metadata, not a gradient rule, cast-back expression, trainable role,
  autograd expansion, optimizer, session, or execution behavior. The model capability baseline
  already inventories explicit cast and correctly separates model representation from executable
  parity, so it requires no task-status edit.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests,
  backend-conformance tests, and integration tests remain accurate unchanged because cast
  construction stays in `modules/model`, preserves dependency direction, and adds no compiler,
  planning, prepare, runtime, backend, training, numerical, or end-to-end behavior. Java 26
  toolchain/release configuration also remains accurate and unchanged.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [prepare glossary](../../../../glossary.md#prepare); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- Existing `DataType`, `TensorDescriptor`, `TensorFactory`, `TensorProvenance`, `Operation`,
  `CastKind`, and `CastAttrs` Javadocs remain accurate: their data-type categories,
  differentiability, descriptor eligibility, central identity allocation, exact provenance,
  generic semantic composition, cast identity, and target-only attributes are composed without
  changing those contracts. Existing expression-family Javadocs likewise remain accurate because
  cast adds a separate helper and public method without changing their validation or result rules.

## Implementation notes

- Added exactly one public fluent `Tensor.cast(DataType)` method and one package-private final,
  stateless, single-entry `TensorCastExpressions` helper.
- Every valid request creates a fresh unlabeled, storage-free explicit `CAST` expression. The
  helper accepts all 36 current pairs, retains the exact input Shape reference, leaves layout
  unresolved, and retains gradient eligibility only when the input already requests it and both
  types are floating.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The result uses one [target](../../../../glossary.md#term-target)-only `CastAttrs`, one `CAST` Operation, exact one-input provenance,
  and the existing central derived-construction/identity path. No values or storage are accessed,
  and no conversion, canonicalization, gradient rule, graph capture, backend support, dependency,
  or execution behavior was introduced.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Final documentation makes the public model behavior current while keeping numerical conversion,
  compiler canonicalization/autograd, backend lowering/support, and execution explicitly deferred.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [lowering glossary](../../../../glossary.md#lowering))</span>

## Completion summary

- Completed changes: Implemented and documented public explicit cast Tensor expression
  construction for every current source/target pair.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [target glossary](../../../../glossary.md#term-target))</span>
- Files changed or created: Exactly the two production, two test, and six documentation/planning
  paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Focused suites passed 9/9 and 14/14; all 362 model tests across 48 suites,
  model Javadoc, root tests, the compiled documentation example, generated-Javadoc inspection,<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `javap`, reflection, import/forbidden-behavior scans, 227 Markdown link/anchor checks,
  fence/terminology/whitespace checks, exact-scope review, status checks, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: `/root/review_model_0015h_docs` independently completed the
  required clean-context pass using General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor API, Compile API, glossary, task evidence, model master plan, and
  roadmap now describe cast expression construction as current while preserving all cross-layer
  boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [cast expression glossary](../../../../glossary.md#cast-expression))</span>
- Javadoc review: Affected Tensor and helper Javadocs are complete without documentation-pass
  wording changes; all reviewed foundational and existing expression Javadocs remain accurate.
- Glossary impact: Cast expression and its distinction from value conversion, graph nodes,
  compiler canonicalization, gradient rules, storage, and executable backend support are current.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node); [compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Unresolved issues: None within task 0015H.
- Follow-up required: None for task 0015H. Task 0016 remains the next Draft planning frontier
  without a detailed specification.

Status: Complete
