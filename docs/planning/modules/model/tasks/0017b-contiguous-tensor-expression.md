# Task 0017B: [Contiguous](../../../../glossary.md#term-contiguous) [Tensor](../../../../glossary.md#tensor) Expression

**Editorial reading key (later context, not a status update).**

Contiguous requests the same logical values in dense row-major, zero-offset geometry: the last axis varies fastest without gaps between rows. For static shape, the helper can derive numeric result strides from shape alone; for dynamic shape they remain unresolved. Neither the resolved descriptor nor a fresh Tensor identity proves separate physical memory or zero-copy execution. The application calls the public Tensor method; its model helper validates available descriptions and returns a fresh storage-free expression with exact input references. It neither copies input values nor calculates the result. Compiler consumes this provenance later; actual storage and execution are separate downstream concerns.

Focused current reading: [Current contiguous-expression boundary](../../../../reference/tensor/reshape-expand.md#contiguous-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add the public `Tensor.contiguous()` expression that records the completed
`ContiguousKind.CONTIGUOUS` semantic request without reading or copying values.

The expression preserves the input's exact logical [shape](../../../../glossary.md#shape), [data type](../../../../glossary.md#data-type--datatype), and [gradient](../../../../glossary.md#term-gradient) eligibility.
For a fully static shape it publishes newly resolved [canonical](../../../../glossary.md#term-canonical-form) dense [row-major](../../../../glossary.md#term-row-major) zero-offset [layout](../../../../glossary.md#layout)
geometry. For a shape containing a dynamic [dimension](../../../../glossary.md#dimension) it leaves layout unresolved because numeric
[strides](../../../../glossary.md#term-stride) and [referenced span](../../../../glossary.md#referenced-element-span) cannot yet be calculated. Every valid call creates a fresh,
storage-free derived [Tensor](../../../../glossary.md#tensor) with exact one-input [provenance](../../../../glossary.md#provenance).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public no-[argument](../../../../glossary.md#term-argument) `Tensor.contiguous()` instance method.
- Add one package-private final stateless `TensorContiguousExpressions` construction boundary.
- Give the helper exactly one package-private static `apply(Tensor)` method and one private
  zero-argument constructor.
- Null-check the helper input before reading [metadata](../../../../glossary.md#term-metadata).
- Retain the exact input [`Shape`](../../../../glossary.md#shape), [`DataType`](../../../../glossary.md#data-type--datatype), and `requiresGrad` value.
- For a fully static shape, create exactly one new `LayoutDescriptor.contiguous(shape)` and store
  it as the resolved result layout.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- For a dynamic shape, store `Optional.empty()` and do not invent numeric [strides](../../../../glossary.md#term-stride), offset, or span.
- Construct exactly `Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE)`.
- Record exact one-input [provenance](../../../../glossary.md#provenance) `[input]` and delegate exactly once to
  `TensorFactory.createDerived` with no label or storage.
- Return a fresh expression even when the input already has canonical dense-contiguous geometry or
  is itself a contiguous expression.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Accept all six current data types, scalar shapes, zero extents, ordinary static shapes, and
  dynamic shapes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Add one focused same-package expression test and update `TensorTest` only for the deliberate
  one-method public API expansion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize affected Javadocs, Tensor API, Compile API, glossary, task evidence, model master plan,
  and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- eager value access, memory copy, materialization, allocation, storage attachment, alias creation,
  mutation, host-storage replacement, or ownership/lifetime changes<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena); [host glossary](../../../../glossary.md#term-host); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- inspecting the input descriptor's layout optional, layout kind, strides, offset, span, view flag,
  contiguity, label, provenance, storage association, storage liveness, or values<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [provenance glossary](../../../../glossary.md#provenance); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- returning the input, reusing its `LayoutDescriptor`, preserving its view flag or offset,
  canonicalizing repeated requests, eliminating an already-contiguous request, or constant folding<span data-glossary-navigation="true"> ([contiguous request glossary](../../../../glossary.md#contiguous-request); [constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- public overloads, static facades, optional copy flags, layout-order arguments, memory-format
  options, destination storage, device arguments, factories, builders, or another public type<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [device glossary](../../../../glossary.md#term-device))</span>
- reshape, expand, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter behavior<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [dimensions glossary](../../../../glossary.md#dimension); [slice glossary](../../../../glossary.md#slice); [gather glossary](../../../../glossary.md#gather))</span>
- data-type conversion or promotion, Shape mutation, symbolic binding, graph-wide inference,
  materialization scoring, backend ownership, kernel or route selection, or executable behavior<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference); [backend ownership glossary](../../../../glossary.md#backend-ownership); [kernel glossary](../../../../glossary.md#kernel))</span>
- gradient formulas, backward operation kinds, saved values, autograd expansion, optimizer, or
  training execution<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- compiler capture, graph traversal, canonicalization implementation, planning output, prepare,
  runtime residency, backend storage, engine composition, tracing, ONNX mapping, or conformance<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend glossary](../../../../glossary.md#backend); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to `ContiguousKind`, [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), [`Operation`](../../../../glossary.md#operation), `TensorDescriptor`,
  `LayoutDescriptor`, `LayoutKind`, `TensorFactory`, `TensorProvenance`, Shape, DataType, storage,
  graph contracts, existing expression helpers, or their tests<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- dependencies, Gradle, Java version, AGENTS, ARCHITECTURE, focused architecture documentation,
  architecture tests, capabilities, another module, or unrelated documentation<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0017C specification or implementation of a later 0017 task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017A](0017a-contiguous-semantic-kind.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes:

```java
Tensor contiguous()
TensorOps.contiguous(Tensor input)
```

Legacy public behavior preserves logical values, [shape](../../../../glossary.md#shape), and [data type](../../../../glossary.md#data-type--datatype) while producing [canonical](../../../../glossary.md#term-canonical-form)
[row-major](../../../../glossary.md#term-row-major) output. Execution evidence covers every selected data type, including [BOOL](../../../../glossary.md#data-type--datatype); already
dense inputs; permuted, expanded, sliced, offset, and other non-dense inputs; chained expressions;
and [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) paths. Legacy implementations may materialize a new [buffer](../../../../glossary.md#term-buffer) or reuse an
already suitable representation.

This task preserves only the public [model](../../../../glossary.md#term-model) expression and deterministic [descriptor](../../../../glossary.md#term-descriptor) facts. It does
not copy the legacy immediate storage/[view](../../../../glossary.md#term-view-meanings) construction, [gradient](../../../../glossary.md#term-gradient) [callback](../../../../glossary.md#term-callback), [graph](../../../../glossary.md#graph) builder,
[operation](../../../../glossary.md#operation) traits, [compiler](../../../../glossary.md#term-compiler) [layout](../../../../glossary.md#layout) flags, [materialization](../../../../glossary.md#materialization) planners, physical view types, [kernels](../../../../glossary.md#kernel),
[lowering](../../../../glossary.md#lowering), fallback, or [runtime](../../../../glossary.md#term-runtime) propagation. Those mechanisms either violate the new boundaries or
belong to later compiler, [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare), runtime, training, and [backend](../../../../glossary.md#backend) tasks.

## Architecture constraints

- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not an intermediate-representation [node](../../../../glossary.md#node).
- The result is a fresh public Tensor carrying immutable expression provenance, not a compiled
  node, runtime tensor, physical buffer, prepared unit, or executable copy.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [runtime glossary](../../../../glossary.md#term-runtime); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- `ContiguousKind.CONTIGUOUS` is the exact [operation](../../../../glossary.md#operation) [identity](../../../../glossary.md#term-identity-meanings) and
  `NoOperationAttrs.INSTANCE` is its complete attributes value.
- The result retains the exact [immutable](../../../../glossary.md#term-immutable) input [`Shape`](../../../../glossary.md#shape) reference, exact [`DataType`](../../../../glossary.md#data-type--datatype), and unchanged
  `requiresGrad` value. This task defines no [gradient](../../../../glossary.md#term-gradient) formula or guarantee that one exists.
- A fully static Shape supplies enough information to resolve canonical dense row-major strides,
  zero logical storage offset, non-view metadata, and referenced span through<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [strides glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset); [view glossary](../../../../glossary.md#term-view-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [referenced span glossary](../../../../glossary.md#referenced-element-span))</span>
  `LayoutDescriptor.contiguous(shape)`.
- A dynamic Shape cannot [carry](../../../../glossary.md#term-carry) numeric `LayoutDescriptor` geometry. The result must use
  `Optional.empty()` without [binding](../../../../glossary.md#term-binding) symbols or guessing concrete [dimensions](../../../../glossary.md#dimension).
- Resolved result layout expresses the requested logical geometry. It does not allocate storage,
  prove that values have been copied, require a distinct physical buffer, or select an executable
  materialization route.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- Input layout and storage are deliberately irrelevant to expression construction. Compiler may
  later eliminate a redundant request; planning may derive a logical materialization requirement;
  backend prepare may choose alias or copy lowering; runtime executes the prepared schedule.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lowering glossary](../../../../glossary.md#lowering); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule))</span>
- Every valid request remains an explicit fresh expression at the model boundary, including an
  already-dense input and nested contiguous requests.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- The helper remains in `model.tensor` because it coordinates the public Tensor, descriptor,
  operation, provenance, and factory contracts. The semantic kind remains in <span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>`model.operation.layout`.
- No project dependency or module boundary changes. Stop if implementation requires storage
  access, mutation, graph state, another type, cross-module behavior, or architecture changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [graph glossary](../../../../glossary.md#graph))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns [`Tensor`](../../../../glossary.md#tensor), its package-private expression helper,
  `TensorDescriptor`, `TensorProvenance`, and `TensorFactory.createDerived`.
- `io.github.pho001.synaptik.model.operation.layout` — supplies the completed
  `ContiguousKind.CONTIGUOUS` [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.model.operation` — supplies [`Operation`](../../../../glossary.md#operation) and
  `NoOperationAttrs.INSTANCE`.
- `io.github.pho001.synaptik.model.layout` — supplies canonical resolved layout construction for
  fully static shapes.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout))</span>
- `io.github.pho001.synaptik.model.shape` — supplies the [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) and its
  `isFullyStatic()` decision.

No package is added or moved.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — gains the one public fluent expression entry.
- `io.github.pho001.synaptik.model.tensor.TensorContiguousExpressions` — package-private stateless
  coordinator colocated with Tensor and its existing derived-construction seam.
- `TensorContiguousExpressionTest` — same-package focused test that can inspect the package-private
  helper without widening production visibility.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>
- `TensorTest` — changes only its exact public-method inventory from 76 to 77 and includes the
  no-argument <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>[`contiguous`](../../../../glossary.md#term-contiguous) [signature](../../../../glossary.md#term-signature).

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) method

Add exactly:

```java
public Tensor contiguous() {
    return TensorContiguousExpressions.apply(this);
}
```

The method is public, final by virtue of `Tensor` being final, non-static, non-synchronized,
parameterless, and returns `Tensor`. Its body contains exactly one delegation to the helper and no
inline [validation](../../../../glossary.md#term-validation), [descriptor](../../../../glossary.md#term-descriptor) construction, storage access, or optimization.

### Helper [shape](../../../../glossary.md#shape)

Create one package-private final class with exactly:

```java
final class TensorContiguousExpressions {
    private TensorContiguousExpressions() {
    }

    static Tensor apply(Tensor input) {
        // Exact construction contract below.
    }
}
```

The helper has no fields, nested types, public or protected members, overloads, factory instance,
[cache](../../../../glossary.md#term-cache), [registry](../../../../glossary.md#term-registry), service, [layout](../../../../glossary.md#layout) policy, or additional methods. The constructor is private and
zero-[argument](../../../../glossary.md#term-argument). `apply` is package-private static and is the sole declared method.

### [Validation](../../../../glossary.md#term-validation) and construction order

`apply` performs exactly this observable order:

1. Null-check `input` with `Objects.requireNonNull(input, "input")`.
2. Read and retain the exact `TensorDescriptor` reference once.
3. Read and retain that [descriptor](../../../../glossary.md#term-descriptor)'s exact [`Shape`](../../../../glossary.md#shape) reference.
4. Call `shape.isFullyStatic()` once.
5. For a static shape, call `LayoutDescriptor.contiguous(shape)` exactly once and use
   `Optional.of` on that exact new descriptor. For a dynamic shape, use `Optional.empty()` and do
   not call layout construction.
6. Create one `TensorDescriptor` from the input descriptor's exact DataType, exact Shape reference,
   resolved-or-unresolved result layout, and unchanged <span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>`requiresGrad` value.
7. Create one [`Operation`](../../../../glossary.md#operation) from exact `ContiguousKind.CONTIGUOUS` and
   `NoOperationAttrs.INSTANCE`.
8. Create one `TensorProvenance` with that exact operation and exact ordered input `[input]`.
9. Call `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` exactly once and
   return its exact result.

No input [layout](../../../../glossary.md#layout), label, [provenance](../../../../glossary.md#provenance), storage, [liveness](../../../../glossary.md#term-liveness), or [tensor](../../../../glossary.md#tensor) value is read. The helper does not
derive geometry itself; on the static branch, `LayoutDescriptor.contiguous(shape)` necessarily
reads the [immutable](../../../../glossary.md#term-immutable) Shape [dimensions](../../../../glossary.md#dimension) and performs its existing checked [stride](../../../../glossary.md#term-stride)/span calculations.
A static layout arithmetic failure propagates before Tensor-ID allocation. A dynamic result does
not inspect symbolic sizes or attempt partial layout resolution.

### Result contract

Every successful call returns a fresh [Tensor](../../../../glossary.md#tensor) with:

| Result fact | Required value |
|---|---|
| ID | fresh factory-allocated [`TensorId`](../../../../glossary.md#tensorid) |
| [DataType](../../../../glossary.md#data-type--datatype) | exact input DataType |
| [Shape](../../../../glossary.md#shape) | exact input Shape reference |
| `requiresGrad` | unchanged input value |
| static [layout](../../../../glossary.md#layout) | new [canonical](../../../../glossary.md#term-canonical-form) `DENSE_CONTIGUOUS`, offset-zero, non-[view](../../../../glossary.md#term-view-meanings) [descriptor](../../../../glossary.md#term-descriptor) |
| dynamic layout | unresolved `Optional.empty()` |
| label | absent |
| [host storage](../../../../glossary.md#host-storage) | absent |
| [operation](../../../../glossary.md#operation) | exact [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) plus `NoOperationAttrs.INSTANCE` |
| [provenance](../../../../glossary.md#provenance) inputs | exact [immutable](../../../../glossary.md#term-immutable) ordered `[input]` |

The resolved descriptor records logical result geometry, not evidence of [eager](../../../../glossary.md#term-eager-evaluation) storage
[materialization](../../../../glossary.md#materialization). Repeated calls and a call on another contiguous expression remain distinct fresh
expressions. The [model](../../../../glossary.md#term-model) layer performs no redundant-request elimination.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorContiguousExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorContiguousExpressionTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- `ContiguousKind`, Operation foundations, DataType, Shape, LayoutDescriptor, LayoutKind,
  TensorDescriptor, TensorFactory, TensorProvenance, storage contracts, graph contracts, existing
  expression helpers, and their Javadocs/tests<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph))</span>
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, Gradle configuration, and every other module<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two test files, three [API](../../../../glossary.md#term-application-programming-interface-api)/glossary files, and three [planning](../../../../glossary.md#term-planning) files:
ten paths total.

Do not modify `ContiguousKind`, any other existing Java source/test, capabilities, Training API,
[Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), completed task, or unrelated
documentation. Stop if another production type, public method, helper method, [dependency](../../../../glossary.md#term-dependency),
cross-layer behavior, or eleventh path is needed. Do not create task 0017C.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update the [`Tensor`](../../../../glossary.md#tensor) type overview so it no longer claims every expression result has unresolved
  layout. Explain the contiguous expression's static-resolved/dynamic-unresolved exception.
- Document `Tensor.contiguous()` as an expression-construction method, not an immediate copy.
- Document unchanged logical values, exact Shape/DataType/gradient-eligibility retention, static
  canonical dense row-major zero-offset layout, and dynamic unresolved layout.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Document the fresh ID, absent label/storage, exact [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) [operation](../../../../glossary.md#operation),
  `NoOperationAttrs.INSTANCE`, and one-input [provenance](../../../../glossary.md#provenance).
- Document `ArithmeticException` for static [canonical](../../../../glossary.md#term-canonical-form) [layout](../../../../glossary.md#layout) [stride](../../../../glossary.md#term-stride)/span [overflow](../../../../glossary.md#term-overflow) and
  `IllegalStateException` for exhausted Tensor [identity](../../../../glossary.md#term-identity-meanings) space.
- Explain that input layout/storage are not inspected, and that already-dense and repeated
  requests remain fresh until compiler canonicalization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Document the package-private helper and its private constructor, input nullability, exact
  construction order, static/dynamic branch, return value, side effects, ownership, and failures.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- State that resolved result geometry does not prove eager allocation, copy, distinct storage,
  runtime residency, or a backend route.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Review `ContiguousKind`, Shape, LayoutDescriptor, LayoutKind, TensorDescriptor, TensorFactory,
  TensorProvenance, and Operation Javadocs and record why they remain accurate, or stop on an
  inconsistency.

## Acceptance criteria

- [`Tensor`](../../../../glossary.md#tensor) exposes exactly one new public method, parameterless `contiguous()`, bringing its exact
  declared-public-method count from 76 to 77.
- The public method delegates exactly once to `TensorContiguousExpressions.apply(this)` and adds no
  inline behavior.
- The helper is package-private final and stateless, with exactly one private zero-argument
  constructor and exactly one package-private static <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`apply(Tensor)` method.
- The helper rejects null input with exact `NullPointerException` message `input` before reading
  metadata or allocating an ID.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- All six current DataTypes are accepted. [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), and `requiresGrad` are retained exactly.
- Every fully static Shape, including scalar and zero-extent shapes, receives a newly constructed
  canonical <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`DENSE_CONTIGUOUS`, offset-zero, non-[view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout) with correct [strides](../../../../glossary.md#term-stride) and span.
- Every dynamic Shape retains the exact Shape reference and receives unresolved layout without
  numeric stride/span calculation.
- Static layout [overflow](../../../../glossary.md#term-overflow) propagates as `ArithmeticException` before identity allocation; no
  unresolved fallback is used.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Result [operation](../../../../glossary.md#operation) and [provenance](../../../../glossary.md#provenance) retain exact `ContiguousKind.CONTIGUOUS`,
  `NoOperationAttrs.INSTANCE`, and input `[input]` references.
- Results are fresh, unlabeled, and storage-free even for already-contiguous input, an input with
  attached/dead storage, repeated calls, or nested contiguous requests.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Input descriptor, label, provenance, storage association, liveness, storage bytes, and layout
  object remain unchanged and are not used to select a code path.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- No eager copy, allocation, storage alias, data access, canonicalization, gradient rule, compiler,
  planning, prepare, runtime, backend, graph, ONNX, dependency, or build behavior is added.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [alias glossary](../../../../glossary.md#aliasing-and-arena); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [graph glossary](../../../../glossary.md#graph); [ONNX glossary](../../../../glossary.md#term-onnx); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused/aggregate tests, model Javadoc, root tests, reflection/javap/bytecode/import/source/scope
  checks, documentation links/examples/formatting, and synchronized statuses pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes all affected Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap and records related no-change
  conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- Task 0017B becomes Complete only after both passes. Task 0017C remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorContiguousExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused expression test must verify:

- exact public method and exact one-method helper reflection surface;
- all six DataTypes and valid [gradient](../../../../glossary.md#term-gradient) eligibility combinations;
- exact Shape reference, static canonical layout kind/strides/offset/view/span, and dynamic
  unresolved layout;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- [scalar](../../../../glossary.md#term-scalar) and zero-[extent](../../../../glossary.md#term-extent) [canonical](../../../../glossary.md#term-canonical-form) geometry;
- exact Operation, no-attributes singleton, one-input provenance, absent label/storage, and fresh
  identity;<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- input layouts covering unresolved static, dense, offset-dense, strided, and broadcast geometry
  without changing result semantics;
- input label, [provenance](../../../../glossary.md#provenance), live or subsequently dead storage, and raw contents remain untouched;
- repeated, already-[contiguous](../../../../glossary.md#term-contiguous), and nested requests are never [model](../../../../glossary.md#term-model)-canonicalized;
- null-helper input and static [layout](../../../../glossary.md#layout) [overflow](../../../../glossary.md#term-overflow) consume no [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings); and
- identifier exhaustion propagates only after valid local [metadata](../../../../glossary.md#term-metadata) construction.

Manually inspect `javap -p -c -s` and reflection. Confirm the public method has one exact helper
call; the helper has one method, no state, one static/dynamic branch, one canonical-layout call on
the static branch, and one [descriptor](../../../../glossary.md#term-descriptor)/[operation](../../../../glossary.md#operation)/provenance/createDerived path. Inspect imports and
source for absence of [host](../../../../glossary.md#term-host)-storage access, values, copies, allocation, input-layout reads,
[graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) types, gradient rules, caches, services, and hidden
[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form). Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) status, glossary terms, examples,
links, anchors, fences, whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), synchronized statuses, and absence of a
task-0017C specification.

## Dependencies

- Task 0002 supplies `Shape.isFullyStatic()` and dynamic/static [shape](../../../../glossary.md#shape) representation.
- Task 0003 supplies `LayoutDescriptor.contiguous(shape)` and [canonical](../../../../glossary.md#term-canonical-form) resolved geometry.
- Task 0007 supplies explicit resolved-or-unresolved `TensorDescriptor` [layout](../../../../glossary.md#layout) state.
- Tasks 0011–0013 supply public Tensor metadata, centralized derived identity allocation, and
  immutable one-input provenance.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance))</span>
- Task 0017A supplies the exact parameterless `ContiguousKind.CONTIGUOUS` semantic [identity](../../../../glossary.md#term-identity-meanings).

## Follow-up tasks

- 0017C remains Draft for [immutable](../../../../glossary.md#term-immutable) reshape and expand semantic kinds/attributes.
- 0017D remains Draft for public reshape and expand expression construction.
- [Compiler](../../../../glossary.md#term-compiler) later owns redundant [contiguous](../../../../glossary.md#term-contiguous)-request elimination and [graph](../../../../glossary.md#graph)-wide [layout](../../../../glossary.md#layout) reasoning.
- [Planning](../../../../glossary.md#term-planning) later derives logical [materialization](../../../../glossary.md#materialization) requirements without selecting concrete routes.
- Backend prepare later chooses whether an owned contiguous request is an alias, copy, fused
  route, or backend-native transform; runtime executes the prepared schedule.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [contiguous request glossary](../../../../glossary.md#contiguous-request); [alias glossary](../../../../glossary.md#aliasing-and-arena); [fused glossary](../../../../glossary.md#specialization-and-fusion); [runtime glossary](../../../../glossary.md#term-runtime); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule))</span>
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns public [Tensor](../../../../glossary.md#tensor) expression construction,
[operation](../../../../glossary.md#operation) semantics, [Shape](../../../../glossary.md#shape), and logical [layout](../../../../glossary.md#layout) values to `modules/model`; [compiler](../../../../glossary.md#term-compiler) optimization to
compiler; logical [materialization](../../../../glossary.md#materialization) requirements to [planning](../../../../glossary.md#term-planning); concrete [lowering](../../../../glossary.md#lowering) and storage to
[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare); and [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) to [runtime](../../../../glossary.md#term-runtime).

If implementation requires [eager](../../../../glossary.md#term-eager-evaluation) storage behavior, physical materialization, [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction),
cross-[module](../../../../glossary.md#term-module) state, a [dependency](../../../../glossary.md#term-dependency), or an architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017A/0017B, Tensor API,
Compile API, Training API, glossary, current Shape/LayoutDescriptor/LayoutKind/TensorDescriptor/
Tensor/TensorFactory/TensorProvenance/Operation/NoOperationAttrs/ContiguousKind contracts and
focused tests, and Java 26 Gradle configuration.

Implement task 0017B exactly. Modify Tensor.java and add package-private final
TensorContiguousExpressions.java for production. Update TensorTest only for the exact one-method
public API expansion and add TensorContiguousExpressionTest. Add exactly public parameterless
contiguous(), delegating once to the helper.

The helper has exactly one package-private static apply(Tensor) method, no fields, and one private
constructor. Null-check input, retain its exact descriptor/Shape/DataType/requiresGrad, branch once
on Shape.isFullyStatic(), create exactly one new LayoutDescriptor.contiguous(shape) for static
Shape or unresolved layout for dynamic Shape, construct exact CONTIGUOUS/NoOperationAttrs
semantics with [input] provenance, and call createDerived once with no label/storage. Every valid
call is fresh, including already-contiguous and nested requests.

Do not inspect input layout/label/provenance/storage/liveness/values, copy or allocate storage,
return input, reuse a layout, canonicalize, add overloads/helpers/types, change existing contracts,
define gradients, capture graphs, or add compiler/planning/prepare/runtime/backend behavior,
dependencies, build/architecture changes, or later specs. Stop beyond ten paths or on
architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/bytecode/import/manual,
documentation/example/link/whitespace/scope/status check. Then hand the actual diff and evidence
to a separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/Compile API/glossary/planning, record
related-contract/capability/Training API/architecture no-change conclusions, and rerun validation.

Update task 0017B, model master plan, and roadmap only for planning status/evidence. Do not mark
0017B Complete until both passes succeed. Leave 0017C Draft without a specification. Do not commit
or push.
```

## Local decisions

- The public [API](../../../../glossary.md#term-application-programming-interface-api) is exactly `Tensor.contiguous()` with no arguments. A caller requests canonical
  row-major geometry; implementation policy is not exposed as flags or storage arguments.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- The helper is `TensorContiguousExpressions` with one method because validation and construction
  form one small cohesive path. Additional factories or layout-policy abstractions have no current
  use.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [layout glossary](../../../../glossary.md#layout))</span>
- Fully static output resolves layout immediately because Shape determines canonical strides,
  offset, view flag, and referenced span without backend or storage facts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [referenced span glossary](../../../../glossary.md#referenced-element-span); [backend glossary](../../../../glossary.md#backend))</span>
- Dynamic output remains unresolved rather than storing symbolic strides or deferring a partially
  numeric descriptor. This preserves the existing LayoutDescriptor contract.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Resolved result layout is newly constructed and non-view. It describes requested logical result
  geometry; later physical buffer reuse is a compiler/planning/backend concern and does not turn
  this model descriptor into an alias view.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Every call is a fresh explicit expression. Returning the input or eliminating nested requests
  would hide semantic provenance and perform compiler canonicalization inside the public model API.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- All DataTypes are accepted because contiguity changes layout representation, not logical element
  type. Gradient eligibility is copied unchanged but no gradient rule is introduced.<span data-glossary-navigation="true"> ([contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [element glossary](../../../../glossary.md#term-element); [Gradient glossary](../../../../glossary.md#term-gradient))</span>
- Layout arithmetic overflow is reported rather than silently converting a fully static result to
  unresolved layout.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>

## Known limitations

- The expression does not copy values or make [host](../../../../glossary.md#term-host)/[device](../../../../glossary.md#term-device) data [contiguous](../../../../glossary.md#term-contiguous). No execution exists yet.
- Dynamic results do not expose numeric strides, offset, layout kind, or referenced span until
  later symbolic binding and preparation.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout); [referenced span glossary](../../../../glossary.md#referenced-element-span); [binding glossary](../../../../glossary.md#term-binding))</span>
- The [model](../../../../glossary.md#term-model) does not eliminate an already-contiguous or repeated request.
- No gradient rule, compiler capture, logical materialization plan, backend lowering, runtime
  residency, storage alias/copy, ONNX mapping, or conformance behavior is implemented.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [alias glossary](../../../../glossary.md#aliasing-and-arena); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-boundary
explanations; documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capabilities and master plan;
tasks 0002, 0003, 0007, 0011, 0012, 0013, 0017A; current [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor),
factory, [provenance](../../../../glossary.md#provenance), [Operation](../../../../glossary.md#operation), and [contiguous](../../../../glossary.md#term-contiguous) semantic source/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs
and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms the public
no-[argument](../../../../glossary.md#term-argument) `Tensor.contiguous()` capability, parameterless semantic [identity](../../../../glossary.md#term-identity-meanings), logical
shape/type/value preservation, support for every selected [data type](../../../../glossary.md#data-type--datatype), and [materialization](../../../../glossary.md#materialization) evidence
for permuted, expanded, sliced, offset, [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) paths. Legacy immediate storage/[view](../../../../glossary.md#term-view-meanings)
construction, [graph](../../../../glossary.md#graph) builders, operation traits, [gradient](../../../../glossary.md#term-gradient) callbacks, layout planners, physical
views, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), and runtime handling are excluded or assigned to later [owners](../../../../glossary.md#term-owner).

Planning selected one public method, one single-method package-private helper, and one focused test.
The current Shape and LayoutDescriptor contracts already support the static/dynamic distinction;
TensorDescriptor, provenance, and `createDerived` already support exact storage-free result
construction. No new public value type, dependency, foundational change, or architecture decision
is required.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- All 186 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: ten in this task, two in the master plan, and zero in
  the roadmap.
- All 20 canonical task-specification headings are present, together with focused Capability
  origin, Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task, model master plan, and roadmap consistently identify 0017B as Ready. Task 0017C remains
  Draft, and no task-0017C specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Package review confirms no new package: the public entry and package-private helper remain in
  `model.tensor`, while the completed semantic kind remains in `model.operation.layout`.
- Scope review confirms exactly ten permitted implementation paths and exactly three planning
  paths in the current diff. No Java, API, glossary, architecture, Gradle, AGENTS, completed-task,
  or other-module file changed during planning.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Dependency review confirms Shape, layout, descriptor, Tensor/provenance/factory, and 0017A
  semantics are real prerequisites. Task 0017B does not depend on later layout/view rows and does
  not create a false sequential dependency chain.
- Static/dynamic review confirms that current public contracts support the planned branch:
  `Shape.isFullyStatic()` gates `LayoutDescriptor.contiguous(shape)`, while `TensorDescriptor`
  already requires dynamic shapes to use unresolved layout.

Implementation and independent documentation validation:

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorContiguousExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed. The focused suites contain 8 and 14
  tests respectively, with zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` passed from fresh execution: 60 suites and 462
  tests, with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` passed. Generated `Tensor.html` contains the
  updated type-level static/dynamic layout exception and the complete public `contiguous()`
  contract. The package-private helper is intentionally absent from the public generated API;
  its source Javadoc was independently reviewed and finalized.
- `./gradlew test --rerun-tasks` passed with all 36 actionable tasks executed. No architecture,
  backend-conformance, or integration behavior needed task-specific changes because this is a
  model-only metadata construction boundary.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `javap -p -c -s` confirms the public method contains only one helper invocation and return. The
  final package-private helper has one private constructor, no fields, and one static `apply`
  method. Its bytecode performs one null check, one descriptor/Shape read, one<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
  `isFullyStatic()` branch, one static-branch `LayoutDescriptor.contiguous` call, and one shared
  descriptor/operation/provenance/`createDerived` path.
- Reflection tests confirm exactly 77 declared public Tensor methods and the exact zero-argument
  public, non-static, non-synchronized `Tensor contiguous()` signature. They also confirm the
  helper is final and package-private, stateless, and has exactly the required constructor and
  method surface.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Source and import inspection found no input-layout read, host-storage/value access, copy,
  storage allocation, graph/compiler/planning/prepare/runtime/backend dependency, gradient rule,
  cache, service, overload, or hidden canonicalization. The implementation imports only existing
  model and Java collection/null-check contracts.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- The complete Tensor API Java example compiled with Java 26 against the built model classes and
  printed the documented static strides, offset, view flag, span, dynamic unresolved state,
  operation kind, exact input provenance, and absent storage.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>
- Local Markdown links across the six affected documentation/planning files resolve. Added
  `contiguous-expressions` and existing `contiguous-semantic-kind` anchors resolve, code fences are
  balanced, and targeted trailing-whitespace checks report no matches.
- Scope inspection reports exactly the ten authorized paths: two production files, two tests,
  three API/glossary files, and three planning files. `git diff --check`, untracked-file diff
  checks, newline, status, and no-task-0017C checks pass.
- Task 0017B, the model master plan, and roadmap consistently identify 0017B as Complete. Task
  0017C remains Draft without a detailed specification.
- Independent review found the Training API and model capabilities accurate without change: this
  task adds no gradient/training promise or capability-baseline category. `ContiguousKind`,
  Operation foundations, DataType, Shape, LayoutDescriptor/LayoutKind, TensorDescriptor,
  TensorFactory, TensorProvenance, storage and graph contracts, existing expression helpers, and
  their tests remain accurate because their behavior and public contracts did not change.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- ARCHITECTURE, focused architecture documents, ADRs, architecture tests, backend-conformance and
  integration material, Gradle/Java 26 configuration, dependencies, AGENTS, completed tasks,
  other modules, and examples outside the affected Tensor API remain accurate without change.

## Implementation notes

- Added the exact one-line `Tensor.contiguous()` delegation and one package-private final,
  stateless helper with the required static/dynamic Shape branch and shared derived-construction
  path.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Added focused coverage for every data type, valid gradient choice, scalar/zero/dynamic Shape,
  every representative input layout state, exact provenance, storage/value non-interference,
  fresh repeated/nested requests, validation order, overflow, and identifier exhaustion.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Updated the exact public [Tensor](../../../../glossary.md#tensor) method inventory from 76 to 77 without [widening](../../../../glossary.md#term-widening) any other [API](../../../../glossary.md#term-application-programming-interface-api).
- Finalized Tensor and helper Javadocs, Tensor API, Compile API, glossary, task evidence, model
  master plan, and roadmap in the mandatory independent documentation context. The documentation
  distinguishes current metadata construction from planned compiler canonicalization and
  materialization policy.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization))</span>
- No architecture decision, dependency, module boundary, storage behavior, gradient rule,
  compiler behavior, backend behavior, execution behavior, or later-task specification was added.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

Completed changes:

- implemented public storage-free contiguous expression construction with exact static-resolved
  and dynamic-unresolved descriptor rules;<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- added exact [operation](../../../../glossary.md#operation)/[provenance](../../../../glossary.md#provenance) construction and focused behavioral/structural tests; and
- finalized all affected Javadocs, API/glossary explanations, planning status, and validation
  evidence.

Files changed or created:

- `Tensor.java` and new `TensorContiguousExpressions.java`;
- `TensorTest.java` and new `TensorContiguousExpressionTest.java`;
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile) API, glossary, this task, [model](../../../../glossary.md#term-model) master plan, and roadmap.

Tests and [validation](../../../../glossary.md#term-validation) performed:

- focused Tensor suites, all model tests, generated model [Javadoc](../../../../glossary.md#term-javadoc), and all root tests passed;
- bytecode, reflection, source/import, generated-Javadoc, example, documentation-integrity,
  whitespace, scope, diff, newline, and synchronized-status checks passed.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [integrity glossary](../../../../glossary.md#term-integrity); [scope glossary](../../../../glossary.md#term-scope))</span>

Unresolved issues: None.

Required follow-up: None for task 0017B. Task 0017C remains a separate Draft [planning](../../../../glossary.md#term-planning) frontier
without a detailed specification.

Status: Complete
