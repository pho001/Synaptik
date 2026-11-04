# Task 0019A1: [Embedding](../../../../glossary.md#embedding) Convenience

**Editorial reading key (current context, not a status update).**

This record concerns the historical addition of a convenience, not a new embedding operation or
evidence of execution support. In the current [Embedding
convenience](../../../../reference/tensor/gather.md#embedding-convenience) description, follow the
receiver and result contract: the caller supplies rank-two floating `weights` and integer `indices`;
`Tensor.embedding` delegates through its validation helper to axis-zero Gather. Ordered provenance
`[weights, indices]` records the exact input references, not copied row values. A producer records
the expression that yields a result. Weights shaped `[10, 4]` and indices shaped `[2, 3]` describe
result Shape `[2, 3, 4]` without reading six indices or selecting rows.

Model semantics fixes the lookup meaning; execution providers realize supported routes rather than
redefine it. This storage-free expression neither executes nor owns a result buffer. Semantic
responsibility is distinct from storage ownership: compiler analysis may reject provably invalid
indices, while preparation or prepared execution must enforce bounds safely. Negative indices do not
wrap. Weight-derived `requiresGrad` denotes eligibility, not a constructed gradient. Historical
counts, scope exclusions, and recorded validation remain historical evidence, not current capability
inventories.

## Status

Complete

## Goal

Add exactly one public [`Tensor`](../../../../glossary.md#tensor) convenience:

```java
public Tensor embedding(Tensor indices)
```

The receiver is the [embedding](../../../../glossary.md#embedding)-[weight](../../../../glossary.md#term-weight) table. A successful call is exactly one validated
[axis](../../../../glossary.md#term-axis)-zero `AxisGatherKind.GATHER` [occurrence](../../../../glossary.md#term-occurrence) with ordered inputs `[weights, indices]`; it does not
introduce an `EMBEDDING` kind, an intermediate Tensor, or a second [producer](../../../../glossary.md#term-producer).

This task makes the common [lookup](../../../../glossary.md#term-lookup)-table spelling explicit while keeping [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings),
[automatic differentiation](../../../../glossary.md#autograd) ([autograd](../../../../glossary.md#autograd)), [lowering](../../../../glossary.md#lowering), bounds enforcement, and execution attached to
the existing [Gather](../../../../glossary.md#gather) semantic.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

An [embedding](../../../../glossary.md#embedding) table is a [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) floating [Tensor](../../../../glossary.md#tensor):

```text
weights: [vocabulary, embeddingSize]
indices: [index axes...]
result:  [index axes..., embeddingSize]
```

This is exactly [Gather](../../../../glossary.md#gather) on [weight](../../../../glossary.md#term-weight) [axis](../../../../glossary.md#term-axis) zero. Naming the composition improves the public [API](../../../../glossary.md#term-application-programming-interface-api) but
does not add a new computation for the [compiler](../../../../glossary.md#term-compiler) or [backends](../../../../glossary.md#backend) to recognize.

### Newcomer example

Given floating [weights](../../../../glossary.md#term-weight) of [Shape](../../../../glossary.md#shape) `[10, 4]` and [INT64](../../../../glossary.md#data-type--datatype) indices of Shape `[2, 3]`:

```java
Tensor embedded = weights.embedding(indices);
```

the result Shape is `[2, 3, 4]`. The two index [axes](../../../../glossary.md#term-axis) are retained first and the exact weight
[Dimension](../../../../glossary.md#dimension) at axis one is appended. The result [operation](../../../../glossary.md#operation) remains [`GATHER`](../../../../glossary.md#gather) with
`new IndexAxisAttrs(0)` and exact [provenance](../../../../glossary.md#provenance) `[weights, indices]`. This example describes [model](../../../../glossary.md#term-model)
[metadata](../../../../glossary.md#term-metadata); it does not read the six index values or the selected weight rows.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `public Tensor embedding(Tensor indices)` to [`Tensor`](../../../../glossary.md#tensor).
- Add exactly one package-private static `embedding(Tensor weights, Tensor indices)` method to
  the existing field-free `TensorAxisGatherExpressions` helper.
- Accept only a [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [weight](../../../../glossary.md#term-weight) receiver with [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), or [`FLOAT64`](../../../../glossary.md#term-float64) [data type](../../../../glossary.md#data-type--datatype).
- Accept only exact [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) indices of any [rank](../../../../glossary.md#term-rank), including a [scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape).
- Validate the embedding-specific receiver contract, then delegate directly to the existing
  package-private <span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding))</span>`gather(weights, indices, 0)` method.
- Preserve the exact existing Gather operation, attributes, result metadata, producer, provenance,
  identity-allocation, and value-bounds boundaries.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [operation glossary](../../../../glossary.md#operation); [metadata glossary](../../../../glossary.md#term-metadata); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add focused tests and update every current exact public/helper inventory affected by one method.
- Finalize public Javadoc, Tensor API, Compile API, glossary impact, and planning records through a
  mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- an [`EMBEDDING`](../../../../glossary.md#embedding) operation kind, attributes type, signature, alias, registry entry, compiler
  decomposition, or fused model semantic<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [alias glossary](../../../../glossary.md#aliasing-and-arena); [registry glossary](../../../../glossary.md#term-registry); [compiler glossary](../../../../glossary.md#term-compiler); [decomposition glossary](../../../../glossary.md#term-decomposition); [fused glossary](../../../../glossary.md#specialization-and-fusion); [model glossary](../../../../glossary.md#term-model))</span>
- [weight](../../../../glossary.md#term-weight) ranks other than two; integral, [BOOL](../../../../glossary.md#data-type--datatype), [FLOAT16](../../../../glossary.md#term-float16), quantized, sparse, or complex weights
- a padding index, sparse-gradient flag, maximum-norm option, frequency scaling, negative-index
  wrapping, clamping, default row, or configurable table axis<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient); [norm glossary](../../../../glossary.md#term-norm); [axis glossary](../../../../glossary.md#term-axis))</span>
- a mutable embedding-table abstraction, parameter wrapper, initializer, eager value lookup,
  host-storage allocation, result storage, resolved result layout, or input mutation<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [parameter glossary](../../../../glossary.md#term-parameter); [eager glossary](../../../../glossary.md#term-eager-evaluation); [lookup glossary](../../../../glossary.md#term-lookup); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout))</span>
- gradient rules, scatter-add adjoints, backward graph construction, optimizer behavior, or
  changes to <span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`requiresGrad` eligibility contracts
- compiler capture or constant propagation, backend lowering or kernels, prepare-time proof,
  runtime bounds checks, execution, backend conformance, or integration support<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to `AxisGatherKind`, `IndexAxisAttrs`, [`Operation`](../../../../glossary.md#operation), `OperationSignature`, [`DataType`](../../../../glossary.md#data-type--datatype),
  [`Shape`](../../../../glossary.md#shape), [`Dimension`](../../../../glossary.md#dimension), `TensorDescriptor`, `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, [Gradle](../../../../glossary.md#term-gradle), another [module](../../../../glossary.md#term-module), `ARCHITECTURE.md`, or focused architecture docs
- [one-hot encoding](../../../../glossary.md#one-hot-encoding), [dropout](../../../../glossary.md#dropout--inverted-dropout)/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate), sorting/top-K, [linear](../../../../glossary.md#term-linear-meanings), attention, or any later task

## Exact [API](../../../../glossary.md#term-application-programming-interface-api) and delegation design

### Receiver and [argument](../../../../glossary.md#term-argument) roles

Add exactly:

```java
public Tensor embedding(Tensor indices)
```

The receiver is named [`weights`](../../../../glossary.md#term-weight) inside the helper and is ordered [producer](../../../../glossary.md#term-producer) input zero. `indices`
is ordered producer input one. Add no static form and no overload. Place the public method next to
the [axis](../../../../glossary.md#term-axis)-[gather](../../../../glossary.md#gather) surface, immediately after `gather(Tensor, int)` and before `gatherElements`.

Weights must have exact [rank](../../../../glossary.md#term-rank) two:

```text
axis 0 = vocabulary extent
axis 1 = embeddingSize
```

The accepted weight types are exactly the current floating types [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`FLOAT32`](../../../../glossary.md#term-float32), and
[`FLOAT64`](../../../../glossary.md#term-float64). Indices are exactly [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype); their `requiresGrad` is necessarily false under
the current [descriptor](../../../../glossary.md#term-descriptor) contract.

### Direct [Gather](../../../../glossary.md#gather) delegation

`Tensor.embedding(indices)` is a one-call [facade](../../../../glossary.md#term-facade) over
`TensorAxisGatherExpressions.embedding(this, indices)`. After the [validation](../../../../glossary.md#term-validation) below, the helper
must return exactly:

```java
gather(weights, indices, 0)
```

Do not call the public `weights.gather(...)` method and do not construct a [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation),
[producer](../../../../glossary.md#term-producer), [Tensor](../../../../glossary.md#tensor), or [eager](../../../../glossary.md#term-eager-evaluation) index [carrier](../../../../glossary.md#term-storage-carrier) in [`embedding`](../../../../glossary.md#embedding). The existing helper call constructs the
sole result. There is no intermediate public call result and no second ID allocation.

The final operation is exactly:

```java
new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0))
```

Its family-owned [signature](../../../../glossary.md#term-signature) remains the existing exact two-input, one-output signature.

## [Shape](../../../../glossary.md#shape) and result [metadata](../../../../glossary.md#term-metadata)

For [weights](../../../../glossary.md#term-weight) Shape `[vocabulary, embeddingSize]`, the result Shape is:

```text
indices.shape + [embeddingSize]
```

This follows the existing [Gather](../../../../glossary.md#gather) replacement formula at [axis](../../../../glossary.md#term-axis) zero. The result:

- retains every indices [`Dimension`](../../../../glossary.md#dimension) reference exactly and in order;
- appends the exact weight axis-one `Dimension` reference;
- omits the weight vocabulary `Dimension` from the result;
- accepts static, named dynamic, and expression Dimensions without [binding](../../../../glossary.md#term-binding) or rewriting them;
- accepts [scalar](../../../../glossary.md#term-scalar) indices, producing [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape `[embeddingSize]`;
- accepts zero extents structurally, including zero vocabulary or zero [embedding](../../../../glossary.md#embedding) size;
- creates one new non-scalar Shape value through the existing Gather path;
- has the exact weight [data type](../../../../glossary.md#data-type--datatype) and `requiresGrad` value;
- always has unresolved [layout](../../../../glossary.md#layout), no label, and no [host storage](../../../../glossary.md#host-storage); and
- leaves weights and indices, including their descriptor, label, storage, provenance, and ID,
  unchanged.

`requiresGrad` comes only from weights because Gather preserves data-input eligibility. Indices do
not contribute eligibility. This [descriptor](../../../../glossary.md#term-descriptor) fact is not a [gradient](../../../../glossary.md#term-gradient) rule: [compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) later
owns any repeated-index scatter-add construction.

Each valid call creates one fresh single-output [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one [provenance](../../../../glossary.md#provenance) object at output
index zero, and one newly allocated [Tensor](../../../../glossary.md#tensor) ID. The [producer](../../../../glossary.md#term-producer) retains the exact existing [operation](../../../../glossary.md#operation),
the exact ordered input references `[weights, indices]`, and the exact sole result descriptor.
Repeated valid calls return [identity](../../../../glossary.md#term-identity-meanings)-distinct results and producers.

## [Validation](../../../../glossary.md#term-validation) order, exact messages, and ID effects

`TensorAxisGatherExpressions.embedding(weights, indices)` performs this exact order:

1. `Objects.requireNonNull(weights, "weights")`;
2. `Objects.requireNonNull(indices, "indices")`;
3. require exact [weight](../../../../glossary.md#term-weight) [rank](../../../../glossary.md#term-rank) two;
4. require a floating weight [data type](../../../../glossary.md#data-type--datatype);
5. call the existing `validateIndexType("embedding", indices.descriptor())`;
6. delegate exactly once to `gather(weights, indices, 0)`.

The unchanged [Gather](../../../../glossary.md#gather) method repeats its own null and index-type checks before [Shape](../../../../glossary.md#shape) construction.
Do not bypass or refactor those checks in this task. They cannot allocate an ID and preserve one
[canonical](../../../../glossary.md#term-canonical-form) Gather construction path; the [embedding](../../../../glossary.md#embedding)-specific precheck exists to provide the exact
public convenience failure name before delegation.

Use these exact task-owned messages:

```text
embedding weights rank must be 2: actual=<rank>
embedding weights data type must be BFLOAT16, FLOAT32, or FLOAT64: <type>
embedding indices data type must be INT32 or INT64: <type>
```

Null failures use exact messages `weights` and `indices`. Rank validation precedes weight-type
validation, which precedes index-type validation. The only embedding-specific Shape validation is
exact weight rank two. There is no indices-rank restriction and no static [extent](../../../../glossary.md#term-extent) restriction;
rank two makes [axis](../../../../glossary.md#term-axis) zero valid, and the unchanged Gather helper owns result-Shape construction.

Every null, rank, type, or local [metadata](../../../../glossary.md#term-metadata) failure occurs before the sole final
`TensorFactory.createDerived` delegation and consumes no [Tensor](../../../../glossary.md#tensor) ID. The successful path allocates
exactly one ID. Identifier exhaustion retains the existing exact message
`tensor identifier space exhausted`; there is no [rollback](../../../../glossary.md#term-rollback) or alternative allocation path.
Focused tests may inspect the shared ID counter for this concrete no-ID-on-validation-failure risk.

## Index values and bounds [ownership](../../../../glossary.md#term-ownership-meanings)

[Model](../../../../glossary.md#term-model) construction validates index representation, not index contents. It reads neither [host
storage](../../../../glossary.md#host-storage) nor values, so even [eager](../../../../glossary.md#term-eager-evaluation) [constant](../../../../glossary.md#term-constant) indices are not inspected here. A valid executed [lookup](../../../../glossary.md#term-lookup)
requires each index value `i` to satisfy:

```text
0 <= i < bound vocabulary extent
```

Negative and out-of-range values are invalid. They do not wrap, clamp, select a [padding](../../../../glossary.md#padding)/default
row, or produce zero. Because the retained [operation](../../../../glossary.md#operation) is ordinary [Gather](../../../../glossary.md#gather), this is not an
[embedding](../../../../glossary.md#embedding)-only execution branch: future Gather [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend) support must apply its common
safe bounds contract. Compiler [validation](../../../../glossary.md#term-validation) may reject captured constant invalid indices when it
can prove them. After dynamic extents are bound, backend preparation or the [prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable)
must prove or check bounds and fail safely rather than perform an out-of-bounds read. The exact
future execution exception/status belongs to the later execution contract and is not invented by
this model task. [Runtime](../../../../glossary.md#term-runtime) executes prepared behavior and must not inspect the original `Operation`.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains inside the model-owned public Tensor and existing Gather-construction boundary plus
  its documentation/planning records.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- [`Operation`](../../../../glossary.md#operation) retains [backend](../../../../glossary.md#backend)-independent [`GATHER`](../../../../glossary.md#gather) meaning only; it gains no embedding identity,
  support metadata, route, kernel, or execution state.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [kernel glossary](../../../../glossary.md#kernel))</span>
- Package direction remains `model.tensor -> model.operation.index`, datatype, [shape](../../../../glossary.md#shape), and [layout](../../../../glossary.md#layout).
- Compiler owns capture, graph-wide validation, constant analysis, autograd, and backward-graph
  construction. Backend prepare owns lowering, fusion, specialization, bounds strategy, and kernel
  choice. Runtime executes prepared work only.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [constant glossary](../../../../glossary.md#term-constant); [autograd glossary](../../../../glossary.md#autograd); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [module](../../../../glossary.md#term-module)-boundary, build, or cross-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation.index`

Packages added or changed:

- No package is added. The existing `model.tensor` package gains one public facade method and one
  package-private helper method.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — existing [owner](../../../../glossary.md#term-owner) of the public fluent convenience.
- `io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressions` — existing package-private
  owner of Gather validation, Shape derivation, and final derived-Tensor construction.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor))</span>

Tests mirror the production package because they inspect package-private helper and [producer](../../../../glossary.md#term-producer)
contracts.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorEmbeddingExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorAxisGatherExpressionTest.java`
  — update only the exact helper method inventory for [`embedding`](../../../../glossary.md#embedding).
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — add the exact
  public signature/name inventory and change the total public method count from 160 to 161.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — change only the shared total public Tensor method count from 160 to 161.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — change only the shared total public Tensor method count from 160 to 161.

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task file
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the final implementation makes a current statement inaccurate:
Training [API](../../../../glossary.md#term-application-programming-interface-api); [axis](../../../../glossary.md#term-axis)-[gather](../../../../glossary.md#gather) semantic Javadocs; [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension), TensorDescriptor,
TensorFactory/createDerived, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance), [operation](../../../../glossary.md#operation)-[signature](../../../../glossary.md#term-signature), and related expression
contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production, five test, and seven documentation/[planning](../../../../glossary.md#term-planning) files: exactly 14 paths.
`Tensor.java` changes only for the exact method and its [Javadoc](../../../../glossary.md#term-javadoc). The [axis](../../../../glossary.md#term-axis)-[gather](../../../../glossary.md#gather) helper changes
only for [`embedding`](../../../../glossary.md#embedding), its documentation, and the helper inventory it necessarily changes. The
three historical [API](../../../../glossary.md#term-application-programming-interface-api)-count tests change only `160` to `161` plus `TensorTest`'s exact name and
[signature](../../../../glossary.md#term-signature) inventory.

Stop before a fifteenth path, a new production helper/type, any semantic/attribute/foundation
edit, another existing test edit, cross-[module](../../../../glossary.md#term-module) work, or architecture/build change.

## [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation requirements

- Add complete Javadoc to `Tensor.embedding` and the helper [`embedding`](../../../../glossary.md#embedding) method, including receiver
  and argument roles, exact signature, accepted types/ranks, Shape formula and Dimension identity,
  metadata, delegation, producer/provenance/ID behavior, validation messages/order, value-bounds
  boundary, failures, and cross-layer exclusions.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Update the helper type Javadoc only as needed to include its embedding validation/composition
  role without weakening the existing Gather contract.
- Add the `[10, 4]` [weights](../../../../glossary.md#term-weight) and `[2, 3]` indices newcomer example to the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api). Explain that
  `[2, 3, 4]` is [metadata](../../../../glossary.md#term-metadata) and that no row values were read.
- Update Compile API only to keep the current public-expression inventory and planned compiler/
  autograd/bounds boundary accurate. Do not claim compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>
- Do not change Training API: its current compiler-owned autograd and backend-independent training
  boundary remains accurate because this convenience adds no parameter, gradient, optimizer,
  publication, session, or training-execution contract. Record that no-change conclusion.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- Update the existing [Gather](../../../../glossary.md#gather) glossary entry or add an `Embedding` entry only if the independent
  documentation pass finds the reusable convenience distinction needs its own navigation term.
- Keep capabilities, task, master plan, and roadmap synchronized. Record reasoned no-change
  conclusions for Training API and every reviewed unchanged contract.

## Acceptance criteria

- Exactly one public `embedding(Tensor)` method exists; the public [Tensor](../../../../glossary.md#tensor) method count is 161.
- [Weights](../../../../glossary.md#term-weight) are [rank](../../../../glossary.md#term-rank) two and exactly floating; indices are exactly [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) and may have any [Shape](../../../../glossary.md#shape).
- Result Shape is the complete indices Shape plus the exact weight axis-one Dimension, including
  scalar, zero, static, named dynamic, and expression-dimension cases.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Dimension glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Validation](../../../../glossary.md#term-validation) follows the exact order/messages and every local failure consumes no ID.
- The public method delegates once to the helper; the helper delegates once directly to existing
  `gather(weights, indices, 0)` and constructs no intermediate Tensor or public call.
- Every success is exactly one [`GATHER`](../../../../glossary.md#gather)/axis-zero/two-input/one-output occurrence, one producer,
  provenance output index zero, and one fresh ID, with exact <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>`[weights, indices]` references.
- Result type and `requiresGrad` come only from weights; layout is unresolved; label and storage are
  absent; inputs are unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Tests establish metadata and construction behavior without eager value execution or inventing a
  gradient/backend/runtime contract.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [eager glossary](../../../../glossary.md#term-eager-evaluation); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Negative/out-of-range values remain unread during construction and are documented as invalid at
  execution, with no wrap/clamp/padding/default behavior and safe enforcement deferred to the
  future ordinary Gather execution contract.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- No [`EMBEDDING`](../../../../glossary.md#embedding) kind, padding/sparse/max-norm/frequency option, mutable table, gradient rule,
  compiler/backend/runtime implementation, dependency, Gradle, architecture, or other-module work
  lands.<span data-glossary-navigation="true"> ([norm glossary](../../../../glossary.md#term-norm); [mutable glossary](../../../../glossary.md#term-mutable); [compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, exactly one final model test after Java stability, final model Javadoc,
  documentation/link/scope/status/formatting checks, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.
- A separate clean-context documentation pass finalizes all authorized documentation, reuses the
  final Java evidence, and records its context and no-change conclusions.
- 0019 and 0019A remain Complete; 0019A1 becomes Complete only after all evidence; 0019A2,
  0019B–0019E, and later tasks remain Draft without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Required focused command during implementation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorEmbeddingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact public/helper surfaces and one-call delegation; [rank](../../../../glossary.md#term-rank)/type/null ordering
and messages; no-ID local failures; all [weight](../../../../glossary.md#term-weight)/index types; [scalar](../../../../glossary.md#term-scalar), zero, static, named dynamic,
and expression [Dimensions](../../../../glossary.md#dimension); exact [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance)/[producer](../../../../glossary.md#term-producer)/freshness; one-ID success; input
non-mutation; and absence of an [embedding](../../../../glossary.md#embedding) kind or value inspection.

The mandatory separate clean-context documentation pass receives and reuses the successful final
model-test evidence. It does not rerun Java tests unless it changes executable Java behavior or
records a concrete risk. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, terminology, generated
Javadoc, final newlines, trailing whitespace, exact 14-path [scope](../../../../glossary.md#term-scope), package placement, the
161-method and seven-helper-method inventories, exact [Ready](../../../../glossary.md#term-ready)/Complete/Draft [synchronization](../../../../glossary.md#term-synchronization), one
Ready model frontier, and absence of a task-0019A2 or later detailed specification.

Repository-wide validation is deferred to the selected modern-[operation](../../../../glossary.md#operation) capability [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0022 and to [CI](../../../../glossary.md#term-continuous-integration-ci). This is a task-tier, single-[module](../../../../glossary.md#term-module) convenience with no [dependency](../../../../glossary.md#term-dependency),
architecture, build, or cross-module change.

## Dependencies

- Completed task 0018K supplies exact [operation](../../../../glossary.md#operation) [signatures](../../../../glossary.md#term-signature) and local [occurrence](../../../../glossary.md#term-occurrence) [validation](../../../../glossary.md#term-validation).
- Completed task 0018L supplies the one-[producer](../../../../glossary.md#term-producer)/output-index [provenance](../../../../glossary.md#provenance) contract.
- Completed task 0018O supplies the final [canonical](../../../../glossary.md#term-canonical-form) [`GATHER`](../../../../glossary.md#gather) taxonomy and public/helper path.
- Completed tasks 0001, 0002, 0007, and 0013 supply DataType, exact Dimension/Shape retention,
  descriptor eligibility, derived construction, identity, and provenance foundations.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Follow-up tasks

- 0019A2: focused [one-hot encoding](../../../../glossary.md#one-hot-encoding) semantics. It remains Draft without a detailed specification.
- 0023: compiler-generated semantic operations may later own Gather adjoints; it remains Draft and
  does not implement autograd traversal in this task.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Gather glossary](../../../../glossary.md#gather); [autograd glossary](../../../../glossary.md#autograd))</span>

Do not create either specification during task 0019A1.

## Architecture impact

Expected impact: None.

This task adds a [model](../../../../glossary.md#term-model)-owned public convenience over an existing model-owned semantic [occurrence](../../../../glossary.md#term-occurrence).
If implementation requires a new semantic [identity](../../../../glossary.md#term-identity-meanings), cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or different
[compiler](../../../../glossary.md#term-compiler)/[backend ownership](../../../../glossary.md#backend-ownership), stop and report the concrete conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/lifecycle/training architecture docs,
docs/developer-guide/documentation-rules.md and applicable profiles, the planning guide and
roadmap, model capabilities/master plan, completed foundation/provenance/signature/indexing tasks,
completed tasks 0019/0019A, this task, Tensor/Compile/Training APIs, glossary, and the current
Tensor, TensorAxisGatherExpressions, Gather semantics/attributes, Shape/Dimension, DataType,
TensorDescriptor, TensorFactory/createDerived, producer/provenance, signatures, tests, and API
inventories.

Implement task 0019A1 exactly. Add only public weights.embedding(indices) and the existing
axis-gather helper's embedding validation/delegation method. Preserve one ordinary axis-zero
GATHER occurrence, exact [weights, indices] provenance, one producer, one ID, weight-only
requiresGrad, exact indices Dimensions plus embedding Dimension, and unresolved layout. Follow
the specified validation order/messages and no-ID failures.

Do not add an EMBEDDING kind, another helper/type/overload, padding/sparse/max-norm/frequency
options, eager value access, a mutable table, gradient/compiler/backend/runtime behavior,
dependencies, build/architecture changes, other modules, or later specs. Stop beyond the exact
14 paths or on an architecture/scope conflict.

Run the focused command and exactly one final model suite after Java stability. Then hand the
actual diff, task, affected API/behavior, architecture constraints, drafted Javadocs/docs, and
exact Java evidence to a separate clean-context documentation-focused agent in the same overall
change. That pass must independently inspect source/tests, finalize permitted Javadocs, Tensor
API, Compile API, glossary impact, capabilities/task/master/roadmap, run final model Javadoc and
documentation validation, reuse Java evidence unless executable behavior changes, and record
reasoned no-change conclusions.

Update task evidence, implementation notes, completion summary, and synchronized status only
after both passes finish. Keep 0019/0019A Complete and 0019A2 plus 0019B-0019E/later Draft without
detailed specs. Mark Complete only when every acceptance criterion passes.
```

## Separate documentation [handoff](../../../../glossary.md#term-handoff)

The implementation agent must provide the documentation-focused clean context with:

- this task and the exact final implementation/test diff;
- the affected `Tensor.embedding` and helper behavior, existing Gather semantics, and exact Java
  test command/results;<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather))</span>
- architecture constraints and the explicit no-new-kind/no-cross-layer boundary;
- drafted Javadocs and expected Tensor API, Compile API, glossary, capabilities, task, master, and
  roadmap impact; and<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- required Javadoc, generated-page, example, Markdown, terminology, scope, status, and whitespace
  checks.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

The documentation pass applies General plus [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) style to public/explanatory content,
[Planning](../../../../glossary.md#term-planning) style to planning records, and Example format to the newcomer example. It must name its
clean context and record reviewed files, changes, commands/results, limitations, glossary outcome,
and reasoned no-change conclusions. The task remains incomplete until this evidence lands in the
same overall change.

## Local decisions

- Kept [`embedding`](../../../../glossary.md#embedding) as a thin public convenience and package-private validation boundary over the
  existing Gather construction path. The helper performs the task-owned receiver checks and then
  returns <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Gather glossary](../../../../glossary.md#gather))</span>`gather(weights, indices, 0)` directly; no descriptor, producer, or Tensor is created by
  the convenience itself.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Added a glossary `Embedding` entry because the public convenience is a reusable navigation term
  whose distinction from a separate operation kind and from configurable training-framework
  embedding layers is useful beyond one API paragraph.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Placed the complete `[10, 4]` by `[2, 3]` metadata example inside the existing axis-gather API
  section. It demonstrates Shape, operation, attributes, ordered inputs, output index, eligibility,
  and storage boundaries without implying eager row selection.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Kept execution ownership phrased as the ordinary Gather contract: compiler constant analysis
  may reject provably invalid captured indices; preparation or prepared execution must enforce
  bounds after binding. The model convenience itself does not select an exception or status.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [constant glossary](../../../../glossary.md#term-constant); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [binding glossary](../../../../glossary.md#term-binding); [model glossary](../../../../glossary.md#term-model))</span>

## Known limitations

- Model construction intentionally accepts stored negative and out-of-range index values because
  it does not inspect values. Such values are invalid for later ordinary Gather execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Gather glossary](../../../../glossary.md#gather))</span>
- No padding index, sparse-gradient flag, maximum-norm option, frequency scaling, negative-index
  wrapping, configurable table axis, gradient rule, compiler capture, backend lowering, or
  execution support is included.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient); [norm glossary](../../../../glossary.md#term-norm); [axis glossary](../../../../glossary.md#term-axis); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- Repository-wide validation remains deferred to the selected modern-operation checkpoint after
  task 0022 and CI. This task changed one module without changing dependencies, architecture, or
  shared build configuration.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [module glossary](../../../../glossary.md#term-module))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context ran the exact focused command:

  ```bash
  ./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorEmbeddingExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorAxisGatherExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
  ```

  It passed. Focused coverage included the exact facade/helper surface, validation order and
  messages, no-ID failures, type/rank/Shape cases, exact Dimension retention, producer/provenance
  identity, input non-mutation, and construction-time non-inspection of stored invalid values.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- After executable Java stabilized, the implementation context ran exactly one final command:

  ```bash
  ./gradlew :modules:model:test
  ```

  It passed with 94 suites and 765 tests, with zero failures, errors, or skips. `git diff --check`
  also passed after Java stability. The documentation context reused this evidence and did not
  rerun Java tests because it changed only Javadocs and Markdown afterward.
- Separate clean-context documentation review [identity](../../../../glossary.md#term-identity-meanings):
  `/root/task_0019a1_documentation`. It applied General style, API/Javadoc style, Planning style,
  and Example format. It independently read the implementation and tests; finalized both
  production Javadocs and all seven authorized documentation/planning paths; and reviewed the
  final Tensor, Gather, descriptor, producer/provenance, Shape/Dimension, operation-signature,
  compiler, training, architecture, testing, and build boundaries.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Documentation context ran:

  ```bash
  ./gradlew :modules:model:javadoc
  ```

  It passed on OpenJDK 26.0.1 (`BUILD SUCCESSFUL`, two executed tasks). The generated public
  `Tensor.html` contains the `embedding(Tensor)` signature, direct ordinary axis-zero Gather
  explanation, accepted types/Shape, value-bound boundary, and unsupported options. The
  package-private helper Javadoc was checked in source because the default public Javadoc task
  does not emit that package-private type.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- A Java 26 [compile](../../../../glossary.md#compile)-and-[run](../../../../glossary.md#term-run) check of the documented `EmbeddingExpressionExample` passed and
  printed exactly `Shape[2, 3, 4]`, [`GATHER`](../../../../glossary.md#gather), `IndexAxisAttrs[axis=0]`, two true ordered-input
  identity checks, output index `0`, true [weight](../../../../glossary.md#term-weight) eligibility, and true empty [layout](../../../../glossary.md#layout)/label/storage.
- `javap`, reflection, import, and source checks passed. They confirmed exactly one public
  `Tensor.embedding(Tensor)`, one package-private static helper, 161 public Tensor methods, seven
  helper methods, the required `AxisGatherKind` imports, the one-call public delegation, and the
  exact `return gather(weights, indices, 0)` helper delegation.
- Final documentation validation checked local Markdown targets and heading anchors, balanced
  fences, final newlines, trailing whitespace, terminology, the generated page, and the runnable
  example. It also confirmed exactly 14 changed paths; no architecture/build/other-module path;
  0019, 0019A, and 0019A1 Complete; 0019A2 and 0019B–0019E/later Draft; no later detailed task
  specification; no Ready model task; and a final successful <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [module path glossary](../../../../glossary.md#term-module-path); [Ready glossary](../../../../glossary.md#term-ready); [model glossary](../../../../glossary.md#term-model))</span>`git diff --check`.
- Training API no-change conclusion: the page already assigns global autograd to the compiler and
  backend-specific lowering to backend owners. Embedding adds no parameter, gradient, optimizer,
  publication, session, or training-execution contract, so changing that page would be inaccurate
  scope expansion.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [Embedding glossary](../../../../glossary.md#embedding); [parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication); [scope glossary](../../../../glossary.md#term-scope))</span>
- Architecture and focused architecture documentation required no change because module ownership,
  dependency direction, lifecycle, and runtime hot-path rules are unchanged. Architecture tests,
  backend conformance, and integration tests required no change because no boundary, backend, or
  end-to-end behavior changed. Compile API required only an inventory and planned-ownership
  clarification; it does not claim compiler support. Gradle and other modules required no change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Implementation notes

- Added `Tensor.embedding(Tensor)` immediately after [`gather`](../../../../glossary.md#gather) and added exactly one package-private
  `TensorAxisGatherExpressions.embedding(weights, indices)` method.
- The helper checks weights null, indices null, weight rank two, floating weight type, and exact
  index type in the specified order, then delegates directly to existing axis-zero Gather.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis))</span>
- Added focused embedding tests and updated only the specified helper/public inventories and three
  historical public-method counts. No executable Java changed after the final model test.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [model glossary](../../../../glossary.md#term-model))</span>
- Final documentation explains receiver/indices/result roles, exact Shape and Dimension identity,
  one ordinary producer/output-index-zero/ID occurrence, weight-only metadata, unresolved result
  layout, value-bound ownership, unsupported options, and planned compiler/autograd/backend/runtime
  responsibilities.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [metadata glossary](../../../../glossary.md#term-metadata); [layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: added the validated rank-two floating embedding convenience as exactly one
  ordinary axis-zero Gather occurrence, plus focused tests, complete Javadocs, API/glossary
  explanation, and synchronized planning status.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [Gather glossary](../../../../glossary.md#gather); [occurrence glossary](../../../../glossary.md#term-occurrence); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: two production Java files, five model test files, Tensor API, Compile
  API, glossary, model capabilities, this task, model master plan, and roadmap; exactly 14 paths.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- Tests and validation: focused tests and the final 94-suite/765-test model run passed in the
  implementation context; final model Javadoc, Java 26 example/metadata checks, generated-page,<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  `javap`, reflection, import/source, Markdown, scope/status, and whitespace checks passed in the
  documentation context.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/task_0019a1_documentation` completed the mandatory targeted
  independent pass without changing executable behavior or rerunning Java tests.
- Documentation impact: Tensor and Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized. Training API, architecture documentation, test guides, and other-module docs
  remain accurate without changes for the reasons recorded above.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: both affected production contracts are complete; public generated Javadoc passed.
- Glossary impact: added [`Embedding`](../../../../glossary.md#embedding) to distinguish the convenience from a semantic kind or
  configurable training-layer abstraction.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
