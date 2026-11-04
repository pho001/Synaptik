# Task 0017A: [Contiguous](../../../../glossary.md#term-contiguous) Semantic Kind

**Editorial reading key (later context, not a status update).**

Contiguous requests the same logical values in dense row-major, zero-offset geometry: the last axis varies fastest without gaps between rows. This kind does not mean COPY or ALIAS; those are realization choices made by later planning/preparation. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current contiguous-expression boundary](../../../../reference/tensor/reshape-expand.md#contiguous-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the typed, [backend](../../../../glossary.md#backend)-independent, parameterless semantic [identity](../../../../glossary.md#term-identity-meanings) for a [contiguous](../../../../glossary.md#term-contiguous)-[layout](../../../../glossary.md#layout)
request: `CONTIGUOUS`.

The [operation](../../../../glossary.md#operation) preserves logical values, [shape](../../../../glossary.md#shape), [data type](../../../../glossary.md#data-type--datatype), and [element](../../../../glossary.md#term-element) order while requesting a
[canonical](../../../../glossary.md#term-canonical-form) dense [row-major](../../../../glossary.md#term-row-major) result. This task defines only that meaning. It does not add
`Tensor.contiguous()`, derive a result [descriptor](../../../../glossary.md#term-descriptor), inspect an input layout, decide whether an [alias](../../../../glossary.md#aliasing-and-arena)
is sufficient, materialize storage, copy values, or select an executable implementation.

## [Scope](../../../../glossary.md#term-scope)

- Add one public enum `ContiguousKind` implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly one [constant](../../../../glossary.md#term-constant), [`CONTIGUOUS`](../../../../glossary.md#term-contiguous).
- Document one logical input and unchanged logical values, shape, data type, and row-major element
  order.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element))</span>
- Document the requested result geometry as canonical dense row-major layout with zero logical
  storage offset, without constructing a <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>`LayoutDescriptor` in this task.
- Establish that the kind has no intrinsic attributes and composes explicitly with
  `NoOperationAttrs.INSTANCE` when represented by [`Operation`](../../../../glossary.md#operation).
- Distinguish a semantic [contiguous request](../../../../glossary.md#contiguous-request) from the current geometric
  `LayoutKind.DENSE_CONTIGUOUS` classification.
- Document that later layers may preserve an already-suitable representation or materialize a
  copy, and that this enum makes neither decision.
- Add one focused same-package test proving exact vocabulary, typed identity, enum behavior,
  parameterless <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`Operation` composition, and absence of [metadata](../../../../glossary.md#term-metadata) or state.
- Add the cohesive `model.operation.layout` package to the [model](../../../../glossary.md#term-model) package map.
- Finalize affected Javadoc, Tensor API semantic reference, glossary, task evidence, model master
  plan, and roadmap through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.contiguous()`, another Tensor method, static facade, overload, factory, builder,
  expression helper, or task-0017B implementation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade))</span>
- input Tensor, provenance, identity, graph values/nodes, labels, host storage, result
  construction, or <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node); [host storage glossary](../../../../glossary.md#host-storage))</span>`TensorFactory.createDerived`
- [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), `LayoutDescriptor`, `LayoutKind`, strides, storage offset, referenced span,
  view flag, result layout construction, or dynamic-shape layout resolution<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset); [referenced span glossary](../../../../glossary.md#referenced-element-span); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout))</span>
- checking whether an input is contiguous, returning the input, aliasing storage, allocating or
  copying storage, materialization policy, mutation, publication, or execution<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [aliasing glossary](../../../../glossary.md#term-aliasing); [materialization glossary](../../../../glossary.md#materialization); [publication glossary](../../../../glossary.md#term-publication))</span>
- reshape, expand, permute, transpose, expand-dimensions, squeeze, slice, pad, tile, concat, stack,
  unstack, unfold, fold, select, gather, or scatter semantics<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [dimensions glossary](../../../../glossary.md#dimension); [slice glossary](../../../../glossary.md#slice); [gather glossary](../../../../glossary.md#gather))</span>
- attributes, layout-requirement records, factories, registries, parsers, aliases, symbols,
  visitors, string dispatch, reflection discovery, maps, services, or compatibility validators<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- arity, family, result-kind, cost, fusion, differentiability, capability, backend support,
  lowering, route, kernel, executable, or physical-buffer metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [buffer glossary](../../../../glossary.md#term-buffer); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- compiler canonicalization, logical materialization requirements, planning ownership, prepare-time
  lowering, runtime residency, backend storage, gradient rules, autograd, or training behavior<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd))</span>
- changes to Operation foundations, layout value contracts, Tensor, graph records, existing Java
  tests, dependencies, Gradle, architecture, or another module<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0017B specification or any later 0017 task specification

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
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
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

Its [operation](../../../../glossary.md#operation) [descriptor](../../../../glossary.md#term-descriptor) identifies one parameterless [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) meaning. The public operation
preserves logical [shape](../../../../glossary.md#shape) and [data type](../../../../glossary.md#data-type--datatype). Legacy execution can reuse an already suitable
representation or materialize logical values from offset, permuted, sliced, expanded, or other
non-dense [views](../../../../glossary.md#term-view-meanings) into [canonical](../../../../glossary.md#term-canonical-form) [row-major](../../../../glossary.md#term-row-major) storage. Tests cover [BOOL](../../../../glossary.md#data-type--datatype) and every numeric type,
permuted/expanded/sliced inputs, chained expressions, [CPU](../../../../glossary.md#term-cpu) routes, [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), and preparation or
[runtime](../../../../glossary.md#term-runtime) [layout](../../../../glossary.md#layout) handling.

Legacy operation traits, `OpType`, result-kind labels, cost/[fusion](../../../../glossary.md#term-fusion) flags, storage [views](../../../../glossary.md#view), physical
[buffer](../../../../glossary.md#term-buffer) policies, materializers, [kernel](../../../../glossary.md#kernel) classes, [lowering](../../../../glossary.md#lowering), runtime propagation, and [gradient](../../../../glossary.md#term-gradient)
callbacks are evidence only and are not copied. The new [model](../../../../glossary.md#term-model) retains only the stable semantic
request. Task 0017B will own public [Tensor](../../../../glossary.md#tensor) expression construction; [compiler](../../../../glossary.md#term-compiler), [planning](../../../../glossary.md#term-planning), [prepare](../../../../glossary.md#prepare),
runtime, and concrete [backends](../../../../glossary.md#backend) retain their architecture-defined responsibilities.

## Architecture constraints

- Production remains in `modules/model`, which owns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics.
- `ContiguousKind` is a typed semantic discriminator, not a layout descriptor, Tensor, graph
  occurrence, materialization requirement, physical copy, executable operation, backend
  capability, or kernel route.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence); [materialization glossary](../../../../glossary.md#materialization); [backend capability glossary](../../../../glossary.md#backend-capability); [kernel glossary](../../../../glossary.md#kernel))</span>
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) through inherited `Enum.name()` and adds no duplicate name
  field or method.
- [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) has no intrinsic [parameters](../../../../glossary.md#term-parameter). Its complete attributes value is
  `NoOperationAttrs.INSTANCE`, never null or an empty map.
- The operation has one logical input and preserves its logical values, Shape, DataType, and
  row-major element order. These facts are documentation, not stored arity or descriptor state.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element); [arity glossary](../../../../glossary.md#term-arity))</span>
- The requested logical result geometry is canonical dense row-major layout with zero logical
  storage offset. A later Tensor expression decides how that request is represented in a<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>
  `TensorDescriptor`, including the unresolved state required by dynamic [shapes](../../../../glossary.md#shape).
- `ContiguousKind.CONTIGUOUS` expresses a computation request. `LayoutKind.DENSE_CONTIGUOUS`
  classifies already-resolved geometry. Neither type replaces or depends on the other.
- Model semantics do not decide whether a request can alias an existing representation, can be
  eliminated by compiler proof, or needs materialization. Planning derives logical materialization
  requirements; backend prepare chooses lowering and a concrete route; runtime executes the
  prepared schedule.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [alias glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule))</span>
- Stable enum text is diagnostic typed vocabulary, not serialization, ONNX mapping, a registry
  key, backend dispatch key, kernel name, or reflection identifier.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [ONNX glossary](../../../../glossary.md#term-onnx); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Package direction is `model.operation.layout -> model.operation` only. This task must not import
  Tensor, datatype, shape, layout-value, storage, graph, compiler, planning, prepare, runtime,
  backend, engine, trace, or training packages.<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- Stop if implementation needs another type, attributes, descriptor inference, Tensor behavior,
  materialization policy, dependency, or architecture change.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.operation` — supplies [`OperationKind`](../../../../glossary.md#operationkind), [`Operation`](../../../../glossary.md#operation), and
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) for the semantic contract and composition test.

Package added:

```text
io.github.pho001.synaptik.model.operation.layout
  Typed backend-independent layout and view operation meanings and immutable parameters.
```

This operation package is distinct from `io.github.pho001.synaptik.model.layout`, which describes
resolved geometry. The operation package states requested computation semantics; the [layout](../../../../glossary.md#layout)-value
package stores already-resolved [shape](../../../../glossary.md#shape)-to-storage geometry without representing an operation.

Type placement:

- `io.github.pho001.synaptik.model.operation.layout.ContiguousKind` — public parameterless
  contiguous-request identity.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ContiguousKindTest` — same-package focused test for vocabulary, contract shape, composition,
  and dependency boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Required contract

### Enum declaration

Create exactly:

```java
public enum ContiguousKind implements OperationKind {
    CONTIGUOUS
}
```

The enum declares no project field, explicit constructor, method, nested type, per-[constant](../../../../glossary.md#term-constant) class
body, [alias](../../../../glossary.md#aliasing-and-arena), [symbol](../../../../glossary.md#term-symbol), or [metadata](../../../../glossary.md#term-metadata). [Compiler](../../../../glossary.md#term-compiler)-generated enum machinery is not additional project [API](../../../../glossary.md#term-application-programming-interface-api).
Inherited `Enum.name()` satisfies `OperationKind.name()` and returns exact text [`CONTIGUOUS`](../../../../glossary.md#term-contiguous).

The sole kind means: preserve the input's logical values, [shape](../../../../glossary.md#shape), [data type](../../../../glossary.md#data-type--datatype), and [element](../../../../glossary.md#term-element) order while
requesting [canonical](../../../../glossary.md#term-canonical-form) dense [row-major](../../../../glossary.md#term-row-major) result geometry with logical [storage offset](../../../../glossary.md#term-storage-offset) zero. It does not
state whether the eventual implementation aliases, copies, allocates, eliminates, fuses, lowers,
or executes the request.

### Parameterless [Operation](../../../../glossary.md#operation) composition

The kind composes explicitly as:

```java
Operation operation = new Operation(
        ContiguousKind.CONTIGUOUS,
        NoOperationAttrs.INSTANCE);
```

Do not add a family factory, enum `operation()` method, attributes record, [layout](../../../../glossary.md#layout) requirement,
result [descriptor](../../../../glossary.md#term-descriptor), or compatibility [registry](../../../../glossary.md#term-registry). Generic `Operation` remains an open typed pairing
and does not enforce family-specific input count, result geometry, or kind-to-attributes pairing.

### Semantic and geometric distinction

The implementation and documentation must preserve this distinction:

| Concept | Meaning | Stored by this task |
|---|---|---|
| `ContiguousKind.CONTIGUOUS` | a request to produce logically equivalent [canonical](../../../../glossary.md#term-canonical-form) dense [row-major](../../../../glossary.md#term-row-major) output | yes |
| `LayoutKind.DENSE_CONTIGUOUS` | classification of resolved [strides](../../../../glossary.md#term-stride) and zero offset for a static [shape](../../../../glossary.md#shape) | no |
| [materialization](../../../../glossary.md#materialization) requirement | [planning](../../../../glossary.md#term-planning) fact that a logical representation needs a copy or conversion | no |
| prepared copy or [alias](../../../../glossary.md#aliasing-and-arena) | [backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime) implementation chosen before execution | no |

The table describes [ownership](../../../../glossary.md#term-ownership-meanings) boundaries. It does not add a Java [dependency](../../../../glossary.md#term-dependency) from the [operation](../../../../glossary.md#operation)
package to the [layout](../../../../glossary.md#layout)-value package.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKind.java`

Test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKindTest.java`

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
- `LayoutKind`, `LayoutDescriptor`, Operation foundations, existing operation families, Tensor
  expressions, and their Javadocs/tests<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- focused architecture documents, ADRs, architecture tests, backend-conformance material,
  integration tests, and Gradle configuration<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production file, one focused test, and five documentation/[planning](../../../../glossary.md#term-planning) files: seven paths
total.

Do not modify existing Java source/tests, capabilities, completed tasks, [Compile](../../../../glossary.md#compile)/Training [API](../../../../glossary.md#term-application-programming-interface-api),
[Gradle](../../../../glossary.md#term-gradle), AGENTS, architecture documents/tests, another [module](../../../../glossary.md#term-module), or unrelated documentation. Stop if
another production concept, attributes, [Tensor](../../../../glossary.md#tensor) expression, [descriptor](../../../../glossary.md#term-descriptor) rule, [dependency](../../../../glossary.md#term-dependency), or eighth
path is needed. Do not create task 0017B.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document the enum as [backend](../../../../glossary.md#backend)-independent parameterless [contiguous](../../../../glossary.md#term-contiguous)-request vocabulary.
- Document `CONTIGUOUS` as preserving logical values, Shape, DataType, and element order while
  requesting canonical dense row-major zero-offset result geometry.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [DataType glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Explain the distinction from resolved `LayoutKind.DENSE_CONTIGUOUS` without importing or linking
  the operation type to the layout-value package.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Explain one logical input without storing or validating [arity](../../../../glossary.md#term-arity).
- Document explicit [`Operation`](../../../../glossary.md#operation) composition with `NoOperationAttrs.INSTANCE` and absence of generic
  family compatibility validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- State that alias reuse, compiler elimination, materialization, allocation, copying, lowering,
  execution, gradient behavior, and backend availability are outside this semantic type.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Explain that enum names are diagnostic and are not [serialization](../../../../glossary.md#term-serialization) or [dispatch](../../../../glossary.md#term-dispatch) contracts.
- Review related Javadocs and record why they remain accurate, or stop on an inconsistency.

## Acceptance criteria

- Exactly one public `ContiguousKind` enum is added in the planned [operation](../../../../glossary.md#operation)-[layout](../../../../glossary.md#layout) package.
- It implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly [`CONTIGUOUS`](../../../../glossary.md#term-contiguous).
- It adds no project field, method, constructor, nested type, constant body, alias, attributes,
  arity, category, result, cost, fusion, differentiability, layout state, backend, inference,
  materialization, or execution metadata.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [alias glossary](../../../../glossary.md#aliasing-and-arena); [arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [inference glossary](../../../../glossary.md#term-inference); [materialization glossary](../../../../glossary.md#materialization); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Inherited name and standard enum equality, hashing, and diagnostic text behavior remain.
- `CONTIGUOUS` constructs a valid `Operation` with the exact kind and
  `NoOperationAttrs.INSTANCE`; no factory or attributes type is added.
- Javadoc explains logical preservation, requested canonical dense row-major zero-offset geometry,
  and the semantic-kind versus resolved-layout distinction.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Production imports only `OperationKind`; no Tensor, DataType, Shape, LayoutDescriptor,
  LayoutKind, provenance, graph, compiler, planning, prepare, runtime, backend, storage, training,
  dependency, or architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No public Tensor API, result descriptor, layout inspection, alias/copy decision, storage access,
  gradient rule, compiler behavior, or execution is implemented.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Focused and aggregate tests, Javadoc, root tests, reflection/javap/import/scope checks,
  documentation links and formatting, and synchronized statuses pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation agent finalizes Javadoc, Tensor API, glossary, evidence,
  master plan, and roadmap and records related no-change conclusions.
- Task 0017A becomes Complete only after both passes. Task 0017B remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.layout.ContiguousKindTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover the exact package, visibility, implemented interface, sole [constant](../../../../glossary.md#term-constant),
name, declaration order, absence of project fields/methods/nested types/constant bodies, standard
enum [identity](../../../../glossary.md#term-identity-meanings), typed distinction from a private test-local equal-name kind, and explicit
[`Operation`](../../../../glossary.md#operation) composition retaining the exact kind and `NoOperationAttrs.INSTANCE` references.

Manually inspect `javap -p -c -s`, reflection, imports, and [Gradle](../../../../glossary.md#term-gradle) dependencies. Confirm that no
[Tensor](../../../../glossary.md#tensor), [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), LayoutDescriptor, LayoutKind, [provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/
[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) type, storage access, [materialization](../../../../glossary.md#materialization) decision, result tag, [gradient](../../../../glossary.md#term-gradient), cost,
[fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, or service appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor [API](../../../../glossary.md#term-application-programming-interface-api) semantic
status, glossary terminology, links, anchors, fences, whitespace, exact seven-path [scope](../../../../glossary.md#term-scope),
synchronized statuses, package-map placement, and absence of a task-0017B specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind) and `NoOperationAttrs.INSTANCE`.
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and exact reference retention.
- Task 0003 supplies the resolved layout vocabulary used only to review and document the ownership
  distinction; this enum has no Java dependency on it.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Task 0013 supplies the future derived-Tensor construction seam for task 0017B; it is
  implementation-order context rather than a dependency of this enum.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Follow-up tasks

- 0017B remains Draft for the exact public `Tensor.contiguous()` expression, result descriptor,
  static/dynamic layout handling, one-input provenance, and storage-free derived construction.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0017C–0017L remain Draft for reshape/expand, axis transforms, slicing, pad/tile, tensor
  composition, and unfold/fold semantic and expression groups.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [tensor composition glossary](../../../../glossary.md#tensor-composition))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns redundant-request [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and [graph](../../../../glossary.md#graph)-wide [layout](../../../../glossary.md#layout) reasoning.
- [Planning](../../../../glossary.md#term-planning) later derives logical [materialization](../../../../glossary.md#materialization) requirements without selecting a [kernel](../../../../glossary.md#kernel).
- Backend prepare later chooses alias/copy lowering and executable routes; runtime executes the
  prepared schedule.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule))</span>
- Training and compiler-generated semantic tasks later own differentiation and backward forms.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) semantics
and logical [layout](../../../../glossary.md#layout) facts to `modules/model`, [materialization](../../../../glossary.md#materialization) requirements to [planning](../../../../glossary.md#term-planning), concrete
[lowering](../../../../glossary.md#lowering) to backend [prepare](../../../../glossary.md#prepare), and [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) to [runtime](../../../../glossary.md#term-runtime). The new package refines [model](../../../../glossary.md#term-model)
[ownership](../../../../glossary.md#term-ownership-meanings) without adding cross-[module](../../../../glossary.md#term-module) dependencies or executable state.

If implementation requires [Tensor](../../../../glossary.md#tensor) behavior, [descriptor](../../../../glossary.md#term-descriptor) [inference](../../../../glossary.md#term-inference), materialization policy, backend
[metadata](../../../../glossary.md#term-metadata), another [dependency](../../../../glossary.md#term-dependency), or architecture change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0003/0005/0006/0013/0016J/0017A, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation and
LayoutKind/LayoutDescriptor contracts and focused tests, and Java 26 Gradle configuration.

Implement task 0017A exactly. Add only ContiguousKind.java and ContiguousKindTest.java under
io.github.pho001.synaptik.model.operation.layout for Java code and tests.

The public enum implements OperationKind and contains exactly CONTIGUOUS, with no project fields,
methods, nested types, aliases, attributes, arity, layout state, materialization state, or metadata.
It is parameterless and composes explicitly with Operation plus NoOperationAttrs.INSTANCE.
Document one logical input, unchanged values/Shape/DataType/element order, and the request for
canonical dense row-major zero-offset result geometry. Keep the semantic request distinct from
resolved LayoutKind.DENSE_CONTIGUOUS geometry and from planning/backend/runtime materialization.

Do not add Tensor methods, result descriptors, LayoutDescriptor dependencies, input-layout
inspection, alias/copy decisions, storage access, provenance, gradients, graph/compiler/planning/
prepare/runtime/backend behavior, other layout kinds/attributes, dependencies, build/architecture
changes, existing Java edits, or later specs. Stop beyond seven paths or on architecture doubt.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-
contract/capability/Compile API/Training API/architecture no-change conclusions, and rerun
validation.

Update task 0017A, model master plan, and roadmap only for planning status/evidence. Do not mark
0017A Complete until both passes succeed. Leave 0017B Draft without a specification. Do not commit
or push.
```

## Local decisions

- The type is `ContiguousKind`, not a broad frozen `LayoutOperationKind`. This task has one stable
  parameterless meaning; later parameterized layout families can receive cohesive typed contracts
  without modifying this completed enum.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- The package is `model.operation.layout`. It owns requested operation semantics and is deliberately
  distinct from <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`model.layout`, which owns resolved geometry values.
- The sole [constant](../../../../glossary.md#term-constant) is [`CONTIGUOUS`](../../../../glossary.md#term-contiguous). No `COPY`, `MATERIALIZE`, `DENSE`, or [`ALIAS`](../../../../glossary.md#aliasing-and-arena) synonym is added
  because those names would conflate the mathematical request with an implementation decision.
- `NoOperationAttrs.INSTANCE` completely represents the absence of intrinsic [parameters](../../../../glossary.md#term-parameter).
- The semantic target is canonical dense row-major zero-offset geometry. Whether that target is
  represented as resolved or unresolved descriptor metadata belongs to task 0017B because dynamic
  Shape handling is part of expression construction.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [Shape glossary](../../../../glossary.md#shape))</span>
- Planning and backend prepare, not this enum, decide whether an actual representation requires a
  copy. An already suitable input does not change the identity of the requested semantic operation.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Known limitations

- No public `Tensor.contiguous()` expression exists until task 0017B.
- No result descriptor, static/dynamic layout rule, provenance, or gradient eligibility is defined
  here.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The enum does not inspect input geometry, prove contiguity, choose aliasing, request a physical
  buffer directly, or enforce one input.<span data-glossary-navigation="true"> ([contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [aliasing glossary](../../../../glossary.md#term-aliasing); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- No compiler canonicalization, planning materialization, backend lowering, storage copy, runtime
  execution, differentiation, or backend support is implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read the [architecture contract](../../../../glossary.md#architecture-contract) and focused [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-boundary
explanations; documentation and planning rules; roadmap; [model](../../../../glossary.md#term-model) capability baseline and master
plan; tasks 0003, 0005, 0006, 0013, and 0016J; current [Operation](../../../../glossary.md#operation) and resolved-[layout](../../../../glossary.md#layout) contracts;
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs and glossary; and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

The read-only `legacy/pre-rewrite` branch was inspected directly. It confirms one public
`Tensor.contiguous()` capability, one parameterless [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) semantic [identity](../../../../glossary.md#term-identity-meanings), unchanged
logical [shape](../../../../glossary.md#shape)/[data type](../../../../glossary.md#data-type--datatype), and execution evidence for numeric and [BOOL](../../../../glossary.md#data-type--datatype) tensors plus permuted,
expanded, sliced, offset, [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), and [CUDA](../../../../glossary.md#term-cuda) paths. Legacy operation traits, materializers,
physical [views](../../../../glossary.md#term-view-meanings), storage policies, [kernels](../../../../glossary.md#kernel), [lowering](../../../../glossary.md#lowering), runtime propagation, [gradients](../../../../glossary.md#term-gradient), and [backend](../../../../glossary.md#backend)
selection are excluded from this semantic-only task.

Planning selected one enum and one focused test because the [contiguous request](../../../../glossary.md#contiguous-request) has one stable
parameterless meaning. Public Tensor construction, [descriptor](../../../../glossary.md#term-descriptor) handling, [provenance](../../../../glossary.md#provenance), and any
static-versus-dynamic layout decision remain in task 0017B. No existing dependency, layout value,
Operation foundation, or architecture contract changes.

Planning validation after synchronizing this task, the model master plan, and roadmap:

- `git diff --check` passed.
- The targeted trailing-whitespace scan returned no matches across the three changed planning
  files.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- All 180 local Markdown file links across the three planning files resolve.
- Markdown code-fence counts are balanced: twelve in this task, two in the master plan, and zero
  in the roadmap.
- All 20 canonical task-specification headings are present, together with the focused Capability
  origin, Required contract, and Javadoc requirements sections.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Task, model master plan, and roadmap consistently identify 0017A as Ready. Tasks 0017B–0017N
  remain Draft, and no task-0017B specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Package-map inspection finds exactly one new planned package,
  `model.operation.layout`, with direction only to `model.operation` for this task.
- Dependency review distinguishes hard Java prerequisites 0005/0006 from layout/provenance context
  0003/0013; later 0017 rows depend on the contracts they actually consume rather than forming a
  false chain.
- Repository scope is exactly this task, the model master plan, and the roadmap. No Java, API,
  glossary, architecture, Gradle, AGENTS, completed task, or other-module file changed during
  planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0017a` added exactly `ContiguousKind.java` and
  `ContiguousKindTest.java`. Clean documentation context
  `/root/implement_model_0017a/review_model_0017a_docs` independently read the required
  architecture, documentation and planning profiles, plans and historical tasks, APIs, glossary,
  Java 26 build configuration, final source and tests, generated Javadoc, XML reports, and the
  complete workspace diff. It applied General and API/Javadoc style to the enum, Tensor API, and
  glossary, and Planning style to this task, the model master plan, and roadmap. No substantive
  example changed, so Example format required no new complete example.
- Independent source, reflection, and [bytecode](../../../../glossary.md#term-bytecode) inspection confirmed one public final enum in
  `io.github.pho001.synaptik.model.operation.layout`, exactly `CONTIGUOUS`, inherited enum
  behavior, and only compiler-generated enum fields, methods, and constructor. Production source
  imports exactly [`OperationKind`](../../../../glossary.md#operationkind); it adds no Tensor, descriptor, layout-value, provenance,
  storage, graph, compiler, planning, prepare, runtime, backend, engine, trace, or training
  dependency or behavior.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [trace glossary](../../../../glossary.md#trace))</span>
- The submitted enum and constant Javadocs are complete unchanged. They document the one logical
  input; preservation of logical values, Shape, DataType, and row-major element order; requested
  canonical dense row-major zero-offset geometry; explicit <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [DataType glossary](../../../../glossary.md#data-type--datatype); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element))</span>`Operation` plus
  `NoOperationAttrs.INSTANCE` composition; the distinction from resolved
  `LayoutKind.DENSE_CONTIGUOUS`; diagnostic enum names; and the deferred alias, copy,
  materialization, compiler, lowering, execution, gradient, and backend boundaries.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization))</span>
- `docs/api/tensor-api.md` now lists `ContiguousKind` as current semantic vocabulary and adds a
  focused semantic reference. It explicitly keeps public `Tensor.contiguous()`, result-descriptor
  construction, input-layout inspection, provenance, and materialization behavior planned.
  `docs/glossary.md` adds the reusable contiguous-request term and synchronizes the operation-kind
  status while distinguishing semantic request, resolved layout classification, and later
  materialization.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.layout.ContiguousKindTest` — `BUILD SUCCESSFUL`; the
  XML report records 5 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — `BUILD SUCCESSFUL`; 59 XML suites record 454 tests with zero
  failures, errors, or skips.
- One preliminary combined shell invocation printed the focused XML report and then failed before
  the aggregate test started because the sandbox denied creation of the Gradle distribution
  lock file. Rerunning each required Gradle command separately used the approved command scope and
  produced the successful focused, aggregate, Javadoc, and root results recorded here.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- `./gradlew :modules:model:javadoc` — `BUILD SUCCESSFUL`. Generated
  `ContiguousKind.html` contains the enum and constant contracts, explicit composition,
  semantic/geometric distinction, and diagnostic-name and cross-layer boundaries.
- `./gradlew test` — `BUILD SUCCESSFUL`; all repository test tasks completed without failure.
- `javap -p -c -s` confirmed exactly `CONTIGUOUS`, [compiler](../../../../glossary.md#term-compiler)-generated `$VALUES`, standard enum
  `values`/`valueOf`, the compiler constructor and initializer, and no project field, method, or
  nested type. Focused reflection tests independently verify the same surface, standard identity,
  exact Operation composition, and typed inequality with an equally named test-local kind.
- Markdown target and heading-anchor validation, balanced-fence inspection, terminology review,
  targeted trailing-whitespace scans, package-map/status checks, and <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
  Exact scope is the seven authorized paths: one production file, one focused test, Tensor API,
  glossary, this task, model master plan, and roadmap. Task 0017B remains Draft and no task-0017B
  specification exists. No commit or push was performed.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Operation/attribute foundations remain accurate unchanged because this task only supplies one
  conforming parameterless kind and composes it through their existing contracts.
  `LayoutKind` and `LayoutDescriptor` remain accurate because resolved geometry is neither changed
  nor imported. Current operation families and Tensor expressions remain accurate because no
  public expression, result descriptor, provenance, input inspection, or storage behavior was
  added. `capabilities.md` already inventories contiguous support and its layer separation.
- Compile API remains accurate unchanged because no public Tensor expression, capture input,
  compiler pass, canonicalization, logical materialization requirement, or artifact was added.
  Training API remains accurate because no gradient, autograd, parameter, optimizer, publication,
  or session behavior changed. <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>`ARCHITECTURE.md`, focused architecture explanations, ADRs, and
  architecture tests remain accurate because module ownership, dependency direction, and
  lifecycle boundaries did not change. Backend conformance and integration tests remain unchanged
  because no backend or end-to-end behavior exists. Root/model Java 26 Gradle configuration and
  other modules remain unchanged because no dependency, language level, source-set, build-task,
  or cross-module behavior changed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added the exact public `ContiguousKind` enum with sole [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) identity and no project
  state or behavior beyond inherited <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`OperationKind`](../../../../glossary.md#operationkind) naming.
- Added the focused five-test suite for exact vocabulary and enum shape, typed identity, canonical
  parameterless Operation composition, and absence of project metadata.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [Operation glossary](../../../../glossary.md#operation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Finalized Tensor API and glossary semantics plus synchronized planning status. The production
  Javadocs required no correction after independent review.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented and documented the parameterless contiguous-layout semantic
  request without public Tensor construction or materialization behavior.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: Exactly one production enum, one focused test, Tensor API, glossary,
  this task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused tests passed 5/5; all 454 model tests across 59 suites, generated
  model Javadoc, root tests, javap/reflection/import/source/generated-documentation/manual checks,
  Markdown link/anchor/fence/terminology/whitespace checks, exact scope/status checks, and<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017a/review_model_0017a_docs` completed the independent pass using
  General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now present the contiguous semantic kind as
  current while keeping public `Tensor.contiguous()`, descriptor construction, materialization,
  and cross-layer behavior planned.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Javadoc review: The enum and constant Javadocs are complete unchanged; related operation,
  resolved-layout, Tensor-expression, and cross-layer contracts remain accurate for the reasons
  recorded above.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation))</span>
- Glossary impact: The contiguous-request term and operation-kind status now distinguish semantic
  intent from resolved `DENSE_CONTIGUOUS` geometry and later [materialization](../../../../glossary.md#materialization).
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0017A. Task 0017B remains Draft without a detailed
  specification.

Status: Complete
