# Task 0011: Public [Tensor](../../../../glossary.md#tensor) Skeleton

**Editorial reading key (later context, not a status update).**

This skeleton gives the application a public Tensor object with stable metadata and one replaceable borrowed-storage reference; the public construction path is still deferred at this checkpoint. Tensor is not a graph node, and its descriptor is not the actual data. Replacing or clearing the association changes a reference rather than copying or freeing values. A caller that supplied arena-backed storage retains scope cleanup responsibility; Tensor itself has no general close operation. Synchronizing that reference does not synchronize reads and writes to the underlying memory.

Focused current reading: [Current public Tensor components](../../../../reference/tensor/identity.md#public-tensor-state); [Current replacement and clearing contract](../../../../reference/tensor/storage-association.md). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the smallest public [mutable](../../../../glossary.md#term-mutable) [`Tensor`](../../../../glossary.md#tensor) state that can safely precede `TensorFactory` and
[provenance](../../../../glossary.md#provenance). Add one final `Tensor` class with stable typed [identity](../../../../glossary.md#term-identity-meanings), one [immutable](../../../../glossary.md#term-immutable) logical
[descriptor](../../../../glossary.md#term-descriptor), an immutable optional diagnostic label, and one synchronized optional [host](../../../../glossary.md#term-host)-storage
association. The tensor remains public [API](../../../../glossary.md#term-application-programming-interface-api) state rather than an intermediate-representation ([IR](../../../../glossary.md#term-intermediate-representation-ir))
[node](../../../../glossary.md#node) and contains no [graph](../../../../glossary.md#graph)-local identity, [operation](../../../../glossary.md#operation), [gradient](../../../../glossary.md#term-gradient) object, [publication](../../../../glossary.md#term-publication) policy,
[runtime](../../../../glossary.md#term-runtime) [residency](../../../../glossary.md#residency), or [backend](../../../../glossary.md#backend) state.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public final [`Tensor`](../../../../glossary.md#tensor) class in the existing `model.tensor` package.
- Give each tensor one exact non-null [`TensorId`](../../../../glossary.md#tensorid), one exact non-null [immutable](../../../../glossary.md#term-immutable)
  `TensorDescriptor`, one normalized immutable `Optional<String>` label, and an optional [mutable](../../../../glossary.md#term-mutable)
  `HostTensorStorage` association.
- Keep construction package-private so task 0012 can introduce the public creation surface and
  tensor-ID allocation policy without preserving a temporary public constructor.
- Expose stable [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), and label accessors.
- Expose [host](../../../../glossary.md#term-host)-storage presence with `Optional`, plus synchronized replacement and clearing that
  return the previous association.
- Validate storage data type and, when layout geometry is resolved, the complete referenced
  element span. Do not invent geometry for an unresolved layout.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [referenced element span glossary](../../../../glossary.md#referenced-element-span))</span>
- Accept read-only storage, reject storage that is already dead when attached, and continue to
  report an attached borrowed storage object if its scope dies later.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Preserve ordinary object [identity](../../../../glossary.md#term-reference-identity) for `Tensor` equality and hashing and provide stable
  metadata-only diagnostic text.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add one focused test class covering API shape, constructor validation, labels, compatibility,
  mutable-state transitions, aliasing, lifetime, identity, diagnostics, and exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [aliasing glossary](../../../../glossary.md#term-aliasing); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  implementation roadmap through the required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `TensorFactory`, ID generation, counters, registries, public constructors, builders, creation
  conveniences, allocation, import, fills, random values, or task 0012 implementation
- provenance, producing operations, predecessor tensors, expression-building methods, graph
  traversal, graph capture, <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>[`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [graph](../../../../glossary.md#graph) membership, or task 0013 implementation
- [gradient](../../../../glossary.md#term-gradient) [tensor](../../../../glossary.md#tensor)/state, [gradient accumulation](../../../../glossary.md#term-gradient-accumulation), [mutable](../../../../glossary.md#term-mutable) `requiresGrad`, trainable-parameter flags,
  optimizer state, autograd behavior, backward flags, or training-session behavior<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [autograd glossary](../../../../glossary.md#autograd))</span>
- [publication](../../../../glossary.md#term-publication) intent, publication policy, publication [target](../../../../glossary.md#term-target), publication execution, or storing a
  `PublicationBinding` on `Tensor`
- [data type](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), or `requiresGrad` mutation; descriptor reconstruction; shape/layout
  inference; symbolic-dimension binding; default-layout synthesis; or view creation<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [dimension glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- typed scalar/indexed access, typed bulk access, copy/export, conversion, mutation/version
  tracking, byte order, alignment, or direct memory operations<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- storage allocation, resizing, slicing, copying, conversion, [ownership](../../../../glossary.md#term-ownership-meanings), retaining, closing,
  `AutoCloseable`, [arena](../../../../glossary.md#term-arena) creation, [leases](../../../../glossary.md#term-lease), [pooling](../../../../glossary.md#term-pooling), or [materialization](../../../../glossary.md#materialization) policy
- runtime residency, device buffers, backend storage, physical slots, transfers, prepared state,
  execution, backend support, kernel selection, or service lookup<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- operation-family methods, compiler behavior, planning behavior, dependencies, new packages,
  preview/incubator features, or Gradle changes<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [features glossary](../../../../glossary.md#term-software-feature); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- changes to existing Java contracts or tests, architecture documents, architecture tests,
  capability baseline, API pages other than the Tensor API, or unrelated documentation<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- creating a detailed task-0012 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially the public mutable Tensor
  invariant, model ownership, immutable graph distinction, and runtime/backend exclusions<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [invariant glossary](../../../../glossary.md#term-invariant); [model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially public Tensor, host storage, typed
  identity, and factory capabilities<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Model master plan](../master-plan.md), especially `model.tensor`, `model.storage`, and package
  dependency direction<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [Task 0004](0004-typed-identifiers.md), which defines [`TensorId`](../../../../glossary.md#tensorid) without allocation policy
- [Task 0007](0007-tensor-descriptor-model.md), which defines immutable logical tensor facts and
  explicit resolved/unresolved layout state<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- [Task 0009](0009-compiled-graph-model.md), which keeps `PublicationBinding` standalone and graph
  identity separate from public tensor identity
- [Task 0010](0010-host-storage-abstraction.md), which defines [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) exact-size [host storage](../../../../glossary.md#host-storage)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [Training API](../../../../api/training-api.md), and
  [glossary](../../../../glossary.md), which distinguish current and planned [tensor](../../../../glossary.md#tensor)/training state

## Legacy evidence and rejected coupling

Read-only inspection of `legacy/pre-rewrite` covered the old public `tensor.Tensor`, [mutable](../../../../glossary.md#term-mutable)
`TensorMetadata`, `TensorInternalAccess`, constructor and data-factory tests, mutation guards,
[shape](../../../../glossary.md#shape) [validation](../../../../glossary.md#term-validation), typed-storage tests, and representative [operation](../../../../glossary.md#operation) tests.

Useful capability evidence is limited to a public [identity](../../../../glossary.md#term-identity-meanings)-bearing [tensor](../../../../glossary.md#tensor) object, human-readable
labels, [host](../../../../glossary.md#term-host)-storage presence and replacement, [view](../../../../glossary.md#term-view-meanings)/storage compatibility, shared-storage [aliases](../../../../glossary.md#aliasing-and-arena),
and the practical need to distinguish tensor [metadata](../../../../glossary.md#term-metadata) from [graph](../../../../glossary.md#graph) occurrences. The old tensor also
used ordinary object [identity](../../../../glossary.md#term-reference-identity), and its public expression methods sometimes returned the same
object for semantic identities.

The following legacy design is deliberately rejected:

- the large public constructor and static-factory surface that allocated storage directly;
- a mutable catch-all metadata object containing data type, shape, strides, label, gradient and
  trainable flags, and indexing behavior;<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- mutable shape/[stride](../../../../glossary.md#term-stride) escape hatches and data-type conversion setters;
- storage classes and access switches coupled to typed arrays;
- operation, predecessor, gradient rule, backward marker, and gradient tensor state on the same
  object;
- [compiler](../../../../glossary.md#term-compiler), prepared-execution, [runtime](../../../../glossary.md#term-runtime)-resource, [backend](../../../../glossary.md#backend)-intent, and [residency](../../../../glossary.md#residency) coupling;
- runtime alias/replacement helpers that treated a view as a dense tensor; and
- legacy [scalar](../../../../glossary.md#term-scalar), empty-shape, package, validation, and accidental-behavior conventions.

The new class composes completed contracts instead of copying the old flat package, metadata
[model](../../../../glossary.md#term-model), internal-access [facade](../../../../glossary.md#term-facade), constructor surface, or runtime behavior.

## Architecture constraints

- The production package remains below `io.github.pho001.synaptik.*`.
- [`Tensor`](../../../../glossary.md#tensor) lives in `io.github.pho001.synaptik.model.tensor`. It may compose [`TensorId`](../../../../glossary.md#tensorid),
  `TensorDescriptor`, and `HostTensorStorage` plus [JDK](../../../../glossary.md#term-java-development-kit-jdk) values only.
- Package direction is `model.tensor -> model.storage` in addition to the descriptor's existing
  foundational dependencies. `model.storage` must not depend back on `model.tensor`. `Tensor` must
  not import `model.graph` or `model.operation`, preserving the acyclic package map.
- `Tensor` is public mutable API state because its host-storage association can change. It is not
  a graph value, graph node, operation occurrence, compiled graph, memory slot, or runtime
  residency record.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph value glossary](../../../../glossary.md#graph-value); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [memory slot glossary](../../../../glossary.md#memory-slot); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- `TensorId`, `TensorDescriptor`, and normalized label are fixed for the object's lifetime.
  Data type, shape, layout state, and <span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [Data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` therefore change only by creating another
  descriptor and tensor through later owning APIs; task 0011 adds no setters.
- `TensorDescriptor.requiresGrad()` remains immutable model-level eligibility/request metadata.
  This task adds no gradient object, accumulation, mutable gradient request, trainable role,
  autograd behavior, or publication rule.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication))</span>
- The package-private constructor is the only constructor. Same-package tests exercise it
  directly. External construction intentionally remains unavailable until task 0012 defines the
  public factory and ID allocation policy.
- A label is optional diagnostic [metadata](../../../../glossary.md#term-metadata). A present label is stripped with `String.strip()` and
  must remain non-empty; explicitly present blank text is invalid rather than silently becoming
  absence. Labels are immutable and do not participate in equality or hashing because `Tensor`
  retains object identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>
- [Host](../../../../glossary.md#term-host)-storage absence is exposed only as `Optional.empty()`. Public methods never accept or
  return a null storage sentinel. Internally, the single mutable storage reference may be
  nullable; that private representation does not become an API contract.
- `hostStorage()`, `replaceHostStorage(...)`, and `clearHostStorage()` are synchronized instance
  methods. Synchronization makes reference reads and replacement/clear transitions visible and
  atomic with respect to those three methods. It does not synchronize raw memory, freeze an arena,
  make <span data-glossary-navigation="true"> ([Synchronization glossary](../../../../glossary.md#term-synchronization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [arena glossary](../../../../glossary.md#term-arena))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) access thread-safe, or eliminate the race between a liveness observation
  and later memory access.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [race glossary](../../../../glossary.md#term-race); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Replacement validates the proposed storage fully before changing the association. A failed
  replacement leaves the previous reference unchanged. Successful replacement and clearing
  return the exact previous storage reference inside an `Optional`.
- The same host-storage object may be associated with multiple tensors. `Tensor` borrows the
  storage, does not claim exclusive ownership, and never closes or retains its segment scope.
  Replacing or clearing one tensor's reference does not alter another tensor or the storage.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Every attached storage must have exactly the same [`DataType`](../../../../glossary.md#data-type--datatype) as the [descriptor](../../../../glossary.md#term-descriptor).
- When `descriptor.layout()` is present, `storage.elementCapacity()` must be greater than or equal
  to `layout.referencedElementSpan()`. The span already includes offset and strided/broadcast
  geometry under the completed `LayoutDescriptor` contract.
- A scalar resolved layout requires capacity at least one. A resolved zero-sized layout has span
  zero and therefore accepts zero capacity even when its stored offset is non-zero. Offset,
  strided, and broadcast layouts use their computed span rather than logical element count.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- When layout is unresolved, required physical capacity cannot be proved. This applies both to a
  fully static unresolved descriptor and to a dynamic shape. Task 0011 therefore performs no
  capacity check in either case and does not use known logical element count as invented row-major
  geometry. Task 0012 may resolve a layout before attachment; later materialization/access work
  owns any policy for unresolved geometry.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [materialization glossary](../../../../glossary.md#materialization))</span>
- Read-only host storage is valid because this task performs no writes and immutable inputs or
  constants may use it. An already dead storage is rejected at construction or replacement. A
  successfully attached borrowed storage may die later when its caller-owned scope closes; it
  remains present and is returned exactly, with <span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [constants glossary](../../../../glossary.md#term-constant); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>`storage.isAlive()` reporting the point-in-time
  state.
- A liveness check during attachment is necessarily point-in-time. The caller can close the
  owning arena immediately after validation, so successful attachment never promises future
  access. JDK scope and thread-access checks remain authoritative.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `Tensor` does not override `equals` or [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests); equality and hashing remain ordinary object
  identity. `TensorId` is stable public identity metadata, but task 0012 still owns allocation and
  uniqueness. Constructing two package-local objects with equal IDs does not make the objects
  equal.
- `toString()` is overridden only to report stable ID, descriptor, and normalized label facts. It
  omits host-storage presence, storage implementation text, segment addresses, liveness, raw
  contents, graph state, and runtime facts, so storage mutation or arena closure does not change
  the diagnostic text. The format is not serialization.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- `PublicationBinding` remains a standalone `TensorId`-to-[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) model data-transfer object for
  a later compiler-owned publication plan. <span data-glossary-navigation="true"> ([data-transfer object glossary](../../../../glossary.md#data-transfer-object--dto); [compiler glossary](../../../../glossary.md#term-compiler); [publication plan glossary](../../../../glossary.md#publication-plan))</span>`Tensor` stores neither `ValueId` nor [`NodeId`](../../../../glossary.md#nodeid), and it
  does not record compiled-graph membership because one tensor may participate in multiple graph
  captures.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-structural-capture))</span>
- If implementation requires a public constructor, ID allocator, second production type,
  descriptor mutation, provenance, graph import, operation reference, gradient/trainable state,
  publication state, storage ownership, typed access, runtime/backend state, dependency, or
  architecture change, stop and report the conflict.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`, and the public
  tensor state introduced here.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.storage` — owns borrowed host-visible storage and remains
  independent of <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../../glossary.md#term-host))</span>`model.tensor`.

Packages added or changed:

- No package is added. The existing `model.tensor` package gains one public final class.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — public mutable API state whose stable logical
  identity/description and mutable borrowed host-storage association belong together without
  becoming graph or runtime state.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime))</span>

Test placement:

- `io.github.pho001.synaptik.model.tensor.TensorTest` — mirrors the production package so it can
  exercise the package-private constructor without introducing a factory or widening visibility.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>

## Required contract

### Type and constructor

Implement exactly one public final non-record class:

```java
public final class Tensor {
    Tensor(
            TensorId id,
            TensorDescriptor descriptor,
            Optional<String> label,
            Optional<HostTensorStorage> hostStorage) { ... }
}
```

The constructor has package-private visibility. Do not add a public, protected, or private
constructor, static creator, builder, nested type, or companion helper. Tests in the same package
call this constructor directly. Task 0012 will use it from `TensorFactory` in the same package.

The class has exactly four instance fields:

```text
private final TensorId id
private final TensorDescriptor descriptor
private final Optional<String> label
private HostTensorStorage hostStorage
```

The nullable private storage field represents absence internally. All other fields are non-null;
the first three are final. Do not add a lock, version counter, [publication](../../../../glossary.md#term-publication) flag, [provenance](../../../../glossary.md#provenance) field,
cached [layout](../../../../glossary.md#layout) facts, [graph](../../../../glossary.md#graph) ID, or derived [mutable](../../../../glossary.md#term-mutable) state. Synchronize on the [tensor](../../../../glossary.md#tensor) instance through
method modifiers rather than a second lock object.

The constructor validates in this deterministic order:

1. Reject null `id` with `NullPointerException` and exact message `id`.
2. Reject null [`descriptor`](../../../../glossary.md#term-descriptor) with `NullPointerException` and exact message `descriptor`.
3. Reject null `label` optional with `NullPointerException` and exact message `label`.
4. Reject null `hostStorage` optional with `NullPointerException` and exact message `hostStorage`.
5. Strip a present label with `String.strip()`. If the result is empty, reject it with
   `IllegalArgumentException` and exact message `label must not be blank`. Store
   `Optional.of(normalized)` when present and `Optional.empty()` when absent. Optional-container
   identity and original unstripped string identity are not part of the contract.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
6. If host storage is present, validate it in the storage-validation order below and store its
   exact reference only after every check passes. Otherwise store the private null absence value.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [validation glossary](../../../../glossary.md#term-validation))</span>

`Optional` cannot contain a null [element](../../../../glossary.md#term-element) through its public construction [API](../../../../glossary.md#term-application-programming-interface-api); no additional null
sentinel is defined.

### Public [API](../../../../glossary.md#term-application-programming-interface-api)

Expose exactly these declared public methods in addition to the inherited `Object` API:

```java
public TensorId id()
public TensorDescriptor descriptor()
public Optional<String> label()
public synchronized Optional<HostTensorStorage> hostStorage()
public synchronized Optional<HostTensorStorage> replaceHostStorage(
        HostTensorStorage hostStorage)
public synchronized Optional<HostTensorStorage> clearHostStorage()
@Override public String toString()
```

The accessors have these contracts:

- `id()` returns the exact non-null [immutable](../../../../glossary.md#term-immutable) [`TensorId`](../../../../glossary.md#tensorid) reference supplied at construction.
- `descriptor()` returns the exact non-null immutable `TensorDescriptor` reference supplied at
  construction.
- `label()` returns the non-null normalized optional label value. Callers use presence/value
  semantics and do not rely on `Optional` container [identity](../../../../glossary.md#term-identity-meanings).
- `hostStorage()` returns empty when no storage is attached and otherwise an optional containing
  the exact currently attached identity-bearing storage object. It does not hide a storage whose
  scope died after attachment.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- `replaceHostStorage(storage)` rejects a null [argument](../../../../glossary.md#term-argument) with `NullPointerException` and exact
  message `hostStorage`, validates the proposed storage, atomically replaces the reference, and
  returns the previous association by exact reference or empty if there was none.
- `clearHostStorage()` atomically removes any association and returns the previous exact reference
  or empty if the tensor was already storage-free. Clearing is valid even when the attached
  storage is now dead.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>

Do not expose [aliases](../../../../glossary.md#aliasing-and-arena) such as `getId`, `dataType`, [`shape`](../../../../glossary.md#shape), [`layout`](../../../../glossary.md#layout), `requiresGrad`, `hasStorage`,
`attach`, `detach`, `setStorage`, or `isAlive`. Callers obtain immutable logical facts from
`descriptor()` and [liveness](../../../../glossary.md#term-liveness)/read-only facts from the returned `HostTensorStorage`.

### [Host](../../../../glossary.md#term-host)-storage [validation](../../../../glossary.md#term-validation)

Use one private validation path for constructor attachment and replacement. Apply these checks in
order before mutating the association:

1. Compare `hostStorage.dataType()` to `descriptor.dataType()` by enum identity. On mismatch,
   throw <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`IllegalArgumentException` with exact message
   `hostStorage data type must match descriptor data type: expected=<expected>, actual=<actual>`.
2. If `descriptor.layout()` is present, compare `hostStorage.elementCapacity()` to
   `layout.referencedElementSpan()`. When capacity is smaller, throw
   `IllegalArgumentException` with exact message
   `hostStorage element capacity is smaller than resolved layout span: required=<required>, actual=<actual>`.
3. If `hostStorage.isAlive()` is false, throw `IllegalStateException` with exact message
   `hostStorage must be alive when attached`.

`<expected>` and `<actual>` data types use enum diagnostic names. Capacity and span use decimal
`long` text. Data-type mismatch wins over capacity or [liveness](../../../../glossary.md#term-liveness); resolved capacity mismatch wins
over liveness. Unresolved [layout](../../../../glossary.md#layout) skips only step 2, not data-type or liveness checks.

Storage capacity may exceed the [referenced span](../../../../glossary.md#referenced-element-span). Exact equality is not required because [views](../../../../glossary.md#term-view-meanings) may
refer to a subregion of a larger shared [host storage](../../../../glossary.md#host-storage). This task validates addressable range, not
exclusive [ownership](../../../../glossary.md#term-ownership-meanings) or logical [element](../../../../glossary.md#term-element)-count equality.

### Mutability, [identity](../../../../glossary.md#term-identity-meanings), and diagnostics

The three storage methods must [carry](../../../../glossary.md#term-carry) the Java `synchronized` modifier. No other public method
needs [synchronization](../../../../glossary.md#term-synchronization) because the other state is final. Returned `Optional` objects are snapshots
of the association; a later replacement does not mutate an already returned optional, though the
contained storage object and its [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) segment remain identity-bearing and externally [mutable](../../../../glossary.md#term-mutable).

Do not override `equals` or [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests). Do not implement `Comparable`, `Cloneable`,
`AutoCloseable`, or [serialization](../../../../glossary.md#term-serialization) interfaces. Do not use [`TensorId`](../../../../glossary.md#tensorid) as structural Java equality.

Override `toString()` with a concise form containing the type name and stable `id`, [`descriptor`](../../../../glossary.md#term-descriptor),
and `label` facts. Do not include storage presence, the storage object's inherited identity text,
segment/address/content, [liveness](../../../../glossary.md#term-liveness), [operation](../../../../glossary.md#operation)/[provenance](../../../../glossary.md#provenance), [graph](../../../../glossary.md#graph) IDs, or [runtime](../../../../glossary.md#term-runtime) facts. Tests assert
the required facts and stability across storage transitions without treating the complete text as
a [wire format](../../../../glossary.md#term-wire-format).

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| [Descriptor](../../../../glossary.md#term-descriptor) and matching live storage, no label | Valid |
| Present label `"  weights  "` | Valid and stored as `"weights"` |
| Present blank label | Invalid; blank is not converted to absence |
| Matching read-only storage | Valid; task 0011 performs no writes |
| Matching resolved [contiguous storage](../../../../glossary.md#contiguous-request) with capacity equal to span | Valid |
| Matching resolved [view](../../../../glossary.md#term-view-meanings) storage with capacity greater than span | Valid |
| Matching resolved storage with capacity below offset/strided/broadcast span | Invalid |
| Resolved [scalar](../../../../glossary.md#term-scalar) [layout](../../../../glossary.md#layout) with zero capacity | Invalid; scalar span is one |
| Resolved zero-sized layout with zero capacity | Valid; completed layout span is zero |
| Fully static [shape](../../../../glossary.md#shape) with unresolved layout and zero capacity | Valid but geometry remains unresolved |
| Dynamic shape with unresolved layout and zero capacity | Valid but required capacity is unknown |
| Storage with a different [data type](../../../../glossary.md#data-type--datatype) | Invalid even when capacity is sufficient |
| Storage whose [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [scope](../../../../glossary.md#term-scope) is already dead | Invalid at attachment |
| Attached storage whose borrowed scope later dies | Remains present and reports not alive |
| Replace with invalid storage while another is attached | Invalid and previous association remains |
| Clear a live or dead attached storage | Valid and returns the exact previous reference |
| Attach one storage object to two [tensors](../../../../glossary.md#tensor) | Valid shared [alias](../../../../glossary.md#aliasing-and-arena); neither tensor owns it |
| Two tensor objects use equal [`TensorId`](../../../../glossary.md#tensorid) values | Objects remain unequal by Java [identity](../../../../glossary.md#term-identity-meanings) |

The unresolved-layout cases are accepted because capacity sufficiency cannot be established, not
because zero capacity is considered sufficient. They remain a known limitation until a later
owning contract resolves geometry.

## Affected files

Expected new production file:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected new test file:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless an inconsistency requires stopping:

- `docs/api/training-api.md` — no change is expected because task 0011 adds no gradient object,
  trainable role, optimizer behavior, training session, or callable training API.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [training session glossary](../../../../glossary.md#training-session--trainingsession); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorId.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/storage/HostTensorStorage.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorage.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/PublicationBinding.java`

Focused architecture documentation and the capability baseline must remain unchanged. They
already authorize public [mutable](../../../../glossary.md#term-mutable) [Tensor](../../../../glossary.md#tensor) state and reserve broader capabilities for later tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- one new production Java file;
- one new focused test Java file; and
- the five documentation and [planning](../../../../glossary.md#term-planning) files listed as expected updates above.

Do not modify any existing Java source or test, [Gradle](../../../../glossary.md#term-gradle) file, `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture document, [architecture test](../../../../glossary.md#term-architecture-test), `capabilities.md`, Training [API](../../../../glossary.md#term-application-programming-interface-api), another [module](../../../../glossary.md#term-module), or
unrelated documentation. Do not create task 0012. If another file, production type, public method,
[dependency](../../../../glossary.md#term-dependency), or architecture clarification is required, stop and report the issue instead of
expanding this task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Every declared type, constructor, and method, including private validation/normalization helpers
  if used, must have meaningful detailed Javadoc. <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization))</span>`{@inheritDoc}` alone is insufficient.
- Type Javadoc must define public mutable Tensor state, identify host-storage association as its
  sole mutation in this task, and distinguish the class from <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host))</span>`TensorDescriptor`, graph values and
  nodes, operations/provenance, publication bindings/plans, device buffers, runtime residency,
  and prepared execution.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [bindings glossary](../../../../glossary.md#term-binding); [device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Type Javadoc must explain package-private construction and task-0012 factory ownership without
  claiming that the factory is already implemented.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Constructor Javadoc must document all four parameters, exact immutable-reference retention,
  optional value semantics, label normalization, storage borrowing, validation order, every
  exception type/message condition, thread/lifetime limits, and resolved/unresolved capacity
  behavior.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable); [borrowing glossary](../../../../glossary.md#term-borrowing); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `id()`, `descriptor()`, and `label()` must document non-nullness, exact-reference or value
  ownership, immutability, and complete `@return` semantics.
- Every storage method must document synchronization, snapshot semantics, exact-reference
  retention/return, absence/presence, atomic failed replacement, borrowed ownership, read-only
  acceptance, late death, liveness races, underlying-memory thread safety, and complete <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [snapshot glossary](../../../../glossary.md#term-snapshot); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [liveness glossary](../../../../glossary.md#term-liveness))</span>`@return`
  and `@throws` behavior where applicable.
- Storage-validation Javadoc must explain data-type equality, resolved span capacity, unresolved
  geometry, scalar/zero-sized/view cases, deterministic ordering, and point-in-time liveness.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Equality/hash Javadoc belongs in the type-level discussion because those methods are inherited.
  It must state ordinary object identity and distinguish that from the stable <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>[`TensorId`](../../../../glossary.md#tensorid) value.
- `toString()` Javadoc must document stable metadata-only diagnostics, omitted storage/resource
  facts, non-null return, and non-serialization status.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Javadoc must not promise ID uniqueness, future liveness, storage ownership, raw-memory safety,
  graph membership, compiler behavior, gradient/trainable/publication state, factory availability,
  typed access, runtime residency, or backend support.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- The documentation-focused pass must review the existing `TensorId`, `TensorDescriptor`,
  `HostTensorStorage`, `MemorySegmentStorage`, and `PublicationBinding` Javadocs and record why
  they remain accurate without edits or stop if a required correction falls outside scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public final non-record [`Tensor`](../../../../glossary.md#tensor) class and one focused `TensorTest` are added; no
  second production concept, nested public type, builder, factory, or helper file appears.
- `Tensor` has exactly the four required fields, with only host storage mutable, and exactly one
  package-private constructor with the required parameter order and types.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [mutable glossary](../../../../glossary.md#term-mutable); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- The declared public API is exactly the seven methods specified by this task. No public
  constructor, static creator, metadata convenience alias, broad setter, operation, graph,
  training, publication, typed-access, execution, or storage-ownership API is introduced.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [metadata glossary](../../../../glossary.md#term-metadata); [alias glossary](../../../../glossary.md#aliasing-and-arena); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Constructor nulls and blank labels fail in the specified order with exact exception types and
  messages. Present labels are stripped once and remain immutable; empty means absent.
- `id()` and `descriptor()` return the exact supplied [immutable](../../../../glossary.md#term-immutable) references. `label()` exposes the
  normalized optional value without promising optional-container or original-string identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Initial absent and present storage states work. Public storage absence is always
  `Optional.empty()` and no public null storage sentinel exists.
- Constructor attachment and replacement apply exact data-type, resolved-span, and liveness
  validation in the specified order and messages. Failed replacement preserves the prior exact
  association.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [validation glossary](../../../../glossary.md#term-validation))</span>
- Resolved contiguous, offset, strided, zero-stride broadcast, scalar, and zero-sized layouts have
  the specified capacity results. Capacity larger than span is accepted; capacity below span is
  rejected.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [stride glossary](../../../../glossary.md#term-stride); [scalar glossary](../../../../glossary.md#term-scalar); [layouts glossary](../../../../glossary.md#layout))</span>
- Fully static unresolved and dynamic unresolved descriptors skip capacity validation without
  synthesizing contiguous geometry or using logical element count as physical span.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Read-only storage attaches successfully. Storage already dead at attachment is rejected. An
  attached storage that dies later remains present, returns by exact identity, reports false from
  its own `isAlive()`, and can be cleared.
- The same storage can be observed by two tensors without ownership transfer. Replacement or
  clearing on one tensor does not alter the other's association, and raw alias effects remain a
  property of the shared storage/segment rather than Tensor copy behavior.
- All three storage methods are declared `synchronized`. Tests and Javadocs do not imply that this
  synchronizes segment contents, prevents arena closure, or makes a confined segment accessible
  from another thread.<span data-glossary-navigation="true"> ([arena glossary](../../../../glossary.md#term-arena))</span>
- `Tensor` inherits `Object.equals` and `Object.hashCode`. Two tensors remain unequal even with
  the same [`TensorId`](../../../../glossary.md#tensorid), [descriptor](../../../../glossary.md#term-descriptor), label, and storage; each tensor equals only itself.
- Diagnostic text names Tensor, ID, descriptor, and normalized label, excludes host-storage and
  graph/runtime/resource details, and remains unchanged across attach, replace, clear, read-only,
  and late-death transitions.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Reflection and `javap` confirm final/non-record shape, exact field and constructor visibility,
  exact public methods, synchronized modifiers, no interface implementation, inherited equality
  and hashing, and no hidden mutable/version/graph/provenance state.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Production imports are limited to `HostTensorStorage`, `Objects`, and `Optional`; `TensorId` and
  `TensorDescriptor` are same-package types. No graph, operation, compiler, planning, runtime,
  prepare, backend, config, trace, training, FFM, or concrete storage implementation is imported.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace); [FFM glossary](../../../../glossary.md#term-ffm))</span>
- No existing Java contract/test, package, dependency, Gradle configuration, preview/incubator
  setting, or architecture rule changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Complete source Javadoc satisfies every requirement. Generated model Javadoc includes the public
  class and public methods; source inspection confirms the package-private constructor contract,
  which the default public/protected Javadoc output does not render.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [model glossary](../../../../glossary.md#term-model))</span>
- A separate documentation-focused agent or thread with clean context independently inspects the
  final source, tests, generated Javadoc, test evidence, and diff; finalizes Tensor Javadoc, Tensor
  API, glossary, task evidence/status, master-plan status, and roadmap status in the same overall
  change.
- The Tensor API moves the public Tensor skeleton from wholly planned to current, explains the
  temporary package-private construction boundary, stable descriptor/identity/label, optional
  mutable host storage, compatibility and lifetime rules, and keeps factory, provenance,
  operations, typed access, gradient/trainable/publication behavior, compiler, runtime, and
  backend work planned.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The glossary marks the Tensor skeleton implemented and updates the Tensor/TensorId/host-storage
  distinctions without inventing a new term. It keeps graph value, publication binding,
  provenance, residency, and training meanings separate.<span data-glossary-navigation="true"> ([graph value glossary](../../../../glossary.md#graph-value); [binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency))</span>
- The Training API is reviewed with a reasoned no-change conclusion because the task adds no
  trainable, gradient, optimizer, autograd, or training-session behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [autograd glossary](../../../../glossary.md#autograd))</span>
- Focused architecture documents, `ARCHITECTURE.md`, architecture tests, and `capabilities.md`
  remain unchanged because this task implements an already authorized subset without changing
  module ownership or dependency rules.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Task, master-plan row/current status/decisions/notes, and roadmap frontier/table have matching
  final status. After task 0011 completes, task 0012 may become the next `Draft` frontier, but no
  detailed task-0012 specification is created.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) after implementation and again after the separate documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only one new production file, one new test file, and the five allowed
  documentation/planning files;<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- reflection reports a public final non-record class, exactly four fields with the specified
  final/mutable modifiers, one package-private constructor with the exact signature, no
  implemented interfaces, and exactly the declared public API;<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [signature glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- reflection method modifiers and `javap -p -c` show synchronization on all three storage methods,
  constructor validation order, label stripping, resolved-span-only capacity validation,
  validation before assignment, exact-reference returns, nullable storage only as private state,
  inherited equality/hashing, and stable metadata-only text;<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- focused tests cover exact null/blank/type/capacity/[liveness](../../../../glossary.md#term-liveness) failure precedence and messages;
- focused tests cover absent/present state, initial attachment, successful replacement, failed
  atomic replacement, repeated clear, exact previous-reference returns, and optional snapshots;<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- layout tests cover contiguous, offset, strided, broadcast, scalar, zero-sized, fully static
  unresolved, and dynamic unresolved descriptors without changing existing layout/descriptor
  contracts;<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [contiguous glossary](../../../../glossary.md#term-contiguous); [scalar glossary](../../../../glossary.md#term-scalar); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- lifetime tests cover read-only acceptance, dead-at-attachment rejection, caller-controlled
  closure after attachment, continued exact dead-reference reporting, JDK access failure, and
  clearing dead storage without Tensor ownership/close behavior;<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- alias tests attach the same storage to two tensors and prove independent association mutation
  plus shared raw-segment observation without adding production typed access;<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- identity tests confirm exact immutable references, inherited object equality/hashing, equal
  TensorId values across unequal tensors, immutable label/descriptor state, and diagnostic text
  stability/exclusions;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [TensorId glossary](../../../../glossary.md#tensorid))</span>
- production source contains no [`Arena`](../../../../glossary.md#term-arena), [`MemorySegment`](../../../../glossary.md#term-memorysegment), concrete `MemorySegmentStorage`, graph,
  operation, compiler, planning, runtime, prepare, backend, gradient, trainable, publication,
  provenance, typed access, allocation, copy, close, version, service lookup, or device state;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [provenance glossary](../../../../glossary.md#provenance); [lookup glossary](../../../../glossary.md#term-lookup); [device glossary](../../../../glossary.md#term-device))</span>
- package direction is `model.tensor -> model.storage`, while `model.storage` remains independent
  of tensor and no `model.tensor -> model.graph` edge is added;
- generated Javadoc documents the public state and method results, failures, ownership,
  synchronization, lifetime, compatibility, identity, diagnostics, and exclusions; source review
  confirms the complete package-private constructor documentation;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- the documentation-focused context follows
  `docs/developer-guide/documentation-rules.md`, applies General plus API/Javadoc style to
  Java/API work and Planning style to planning updates, inspects actual source/tests/diff, and
  records its identity, selected profiles, commands, outcomes, limitations, existing-Javadoc
  review, Training API no-change rationale, and glossary impact;
- Tensor API and glossary current/planned language agrees with implementation without claiming
  factory, provenance, operation, training, compiler, runtime, or backend behavior;
- all local Markdown links and anchors in the five changed documentation/planning files resolve,
  fences are balanced, terminology agrees with the glossary, and changed files have no trailing
  whitespace;
- task 0011 status matches the master plan and roadmap, task 0012 remains only a `Draft` row, and
  no task-0012 specification exists; and
- no [commit](../../../../glossary.md#term-commit) or push occurs.

## Dependencies

- Task 0004 is complete and provides [`TensorId`](../../../../glossary.md#tensorid) without generation policy.
- Task 0007 is complete and provides the exact [immutable](../../../../glossary.md#term-immutable) `TensorDescriptor` logical contract.
- Task 0009 is complete and provides standalone `PublicationBinding` while preserving distinct
  tensor and graph-value identity domains.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Task 0010 is complete and provides [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) `HostTensorStorage` plus [liveness](../../../../glossary.md#term-liveness)/read-only facts.
- The repository Java toolchain and release are 26; this task uses no new Java feature or build
  option.<span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release); [feature glossary](../../../../glossary.md#term-software-feature))</span>

## Follow-up tasks

- Task 0012 remains the next ordered task. It will define the public `TensorFactory`, tensor-ID
  allocation/uniqueness policy, descriptor/layout construction choices, host-storage allocation or
  import, and logical-count/data validation.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [host glossary](../../../../glossary.md#term-host); [validation glossary](../../../../glossary.md#term-validation))</span>
- Task 0013 will define minimal provenance and operation/input relationships without putting
  graph-local IDs or compiler membership on <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler))</span>[`Tensor`](../../../../glossary.md#tensor).
- Tasks 0014 and later will add [operation](../../../../glossary.md#operation)-family expression methods only after [provenance](../../../../glossary.md#provenance) exists.
- Later focused tasks own typed scalar/bulk access, copying, conversions, mutation/version
  semantics, gradients/publication, compiler capture, runtime residency, and backend storage.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [capture glossary](../../../../glossary.md#term-capture-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend glossary](../../../../glossary.md#backend))</span>

Do not create a detailed task-0012 specification as part of task 0011.

## Architecture impact

Expected impact: None.

The architecture already requires public [mutable](../../../../glossary.md#term-mutable) [`Tensor`](../../../../glossary.md#tensor) state, assigns [host storage](../../../../glossary.md#host-storage) and tensor
contracts to `modules/model`, separates Tensor from [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) state, and forbids [runtime](../../../../glossary.md#term-runtime)
[device](../../../../glossary.md#term-device) [residency](../../../../glossary.md#residency) in the [model](../../../../glossary.md#term-model). This task implements only stable logical [metadata](../../../../glossary.md#term-metadata) plus a mutable
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host](../../../../glossary.md#term-host)-storage reference through the planned `model.tensor -> model.storage` package
direction. It adds no [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), graph coupling, runtime/[backend](../../../../glossary.md#backend) state, training behavior,
or architecture decision. Therefore `ARCHITECTURE.md`, focused architecture documents, ADRs, and
architecture tests require no update. If implementation reveals otherwise, stop and report the
conflicting rule and required decision before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/architecture/current-architecture-plan.md
- docs/architecture/overview.md
- docs/architecture/lifecycle.md
- docs/architecture/module-boundaries.md
- docs/architecture/dependency-rules.md
- docs/architecture/training-graph.md
- docs/developer-guide/documentation-rules.md
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0004-typed-identifiers.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0009-compiled-graph-model.md
- docs/planning/modules/model/tasks/0010-host-storage-abstraction.md
- docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md
- docs/api/tensor-api.md
- docs/api/training-api.md
- docs/glossary.md
- current production/tests for TensorId, TensorDescriptor, PublicationBinding,
  HostTensorStorage, MemorySegmentStorage, Shape, and LayoutDescriptor
- root/model Gradle configuration only to confirm Java 26

Implement task 0011 exactly as specified. Create only Tensor.java and TensorTest.java for code and
tests. Tensor must be one public final non-record class with exactly TensorId, TensorDescriptor,
normalized Optional<String> label, and a nullable private HostTensorStorage reference as its four
fields. Use exactly one package-private constructor with the task's signature, deterministic
validation, and exact messages. Do not add a public constructor, factory, builder, ID allocator,
second production type, or helper file.

Keep ID, descriptor, and label immutable. Make only host-storage association mutable. Expose
exactly id(), descriptor(), label(), synchronized hostStorage(), synchronized
replaceHostStorage(HostTensorStorage), synchronized clearHostStorage(), and metadata-only
toString(). Use Optional at every public storage boundary and never accept or return null. Validate
replacement before assignment and return the exact previous reference. Require matching data type;
for resolved layout require capacity at least referencedElementSpan; for static or dynamic
unresolved layout perform no capacity check and do not invent row-major geometry. Accept
read-only storage, reject already-dead storage, and keep a storage association observable if its
borrowed scope dies later. Do not own, retain, close, allocate, copy, convert, or type-access
storage.

Preserve ordinary object equality and hashing. TensorId is stable metadata but task 0012 owns ID
allocation and uniqueness. Diagnostic text must include only stable ID, descriptor, and normalized
label facts and must not expose storage identity/address/content/presence/liveness or graph/runtime
state.

Do not add gradient tensors/state, mutable requiresGrad, trainable/optimizer/autograd behavior,
publication intent/policy, provenance, operation/input references, NodeId, ValueId, graph
membership, expression operations, dtype/shape/layout mutation, typed scalar/bulk access,
materialization, runtime residency, device/backend storage, compiler/planning/prepare/execution,
service lookup, dependencies, preview/incubator features, or Gradle changes. Do not modify existing
Java/tests, AGENTS.md, ARCHITECTURE.md, focused architecture docs, architecture tests,
capabilities.md, Training API, another module, or unrelated docs. Do not create task 0012. Stop and
report if any requirement exceeds the affected-file or maximum-scope list or if architecture
uncertainty appears.

Add every Javadoc contract required by the task. Run the focused Tensor test, all model tests,
model Javadoc, full repository tests, git diff checks, and every reflection, javap, import,
compatibility, alias, lifetime, identity, documentation, scope, and status check in the task.

After code implementation and initial validation, hand the actual diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. Keep task
0011 incomplete until that pass finishes. The handoff must include this task specification, the
implementation/test diff, stable and mutable Tensor behavior, package-private construction,
storage compatibility/lifetime/synchronization boundaries, identity/diagnostic behavior,
architecture constraints, expected Tensor API and glossary updates, Training API no-change
expectation, existing-Javadoc review list, and every validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md,
docs/developer-guide/documentation-rules.md, the documentation profile index, General style,
API/Javadoc style, Planning style, Example format if an example changes, this task, final
source/tests, generated Javadoc, Tensor API, Training API, glossary, model master plan, roadmap,
and the existing TensorId/TensorDescriptor/HostTensorStorage/MemorySegmentStorage/
PublicationBinding contracts. It must inspect the actual diff and test evidence rather than rely
on the handoff summary. It must finalize Tensor Javadoc, move only the implemented Tensor skeleton
from planned to current in the Tensor API and glossary, review links/anchors/fences/whitespace and
terminology, record reasoned no-change conclusions for Training API, existing component Javadocs,
focused architecture, and capabilities.md, and synchronize only the allowed planning files.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, exact validation evidence including the documentation
agent identity and results, implementation notes, and the canonical completion summary. Do not
mark task 0011 Complete until implementation, tests, Javadoc, the documentation pass, scope review,
and status synchronization all pass. Task 0012 then remains Draft. Do not create a task-0012
specification and do not commit or push.
```

## Local decisions

- Construction is package-private rather than public. This lets same-package tests and the next
  factory task use the class while avoiding a temporary user-facing constructor that requires
  callers to invent Tensor IDs and raw storage policy.
- `TensorDescriptor` is retained as one immutable value rather than duplicating data type, shape,
  layout, or <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` fields and [validation](../../../../glossary.md#term-validation) on [Tensor](../../../../glossary.md#tensor).
- Label is present because it is selected capability evidence and useful stable diagnostic state.
  It is optional, stripped, non-blank when present, and immutable; legacy label setters are not
  retained.
- The only mutation is one host-storage reference. Synchronized access provides a small explicit
  reference-state contract without a lock field, atomics, versioning, or a false promise about raw
  memory thread safety.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- Public absence uses `Optional`; a private nullable field keeps the mutable implementation small
  without exposing null as a sentinel.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Replacement returns the previous optional association. This makes attach, replace, and clear
  observable and composable while preserving exact identity-bearing storage references and
  avoiding a broad nullable setter.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Resolved layout span is the only current proof of required physical capacity. Static shape
  element count is not substituted for unresolved layout because offset, strides, and alias
  geometry are unknown.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Read-only storage is valid; dead-at-attachment storage is not. Later death remains visible
  because Tensor borrows rather than owns storage and cannot make a point-in-time liveness check a
  lifetime guarantee.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Tensor equality and hashing remain object identity. TensorId is exposed as stable domain
  identity, but ID allocation and uniqueness remain task 0012 responsibilities.<span data-glossary-navigation="true"> ([TensorId glossary](../../../../glossary.md#tensorid))</span>
- Diagnostic text deliberately excludes all mutable/resource-bearing storage facts so replacing,
  clearing, or invalidating borrowed storage does not leak addresses or destabilize diagnostics.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

## Known limitations

- No external caller can construct a [Tensor](../../../../glossary.md#tensor) until task 0012 supplies the public factory.
- Task 0011 cannot enforce TensorId uniqueness; its package-private constructor accepts any valid
  identifier supplied by same-package code.<span data-glossary-navigation="true"> ([TensorId glossary](../../../../glossary.md#tensorid))</span>
- An unresolved layout permits any matching, live storage capacity, including zero. This does not
  prove compatibility; later owning work must resolve geometry before access or materialization.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- Attachment liveness is only a snapshot. Caller-controlled arena closure can make storage dead
  immediately after a successful constructor or replacement call.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [snapshot glossary](../../../../glossary.md#term-snapshot); [arena glossary](../../../../glossary.md#term-arena))</span>
- Synchronization covers only the Tensor's storage reference. It does not coordinate segment
  reads/writes, external mutation, scope closure, or access to a confined segment.<span data-glossary-navigation="true"> ([Synchronization glossary](../../../../glossary.md#term-synchronization); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scope glossary](../../../../glossary.md#term-scope))</span>
- The class provides no typed access, mutation version, storage copy, allocation, ownership, close,
  provenance, operations, gradients, trainable role, publication behavior, graph capture, runtime
  residency, or backend support.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance); [gradients glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean planning context read the complete agent instructions, architecture contract and focused
  architecture documents, documentation workflow and Planning profile, planning guide, roadmap,
  capability baseline, model master plan, tasks 0004/0007/0009/0010, Tensor API, Training API,
  glossary, Java 26 Gradle configuration, and all requested current production/test contracts
  before defining this task.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [architecture contract glossary](../../../../glossary.md#architecture-contract); [model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Read-only legacy inspection used `git ls-tree`, `git grep`, and
  `git show legacy/pre-rewrite:<path>` for [`Tensor`](../../../../glossary.md#tensor), `TensorMetadata`, `TensorInternalAccess`,
  constructor/data-factory/mutation/shape/storage tests, and representative API tests. The branch
  was not checked out or modified, and no legacy source, flat package, constructor surface,
  mutable metadata, runtime/compiler coupling, or accidental behavior was copied.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [mutable glossary](../../../../glossary.md#term-mutable); [metadata glossary](../../../../glossary.md#term-metadata); [runtime glossary](../../../../glossary.md#term-runtime); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Root and model Gradle configuration review confirmed Java toolchain and release 26 with common
  configuration in the root and no model-specific override. No build file was modified.<span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- `git status --short --untracked-files=all`, `git diff --name-only`, and exact path review
  confirmed exactly three planning paths changed: this new task, the model master plan, and the
  roadmap. No Java, test, Gradle, agent instruction, architecture, API, glossary,
  capability-baseline, other-module, or unrelated documentation file changed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- A targeted Ruby path-and-heading check resolved all 63 local Markdown links and anchors in the
  three changed planning files.
- Fence inspection reported balanced backtick fences: ten markers in this task and two each in the
  model master plan and roadmap. `rg -n '[[:blank:]]+$'` found no trailing whitespace in any
  changed file.
- `git diff --check` passed for tracked changes. `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0011-public-tensor-skeleton.md` emitted no whitespace
  diagnostic; exit status `1` was expected because the complete new file differs from
  `/dev/null`.
- Status review confirmed task 0011 is linked and [`Ready`](../../../../glossary.md#term-ready) in this task, the master-plan row/current
  status/decisions/notes, and the roadmap frontier/table. Task 0012 remains `Draft`, task order and
  dependencies are unchanged, and no task-0012 specification exists.
- Gradle tests and Javadoc are not run for this planning-only change because no Java, test, build,
  API, or glossary file changes. The implementation task requires focused and aggregate tests and
  Javadoc both before and after its separate documentation-focused pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [run glossary](../../../../glossary.md#term-run))</span>
- Implementation context `/root/implement_model_0011` added the final `Tensor` class and focused
  `TensorTest`. Documentation-focused context
  `/root/implement_model_0011/review_model_0011_docs` independently inspected the final source,
  final 13-test class including replacement from an absent association, generated Javadoc, XML
  reports, bytecode, imports, dependency direction, status, and complete diff before finalizing
  documentation. The selected profiles were General plus API/Javadoc for Java and API reference
  work, and Planning for this task, the model master plan, and the roadmap.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- The documentation-focused pass read and compared `Tensor`, [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`,
  `HostTensorStorage`, `MemorySegmentStorage`, and `PublicationBinding` source Javadocs. The five
  existing contracts remain accurate without edits: Tensor composes them without changing their
  identity, immutable-value, raw-storage, borrowed-lifetime, or publication-binding semantics.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../../../../glossary.md#term-lifetime); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding))</span>
- `docs/api/training-api.md` remains unchanged because this task adds no gradient object,
  trainable role, optimizer, autograd, training-session, or callable training behavior.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [autograd glossary](../../../../glossary.md#autograd))</span>
  `ARCHITECTURE.md`, the focused overview/lifecycle/module-boundary/dependency/training documents,
  architecture tests, and <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>`capabilities.md` also remain unchanged because the implementation
  realizes an already authorized model-owned Tensor subset without changing ownership,
  dependency rules, architecture intent, or the broader capability baseline.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed with `BUILD SUCCESSFUL`; the generated
  XML reports 13 tests, zero failures, zero errors, and zero skipped tests. The reflection test
  confirms the final non-record type, four exact fields, single package-private constructor,
  exact seven-method public API, no interfaces, synchronized storage methods, and inherited
  equality and hashing.
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL`. Aggregating the 22 XML reports
  found 154 tests, zero failures, zero errors, and zero skipped tests.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL` and regenerated
  `modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/tensor/Tensor.html`. Rendered
  output review confirmed public purpose, package-private-factory boundary, identity, borrowed
  ownership, read-only and late-death behavior, synchronization limits, compatibility failures,
  exact-reference results, and metadata-only diagnostics. Source review confirmed the complete
  package-private constructor and private-helper Javadocs omitted by default public Javadoc.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- `./gradlew test` passed with `BUILD SUCCESSFUL`; Gradle reported 36 actionable tasks, one
  executed and 35 up-to-date. No test task failed.
- `javap -classpath modules/model/build/classes/java/main -p -c
  io.github.pho001.synaptik.model.tensor.Tensor` confirmed the exact fields and constructor, three
  synchronized methods, parameter/label validation order, storage validation before assignment,
  resolved-layout-only span comparison, exact previous-reference snapshots, private nullable
  storage representation, and metadata-only <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [layout glossary](../../../../glossary.md#layout))</span>`toString()` [bytecode](../../../../glossary.md#term-bytecode).
- Source/import and package-direction checks found exactly `HostTensorStorage`, `Objects`, and
  `Optional` imports in `Tensor`; no concrete storage, Foreign Function and Memory API, graph,
  operation, compiler, planning, runtime, prepare, backend, config, trace, or training import; no<span data-glossary-navigation="true"> ([Foreign Function and Memory API glossary](../../../../glossary.md#application-binary-interface-and-native-bridge); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace))</span>
  `model.storage -> model.tensor` import; and no `model.tensor -> model.graph` import. Manual source
  inspection found no extra fields, methods, ownership, access, allocation, copy, close, version,
  graph, provenance, gradient/trainable/publication, runtime, backend, or device implementation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [device glossary](../../../../glossary.md#term-device))</span>
- Focused test and source inspection confirmed deterministic null/blank/type/capacity/liveness
  failures; contiguous, offset, strided, broadcast, scalar, zero-sized, static-unresolved, and
  dynamic-unresolved layouts; absent and present attachment; replacement from absence; atomic
  valid/invalid replacement; repeated clear; optional snapshots; read-only and late-dead storage;
  shared-storage alias observation with independent tensor associations; ordinary object identity;
  immutable metadata; and stable diagnostic exclusions.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [contiguous glossary](../../../../glossary.md#term-contiguous); [scalar glossary](../../../../glossary.md#term-scalar); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The first two targeted Ruby link-check invocations stopped before checking repository content
  because the initial regular expression used Ruby interpolation syntax and the installed Ruby
  lacks `Array#filter_map`. The compatible rerun resolved 129 local Markdown targets and anchors
  across the five changed documentation/planning files with zero errors. Fence validation found
  24, 0, 10, 2, and 2 backtick markers respectively, all balanced, and no tilde fences.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `rg -n '[[:blank:]]+$'` found no trailing whitespace in the seven changed files. No-index
  `git diff --no-index --check /dev/null <path>` checks for the three untracked files each returned
  the expected difference exit status `1` with no whitespace diagnostic. Final
  `git diff --check` passed with exit status `0` and no diagnostic.
- Exact scope review found only the allowed seven paths: one new production file, one new focused
  test, Tensor API, glossary, this task, model master plan, and roadmap. The forbidden transient<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
  `docs/design/README.md` path and every existing Java/test, Training API, architecture document,
  architecture test, capability baseline, Gradle file, and other module remain unchanged. No
  task-0012 specification exists, task 0012 remains a <span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>`Draft` row, and no [commit](../../../../glossary.md#term-commit) or push occurred.

## Implementation notes

- Added one final public [`Tensor`](../../../../glossary.md#tensor) with exactly the specified stable metadata and synchronized
  optional borrowed host-storage association. No factory, ID allocator, second production type,
  graph/provenance/training/publication state, typed access, runtime state, or backend behavior was
  added.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../../glossary.md#term-host); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Added one package-mirroring 13-test class covering API shape, validation order and messages,
  capacity geometry, association transitions, aliasing, borrowed lifetime, object identity, and
  diagnostics.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [aliasing glossary](../../../../glossary.md#term-aliasing); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The documentation-focused pass finalized `Tensor` Javadoc, moved only the implemented skeleton
  into current Tensor API and glossary language, and synchronized planning status. All deferred
  factory, provenance, operations, typed access, gradient/trainable/publication, compiler,
  runtime, and backend work remains planned.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Completion summary

- Completed changes: Implemented and documented the bounded public Tensor skeleton with stable
  identity, descriptor, normalized label, and synchronized optional borrowed host storage.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host storage glossary](../../../../glossary.md#host-storage))</span>
- Files changed or created: `Tensor.java`, `TensorTest.java`, Tensor API, glossary, this task,
  model master plan, and roadmap; no other path changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 13-test suite, all 154 model tests, model Javadoc, full repository
  tests, reflection, bytecode, import/dependency, compatibility, alias, lifetime, identity,
  diagnostics, generated-documentation, link/anchor, fence, whitespace, scope, and status checks
  passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [dependency glossary](../../../../glossary.md#term-dependency); [alias glossary](../../../../glossary.md#aliasing-and-arena); [lifetime glossary](../../../../glossary.md#term-lifetime); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean context
  `/root/implement_model_0011/review_model_0011_docs` completed the required independent pass using
  General, API/Javadoc, and Planning profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe only the current skeleton; Training
  API, architecture documentation, capability baseline, and unrelated documentation require no
  change for the reasons recorded above.
- Javadoc review: New Tensor Javadoc was finalized; existing TensorId, TensorDescriptor,
  HostTensorStorage, MemorySegmentStorage, and PublicationBinding Javadocs remain accurate.<span data-glossary-navigation="true"> ([TensorId glossary](../../../../glossary.md#tensorid))</span>
- Glossary impact: Tensor, TensorId, host-storage association, and Tensor-versus-graph-value status
  now reflect the implemented skeleton without adding a new term.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [graph glossary](../../../../glossary.md#graph))</span>
- Unresolved issues: None within task 0011.
- Follow-up required: None for task 0011. Task 0012 remains the next `Draft` planning frontier and
  owns the public factory and identifier-allocation policy.

Status: Complete
