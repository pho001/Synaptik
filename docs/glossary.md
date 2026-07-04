# Synaptik glossary

This glossary is the central dictionary for Synaptik terminology. It explains how terms are used in this repository; it is not an independent architecture contract. If a definition conflicts with [`ARCHITECTURE.md`](../ARCHITECTURE.md), the architecture contract wins.

Some entries describe contracts planned by the architecture but not yet implemented. Planning status belongs in the [implementation roadmap](planning/roadmap.md), not in this glossary.

## Implementation-status convention

The currently implemented terms are the model foundations: data type, dimension, shape,
broadcasting, layout, element stride, referenced element span, view, `TensorDescriptor`, typed
`TensorId`, `NodeId`, and `ValueId` values, `OperationKind`, `OperationAttrs`, `NoOperationAttrs`,
the `Operation` descriptor, the `GraphValue` and `CompiledNode` graph-element records,
`GraphPhase`, `CompiledGraphModel`, `PublicationBinding`, and the raw host-storage contracts
`HostTensorStorage` and `MemorySegmentStorage`, plus public `Tensor` state and the descriptor-based
`TensorFactory` construction boundary with JVM-scoped factory-assigned identity and exact-span
JVM-managed heap allocation for resolved layouts, plus copied flat typed import for resolved
dense-contiguous layouts and copied rectangular nested primitive-array import with exact carrier,
static-shape, and dense-layout inference, plus exact typed rank-0 scalars and independent dense
zero, one, zero-like, and one-like constants, plus exact typed full-value tensors and dense
rectangular identity matrices with `eye` as a pure alias, plus eager typed integer ranges, strict
or cyclic flat-prefix population for all six exact primitive carriers, and explicit-source normal
random
and bounded continuous-uniform population for the three floating types, plus bounded integral
random population for exact `INT32` and `INT64` output, plus BOOL Bernoulli population from a
finite scalar probability, plus immutable `TensorProvenance` origin metadata. Concrete operation
kind support now includes the parameterless `BinaryArithmeticKind` vocabulary for `ADD`, `SUB`,
`MUL`, `DIV`, `MIN`, `MAX`, and `POW`, plus matching public floating binary Tensor expression
construction with local promotion, broadcasting, descriptor derivation, and ordered provenance.
The parameterless `UnaryElementwiseKind` vocabulary is also implemented for fifteen unary
arithmetic, transcendental, activation, and explicit fast-approximation meanings; matching unary
Tensor expression methods remain planned. Other concrete kind and expression families, family
attributes, random Operations, typed access and export, native/runtime/backend allocation,
gradient and publication behavior, compiler entry points and artifacts, planning, prepare,
runtime, concrete backends, traces, and training remain architecture or planning contracts. A
definition explains intended meaning; it is not by itself evidence that a Java type exists.

## Terms

### Architecture contract

The normative rules that define Synaptik's module responsibilities, dependency direction, lifecycle boundaries, and core invariants. The repository's architecture contract is [`ARCHITECTURE.md`](../ARCHITECTURE.md). Guides, plans, ADRs, and this glossary may explain those rules but do not override them.

### Autograd

Automatic differentiation: a compiler transformation that derives gradient computations from the forward computation when the compile mode requires them. In Synaptik, the compiler performs global autograd and constructs the backward graph. A concrete backend does not perform global autograd; it prepares and executes only its assigned regions. See [Training graph](architecture/training-graph.md).

### Backend

An execution target identified at planning time, such as CPU, Metal, or CUDA. A backend is responsible for reporting capabilities and preparing the partitions assigned to it. Generic architecture discussions use “backend” for this role; a [concrete backend](#concrete-backend) is the module that implements it.

### Backend capability

A declarative statement about computation a backend can accept, based on facts such as operation kind, data type, shape, layout, or device availability. Planning queries capabilities when choosing ownership. A capability is not a kernel, a live executable service, or a promise that one fixed implementation route will always be selected. See [Partition scoring](architecture/partition-scoring.md).

### Backend ownership

The compile-time decision that assigns a node or segment to a backend identity such as CPU, Metal, or CUDA. Ownership answers “where should this work run?” It does not answer “which kernel should run it?” The owning concrete backend makes that implementation choice during [prepare](#prepare).

### Backend-owned lowering

The rule that each concrete backend translates its assigned planned partitions into its own executable representation during prepare. Lowering, fusion, specialization, and kernel selection stay together because they depend on backend-specific execution and storage models. There is no shared lowering module. See [Runtime, Prepare, and Backend Boundary](architecture/runtime-prepare-backend-boundary.md).

### Backend route

A concrete implementation choice inside one backend, such as a CPU scalar loop, Vector API routine, OpenBLAS call, MPSGraph executable, custom Metal kernel, or CUDA kernel. A route is selected during backend prepare after planning has chosen the backend owner. Routes are not separate backend identities.

### Backward graph

The graph of gradient computations derived by autograd from a forward graph. It propagates derivatives from requested outputs toward differentiable inputs or parameters. In backward-capable compile modes, the compiler may combine it with the forward graph before post-autograd optimization and planning. See [Training graph](architecture/training-graph.md).

### Broadcasting

A shape rule that combines compatible inputs by aligning axes from the right and expanding static singleton dimensions as needed. In the implemented local shape model, equal dimensions remain equal, size `1` expands to the opposing dimension, and symbolic compatibility must be provable from equal names or a singleton. Broadcasting describes logical repetition; a resolved layout may represent that repetition with a zero element stride.

### Compile

The lifecycle stage that captures a tensor expression, builds and validates graph semantics, applies graph transformations, adds autograd when requested, assigns backend ownership, and creates immutable compile artifacts. Compile creates a logical recipe; it does not allocate physical buffers, choose concrete kernels, or create backend executables. See [Lifecycle](architecture/lifecycle.md).

### Compile artifacts

The planned immutable output of compilation: an implemented compiled graph model plus planned
backend-owned partitions, logical memory requirements, a compiler-owned publication plan, and
compile diagnostics. `CompileArtifacts` is not implemented. It will be a recipe for prepare, not
executable state, and will contain no physical buffers, concrete kernel routes, runtime residency,
or mutable run state.

### Compiled graph / `CompiledGraphModel`

The implemented immutable compile-time graph representation. `CompiledGraphModel` stores ordered
`GraphValue` values, topologically ordered `CompiledNode` nodes, ordered input and output
`ValueId` boundaries, and an exact `NodeId`-to-[`GraphPhase`](#graph-phase) mapping. Construction
validates structural closure, producer rules, topology, and phase coverage and then stores
immutable collection snapshots without derived indexes. It performs no graph capture, compiler
transformation, planning, preparation, or execution. The planned public engine facade named
`CompiledGraph` will expose compiled artifacts and lifecycle orchestration; it is not the graph
model. See the [Compile API](api/compile-api.md#current-model-contracts).

### Concrete backend

A module that implements a backend, such as `backends/cpu`, `backends/metal`, or `backends/cuda`. It owns backend-specific capability reporting, prepare-time lowering, fusion, specialization, kernel selection, executable units, storage, workspaces, and native integration. Concrete backends do not own public tensor semantics or global graph compilation. See [Module boundaries](architecture/module-boundaries.md).

### Data type / `DataType`

The logical kind of scalar stored in each element of a tensor, such as `FLOAT32`, `INT64`, or `BOOL`. `DataType` records model-level facts including category and width, which lets model and compiler code interpret values consistently. It does not claim that every backend supports the type, prescribe a physical allocation alignment, or select a conversion route. See [Data types](api/tensor-api.md#data-types).

### Data-transfer object / DTO

A value whose purpose is to carry structured data across a boundary without owning the behavior that produced it. Synaptik's planned trace module uses typed DTOs so diagnostic consumers receive explicit fields without importing compiler, runtime, or backend business objects.

### Dimension

The size description for one axis of a [shape](#shape). A static dimension has a known non-negative `long` size; zero is valid and represents an empty extent. A dynamic dimension has a non-blank symbolic name because its numeric size is not yet known. Dynamic dimensions are explicit values, not negative-number sentinels, and two different symbols are not assumed equal. See [Shapes and dimensions](api/tensor-api.md#shapes-and-dimensions).

### Element stride

The number of storage elements advanced when one logical index advances by one position along an axis. Strides are measured in elements, not bytes. A stride of zero can represent repeated data, while canonical row-major strides describe contiguous geometry.

### Forward graph

The graph that computes outputs from user inputs in the original direction of the tensor expression. It exists before any gradient computation is added. A forward-only compile uses this computation; a backward-capable compile may expand it with a [backward graph](#backward-graph).

### Graph

A directed dataflow model of a computation. [Nodes](#node) represent computation occurrences, and [values](#graph-value) carry logical data from graph inputs or producing nodes to consuming nodes and graph outputs. A graph describes computation and data dependencies; it is not a runtime schedule or a collection of physical buffers.

### Graph value

Logical data flowing through a graph. The implemented immutable `GraphValue` record stores exactly
one graph-local [`ValueId`](#valueid) and one [`TensorDescriptor`](#tensor-descriptor). A value may
be a graph input, an intermediate result, or a graph output. It can exist without a producing node,
one node can produce multiple values, and one value can have multiple consumers. The record stores
no producer, consumer, or role flag. The owning implemented
[`CompiledGraphModel`](#compiled-graph--compiledgraphmodel) derives producers during construction
and validates graph-wide existence, uniqueness, and topology without storing an index. A graph
value is compile-time model state, not the implemented public mutable [`Tensor`](#tensor), a physical
memory slot, storage, or runtime residency. See [Graph values and compiled
nodes](api/tensor-api.md#graph-values-and-compiled-nodes).

### Graph phase

The implemented `GraphPhase` enum classifies compiled nodes as exactly `FORWARD` or `BACKWARD`
compile-time work. It helps later compiler, publication, planning, and diagnostic work distinguish
the two regions. It is not a compile mode, runtime schedule, prepared-execution boundary, ordinal
serialization, or optimizer-update phase.

### Host storage

Implemented model-level raw storage for physical tensor-element capacity visible in host memory.
The sealed `HostTensorStorage` abstraction currently permits exactly the final
`MemorySegmentStorage` identity wrapper. That wrapper binds a non-null logical data type and a
non-negative physical element capacity to one exact-size, initially live JDK `MemorySegment`.
Capacity is measured in complete physical elements and is independent of logical element count,
layout offset, and referenced span. Byte size is the checked product of capacity and data-type byte
width; zero capacity requires a zero-byte segment.

The wrapper borrows and returns the exact segment. It does not allocate memory, own or close an
arena, extend a scope's lifetime, or implement `AutoCloseable`. For a caller-supplied arena-backed
segment, the caller remains responsible for scope lifetime and JDK thread-access rules. A
factory-created primitive-array segment instead has an automatic scope that keeps its heap base
reachable and is accessible from any thread. Read-only state is descriptive, and liveness is a
point-in-time observation: after caller-controlled closure, the wrapper reports not alive but still
returns the exact dead segment so JDK access rules enforce failure. It defines no typed element
access, alignment, byte order, conversion, synchronization, or mutation-version policy.

Host storage is distinct from logical [layout](#layout), public [`Tensor`](#tensor) state,
device/backend storage, prepared memory, workspaces, and runtime [residency](#residency). The
implemented `Tensor` may borrow the exact storage object and validates matching data type,
resolved referenced span when geometry is available, and attachment-time liveness. The
implemented `TensorFactory` may pass an existing caller-supplied borrowed object through that same
construction path or create a matching primitive-array heap segment whose capacity is exactly the
resolved referenced span. The factory introduces no arena, close operation, external owner, or
deterministic lifetime. Neither path changes the wrapper's sizing, borrowed/non-closing ownership,
or raw-access contract. See [Host-visible storage](api/tensor-api.md#host-visible-storage).

### Kernel

A concrete backend implementation route for executing prepared work, such as a scalar CPU routine, an OpenBLAS call, a Metal kernel, or a CUDA kernel. Planning never selects kernels. The owning backend selects a kernel or other executable route during prepare.

### Layout

The logical mapping from a tensor's multidimensional indices to positions in storage. A layout can describe contiguous, offset-contiguous, strided, or broadcast views using facts such as element strides and storage offset. Layout describes geometry and aliasing; it does not own storage or decide whether a copy must be materialized. See the [Tensor API](api/tensor-api.md#resolved-layouts).

### Logical memory plan

Compile-time requirements derived from graph values and lifetimes, such as logical storage or materialization needs. It does not allocate buffers or assign physical addresses. Prepare turns these requirements into a prepared physical memory plan and slots.

### Lifecycle

The ordered stages through which Synaptik state moves. The core lifecycle is compile, prepare, and run: compile creates immutable artifacts, prepare creates reusable executable state, and each run creates or uses invocation-specific mutable state. “Lifecycle” also includes ownership and validity rules for objects and resources within those stages. See [Lifecycle](architecture/lifecycle.md).

### Lowering

Translation from a planned, backend-neutral graph region into a backend-specific executable form. Lowering may include backend-specific decomposition, fusion, specialization, and representation building. In Synaptik it happens during prepare inside the owning concrete backend, not in planning or the runtime hot path. See [Backend-owned lowering](#backend-owned-lowering).

### Materialization

Creating a concrete stored representation when a logical value or view cannot be consumed in its current form. For example, a backend route may require a contiguous copy of a strided view. Planning expresses logical materialization requirements; prepare and backend/runtime mechanisms realize the required storage and copy work. Materialization is not a property decided by `LayoutDescriptor` alone.

### Memory slot

A position in a prepared memory plan for a physical buffer or workspace used during execution. Slots let a prepared schedule refer to reusable storage without embedding raw addresses. A memory slot is a physical execution resource and must not be confused with a logical [`ValueId`](#valueid).

### `NoOperationAttrs`

The implemented canonical immutable attribute value for an operation kind that has no semantic parameters. It is a single-value enum whose only value is `NoOperationAttrs.INSTANCE`. The singleton makes “no parameters” explicit and non-null rather than representing absence with `null`, an empty map, or a newly allocated placeholder. See [`OperationAttrs`](#operationattrs).

### Node

One occurrence of computation in a graph. The implemented immutable `CompiledNode` record stores a
graph-local [`NodeId`](#nodeid), one [`Operation`](#operation), and ordered immutable input and
output `ValueId` snapshots. Empty and repeated inputs are valid; outputs must be non-empty and
unique within that node. Reusing the same operation kind in two places creates two node
occurrences. A node is not the operation semantics alone and is not the data flowing between
computations; that data is represented by [graph values](#graph-value). The record validates only
its local list invariants. The owning implemented `CompiledGraphModel` validates referenced-value
existence, producer uniqueness, topology, and graph boundaries, while planned operation-family
contracts own arity and descriptor compatibility. A compiled node is compile-time model state and
must not enter runtime hot paths. See [Graph values and compiled
nodes](api/tensor-api.md#graph-values-and-compiled-nodes).

### `NodeId`

A validated non-negative identifier for a node occurrence within one owning graph. It identifies where operation semantics occur, not the operation kind itself. Its numeric value may be reused in another graph, so it has meaning only with its graph context. See [Identifiers](api/tensor-api.md#typed-identifiers).

### Operation

The implemented immutable value that keeps two parts of a computation description together: an [`OperationKind`](#operationkind), which says which computation is meant, and [`OperationAttrs`](#operationattrs), which carries its typed parameters. Both parts must be non-null and are retained unchanged. Record equality and hashing use both parts, while its text form is for diagnostics rather than serialization. The descriptor does not verify that a particular attributes type is compatible with a kind; future operation-family contracts own that validation. It also does not report backend support, perform compiler work, choose a kernel, execute computation, own runtime state, or identify a particular occurrence in a graph; an implemented [`CompiledNode`](#node) represents that occurrence.

### `OperationAttrs`

The implemented zero-method marker contract for immutable, typed parameters that refine which computation an [`Operation`](#operation) describes. A future family-specific attribute record might hold axes, padding, or another operation-specific value. Implementations use typed fields, defensively isolate mutable inputs, and provide structural equality and hashing; they do not use a primary string-keyed map or contain backend, compiler-service, mutable tensor, storage, or runtime state. Kinds without parameters use [`NoOperationAttrs.INSTANCE`](#nooperationattrs). The marker identifies the role of a value but does not enforce immutability at runtime.

### `OperationKind`

The implemented open typed discriminator that supplies the “which computation” part of an [`Operation`](#operation). Its only method, `name()`, provides a stable, non-null, non-blank diagnostic name. Equality belongs to the typed kind value, so equal name text from unrelated kind types does not create implicit equivalence or a global string registry. An operation kind does not describe attributes, backend support, cost, fusion, storage, execution behavior, or a kernel route.

The first production family is `BinaryArithmeticKind`, an enum containing exactly `ADD`, `SUB`,
`MUL`, `DIV`, `MIN`, `MAX`, and `POW`. These values identify ordered tensor-to-tensor elementwise
arithmetic meanings. They have no intrinsic parameters and therefore compose with
`NoOperationAttrs.INSTANCE`. The enum stores no operands or broadcast metadata and does not build
Tensor expressions by itself, infer shapes or data types, identify graph occurrences, execute
computation, or report backend support. The implemented public Tensor methods consume these values
while separately owning local expression construction. The enum's inherited names are diagnostics
rather than serialization or dispatch keys.

The second production family is `UnaryElementwiseKind`, an enum containing exactly `ABS`, `NEG`,
`INV`, `LOG`, `EXP`, `ERF`, `SQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`,
`FAST_EXP`, and `FAST_TANH`. These values identify one-input elementwise mathematical or activation
meanings and compose with `NoOperationAttrs.INSTANCE`. One-input arity is family context rather
than stored metadata. `FAST_EXP` and `FAST_TANH` are distinct approximate requests, not aliases or
backend flags; the enum defines no algorithm, accuracy, descriptor inference, provenance,
gradient, execution, or backend support. Public unary Tensor expression methods remain planned.

### Partition

A planned graph region whose nodes share one backend owner. Planning forms maximal same-owner partitions after ownership decisions so each backend can prepare a coherent region. A partition is still compile-time planning data, not a selected kernel or prepared executable.

### Planning

The backend-neutral compile-time work that decides backend ownership, forms same-owner partitions, and derives logical memory or materialization requirements from graph facts, configuration, and declarative backend capabilities. Planning answers where a node or segment should run. It does not choose concrete kernels, perform backend-specific lowering, allocate physical buffers, or inspect runtime residency. See [Partition scoring](architecture/partition-scoring.md).

### Prepare

The lifecycle stage that turns immutable compile artifacts into reusable runtime-ready state. Shared prepare orchestration validates coverage and schedules, while each concrete backend lowers its partitions and selects executable routes. Prepare creates prepared partitions, memory plans, schedules, and `PreparedExecution`. See [Lifecycle](architecture/lifecycle.md#prepare-lifecycle).

### Prepared execution / `PreparedExecution`

Reusable runtime-ready state produced by prepare. It contains or refers to prepared partitions and executables, a physical memory plan, and an execution schedule. It can serve multiple runs; per-run mutable inputs and state belong to `RunState`, not to the immutable compile-time graph. See [Runtime, Prepare, and Backend Boundary](architecture/runtime-prepare-backend-boundary.md#what-prepare-creates).

### Prepared executable / `PreparedExecutable`

The backend-independent runtime call boundary implemented by a concrete backend for one prepared region. It contains the implementation choice made during prepare and computes only that region. Runtime invokes it without passing compile-time `Operation` or `CompiledNode` objects or asking it to select another backend.

### Provenance

Implemented immutable expression-origin metadata carried by an optional `TensorProvenance` value
on public [`Tensor`](#tensor) state. The record retains one exact backend-independent
[`Operation`](#operation) reference and an ordered immutable snapshot of exact input-Tensor
references. Empty inputs represent a valid local zero-input origin, repeated references preserve
distinct ordered roles, and caller mutation of the source list cannot change the snapshot.

Provenance is not intermediate representation (IR), producer-occurrence identity, graph
membership, graph capture, or executable behavior. It contains no graph-local `NodeId` or
`ValueId`, does not validate operation arity, descriptors, cycles, or graph structure, and does not
change when Tensor host storage is replaced, cleared, or becomes dead. Record equality compares
the operation value and ordered input objects using ordinary equality; it does not perform
common-subexpression elimination. A later compiler owns traversal and conversion into immutable
graph records. See [Public Tensor state](api/tensor-api.md#public-tensor-state).

The implemented binary Tensor methods create provenance whose operation uses the exact matching
`BinaryArithmeticKind` and `NoOperationAttrs.INSTANCE`, and whose two input positions preserve the
receiver as left and argument as right. That current construction does not change provenance's
general role or make it graph membership.

### Publication binding

The implemented immutable `PublicationBinding` record associates one [`TensorId`](#tensorid) with
one graph-local [`ValueId`](#valueid). It is standalone model data for a later compiler-owned
`PublicationPlan`, not a component of `CompiledGraphModel`. A binding cannot by itself prove that
its value belongs to a particular graph, and it carries no public `Tensor`, gradient role,
publication policy or target, storage, backend, or runtime state. The planned publication plan,
prepare, and run layers will add their own owning context and behavior.

### Residency

Runtime knowledge of where a value's current physical representation exists, such as host or device storage, and which representation is valid after transfers or execution. Residency is dynamic per-run/runtime state. It does not belong to the model `Tensor`, compile-time planning, or `LayoutDescriptor`.

### Run / runtime

**Run** is one invocation of a `PreparedExecution`: it binds inputs, creates or reuses `RunState`, follows the prepared schedule, executes prepared units, manages scheduled transfers or materialization, and publishes results. **Runtime** is the module and machinery that performs this work. Run does not optimize graphs, discover backends, lower partitions, or choose kernels. See [Lifecycle](architecture/lifecycle.md#run-lifecycle).

### Run state / `RunState`

Mutable state for one invocation of prepared execution, including input bindings, runtime slots, resources, and current residency facts as defined by future runtime contracts. It is separate from reusable `PreparedExecution` and immutable compile artifacts.

### Shape

An immutable ordered collection of [dimensions](#dimension) describing the logical size of a tensor along each axis. The number of dimensions is the shape's rank; a rank-0 shape represents a scalar. A shape describes extents only: it does not define strides, storage, layout, backend support, or runtime allocation. Its total element count is known only when every dimension is static. See [Shapes and dimensions](api/tensor-api.md#shapes-and-dimensions).

### Referenced element span

The minimum count of storage elements needed to include every index referenced by a resolved layout. For non-empty shapes it is the greatest referenced element index plus one; for shapes with a zero-sized dimension it is zero. The span includes a storage offset and is not necessarily equal to the logical element count.

### Tensor

The implemented public mutable API object for stable tensor metadata and optional host-visible
state. The current final `Tensor` retains one exact immutable [`TensorId`](#tensorid), one exact
immutable [`TensorDescriptor`](#tensor-descriptor), one normalized immutable optional label, and
immutable optional [`TensorProvenance`](#provenance). Its only current mutation is a synchronized
optional borrowed [`HostTensorStorage`](#host-storage) association. Replacement validates matching
data type, resolved referenced span when layout is available, and point-in-time liveness before
changing the exact reference. Read-only storage is accepted; later caller-controlled scope death
remains observable; and the tensor retains the wrapper reference without allocating backing
memory itself, copying or accessing contents, owning a closeable resource, or closing storage.
Provenance remains the same exact value across every storage transition and is accessed without
synchronization because it is final.

Construction remains package-private, and the implemented [`TensorFactory`](#tensor-factory) is
the supported public construction boundary. The object uses ordinary identity equality and
hashing, while its diagnostic text contains stable ID, descriptor, and label facts without
provenance expansion, storage, or runtime state. Copied flat typed import is implemented through
the factory for resolved
dense-contiguous layouts. Copied rectangular nested primitive-array import is also implemented;
the factory infers its exact type, fully static shape, and dense-contiguous layout before returning
a Tensor. Exact typed scalar, zero, one, zero-like, and one-like creation is implemented with new
dense storage and explicit label and gradient intent. Type-safe full-value creation is implemented
for every current primitive meaning, and rectangular identity creation is implemented for all six
data types with typed main-diagonal ones and off-diagonal zeros. Eager non-empty `INT32` and
`INT64` range creation, strict or cyclic typed flat-prefix creation, and caller-source normal
random and bounded
continuous-uniform creation are also implemented as copied canonical dense leaf data.
Caller-source bounded integral creation is implemented for exact `INT32` and `INT64` output with
false gradient intent. Caller-source Bernoulli creation is implemented for canonical BOOL output
with false gradient intent and a finite scalar probability. Random Operations, typed access and
export, and deterministic native-resource ownership remain planned. The current `add`, `sub`,
`mul`, `div`, `min`, `max`, and tensor-valued `pow` methods create fresh storage-free binary
arithmetic expression tensors from floating operands. They promote data type, broadcast shape,
leave layout unresolved, propagate gradient eligibility as input OR, and retain exact matching
operation semantics plus ordered provenance. Other expression families, gradient rules and
objects, trainable role, publication behavior, compiler integration, device buffers, numerical
execution, and runtime residency remain planned.
A `Tensor` is not an
intermediate-representation node or [graph value](#graph-value). See [Public Tensor
state](api/tensor-api.md#public-tensor-state).

### Tensor factory

The implemented public static `TensorFactory` creates a fresh [`Tensor`](#tensor) from a completed
[`TensorDescriptor`](#tensor-descriptor), with optional diagnostic label text and optional
existing borrowed [host storage](#host-storage). It is the public construction boundary while the
Tensor constructor remains package-private. For a descriptor with a resolved layout, the factory
can also allocate one matching JVM primitive array whose length is exactly the referenced element
span, wrap its heap segment, and attach that writable storage. `FLOAT64`, `FLOAT32`, `BFLOAT16`,
`INT32`, `INT64`, and `BOOL` use `double[]`, `float[]`, raw `short[]`, `int[]`, `long[]`, and raw
`byte[]`, respectively. The raw array starts at the JVM default zero representation.

The factory can import those same six flat carriers into a new resolved dense-contiguous tensor.
Carrier data type and logical element count must match exactly. Numeric carriers and raw BFLOAT16
bits are copied unchanged; BOOL treats zero as false and canonicalizes every non-zero byte to one.
The source is not retained or mutated, and later source mutation cannot change the tensor. Scalar
and empty dense imports follow their logical counts. Offset, strided, broadcast, and unresolved
layouts are rejected because they require a separate scatter or view-population policy.

The factory also accepts one rank-two-or-greater rectangular Java primitive-array graph through an
`Object` parameter, which is necessary because arbitrary array rank has no finite Java overload
family. Runtime class metadata must prove an ultimate `double`, `float`, `short`, `int`, `long`, or
`byte` carrier. The factory validates every reachable subarray for non-null rectangular structure,
rejects an empty non-final axis whose trailing extents are unobservable, accepts an empty final
primitive axis, and infers an exact fully static dense-contiguous descriptor. It flattens leaves in
row-major order into a fresh matching carrier and delegates to flat import. Numeric and raw
BFLOAT16 values remain unchanged; BOOL normalization remains centralized in flat import. No source
level is retained or mutated, later source mutation cannot change the tensor, and concurrent
mutation during import has no atomic deep-snapshot guarantee.

The heap segment's automatic scope keeps the primitive array reachable and permits access from
any thread. No arena, close operation, external owner, native fallback, or deterministic release
is introduced. Descriptor-based creation and flat import do not construct descriptors or resolve
absent layouts; nested import constructs only the exact static dense descriptor inferred from the
validated source. Constant creation constructs only canonical dense descriptors for exact
primitive rank-0 scalars or fully static requested/template shapes. Scalars infer data type from
their declared primitive inputs; `scalarBFloat16(float)` alone converts with BFLOAT16
round-to-nearest, ties-to-even semantics. Zeros use default-zero allocation, while scalars and
ones use exact typed flat import. Like methods read only template shape and data type and preserve
neither layout nor mutable or diagnostic state. Full-value methods infer exact type from primitive
values, with only `fullBFloat16` converting, and fill one exact source before one flat import.
`identityMatrix` creates square or rectangular dense matrices for every current data type with
typed one on the main diagonal and typed zero elsewhere; `eye` delegates unchanged to that
canonical method. Separate calls create fresh metadata, storage, Tensor identity, and backing
arrays. Every public factory path creates a
provenance-free leaf. One package-private derived-construction seam attaches an already-created
provenance value through the existing ID allocator without storage, graph capture, traversal,
inference, or semantic validation. The factory retains no tensor, graph, runtime, backend,
registry, or service state.

The factory's two eager range overloads map `int` bounds and step to `INT32`, and `long` bounds and
step to `INT64`. Each result is non-empty, rank one, non-differentiable, inclusive at the start,
exclusive at the end, and stored in new canonical dense storage. Positive and negative non-zero
steps are accepted only when they advance toward the end. Exact overflow-safe sizing rejects a
count above `Integer.MAX_VALUE` before allocation and does not evaluate an unused addition after
the final emitted value.

Strict and cyclic flat-prefix creation each have six overloads for `double[]`, `float[]`, raw
BFLOAT16 `short[]`, `int[]`, `long[]`, and BOOL `byte[]`. They require a fully static caller shape,
infer the exact data type, and create a new canonical dense descriptor with explicit label and
gradient intent. Strict mode copies exactly the requested leading values and ignores a tail.
Cyclic mode repeats `source[i % source.length]`; an empty source is accepted only for an empty
result. No source is retained or mutated. Numeric and raw BFLOAT16 values remain unchanged, while
flat import normalizes BOOL zero/non-zero bytes to canonical zero/one storage. Neither mode adds
shape inference, conversion, view scattering, or a general fill/repeat/tile operation.

Normal random creation accepts one transient caller-owned `RandomGenerator`, fully static
Java-array-sized shape, explicit `FLOAT64`, `FLOAT32`, or `BFLOAT16` output, finite mean, finite
numerically non-negative standard deviation, label, and gradient intent. It consumes exactly one
`nextGaussian()` call per logical row-major element, transforms with ordinary binary64
multiplication then addition, converts to one exact carrier, and delegates once to flat import.
The factory never selects, seeds, retains, substitutes, synchronizes, resets, splits, or closes
the source. Reproducibility is consequently bounded to equivalent generator implementation and
state, identical arguments, and no interfering use. Random Operations, typed access or export,
general numeric conversion, native/runtime/backend allocation, and deterministic resource
ownership remain planned.

Continuous-uniform random creation accepts the same transient caller-owned source, static
Java-array-sized shapes, and three floating output types. Its finite binary64 lower bound must be
strictly less than its finite upper bound. Each row-major element consumes exactly one
`nextDouble(lower, upper)` call. A conforming source result is in the binary64 half-open interval;
FLOAT64 stores it directly, FLOAT32 narrows once, and BFLOAT16 narrows to binary32 before
`BFloat16Bits.fromFloat`. Narrowing may produce a stored value equal to the corresponding narrowed
upper bound. The factory does not clamp or resample, post-validate custom source results, or retain
the generator. The same caller ownership, no-synchronization, and bounded reproducibility policy
applies.

Bounded integral random creation has two `randomInt` overloads. Primitive `int` bounds infer
`INT32`, primitive `long` bounds infer `INT64`, and both results disable gradients. Each row-major
element consumes exactly one matching bounded `nextInt(origin, bound)` or
`nextLong(origin, bound)` call and is stored directly in one exact carrier before one flat import.
Bounds define a strict half-open interval and are validated even for empty output. No modulo,
unbounded draw, floating conversion, data-type parameter, gradient parameter, or default source is
added. Because the exclusive bound uses the result carrier, the API cannot express a mathematical
exclusive bound above `Integer.MAX_VALUE` or `Long.MAX_VALUE`; no full-domain convenience is
provided. The same caller ownership, no-synchronization, bounded reproducibility, and late
failure/no-rollback rules apply.

Bernoulli random creation has one `randomBernoulli` method. It requires a fully static
Java-array-sized shape and a finite binary64 probability in the closed interval `[0, 1]`, always
produces canonical BOOL storage, and always disables gradients. Each row-major element consumes
exactly one unbounded `nextDouble()` call, including when probability is zero or one, and stores
byte one exactly when the draw is strictly less than the probability. Equal or custom
non-conforming draws are not post-validated or coerced. Positive and negative zero are both
accepted as probability zero. The factory builds one complete byte carrier and delegates once to
BOOL flat import; it exposes no data-type, gradient, numeric-truthiness, default-source, or
probability-tensor option. The same caller ownership, no-synchronization, bounded reproducibility,
and late failure/no-rollback rules apply.

No random package, source service, seed API, or distribution enum is introduced because the
distribution-specific factory methods share one cohesive package-private helper and add no
independent public random-domain model.

For every attempted construction that reaches identifier allocation, the factory issues one
non-negative [`TensorId`](#tensorid) unique among its allocations in the current Java virtual
machine (JVM), including concurrent calls. Numeric values are opaque: semantic construction
failures create permanent gaps, and numeric order need not match method-completion order. Null
argument containers fail before allocation, while delegated label or storage validation fails
afterward and consumes the candidate. `Long.MAX_VALUE` can be claimed once; every later allocation
fails permanently instead of wrapping or reusing an ID. The guarantee does not cover manually
constructed IDs, another JVM, process restarts, persisted artifacts, or distributed identity.

Heap allocation additionally requires resolved layout and a referenced span no greater than
`Integer.MAX_VALUE`. Null, unresolved-layout, over-limit, JVM array, segment, and storage-wrapper
failures occur before identifier allocation and consume no ID. Heap allocation and wrapping occur
before delegated creation, so a blank label or exhausted identifier space is observed only after
the heap work; a blank-label failure consumes its allocated ID.

Flat import performs carrier, dense-layout, and logical-count validation before destination or ID
allocation. A blank label and identifier exhaustion are observed after destination allocation and
before copying. Unexpected population failures occur after ID allocation and are not rolled back.

Nested import performs complete structural and checked-count validation before allocating its
intermediate flat carrier. Descriptor gradient eligibility is checked after flattening but before
destination or ID allocation. A blank label and identifier exhaustion then have the same
destination-allocation and ID side effects as delegated flat import. The intermediate carrier is
never exposed or retained.

Constant creation validates static shape, checked logical count, the Java array limit, dense
layout geometry, and gradient eligibility before destination or ID allocation. Scalar and one
source carriers are allocated after descriptor validation; zeros have no source carrier. Blank
labels fail after destination and ID allocation and consume that ID. Exhaustion is also observed
after destination allocation. Every successful constant has a new Tensor, descriptor, layout,
storage wrapper, backing array, and factory ID; like-shaped results retain no template object or
template state beyond the immutable shape and data-type values used to build the result.

Range label and argument validation and prefix shape/count/source/gradient validation run before
result-carrier, destination, or ID allocation. Each successful path builds one complete exact
carrier and delegates once to flat import. A blank label is rejected after carrier, destination,
and ID allocation but before copying, and consumes that ID. Exhaustion is observed after both
arrays exist. These failures do not roll back identifiers.

Normal-random null, shape, count, type, distribution, layout, and descriptor validation completes
before source-carrier allocation, sampling, destination allocation, or ID allocation. Source
allocation failure consumes neither calls nor an ID. A generator exception preserves completed
source calls but creates no destination or ID. After sampling, delegated flat import allocates the
destination and then the ID; blank-label failure consumes all calls and one ID, while exhaustion
consumes all calls without rollback.

Continuous-uniform null, shape, count, type, bound, layout, and descriptor validation likewise
completes before source-carrier allocation, sampling, destination allocation, or ID allocation.
Source allocation failure consumes neither calls nor an ID. A generator exception preserves prior
bounded calls but creates no destination or ID. After sampling, delegated flat import allocates the
destination and then the ID; blank-label failure consumes all calls and one ID, while exhaustion
consumes all calls without rollback.

Bounded-integral null, shape, count, and bound validation likewise completes before source-carrier
allocation, sampling, destination allocation, or ID allocation. Source allocation failure
consumes neither calls nor an ID. A generator exception preserves prior bounded calls but creates
no destination or ID. After sampling, delegated flat import allocates the destination and then the
ID; blank-label failure consumes all calls and one ID, while exhaustion consumes all calls without
rollback.

Bernoulli null, shape, count, and probability validation likewise completes before source-carrier
allocation, sampling, destination allocation, or ID allocation. Source allocation failure
consumes neither calls nor an ID. A generator exception preserves prior unbounded calls but
creates no destination or ID. After sampling, delegated BOOL flat import allocates the destination
and then the ID; blank-label failure consumes all calls and one ID, while exhaustion consumes all
calls without rollback.

### Tensor descriptor

The implemented immutable combination of one non-null data type, one non-null shape, an explicit
resolved-or-unresolved layout state, and a `requiresGrad` flag. A present
`Optional<LayoutDescriptor>` contains resolved numeric geometry; `Optional.empty()` means geometry
is unresolved. Dynamic shapes must be unresolved, while fully static shapes may also remain
unresolved without implying a default layout. A present layout is reconstructed against the paired
shape and accepted only when its complete public geometry remains equal. This proves geometric
compatibility, not the identity of the shape that originally created the layout. `requiresGrad`
may be true only for a differentiable data type and records model eligibility, not the existence of
a gradient rule or backend support. The optional is compared by value rather than container
identity; when present, it contains the exact immutable layout object supplied at construction. A
descriptor describes logical facts and does not own a public mutable `Tensor`, graph identity, host
storage, device buffers, runtime residency, materialization policy, or backend execution state. See
[Tensor descriptors](api/tensor-api.md#tensor-descriptors).

### `TensorId`

A validated non-negative identifier retained by implemented public mutable `Tensor` state. It
belongs to the tensor identity domain and is distinct from graph-local node and value identities.
The value type itself does not allocate or guarantee uniqueness. The implemented
[`TensorFactory`](#tensor-factory) provides the narrower guarantee that IDs it allocates are unique
among its allocations in one JVM; callers may still construct equal numeric values manually. Two
tensor objects remain unequal even when their IDs compare equal. An implemented
`PublicationBinding` can associate an ID with a graph value without storing graph-local IDs on the
tensor. See [Identifiers](api/tensor-api.md#typed-identifiers).

### Trace

Structured diagnostic information emitted by compile, prepare, run, and backend activity. A trace helps people and tools understand what happened without becoming business logic or execution state. The trace module is a dependency leaf and uses trace-local identifiers rather than importing producer-layer domain objects. See [Tracing](architecture/tracing.md).

### Training graph

The compile-time computation used for a training-capable mode. It contains the forward computation and backward gradient computation, and a later architecture version may also represent optimizer updates as graph operations. It remains compile-time graph state; backends only prepare and execute their assigned regions. See [Training graph](architecture/training-graph.md).

### Typed trace DTO

A typed data-transfer object used to carry one defined category of diagnostic facts, such as compile, prepare, run, or backend payloads. Typed fields preserve meaning and machine-readable types. They are preferred over a primary `Map<String,String>` model, while typed trace attributes provide a limited escape hatch for backend-specific details. See [Tracing](architecture/tracing.md#typed-diagnostic-dtos).

### `ValueId`

A validated non-negative identifier for an input, intermediate, or output logical value within one owning graph. It does not identify a computation occurrence or a physical buffer. Its numeric value may be reused in another graph. See [Identifiers](api/tensor-api.md#typed-identifiers).

### View

A tensor or layout interpretation that aliases storage also used by another logical tensor representation. A view may change shape, strides, or offset without copying elements. In the implemented `LayoutDescriptor`, view is explicit metadata independent of geometric kind, except that broadcast repetition through a zero stride must be marked as a view.

## Common distinctions

### Tensor versus graph value

| `Tensor` | Graph value |
|---|---|
| Implemented public mutable API state | Immutable compile-time graph state |
| Retains stable ID, descriptor, label, and optional provenance plus optional mutable host storage | Represents logical data flowing between graph nodes |
| Identified by `TensorId` | Identified by graph-local `ValueId` |
| Can participate in more than one separately compiled graph | Belongs to one owning graph context |
| Must not become runtime device residency | Must not be confused with a physical buffer or slot |

The implemented standalone `PublicationBinding` connects the two identity domains. The planned
compiler-owned publication plan will provide owning-graph and publication-policy context. Public
descriptor-based leaf construction, immutable provenance, and floating binary Tensor expression
construction are implemented. Gradient rules and objects, compiler graph capture, numerical
execution, and publication behavior are not part of the current Tensor contract.

### Node versus value

| Node | Value |
|---|---|
| Implemented as `CompiledNode` | Implemented as `GraphValue` |
| A computation occurrence | Logical data consumed or produced by computation |
| Identified by `NodeId` | Identified by `ValueId` |
| Stores an `Operation` and ordered value-ID positions | Stores a `TensorDescriptor`, not an operation |
| May produce multiple values | May exist without a producer and may have multiple consumers |

Neither local record is the graph container. The implemented `CompiledGraphModel` owns producer
derivation and validates whole-graph uniqueness, topology, boundaries, and exact phase coverage
without storing derived indexes.

### Operation kind versus attributes versus operation

| Concept | Meaning | Current status |
|---|---|---|
| `OperationKind` | Which backend-independent computation is meant | Interface, binary arithmetic family, and unary elementwise family implemented; other families planned |
| `OperationAttrs` | Immutable typed parameters that refine that meaning | Marker implemented; family-specific values planned |
| `NoOperationAttrs.INSTANCE` | Explicit parameter value for a kind with no parameters | Implemented canonical singleton |
| `Operation` | Immutable pairing of one kind with one caller-supplied `OperationAttrs` value | Implemented descriptor |

A kind distinguishes computations, while attributes carry parameters within a computation family.
`Operation` stores both as one value but does not validate family compatibility. None of these
values identifies where computation occurs in a graph; an implemented [node](#node) represents
that occurrence. Binary arithmetic kinds and their public Tensor construction path are implemented.
Unary elementwise kinds are also implemented, while their public Tensor construction path remains
planned. Other concrete families, family-specific attributes, compiler capture, and execution
remain planned. The compiled graph container is implemented model state.

### Compile versus prepare versus run

| Stage | Main question | Produces | Must not do |
|---|---|---|---|
| Compile | What does the graph mean, and which backend owns each region? | Immutable compile artifacts | Allocate physical buffers or choose kernels |
| Prepare | How will each owned region execute? | Reusable prepared executables, memory plan, and schedule | Perform global graph optimization or per-run work |
| Run | Execute this prepared schedule with these inputs. | Per-run state and published results | Discover backends, lower partitions, or select kernels |

### Logical `ValueId` versus physical memory slot

A `ValueId` names logical data in one compiled graph. A memory slot names reusable physical storage in one prepared execution. Preparation may map values to slots, and different values may reuse a slot when their lifetimes do not overlap. The identifier therefore remains stable as a logical graph fact even when physical storage decisions change.
