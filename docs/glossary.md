# Synaptik glossary

This glossary is the central dictionary for Synaptik terminology. It explains how terms are used in this repository; it is not an independent architecture contract. If a definition conflicts with [`ARCHITECTURE.md`](../ARCHITECTURE.md), the architecture contract wins.

Some entries describe contracts planned by the architecture but not yet implemented. Planning status belongs in the [implementation roadmap](planning/roadmap.md), not in this glossary.

## Implementation-status convention

The currently implemented terms are the model foundations: data type, dimension, shape, broadcasting, layout, element stride, referenced element span, view, typed `TensorId`, `NodeId`, and `ValueId` values, plus `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, and the `Operation` descriptor. Concrete operation kinds and family attributes, graph integration, public `Tensor`, compiled graphs, planning, prepare, runtime, concrete backends, traces, and training remain architecture or planning contracts. A definition explains intended meaning; it is not by itself evidence that a Java type exists.

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

The immutable planned output of compilation: a compiled graph model, backend-owned planned partitions, logical memory requirements, publication bindings, and compile diagnostics. Compile artifacts are a recipe for prepare, not executable state, and contain no physical buffers, concrete kernel routes, runtime residency, or mutable run state.

### Compiled graph / `CompiledGraphModel`

The immutable compile-time graph representation: nodes, logical values, phases, and relationships after graph capture and compiler transformations. `CompiledGraphModel` is model state inside the compiler's output and is not prepared executable state. The public engine facade named `CompiledGraph` represents access to compiled artifacts and lifecycle orchestration; it should not be confused with the graph model itself. See [Architecture overview](architecture/overview.md).

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

Logical data flowing through a graph. A value may be a graph input, an intermediate result, or a graph output. It can exist without a producing node, one node can produce multiple values, and one value can have multiple consumers. A graph value is compile-time graph state, not the public mutable [`Tensor`](#tensor) object and not a physical memory slot.

### Graph phase

The compile-time role of graph work, such as forward or backward computation. A phase helps optimization, publication, planning, and diagnostics distinguish related regions without turning them into separate runtime programs by definition.

### Host storage

Model-level storage for tensor elements that are visible in host memory, represented through the `HostTensorStorage` abstraction and its implementations. Host storage has explicit data type, size, mutability, ownership, and lifetime rules. It is distinct from backend device buffers, workspaces, and runtime [residency](#residency). See the [model capability baseline](planning/modules/model/capabilities.md#host-storage-baseline).

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

One occurrence of computation in a graph. A node applies operation semantics to input values and produces one or more output values. Reusing the same operation kind in two places creates two node occurrences. A node is not the data flowing between computations; that data is represented by graph values.

### `NodeId`

A validated non-negative identifier for a node occurrence within one owning graph. It identifies where operation semantics occur, not the operation kind itself. Its numeric value may be reused in another graph, so it has meaning only with its graph context. See [Identifiers](api/tensor-api.md#typed-identifiers).

### Operation

The implemented immutable value that keeps two parts of a computation description together: an [`OperationKind`](#operationkind), which says which computation is meant, and [`OperationAttrs`](#operationattrs), which carries its typed parameters. Both parts must be non-null and are retained unchanged. Record equality and hashing use both parts, while its text form is for diagnostics rather than serialization. The descriptor does not verify that a particular attributes type is compatible with a kind; future operation-family contracts own that validation. It also does not report backend support, perform compiler work, choose a kernel, execute computation, own runtime state, or identify a particular occurrence in a graph; the occurrence is a [node](#node).

### `OperationAttrs`

The implemented zero-method marker contract for immutable, typed parameters that refine which computation an [`Operation`](#operation) describes. A future family-specific attribute record might hold axes, padding, or another operation-specific value. Implementations use typed fields, defensively isolate mutable inputs, and provide structural equality and hashing; they do not use a primary string-keyed map or contain backend, compiler-service, mutable tensor, storage, or runtime state. Kinds without parameters use [`NoOperationAttrs.INSTANCE`](#nooperationattrs). The marker identifies the role of a value but does not enforce immutability at runtime.

### `OperationKind`

The implemented open typed discriminator that supplies the “which computation” part of an [`Operation`](#operation). Its only method, `name()`, provides a stable, non-null, non-blank diagnostic name. Equality belongs to the typed kind value, so equal name text from unrelated kind types does not create implicit equivalence or a global string registry. An operation kind does not describe attributes, backend support, cost, fusion, storage, execution behavior, or a kernel route. No production concrete kind is implemented yet.

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

The minimal origin information associated with a public tensor expression so graph capture can discover how a tensor was produced and from which inputs. Provenance supports later graph construction but does not turn `Tensor` into an IR node or assign graph-local `NodeId` or `ValueId` values to the tensor itself. See the [model capability baseline](planning/modules/model/capabilities.md#public-tensor-baseline).

### Publication binding

An immutable compile-time association between public tensor identity and a logical graph result. Conceptually, it connects a `TensorId` requested for publication with the final `ValueId` that the compiled graph produces. Prepare and run use the publication plan to make requested outputs or gradients visible without treating the public `Tensor` as compiled IR.

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

The public mutable API object used to describe tensor metadata and host-visible state and to build expressions. It may carry data type, shape, layout, host storage, gradient/publication state, identity, and provenance. A `Tensor` is not an intermediate-representation node, a graph value, a physical device buffer, or runtime residency state. See the [Tensor API](api/tensor-api.md).

### Tensor descriptor

The planned immutable combination of model-level tensor facts such as data type, shape, layout, and gradient requirements. A descriptor describes a logical value and does not own host storage, device buffers, runtime residency, or a backend.

### `TensorId`

A validated non-negative identifier for public mutable tensor state. It belongs to the tensor's lifecycle and is distinct from graph-local node and value identities. A future publication binding can associate it with a compiled graph's result without storing graph-local IDs on the tensor. See [Identifiers](api/tensor-api.md#typed-identifiers).

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
| Public mutable API state | Immutable compile-time graph state |
| Can carry host storage, labels, gradient state, and provenance | Represents logical data flowing between graph nodes |
| Identified by `TensorId` | Identified by graph-local `ValueId` |
| Can participate in more than one separately compiled graph | Belongs to one owning graph context |
| Must not become runtime device residency | Must not be confused with a physical buffer or slot |

A publication binding connects the two domains when a compiled logical result must be exposed through public tensor identity.

### Node versus value

| Node | Value |
|---|---|
| A computation occurrence | Logical data consumed or produced by computation |
| Identified by `NodeId` | Identified by `ValueId` |
| Applies operation semantics | Carries an input, intermediate result, or output |
| May produce multiple values | May have multiple consumers |

### Operation kind versus attributes versus operation

| Concept | Meaning | Current status |
|---|---|---|
| `OperationKind` | Which backend-independent computation is meant | Interface implemented; concrete kinds planned |
| `OperationAttrs` | Immutable typed parameters that refine that meaning | Marker implemented; family-specific values planned |
| `NoOperationAttrs.INSTANCE` | Explicit parameter value for a kind with no parameters | Implemented canonical singleton |
| `Operation` | Immutable pairing of one kind with one caller-supplied `OperationAttrs` value | Implemented descriptor |

A kind distinguishes computations, while attributes carry parameters within a computation family. `Operation` stores both as one value but does not validate family compatibility. None of these values identifies where computation occurs in a graph; a [node](#node) represents that occurrence. Concrete kinds, family-specific attributes, and graph integration remain planned.

### Compile versus prepare versus run

| Stage | Main question | Produces | Must not do |
|---|---|---|---|
| Compile | What does the graph mean, and which backend owns each region? | Immutable compile artifacts | Allocate physical buffers or choose kernels |
| Prepare | How will each owned region execute? | Reusable prepared executables, memory plan, and schedule | Perform global graph optimization or per-run work |
| Run | Execute this prepared schedule with these inputs. | Per-run state and published results | Discover backends, lower partitions, or select kernels |

### Logical `ValueId` versus physical memory slot

A `ValueId` names logical data in one compiled graph. A memory slot names reusable physical storage in one prepared execution. Preparation may map values to slots, and different values may reuse a slot when their lifetimes do not overlap. The identifier therefore remains stable as a logical graph fact even when physical storage decisions change.
