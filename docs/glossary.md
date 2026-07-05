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
arithmetic, transcendental, activation, and explicit fast-approximation meanings, plus matching
public floating unary Tensor expression construction with exact type/shape retention and one-input
provenance. The parameterized `ScalarElementwiseKind` vocabulary is implemented for scalar
`MUL`, `POW`, `CLAMP`, `CLAMP_MIN`, and `CLAMP_MAX`, together with exact-double
`ScalarValueAttrs` and `ClampRangeAttrs`, plus matching public floating Tensor expression
construction with exact type/shape retention, exact binary64 attributes, and one-input provenance.
The parameterless `BinaryComparisonKind` vocabulary is implemented for ordered `GREATER_THAN`,
`GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL` meanings, plus matching
public floating comparison Tensor expression construction with local broadcasting, fixed
non-differentiable BOOL descriptors, and ordered provenance. The parameterless
`BooleanLogicalKind` vocabulary is implemented for elementwise `AND`, `OR`, and `NOT` truth
meanings, plus matching public BOOL-only logical Tensor expression construction. Binary AND and OR
use local broadcasting and ordered provenance; unary NOT retains the exact input shape; and every
logical result has fixed non-differentiable BOOL descriptor facts.
The parameterless `WhereSelectionKind` vocabulary is implemented with the sole `WHERE` identity
and ordered condition, true-branch, and false-branch roles, plus matching static public
`Tensor.where` expression construction. The method requires an exact BOOL condition, promotes two
floating branches, composes branch-first and condition-second local broadcasts, derives an
unresolved result with branch-only gradient eligibility, and records exact three-input provenance.
Value selection, gradient routing and rules, compiler capture, ONNX/backend execution, and
scalar-index `select` remain separate or planned concerns.
The parameterized `CastKind` vocabulary is implemented with the sole `CAST` identity, together
with `CastAttrs` carrying one exact non-null target `DataType`. All six current data types are
representable targets, and public `Tensor.cast` now creates a fresh explicit storage-free
expression for all 36 source/target pairs. It retains the exact input Shape, leaves layout
unresolved, preserves a true gradient request only for floating-to-floating casts, and records
typed target attributes plus exact one-input provenance. Numerical conversion behavior, gradient
rules, compiler capture and canonicalization, and backend execution remain planned or separately
owned.
The `AggregateReductionKind` vocabulary is implemented for `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`,
`ALL`, `ANY`, and `ARG_MAX`, together with normalized single-axis `AxisReductionAttrs`, explicit
full-form `NoOperationAttrs.INSTANCE`, `ArgMaxAttrs`, `ArgMaxTiePolicy`, and masked SUM/MEAN
`MaskedReductionAttrs`. The masked attributes preserve an ordered mask-dimension-to-input-axis
mapping and fixed all-false zero semantics. Public floating
`Tensor.sum`, `mean`, `prod`, reduction `min`, and reduction `max` now construct full,
axis-removing, and retained-axis expressions with locally derived shapes and one-input provenance.
The masked `sum(axis, mask)` and `mean(axis, mask)` overloads now resolve their ordered Shape
mapping locally, remove the selected axis, and record exact `[input, mask]` provenance.
Aggregate `MIN`/`MAX` remain typed separately from equally named binary elementwise kinds. Boolean
`Tensor.all` and `Tensor.any` now provide the corresponding exact-BOOL expressions with false
gradient eligibility and one-input provenance; aggregate `ALL`/`ANY` remain typed separately from
elementwise `AND`/`OR`. Axis-only `Tensor.argMax` now accepts floating or integral input and
produces fixed non-differentiable INT64 expressions with explicit first- or last-index policy.
Numerical or truth evaluation, empty-domain behavior, extrema comparison and tie execution,
gradients, compiler capture, backend support, and execution remain planned.
The `CumulativeSumKind` vocabulary is implemented with the sole `CUM_SUM` semantic identity,
together with `CumulativeSumAttrs` carrying one normalized axis and exact exclusive/reverse mode
flags. Public `Tensor.cumSum` now constructs all four traversal/inclusion modes for floating and
integral inputs, retaining exact Shape/type/eligibility metadata in an unresolved descriptor and
recording one-input provenance. Value accumulation, gradients, compiler behavior, and execution
remain planned.
Other concrete kind families and expression families, their family attributes, random Operations,
typed access and export, native/runtime/backend allocation,
gradient and publication behavior, compiler entry points and artifacts, planning, prepare,
runtime, concrete backends, traces, and training remain architecture or planning contracts. A
definition explains intended meaning; it is not by itself evidence that a Java type exists.

## Terms

### Aggregate reduction

A computation that combines values from a selected reduction domain into fewer logical positions.
The implemented `AggregateReductionKind` vocabulary includes numeric `SUM`, `MEAN`, `PROD`,
`MIN`, and `MAX`, boolean `ALL` and `ANY`, and index-producing `ARG_MAX`. An ordinary full
reduction selects every input axis and uses `NoOperationAttrs.INSTANCE`; an ordinary single-axis
reduction uses `AxisReductionAttrs`. Representing the full form through parameter absence avoids a
negative numeric all-axis sentinel.

A masked, axis-removing `SUM` or `MEAN` instead uses `MaskedReductionAttrs`. Its immutable mapping
states which ordered input axis receives each mask dimension. Public `sum(axis, mask)` and
`mean(axis, mask)` resolve that mapping from the input and mask Shapes, remove the normalized
reduction axis, and use exact provenance order `[input, mask]`. Equal dimensions are compatible,
and a static singleton mask dimension may align to any input dimension. Resolution prefers a
mapping that contains the reduction axis, then minimum positional displacement, then
lexicographic axis order. False mask positions are excluded. Selecting no values produces zero
for masked sum; masked mean divides by the selected true-count and also produces zero when that
count is zero. Expression construction records this meaning but performs no storage alignment,
value selection, counting, aggregation, division, gradient work, or execution.

For a single-axis form, `keepDimensions == false` requests removal of the selected axis, while
`true` requests retaining it with extent one. `ARG_MAX` instead uses `ArgMaxAttrs` because its tie
policy is an intrinsic semantic parameter, and it has no full form in the current contract. These
semantic types describe requested meaning only. The current public `sum`, `mean`, `prod`,
reduction `min`, and reduction `max` methods add floating input eligibility, local result-shape
derivation, exact type and gradient-eligibility retention, and provenance. Aggregate extrema use
one input and `AggregateReductionKind.MIN` or `AggregateReductionKind.MAX`; binary elementwise
extrema use two ordered inputs and the distinct `BinaryArithmeticKind` constants. The aggregate
methods still define no numerical or empty-domain policy, comparison, NaN or signed-zero handling,
extrema-tie gradient rule, or executable behavior. Public `all` and `any` instead require exact
BOOL input and produce exact BOOL with false gradient eligibility, using the same shape and
provenance rules. They do not inspect truth values or define empty-domain identities. Aggregate
ALL/ANY use one input and `AggregateReductionKind`; elementwise AND/OR use two ordered inputs and
the distinct `BooleanLogicalKind` constants. Public `argMax` accepts floating or integral input,
normalizes one axis, and produces exact INT64 with false gradient eligibility. Its convenience
forms explicitly supply `FIRST_INDEX`; the complete form retains the caller's exact non-null
policy. It does not compare values, select an index, or define empty-axis behavior. See [Numeric
aggregate expressions](api/tensor-api.md#numeric-aggregate-expressions) and [Aggregate reduction semantic
kinds and attributes](api/tensor-api.md#aggregate-reduction-semantic-kinds-and-attributes), plus
[Boolean aggregate expressions](api/tensor-api.md#boolean-aggregate-expressions) and [Arg-max
expressions](api/tensor-api.md#arg-max-expressions). The masked forms are described separately
under [Masked sum and mean expressions](api/tensor-api.md#masked-sum-and-mean-expressions).

### Arg-max tie policy

The implemented `ArgMaxTiePolicy` choice for selecting a logical axis index when several values
share a maximum. `FIRST_INDEX` requests the smallest logical index, and `LAST_INDEX` requests the
largest. A logical index is a position along the selected axis rather than a physical storage
offset. `ArgMaxAttrs` requires an explicit non-null policy; the semantic value does not supply a
default. The public `Tensor.argMax` convenience overloads supply `FIRST_INDEX` explicitly, while
the complete overload retains an explicit caller policy. Equality, NaN, comparison, and
empty-axis behavior remain later numerical and execution contracts.

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

### Cast expression

An implemented public `Tensor.cast(targetDataType)` request that records elementwise conversion
metadata without converting a value. Every current source/target data-type pair is representable.
The result is a fresh unlabeled, storage-free [`Tensor`](#tensor), including when source and target
types are equal. It retains the input descriptor's exact [`Shape`](#shape), leaves layout
unresolved, and records `CastKind.CAST`, `CastAttrs(targetDataType)`, and the exact input as its
sole [provenance](#provenance) reference.

Gradient eligibility survives only when it was already requested and both source and target are
floating. That descriptor fact is not a gradient rule or backend differentiability promise.
Numerical conversion, redundant-cast and cast-chain canonicalization, autograd expansion, backend
support, and execution belong to later owning layers. A cast expression is therefore not converted
storage, a compiler graph node, or proof that a requested conversion can execute.

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

### Cumulative sum / scan

An ordered one-input operation that cumulatively adds values along one axis while preserving one
output position for every input position. The implemented `CumulativeSumKind.CUM_SUM` identifies
this meaning, and `CumulativeSumAttrs(axis, exclusive, reverse)` carries an already normalized
non-negative axis plus the two independent mode choices.

For logical input `[1, 2, 3]`, inclusive forward produces `[1, 3, 6]` from `1`, `1 + 2`, and
`1 + 2 + 3`. Exclusive forward produces `[0, 1, 3]` from the empty prefix, `1`, and `1 + 2`.
Inclusive reverse produces `[6, 5, 3]` from `1 + 2 + 3`, `2 + 3`, and `3`. Exclusive reverse
produces `[5, 3, 0]` from `2 + 3`, `3`, and the empty reverse prefix. Reverse changes traversal
direction, not output order.

Public `Tensor.cumSum(axis)` selects inclusive forward mode, and the complete overload preserves
explicit exclusive and reverse flags. Both validate numeric input, normalize one caller axis,
retain exact Shape/type/gradient-eligibility metadata with unresolved layout, and record exact
one-input provenance in a fresh unlabeled, storage-free Tensor. They calculate none of the example
values and define no accumulation, gradient, compiler, backend, or execution behavior. See
[Cumulative-sum semantic kind and attributes](api/tensor-api.md#cumulative-sum-semantic-kind-and-attributes).

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

### Masked reduction

An aggregate `SUM` or `MEAN` that excludes input positions whose aligned boolean mask position is
false. The implemented `MaskedReductionAttrs` semantic value stores one already normalized,
non-negative reduction axis and an immutable [mask-to-input axis mapping](#mask-to-input-axis-mapping).
The selected public baseline removes the reduction axis. A masked sum with no selected values is
zero. A masked mean divides by the number of true selected positions for each output and is zero
when that count is zero.

Public masked Tensor expressions are also current. They validate floating input and an exact BOOL
mask, resolve a deterministic ordered mapping from the two Shapes, remove the normalized reduction
axis, preserve input type and gradient eligibility, and record exact `[input, mask]` provenance.
Storage alignment, value selection, counting, aggregation, division, gradient rules, compiler
capture, backend behavior, and numerical execution remain planned.

### Mask-to-input axis mapping

The ordered structural mapping stored by `MaskedReductionAttrs`. Element
`maskInputAxes[i]` names the zero-based input axis aligned with mask dimension `i`. Values must be
non-negative and strictly increasing, which preserves mask-dimension order and prevents two mask
dimensions from claiming one input axis. Omitted input axes are implicit broadcast dimensions;
an empty mapping represents a scalar mask. For example, `[0, 1]` maps mask `[batch, time]` onto
the first two axes of input `[batch, time, features]` and leaves the features axis implicit.

The mapping value does not contain either Shape. It therefore cannot prove input-rank bounds,
dimension compatibility, or that one mapping should be selected over another. The public masked
expression methods separately own deterministic resolution from concrete input and mask Shapes.

### Memory slot

A position in a prepared memory plan for a physical buffer or workspace used during execution. Slots let a prepared schedule refer to reusable storage without embedding raw addresses. A memory slot is a physical execution resource and must not be confused with a logical [`ValueId`](#valueid).

### `NoOperationAttrs`

The implemented canonical immutable attribute value for an operation kind that has no semantic parameters. It is a single-value enum whose only value is `NoOperationAttrs.INSTANCE`. The singleton makes “no parameters” explicit and non-null rather than representing absence with `null`, an empty map, or a newly allocated placeholder. See [`OperationAttrs`](#operationattrs).

### Normalized axis

A non-negative axis index in the range established by a tensor's rank. The implemented
`Shape.normalizeAxis` method accepts a caller-facing positive or negative axis and returns this
form. Reduction attributes store only an already normalized non-negative `int`; they do not retain
a Shape or prove that the index exists for a particular input. A negative stored axis is invalid,
and no negative value is reused as an all-axis sentinel. Current `sum`, `mean`, `prod`, reduction
`min`, reduction `max`, boolean `all`, and boolean `any` axis methods normalize against the input
Shape before constructing semantic attributes. The current axis-only `argMax` methods use the
same boundary before constructing `ArgMaxAttrs`.
`MaskedReductionAttrs` follows the same normalized-axis boundary, and current public masked
expression construction resolves that axis and its mask mapping before creating the attributes.
`CumulativeSumAttrs` also stores only an already normalized non-negative axis; current public
Shape-aware `Tensor.cumSum` construction normalizes the caller axis before creating it.

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

The implemented zero-method marker contract for immutable, typed parameters that refine which
computation an [`Operation`](#operation) describes. Implemented scalar-family values are
`ScalarValueAttrs`, which holds one exact Java `double`, and `ClampRangeAttrs`, which holds exact
ordered inclusive lower and upper bounds. Implemented cast-family `CastAttrs` holds one exact
non-null target `DataType` without duplicating a source type. Implemented reduction-family
`AxisReductionAttrs` holds one normalized axis and retained-dimension choice, while `ArgMaxAttrs`
adds an explicit tie policy and `MaskedReductionAttrs` holds one reduction axis plus an immutable
ordered mask-to-input axis mapping. Implemented scan-family `CumulativeSumAttrs` holds one
normalized axis plus exact exclusive and reverse flags. Other families may define records for
padding or another operation-specific value. Implementations use typed fields, defensively
isolate mutable inputs, and provide structural equality and hashing; they do not use a primary
string-keyed map or contain backend, compiler-service, mutable tensor, storage, or runtime state.
Kinds without parameters use [`NoOperationAttrs.INSTANCE`](#nooperationattrs). The marker
identifies the role of a value but does not enforce immutability at runtime.

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
gradient, execution, or backend support. The implemented public unary Tensor methods consume these
values while separately owning local expression construction.

The third production family is `ScalarElementwiseKind`, an enum containing exactly `MUL`, `POW`,
`CLAMP`, `CLAMP_MIN`, and `CLAMP_MAX`. These values identify parameterized one-input elementwise
meanings. `MUL`, `POW`, `CLAMP_MIN`, and `CLAMP_MAX` pair with `ScalarValueAttrs`; `CLAMP` pairs
with `ClampRangeAttrs`. The scalar values are attributes rather than additional Tensor inputs, and
the generic `Operation` descriptor does not enforce family compatibility. The attributes retain
exact Java `double` values. Clamp-range construction rejects only a primitive
`minValue > maxValue` comparison, so equal bounds, both signed-zero orderings, ordered infinities,
and NaN endpoints are valid. The enum and attributes perform no Tensor expression construction,
descriptor inference, numerical execution behavior, gradients, or backend support by themselves.
The implemented public scalar Tensor methods consume these values while separately owning
floating validation, descriptor derivation, exact attribute composition, and one-input
provenance.

The fourth production family is `BinaryComparisonKind`, an enum containing exactly
`GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL`. These
values identify ordered, parameterless tensor-to-tensor comparison meanings and compose with
`NoOperationAttrs.INSTANCE`. The enum stores no operands, broadcast geometry, BOOL result facts,
numeric edge policy, or execution metadata. The implemented public comparison Tensor methods
consume these values while separately owning floating input validation, local broadcasting, fixed
BOOL result derivation, and ordered provenance. Numerical comparison policy, compiler capture,
gradients, execution, and backend support remain planned. Its inherited names are diagnostic
rather than serialization or dispatch keys, and an equally named kind from another family remains
a different typed value.

The fifth production family is `BooleanLogicalKind`, an enum containing exactly `AND`, `OR`, and
`NOT`. These values identify parameterless elementwise boolean conjunction, disjunction, and
negation and compose with `NoOperationAttrs.INSTANCE`. `AND` and `OR` have two logical input roles;
`NOT` has one. Those roles are family context rather than stored or generically validated arity
metadata. The enum itself defines no BOOL descriptor eligibility, binary broadcasting, unary shape
preservation, provenance, storage representation, numeric truthiness, gradient, execution, or
backend support. The implemented public logical Tensor methods separately own exact BOOL input
validation, binary broadcast or unary shape rules, fixed BOOL results, and provenance. Its
inherited names are diagnostic rather than serialization or dispatch keys, and an equally named
kind from another family remains a different typed value.

The sixth production family is `WhereSelectionKind`, an enum containing exactly `WHERE`. This
parameterless value identifies elementwise conditional choice with three ordered logical roles:
condition, true branch, and false branch. A true condition chooses the corresponding true-branch
value; otherwise it chooses the false-branch value. The roles are ternary family context rather
than stored or generically validated arity metadata, and the kind composes with
`NoOperationAttrs.INSTANCE`. It is distinct from scalar-index `select`, gather, take, and scatter.
The enum itself defines no Tensor construction, condition or branch eligibility, promotion,
three-way broadcasting, result descriptor, provenance, evaluation order, gradient, compiler,
ONNX, execution, or backend-support behavior. Static `Tensor.where` separately owns local
BOOL/floating validation, branch promotion, ordered pairwise broadcasting, descriptor derivation,
and exact three-input provenance. Its inherited name is diagnostic rather than a serialization or
dispatch key, and an equally named kind from another family remains a different typed value.

The seventh production family is `CastKind`, an enum containing exactly `CAST`. This value
identifies parameterized elementwise conversion of one logical input and pairs with `CastAttrs`,
whose sole component is the exact non-null target `DataType`. Every current data type is a valid
target. The source type remains a fact of the later input descriptor rather than duplicated
attribute state, and generic `Operation` does not enforce the family pairing. The kind and
attributes alone define no source compatibility, same-type handling, result descriptor, numerical
conversion rules, gradients, provenance, compiler capture, execution, or backend support. The
implemented public `Tensor.cast` method separately owns fresh expression construction, exact shape
retention, unresolved layout, floating-only eligibility retention, and one-input provenance. Their
text forms are diagnostic rather than serialization or dispatch contracts.

The eighth production family is `AggregateReductionKind`, an enum containing exactly `SUM`,
`MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, `ANY`, and `ARG_MAX`. The first seven ordinary kinds pair
with `NoOperationAttrs.INSTANCE` for a full reduction over every input axis or with
`AxisReductionAttrs` for one already normalized axis. `ARG_MAX` pairs with `ArgMaxAttrs`, which
adds an explicit `FIRST_INDEX` or `LAST_INDEX` tie policy. Generic `Operation` does not enforce
these family pairings. Masked axis-removing `SUM` and `MEAN` pair with
`MaskedReductionAttrs`, whose strictly increasing list aligns mask dimensions to input axes and
whose semantic contract fixes false-value exclusion and zero output when the selected count is
zero. The family stores no Tensor input, result descriptor, negative all-axis
sentinel, numerical or empty-domain policy, gradient rule, executable behavior, or backend
support. Public `sum`, `mean`, `prod`, reduction `min`, and reduction `max` separately own
floating eligibility, while public `all` and `any` own exact BOOL eligibility and fixed false
gradient eligibility. All seven ordinary families own axis normalization, result-shape derivation,
and one-input provenance. Public `argMax` separately owns floating-or-integral eligibility, fixed
INT64 false-gradient results, explicit tie policy, axis-only shape derivation, and one-input
provenance. Public masked `sum` and `mean` separately own floating/BOOL validation, deterministic
ordered Shape mapping, axis-removing result derivation, exact input type and gradient eligibility,
and `[input, mask]` provenance.

The ninth production family is `CumulativeSumKind`, an enum containing exactly `CUM_SUM`. It
identifies one-input cumulative addition along the already normalized axis in
`CumulativeSumAttrs`. The attributes also select inclusive or exclusive output and forward or
reverse traversal. The family preserves logical positions; reverse traversal does not reverse
output order, and exclusive traversal emits additive zero at its first visited position. Generic
`Operation` does not enforce the kind/attributes pairing. The kind and attributes contain no
Tensor, Shape, result descriptor, provenance, data-type policy, gradient rule, algorithm,
executable behavior, or backend support. Public `Tensor.cumSum` separately owns numeric
validation, Shape-aware axis normalization, descriptor construction, fresh identity, and
one-input provenance without accumulating values.

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
receiver as left and argument as right. Binary comparison methods use the same exact ordered input
contract with `BinaryComparisonKind`, including for symmetric equality and inequality, while
producing a storage-free non-differentiable BOOL result. Boolean logical AND and OR likewise
retain exact ordered receiver/argument inputs with `BooleanLogicalKind`; NOT retains exactly the
receiver as its one input. The implemented floating unary methods use the exact matching
`UnaryElementwiseKind` and canonical no-attributes value, also with exactly the receiver as their
one input. Static `Tensor.where` uses `WhereSelectionKind.WHERE` and retains exact ordered inputs
`[condition, ifTrue, ifFalse]`, including repeated branch references. These construction paths do
not change provenance's general role or make it graph membership. `Tensor.cast` uses
`CastKind.CAST`, retains its exact target in `CastAttrs`, and records exactly the receiver as its
one immediate input, including in same-type and chained requests. `Tensor.cumSum` uses
`CumulativeSumKind.CUM_SUM`, retains its normalized axis and exact mode flags in
`CumulativeSumAttrs`, and likewise records exactly the receiver as its sole input.

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
operation semantics plus ordered provenance. The current `greaterThan`, `greaterOrEqual`,
`lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo` methods also accept ordered floating pairs
and broadcast shapes, but create fixed BOOL descriptors with false gradient eligibility while
retaining exact comparison semantics and ordered provenance. The current `logicalAnd` and
`logicalOr` methods accept only exact BOOL inputs, broadcast their shapes, retain ordered
provenance, and create fresh fixed BOOL results. `logicalNot` also requires exact BOOL, but retains
the exact input Shape and one-input provenance without broadcasting. All logical results have
unresolved layout, false gradient eligibility, no label, and no storage, and construction does not
read truth bytes, short-circuit, simplify, or execute them. Static `Tensor.where` requires an exact
BOOL condition and two floating branches. It promotes the branch types, broadcasts the branches
before combining the condition shape, creates a fresh unresolved storage-free result with
branch-only gradient eligibility, and records exact ordered
`[condition, ifTrue, ifFalse]` provenance. It does not inspect values, choose or evaluate a branch,
define gradient routing, capture a graph, or execute selection. The current `cast` method accepts
all current source/target pairs and creates a fresh explicit result even for a same-type request. It
retains the exact input Shape, leaves layout unresolved, retains gradient eligibility only across
an already-eligible floating-to-floating cast, and records typed target attributes plus exact
one-input provenance. It does not inspect or convert values/storage, define numerical or gradient
rules, canonicalize casts, capture a graph, or execute conversion. The current fifteen
zero-argument unary methods also create fresh floating expression tensors. They retain the exact
input data type and Shape,
leave layout unresolved, preserve gradient eligibility, and record the matching parameterless kind
plus exactly one input reference without domain checks or canonicalization. The current scalar
`mul`, scalar `pow`, `clamp`, `clampMin`, and `clampMax` methods likewise create fresh floating
one-input expressions. They retain the exact type and Shape, preserve gradient eligibility, and
store exact binary64 attributes without conversion or canonicalization; range clamp remains one
first-class `CLAMP` operation. Other expression families, gradient rules and objects, trainable
role, publication behavior, compiler integration, device buffers, numerical execution, and
runtime residency remain planned. The current `sum`, `mean`, `prod`, reduction `min`, and
reduction `max` methods create floating full or single-axis aggregate expressions. Current `all`
and `any` create exact-BOOL, non-differentiable forms. Full forms produce canonical rank-zero
shape; axis forms normalize, then remove or retain the selected axis with extent one. Results
preserve the family-specific type and eligibility, leave layout unresolved, and record one-input
provenance without aggregating, comparing, or evaluating values or defining empty-domain,
tie-gradient, compiler, backend, or executable behavior. Current `argMax` methods accept one axis
of floating or integral input and create fixed INT64, non-differentiable results. Convenience
forms request the first equal maximum; the complete form retains an explicit policy. They perform
no comparison or actual index selection and define no NaN, equality, or empty-axis behavior.
A masked `sum(axis, mask)` or `mean(axis, mask)` requires an exact BOOL mask, resolves an ordered
mapping from mask dimensions to input axes, removes the selected axis, and records exact
`[input, mask]` provenance. It preserves input type and gradient eligibility but does not align
storage, select or aggregate values, count true positions, divide, or define a gradient rule.
A `cumSum` request accepts floating or integral input, preserves its exact Shape, data type, and
gradient eligibility in an unresolved descriptor, and records one normalized axis plus exact
exclusive/reverse flags. Each valid call is fresh and storage-free; construction performs no
addition and defines no gradient, compiler, backend, or execution behavior.
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
descriptor-based leaf construction, immutable provenance, and floating binary arithmetic,
comparison, unary, and scalar Tensor expression construction are implemented, as is BOOL-only
logical expression construction, explicit cast expression construction, floating numeric
aggregate expression construction for sum, mean, product, minimum, and maximum, and BOOL
aggregate expression construction for all and any. Axis-only index-producing construction for
arg-max and shape-preserving cumulative-sum construction are also implemented. Gradient rules and
objects, compiler graph capture, truth or
numerical execution, and publication behavior are not part of the current Tensor contract.

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
| `OperationKind` | Which backend-independent computation is meant | Interface plus binary arithmetic, binary comparison, boolean logical, conditional selection, unary elementwise, scalar elementwise, cast, aggregate reduction, and cumulative-sum scan families implemented; other families planned |
| `OperationAttrs` | Immutable typed parameters that refine that meaning | Marker plus scalar-value, clamp-range, cast-target, ordinary reduction-axis, arg-max, masked-reduction, and cumulative-sum values implemented; other family-specific values planned |
| `NoOperationAttrs.INSTANCE` | Explicit parameter value for a kind with no parameters | Implemented canonical singleton |
| `Operation` | Immutable pairing of one kind with one caller-supplied `OperationAttrs` value | Implemented descriptor |

A kind distinguishes computations, while attributes carry parameters within a computation family.
`Operation` stores both as one value but does not validate family compatibility. None of these
values identifies where computation occurs in a graph; an implemented [node](#node) represents
that occurrence. Binary arithmetic, binary comparison, boolean logical, conditional selection,
unary elementwise, scalar elementwise, cast, and aggregate reduction kinds are implemented.
Arithmetic, unary, scalar, and comparison public Tensor construction paths are also implemented,
together with boolean logical, conditional-selection, cast, and
sum/mean/product/minimum/maximum/all/any/arg-max aggregate Tensor construction, including masked
sum/mean. Cumulative-sum semantics and public Tensor construction are also implemented. Other
concrete families and their family-specific attributes, compiler capture, and execution remain
planned.
The compiled graph container is implemented model state.

### Compile versus prepare versus run

| Stage | Main question | Produces | Must not do |
|---|---|---|---|
| Compile | What does the graph mean, and which backend owns each region? | Immutable compile artifacts | Allocate physical buffers or choose kernels |
| Prepare | How will each owned region execute? | Reusable prepared executables, memory plan, and schedule | Perform global graph optimization or per-run work |
| Run | Execute this prepared schedule with these inputs. | Per-run state and published results | Discover backends, lower partitions, or select kernels |

### Logical `ValueId` versus physical memory slot

A `ValueId` names logical data in one compiled graph. A memory slot names reusable physical storage in one prepared execution. Preparation may map values to slots, and different values may reuse a slot when their lifetimes do not overlap. The identifier therefore remains stable as a logical graph fact even when physical storage decisions change.
