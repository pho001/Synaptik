# Foundational module contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns the detailed responsibilities and prohibitions for Trace, Backend Contract,
Model, Config, and Planning. Model's scope includes the sole normative definition of graph
numerical profiles and profile-indexed allowed-result sets. Planning scope here is the module
boundary and ownership question; the detailed partition-scoring algorithm boundary belongs to the
compiler/autograd contract. This file does not own recurrent-scan semantics, compiler behavior,
execution lifecycles, concrete backend implementation, or extensions.

## Module responsibilities

### `modules/trace`

`modules/trace` owns typed diagnostic DTOs only.

Allowed:

- trace event envelopes
- typed compile payloads
- typed prepare payloads
- typed run payloads
- typed backend payloads
- trace-local IDs
- typed trace attributes

Forbidden:

- importing model
- importing planning
- importing compiler
- importing runtime
- importing prepare
- importing engine
- importing concrete backends
- graph traversal
- backend execution
- business logic
- runtime state

Trace must use typed DTOs.

`Map<String,String>` must not be used as the primary trace model.

Backend-specific details may use typed `TraceAttributes` as an escape hatch.

### `modules/backend-contract`

`modules/backend-contract` owns minimal backend identities and declarative requirements.

Allowed:

- `BackendId`
- `BackendDeviceId`
- `BackendAvailabilitySnapshot`
- `BackendRequirement`
- `DeviceClass`

Forbidden:

- kernel registry
- operation support logic
- backend prepare services
- executable units
- runtime storage
- physical buffers
- cost model implementation

Compile-time plans must hold backend identity, not live backend services.

Use `BackendId`, not concrete backend objects, in compile-time ownership and partitioning data.

### `modules/model`

`modules/model` owns the public tensor model, operation semantics, shape/data type/layout model, host storage abstraction, and immutable graph model.

Allowed:

- `Tensor`
- `TensorId`
- `TensorFactory`
- `DataType`
- `Shape`
- `LayoutDescriptor`
- `HostTensorStorage`
- `MemorySegmentStorage`
- `Operation`
- `OperationAttrs`
- `CompiledGraphModel`
- `CompiledNode`
- `GraphValue`
- `NodeId`
- `ValueId`
- `GraphPhase`
- `ForwardPublicationBinding`
- `TensorDescriptor`
- profile-indexed operation allowed-result sets

Forbidden:

- backend support
- `supportedBackends()`
- device residency
- runtime workspaces
- physical device buffers
- kernel selection
- backend-specific storage
- prepared execution
- runtime state

`Tensor` may retain its existing mutable borrowed host-storage association, but it must not own
gradient state or runtime device residency. Its identity, descriptor, and expression provenance
remain immutable.

Every derived expression producer owns one operation occurrence, its ordered input tensors, its
ordered output descriptors, and the canonical exact `Tensor` wrapper for every output slot.
`TensorFactory` constructs those wrappers atomically with their indexed provenance. A producer
must return the retained wrapper for a slot rather than reconstructing an equal wrapper. This
pre-capture model relationship is not graph IR, graph membership, or graph-local identity.

Device storage belongs to runtime/backend layers, not model.

### Numerical profiles

Model is the sole semantic owner of graph numerical profiles and the allowed-result set for every
operation under each profile. The profile names in this contract establish semantic identities;
they do not by themselves add a Java enum, configuration API, backend capability, runtime policy,
native schema, cache codec, or trace field.

`STRICT_IEEE` means the exact current per-operation Model contract. It preserves every existing
family-specific rounding, approximation, NaN-payload, signed-zero, accumulation, reassociation,
and fused-multiply-add promise or freedom. The name does not promise universal bitwise identity,
correct rounding, Java `strictfp`, a fixed instruction sequence, or cross-backend equality beyond
each operation's existing contract.

`ACCELERATOR` is an opt-in superset of the complete `STRICT_IEEE` allowed-result set. A backend may
always produce a strict result. For `FLOAT32`, it may additionally evaluate any current operation
through the three recursive floors below. Every other data type retains strict behavior. For one
operation occurrence and descriptor domain, strict capability and behavior must remain subsets of
accelerator capability and behavior.

#### Exact and discrete floor

Kind and attributes, input/output arity and descriptors, Shape/layout/axis mapping, contributor
membership, masks, indices, graph-RNG state transitions, traversal, casts, conversions, ordering,
stability, ties, empty-domain behavior, identities, divisors, and publication remain exact. A
selected stored value is one original candidate representation. Predicates gain no epsilon, and
approximation cannot choose an index, mask, winner, ordering, or state transition. A floating
comparison may compare DAZ-normalized operands, and an extrema operation may treat DAZ-normalized
values as tied; its existing NaN and signed-zero rules still apply and a selected result remains an
original candidate. Movement, copying, `CONTIGUOUS`, `WHERE`, classification, Boolean logic,
casts, indexing, ordering, arg-extrema, RNG state/masks, and max-pooling selection otherwise gain
no arithmetic relaxation from adjacent work.

#### Primitive `FLOAT32` floor

At each arithmetic primitive site named by the operation's Model formula, a subnormal operand may
be read as same-signed zero (denormals-are-zero, DAZ), and a finite subnormal site result may become
either signed zero (flush-to-zero, FTZ). These choices do not rewrite a stored input or another
observable value. Basic `ADD`, `SUB`, `MUL`, and `DIV` still perform one `FLOAT32`
round-to-nearest-even operation. A multiply and corresponding add with no Model-observable
intermediate may use one `FLOAT32` fused multiply-add; contraction never crosses graph nodes.

An elementary-function site — `POW`, logarithmic, exponential, error-function,
root/reciprocal-root, sigmoid/tanh, or a formula-named activation site — may return an ordinary
finite value at most five monotonically ordered `FLOAT32` representations from the correctly
rounded exact mathematical result after DAZ and before FTZ. Five ULP is the smallest single
ceiling covering current direct-route evidence: certified CPU primitive gates are one or two ULP
except `TANH` at five, while retained Metal `EXP` and `SIGMOID` evidence is one and two ULP. It is
not the historical four-ULP softmax output oracle or the looser composite ERF/GELU test tolerance.
The bound is a primitive-site construction rule, never a final-output `allclose` envelope.

Domain and special classes remain formula-derived. NaN cannot become ordinary. An ordinary finite
result cannot become NaN or infinity except through genuine overflow or a domain path after DAZ.
Required infinity and zero signs remain intact except for explicit DAZ/FTZ, the established
extrema tie rule, or an operation's existing final exact-zero publication freedom. Five ULP
applies only to an ordinary finite non-subnormal reference; a subnormal reference is exact or FTZ.

#### Aggregate floor

After exact mapping, masking, padding, bounds, and selection determine one logical cell, every
declared contributor participates exactly once. `FLOAT32` evaluation may choose any binary tree,
reassociate, round each arithmetic step to `FLOAT32`, use the primitive DAZ/FTZ rules, and fuse
only a corresponding multiply/add. Exact empty/point identities and mandatory divisors remain.
No evaluation may drop, duplicate, invent, pretruncate, or replace a term.

This floor applies recursively to ordinary and masked reductions, sum-to-Shape, scans,
overlap/scatter reductions, statistics and norms, loss reductions, matrix/convolution/attention/
recurrent contractions, and average pooling. Extrema and winner selection use the exact/discrete
floor. The established final exact-zero publication freedoms for qualifying SUM, MEAN,
SUM-to-Shape, and nonempty MATMUL results remain part of their Model formulas; they do not grant
intermediate or neighboring-operation sign freedom.

#### Composite inheritance

Normalization, activation, loss, attention, pooling, convolution, random/dropout, recurrent,
visible convenience composition, and Compiler-generated gradient formulas receive no separate
error envelope. Their exact guards run first, and their authoritative Model formulas recurse
through the exact/discrete, primitive, and aggregate floors. Saved outputs, statistics, masks, and
indices are exact stored values. An opaque vendor selector is eligible only when its complete
output set is proved to be a subset of this recursive set.

Neither profile permits reduced precision, reciprocal substitution for division, algebraic
identities absent from the Model formula, cross-node contraction, term loss, hidden state changes,
or tolerance-based predicate/selection changes. In particular, `maxFinite / maxFinite -> 0`,
`+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and `TANH(NaN) -> +1` remain invalid.
`FLOAT64`, `BFLOAT16`, future `FLOAT16`, integral values, and `BOOL` acquire no relaxed arithmetic
meaning.

The selected profile is graph-wide and cold. Model defines its result sets. A later Config change
may own only the immutable declarative selector, without semantic or route logic. Planning may
later ask profile-qualified capability questions but must not reinterpret a result set. Later
concrete-backend preparation may realize any result allowed by the selected profile and must fail
closed when it cannot; shared Prepare transports the selection without owning numerical meaning.
The selection must participate in backend route, specialization, generated-artifact, and tuning-
cache identity before relaxed capability is advertised. Runtime and Trace remain profile-free
unless a later coordinated architecture update explicitly changes their contracts.

### `modules/config`

`modules/config` owns declarative configuration only.

Allowed:

- `CompileConfig`
- `CompileMode`
- `BackendIntent`
- `GraphOptimizationConfig`
- `PartitionScoringConfig`
- `PartitionScoringPolicy`
- `PrepareConfig`
- `CpuPrepareConfig`
- `AcceleratorPrepareConfig`
- `RunOptions`
- `PublicationPolicy`
- immutable declarative planning-cost inputs, after their planning consumer is stable
- immutable declarative model-autotuning inputs, after their owning contracts are stable

Forbidden:

- live services
- concrete backend classes
- kernel class references
- runtime state
- executable units
- backend-specific implementation logic
- benchmark runners, model-autotuning search, cache mutation, or measurement algorithms
- live platform discovery or mutable measurement evidence

Backend-specific interpretation of config belongs to backend prepare.

### `modules/planning`

`modules/planning` owns backend-neutral compile-time planning.

Allowed:

- backend intent propagation
- capability query contracts
- capability matrix construction
- backend-neutral partition scoring
- node ownership decisions
- segment ownership decisions
- maximal same-owner partitioning
- logical materialization requirements
- logical memory requirements
- backend-neutral planning cost estimates and profiles

Forbidden:

- fusion implementation
- specialization
- concrete kernel selection
- OpenBLAS route selection
- Vector API route selection
- MPSGraph route selection
- CUDA kernel selection
- physical memory allocation
- runtime residency
- prepared schedules
- prepared executables
- backend-specific DAG construction
- backend-specific lowering
- runtime execution units
- concrete kernel/runtime scoring
- backend route names or route-selection parameters
- vector species, lane counts, unroll factors, thread counts, chunk sizes, or tile sizes

Planning answers:

```text
Where should this node or segment run?
```

Planning must not answer:

```text
Which concrete kernel, executable, BLAS route, MPSGraph route, or CUDA implementation should run it?
```

## Numerical-profile transport

Planning has a public API dependency on Config so each `OperationCapabilityQuery` carries the exact
non-null graph-wide `NumericalProfile`. Planning treats that value as query identity only; it does
not define allowed results, select a default, or infer a profile from backend or device facts.
