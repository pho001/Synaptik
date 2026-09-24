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

`ACCELERATOR` is an opt-in superset of the `STRICT_IEEE` allowed-result set. A backend may always
produce a strict result. It may produce an additional result only through the operation-specific
`FLOAT32` transformation in the following table. All unlisted data types and operations retain
their `STRICT_IEEE` allowed-result set.

Where a row permits DAZ/FTZ, DAZ means that a declared arithmetic `FLOAT32` subnormal input may be
interpreted as same-signed zero, and FTZ means that a finite subnormal arithmetic result may be
flushed to zero with either sign. These permissions affect only the arithmetic evaluation named by
that row; they do not rewrite a stored input or an observable value outside it.

This is the sole normative operation-family table for numerical-profile result sets:

| Operation family | Additional `ACCELERATOR` allowed results for `FLOAT32` |
|---|---|
| Binary/scalar `ADD`, `SUB`, `MUL`, `DIV` | A declared arithmetic subnormal input may be interpreted as same-signed zero (denormals-are-zero, DAZ); a finite subnormal arithmetic result may be flushed to zero (flush-to-zero, FTZ). An FTZ zero may use either sign. An exact-zero `ADD` or `SUB` result may use either zero sign. `MUL` and `DIV` otherwise retain the sign required by the selected operands. |
| `SUM`, `MEAN`, `SUM_TO_SHAPE` | Use all and only the declared terms for each output coordinate in a binary tree with `FLOAT32` per-step rounding and this row's DAZ/FTZ rules. Reordering or reassociation may change the tree but may not drop, duplicate, or invent a term. Masking and target-Shape mapping still determine the declared terms. `MEAN` divides the selected sum by the declared positive count. |
| `MATMUL`, `CONV2D`, `CONV3D` | Retain the current family-owned reassociation and FMA permission and additionally allow this row's DAZ/FTZ. An FMA may contract only a corresponding multiply and add in the declared contraction; it may not fuse arbitrary graph nodes or erase an observable intermediate value. |
| Floating comparisons | Compare operands after this row's DAZ normalization. Preserve the current NaN and signed-zero truth tables and canonical `BOOL` result. |
| Binary/scalar/reduction `MIN` and `MAX` | NaN still propagates. Opposite-zero ties may return either zero sign. Values equal after DAZ normalization may return either original operand bit pattern; unequal normalized values retain numeric ordering. |
| `CUM_SUM`, `CUM_PROD` | Preserve axis, direction, traversal, inclusive/exclusive placement, and the exact exclusive positive-zero or positive-one identity. Only arithmetic steps may use this row's DAZ/FTZ, and an FTZ zero may use either sign. |
| Affine/layout movement, `CONTIGUOUS`, `WHERE`, classification, Boolean logic, casts, indexing, ordering, and arg-extrema | No relaxation. Preserve their current bit, conversion, truth, selected-value, index, and tie contracts. |
| Unlisted unary/transcendental, normalization, loss, attention, pooling, scatter/fold, random, and recurrent families | No relaxation. A later coordinated architecture update must give a named operation its own finite-domain and special-value envelope before any backend may use `ACCELERATOR` to widen capability. |

For one fixed permitted evaluation, NaN sign, payload, and quiet/signaling representation may vary
only where the current operation has no stronger promise. NaN may not become an ordinary value,
and an ordinary value may not become NaN, except through a genuine IEEE exceptional path created
by one listed DAZ, reassociation, or FMA choice. Transcendentals receive no generic epsilon or ULP
budget.

The following exclusions are mandatory under both profiles:

- no term dropping, reciprocal substitution, algebraic identity rewrite, arbitrary reduced
  precision, cross-node contraction, or tolerance-based acceptance
- `maxFinite / maxFinite -> 0` and `+infinity / maxFinite -> NaN` are invalid; the selected operands
  still require one and positive infinity, respectively
- `RELU(NaN) -> +0` and `TANH(NaN) -> +1` are invalid; zero-sign freedom never changes NaN
  classification
- `CAST`, affine/layout movement, `CONTIGUOUS`, and selected `WHERE` values acquire no DAZ/FTZ or
  NaN-payload freedom from surrounding arithmetic
- `FLOAT64`, `BFLOAT16`, future `FLOAT16`, integral values, and `BOOL` acquire no relaxed arithmetic
  meaning; future IEEE `FLOAT16` semantics remain independently owned

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
