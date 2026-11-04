# Task 0006A: Portable Pad, Tile, and [Tensor](../../../../glossary.md#tensor)-Composition Movement

## Status

Complete

## Goal

Add the next bounded [CPU](../../../../glossary.md#term-cpu) portable family after completed CPU 0006: one fully static,
resolved-[layout](../../../../glossary.md#layout) [occurrence](../../../../glossary.md#term-occurrence) of [`PAD`](../../../../glossary.md#padding), `TILE`, `CONCAT`, or `STACK` lowers to one CPU-private
non-affine data-movement unit, one generated [scalar](../../../../glossary.md#term-scalar) or parallel-scalar [artifact](../../../../glossary.md#term-artifact), and one
[partition](../../../../glossary.md#partition)-level executable.

This task is the first executable [slice](../../../../glossary.md#slice) of the former broad “non-affine movement, [gather](../../../../glossary.md#gather), and
one-hot” row. The row is split because static pad/tile/composition mapping, sliding-window
extraction, and value-dependent indexing need different [lowering](../../../../glossary.md#lowering) state, [validation](../../../../glossary.md#term-validation), and failure
proofs. CPU 0006A establishes only the compact static movement and [variadic](../../../../glossary.md#term-variadic) [binding](../../../../glossary.md#term-binding) foundation.
Draft CPU 0006A1 adds `UNFOLD_AXIS` and both `UNFOLD2D` variants. Draft CPU 0006A2 then adds
`GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, and `ONE_HOT` with complete execution-time index
prevalidation. Only this task has a detailed specification.

[Mental model](../../../../glossary.md#term-mental-model):

```text
one static PAD, TILE, CONCAT, or STACK occurrence
  -> family-specific CPU lowering and compact cold geometry
  -> unique input-value declarations in first-occurrence order + one output declaration
  -> shared CPU-blind slot assignment
  -> one scalar-compute generated artifact
  -> direct invocation or deterministic disjoint parallel ranges
```

## [Scope](../../../../glossary.md#term-scope)

### Exact semantic and [occurrence](../../../../glossary.md#term-occurrence) [matrix](../../../../glossary.md#term-matrix-meanings)

- Admit exactly `PadKind.PAD` with `PadAttrs`, `TileKind.TILE` with `TileAttrs`, and
  `TensorCompositionKind.CONCAT` or `STACK` with `CompositionAxisAttrs`.
- Admit exactly one [node](../../../../glossary.md#node) and one output per [CPU](../../../../glossary.md#term-cpu) [partition](../../../../glossary.md#partition). [`PAD`](../../../../glossary.md#padding) and `TILE` have one ordered input.
  `CONCAT` and `STACK` have between one and sixteen ordered input occurrences. The sixteen-input
  ceiling bounds generated entry descriptors, cold geometry, branch count, tests, and code size;
  a larger valid Model occurrence remains unadvertised and fails before artifact access.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Require every projected input and output Shape to be fully static and every layout to be
  resolved. Validate the exact current Model relationship rather than normalizing caller syntax,
  repairing descriptors, choosing a physical layout, or binding a symbolic extent.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [binding glossary](../../../../glossary.md#term-binding); [extent glossary](../../../../glossary.md#term-extent))</span>
- For `PAD`, require `before` and `after` cardinality to equal input rank, require the exact typed
  constant to match the input/output data type, and require each output extent to equal the
  checked sum <span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [rank glossary](../../../../glossary.md#term-rank); [constant glossary](../../../../glossary.md#term-constant); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`before + input + after`.
- For `TILE`, require repeat cardinality to equal input rank and each output extent to equal the
  checked product `input * repeat`. Complete input patterns repeat; individual scalar values do
  not become adjacent runs unless that follows from the Shape.
- For `CONCAT`, require same data type and rank, exact equality on every non-concat Dimension, a
  normalized existing axis, and the checked encounter-order sum of selected input extents.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis))</span>
- For `STACK`, require same data type and exact Shape for every ordered input, a normalized result
  insertion axis, and an output Shape that inserts the exact input-occurrence count.
- Accept [scalar](../../../../glossary.md#term-scalar) and zero-[element](../../../../glossary.md#term-element) [Shapes](../../../../glossary.md#shape) wherever the [Model](../../../../glossary.md#term-model) contract admits them. Checked `long`
  element, extent, offset, stride, span, and byte arithmetic must fail before artifact access.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride))</span>
- Accept all six current Model data types. Movement copies represented bits without conversion:
  FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL use the seven carrier forms completed by CPU
  0006. <span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`PAD` emits the exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) bits, including NaN payload, signed zero, BFLOAT16 raw
  bits, integral bits, and canonical BOOL. This adds no numerical, conversion, or vector semantics.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [canonical glossary](../../../../glossary.md#term-canonical-form); [vector glossary](../../../../glossary.md#term-vector))</span>

### [Layout](../../../../glossary.md#layout), addressing, and output [materialization](../../../../glossary.md#materialization)

- Consume the exact resolved input and output `LayoutDescriptor` values, including storage
  offsets, positive strides, read-side zero strides, referenced spans, and view classification.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Require the output layout to be injective for every non-empty admitted occurrence. This task
  does not extend CPU 0006's affine same-source equivalence proof to non-affine repeated writes.
  A zero stride on an output axis is legal only when the axis extent cannot repeat an address.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [occurrence glossary](../../../../glossary.md#term-occurrence); [CPU glossary](../../../../glossary.md#term-cpu); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- Materialize exactly one distinct output buffer. No result aliases an input, even for identity
  padding, all-one tiling, or one-input composition. Current shared preparation has no
  distinct-<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [identity glossary](../../../../glossary.md#term-identity-meanings); [padding glossary](../../../../glossary.md#padding); [tiling glossary](../../../../glossary.md#tiling))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) [alias](../../../../glossary.md#aliasing-and-arena) assignment, and this task does not invent one.
- Reject every actual or ambiguous output/input accessed-byte-span overlap during cold binding
  before generated execution. Input/input overlap is legal because every input is read-only.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Use the logical output element range as the generated range domain. Zero elements touch no
  carrier. Arbitrary legal <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`start`/`end` ranges must initialize their primitive coordinates on
  the cold path and cover each logical output coordinate exactly once.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [cold path glossary](../../../../glossary.md#term-cold-path))</span>

### Multi-input resources and [bindings](../../../../glossary.md#term-binding)

- Derive an ordered input-occurrence-to-boundary mapping during lowering. Declare each distinct
  external input <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [lowering glossary](../../../../glossary.md#lowering))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) exactly once, in first occurrence order, then declare the sole output.
  If one tensor appears multiple times in <span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>`CONCAT` or `STACK`, each semantic occurrence maps to
  the same declared boundary and direct carrier; shared Prepare receives no duplicate buffer
  requirement for that <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer))</span>`ValueId`.
- Preserve logical input occurrence order independently of unique declaration order. Reordering,
  deduplicating, or sorting semantic inputs is forbidden.
- Require the explicit carrier pattern, when supplied, to match the unique derived boundary list.
  The default empty pattern still selects one `MEMORY_SEGMENT` form per derived boundary. Each
  generated class has one direct static entry for its exact unique-boundary carrier pattern.
- Declare no workspace. [`PAD`](../../../../glossary.md#padding) constants and static mapping geometry are immutable selected-plan
  facts, not graph values, Runtime buffers, or workspaces.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Finalization validates every assigned unique input and output buffer before artifact lookup.
  The executable selects every unique input as <span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>`READ_ONLY` and the output as `WRITE_ONLY`.
  Runtime's existing repeated-selection support remains available but is not needed to duplicate
  a semantic input occurrence because the CPU-private occurrence map owns that reuse.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

### [CPU](../../../../glossary.md#term-cpu)-private [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), and emission

- Extend the [sealed](../../../../glossary.md#term-sealed) `CpuPortableKernelIr` with one new `CpuDataMovementIr`. It has a closed nested
  plan variant for [`PAD`](../../../../glossary.md#padding), `TILE`, `CONCAT`, and `STACK`; records the data type, input occurrence
  roles, structural access forms, mapping family, ranks, one output store, and the exact typed pad
  constant when applicable; and contains no <span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [occurrence glossary](../../../../glossary.md#term-occurrence); [constant glossary](../../../../glossary.md#term-constant))</span>[`Operation`](../../../../glossary.md#operation), `CompiledNode`, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), Runtime slot,
  concrete carrier, address, worker, route, or artifact-store state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Add one dedicated `CpuNonAffineMovementLowering`. It owns the four-family semantic dispatch,
  descriptor validation, unique-boundary derivation, input-occurrence mapping, compact cold
  geometry, and lowering fingerprint. <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>`CpuPartitionLowering` only routes a one-node supported
  family to it; it must not absorb the family algorithms into one growing switch.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Represent concrete extents, offsets, stride magnitudes, normalized axis, padding widths,
  repeats, segment prefixes, and arbitrary-range initial coordinates in compact immutable cold
  geometry. Do not precompute or retain one address or selector entry per output element.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [normalized axis glossary](../../../../glossary.md#normalized-axis); [padding glossary](../../../../glossary.md#padding); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [immutable glossary](../../../../glossary.md#term-immutable); [element glossary](../../../../glossary.md#term-element))</span>
- Add one family-specific `CpuDataMovementEmitter`. Reuse exact carrier load/store and primitive
  loop/odometer primitives where their contracts match. Emit direct family-specialized bytecode;
  generated code contains no operation-kind, data-type, carrier, route, or strategy switch.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Generated loops may branch on the current coordinate to select a pad fill or one ordered
  composition segment. They must not allocate a cursor or coordinate object, call a virtual
  callback, perform reflection or map lookup, or use division/modulo per output element.
  Carry/reset state implements tile wrapping and arbitrary-range traversal.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [lookup glossary](../../../../glossary.md#term-lookup); [Carry glossary](../../../../glossary.md#term-carry))</span>
- Extend the scalar reference with the same compact geometry and occurrence mapping for
  differential tests. It remains a conformance path, never a Runtime <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [conformance glossary](../../../../glossary.md#term-conformance))</span>`Operation` or IR
  interpreter.
- Preserve the one-unit, one-artifact, four-complete-candidate, zero-fixed-shape, and zero-unroll
  budgets. The movement family has one direct candidate and no optional materialization candidate.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [materialization glossary](../../../../glossary.md#materialization))</span>

### Strategy, [schema](../../../../glossary.md#term-schema), and [cache](../../../../glossary.md#term-cache) boundaries

- Use scalar compute for every admitted row. Scalar orchestration covers all cases;
  parallel-scalar is eligible only for a non-empty injective output and the already-selected
  disjoint range rules. <span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective))</span>`VECTOR_IF_ELIGIBLE` deterministically falls back to [scalar](../../../../glossary.md#term-scalar) compute.
- Add no [Vector API](../../../../glossary.md#term-vector-api) load/store, [gather](../../../../glossary.md#gather), scatter, [lane](../../../../glossary.md#term-lane) map, masked tail, or performance claim.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from 11 to 12. Schema 12 rejects every
  older private artifact without migration.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- Include every bytecode- or compatibility-changing fact in specialization: IR family and
  movement variant, data type, ranks and structural axis roles, ordered input-occurrence mapping,
  unique boundary data types and carrier pattern, structural access plans, output-store form,
  exact pad constant bits when emitted, and selected scalar compute form.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [data type glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence); [carrier glossary](../../../../glossary.md#term-storage-carrier); [constant glossary](../../../../glossary.md#term-constant))</span>
- Keep compatible concrete extents, offsets, stride magnitudes, normalized axis value, pad widths,
  repeats, composition segment lengths, cold initial coordinates, exact carriers, slots,
  addresses, workers, graph/run identity, and artifact root outside class/cache identity when
  emitted code consumes them only through the fixed cold geometry schema. A fact that
  implementation bakes into bytecode must instead enter compatibility identity; it may not be
  omitted silently.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [normalized axis glossary](../../../../glossary.md#normalized-axis); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Keep optional generated-class persistence disabled by default, with one current schema and no
  migration reader. Artifact lookup remains finalization-only and occurs after assignment checks.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Failure behavior and index boundary

- This slice has no index tensor and therefore performs no execution-time index validation.
  Shape, attribute, mapping, count, arithmetic, and layout failures occur during capability or CPU
  analysis before artifact access. Carrier, span, liveness, accessibility, alignment, output
  writability, canonical external BOOL, and overlap failures occur during cold binding before the
  first output write. Runtime propagates the existing unchecked failure and preserves its current
  output-invalidity rules.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [tensor glossary](../../../../glossary.md#tensor); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [artifact glossary](../../../../glossary.md#term-artifact); [Carrier glossary](../../../../glossary.md#term-storage-carrier); [liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- CPU 0006A2, not this task, owns value-dependent indices. Its Draft contract requires a complete
  pre-write pass over INT32/INT64 indices, deterministic rejection of the first invalid logical
  index, and no wrapping, clamping, default selection, partial Gather output, or all-false one-hot
  row. Do not add a partial index-validation seam or placeholder exception in 0006A.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Gather glossary](../../../../glossary.md#gather))</span>

### Stop conditions

Stop and report the exact conflict before [widening](../../../../glossary.md#term-widening) the task if implementation requires:

- a shared physical-[layout](../../../../glossary.md#layout), [alias](../../../../glossary.md#aliasing-and-arena), dynamic-[binding](../../../../glossary.md#term-binding), or index-[validation](../../../../glossary.md#term-validation) contract;
- a Model, Compiler, Planning, shared Prepare, Runtime, public API, dependency, Gradle,
  architecture, backend-conformance, or integration change;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- more than one node, more than one output, more than sixteen input occurrences, a workspace,
  another route, another artifact, or general partition-DAG decomposition;<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [artifact glossary](../../../../glossary.md#term-artifact); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- per-output-[element](../../../../glossary.md#term-element) prepared address/selector storage or hot per-element division/modulo;
- `UNFOLD_AXIS`, `UNFOLD2D`, any [Gather](../../../../glossary.md#gather), `ONE_HOT`, scatter/fold, ordering, random work, or a later
  task specification; or
- a path outside the exact affected-file map and maximum [scope](../../../../glossary.md#term-scope) below.

## Out of [scope](../../../../glossary.md#term-scope)

- `UNFOLD_AXIS` and both `UNFOLD2D` variants; [CPU](../../../../glossary.md#term-cpu) 0006A1 owns their window and [padding](../../../../glossary.md#padding) geometry.
- [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, `GATHER_ND`, and `ONE_HOT`; CPU 0006A2 owns their index loads,
  execution-time prevalidation, failure contract, and indexed value generation.
- `SLICE_UPDATE`, scatter families, `FOLD_AXIS`, `FOLD2D`, ordering, top-K, RNG, dropout,
  reductions, scans, normalization, linear algebra, convolution, pooling, attention, and loss.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [normalization glossary](../../../../glossary.md#term-normalization); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss))</span>
- Chaining or fusing movement with affine, pointwise, window, index, or another movement node;
  branches, joins, fan-out, multiple outputs, general DAG decomposition, and profitability work.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [node glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Dynamic or symbolic Shapes, unresolved layouts, physical-layout selection, output aliasing,
  non-injective output materialization, pooling, packing, transfer, or broad materialization.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [aliasing glossary](../../../../glossary.md#term-aliasing); [injective glossary](../../../../glossary.md#term-injective); [materialization glossary](../../../../glossary.md#materialization))</span>
- Vector movement, native/vendor routes, tuning, benchmarks, relaxed math, fixed-shape variants,
  unrolling, public configuration, a registry, service locator, or broad facade.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [unrolling glossary](../../../../glossary.md#term-unrolling); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [facade glossary](../../../../glossary.md#term-facade))</span>
- Changes outside the CPU module and the authorized explanatory/planning documents; commits and
  pushes; or creation of a detailed 0006A1-or-later task specification.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [planning glossary](../../../../glossary.md#term-planning); [commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Completed CPU 0006](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Runtime API](../../../../api/runtime-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model remains the sole owner of PAD, TILE, CONCAT, and STACK meaning. CPU consumes exact
  captured semantics and descriptors; it does not reinterpret public requests.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [PAD glossary](../../../../glossary.md#padding); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- [Planning](../../../../glossary.md#term-planning) selects only `BackendId("cpu")`. CPU analysis owns family lowering, route/strategy
  selection, specialization, and exact declarations before shared assignment.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared Prepare sees one opaque selected plan plus unique graph-value buffer requirements. It
  does not inspect input occurrences, padding, repeats, composition segments, carriers, or code.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [graph glossary](../../../../glossary.md#graph); [buffer glossary](../../../../glossary.md#term-buffer); [padding glossary](../../../../glossary.md#padding))</span>
- Finalization verifies the selected assignments and realizes exactly the selected artifact. It
  does not change route, strategy, mapping, boundary set, or geometry.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- [Runtime](../../../../glossary.md#term-runtime) cold-binds [nominal](../../../../glossary.md#term-nominal) selections to direct [CPU](../../../../glossary.md#term-cpu) representations. The [hot path](../../../../glossary.md#term-hot-path) receives no
  [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), movement [IR](../../../../glossary.md#term-intermediate-representation-ir), resource map, or selection policy.
- `LogicalMemoryPlan` remains complete, and every final movement result has its own Runtime slot.
  This task creates no CPU-private virtual graph value and no shared alias.<span data-glossary-navigation="true"> ([graph value glossary](../../../../glossary.md#graph-value); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable) and reusable; concurrent runs retain distinct [`RunState`](../../../../glossary.md#run-state--runstate) and
  bound invocation state. Existing caller-owned worker-group lifetime remains unchanged.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- If any authoritative rule must change, stop instead of editing architecture from this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence)-local capability.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed portable [IR](../../../../glossary.md#term-intermediate-representation-ir) family.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — family-specific whole-partition
  validation and compact cold geometry.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — family-specific Class-File
  emission plus shared carrier/loop primitives.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — selected plan, requirements,
  specialization, and post-assignment finalization.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold [binding](../../../../glossary.md#term-binding) and direct invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — [scalar](../../../../glossary.md#term-scalar) differential realization.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) and compatibility.

Packages added or changed:

- No package is added and no responsibility moves.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr` — new sealed-family member
  containing the closed structural PAD/TILE/CONCAT/STACK plans.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [PAD glossary](../../../../glossary.md#padding))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLowering` — new
  family owner for semantic validation, unique boundaries, occurrence roles, and cold geometry.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementEmitter` — new
  family-specific generated scalar body.
- Existing `CpuPartitionLowering`, preparation/finalization, executable, reference, and cache
  types are extended only at their current family seams; do not add a manager, registry, or
  duplicate carrier abstraction.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [registry glossary](../../../../glossary.md#term-registry))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIr.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLowering.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIrTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 38 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 22 | Nineteen existing paths plus three named new types |
| CPU tests | 11 | Eight existing paths plus three named new tests |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **38** | **22 + 11 + 2 + 3** |

The ceiling is larger than the normal task guardrail for one documented technical reason: adding
one generated [IR](../../../../glossary.md#term-intermediate-representation-ir) family must update the [sealed](../../../../glossary.md#term-sealed) IR, [lowering](../../../../glossary.md#lowering) [dispatch](../../../../glossary.md#term-dispatch), Class-File dispatch,
selected-plan geometry, finalization, cold [binding](../../../../glossary.md#term-binding), reference path, [schema](../../../../glossary.md#term-schema), and their existing
[regression](../../../../glossary.md#term-regression) [owners](../../../../glossary.md#term-owner) atomically. Splitting [PAD](../../../../glossary.md#padding)/TILE from CONCAT/STACK would repeat those same seams
and schema transition while leaving a half-family with no independent architectural boundary.
Only three production types and three focused tests are new; the other listed paths are existing
family seams or regression owners and need not change unless their contract is actually affected.

If implementation needs another path, a shared contract, or a larger family, stop and revise the
plan before coding. Do not spend unused ceiling on unrelated cleanup.

## Acceptance criteria

- Capability is true only for the exact one-node, fully static, resolved-layout PAD/TILE/CONCAT/
  STACK matrix and false for wrong attributes, descriptors, layouts, data types, input counts,
  non-injective output, symbolic geometry, window/index operations, and every excluded row.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [layout glossary](../../../../glossary.md#layout); [PAD glossary](../../../../glossary.md#padding); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [injective glossary](../../../../glossary.md#term-injective))</span>
- One through sixteen composition input occurrences preserve exact semantic order. Repeated input
  values produce one declaration per unique [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), one direct carrier per unique boundary,
  and an exact occurrence-role map; the output remains last and distinct.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- PAD, TILE, CONCAT, and STACK generated/reference results agree for all six data types across
  scalar, zero-element, offsets, positive/read-zero strides, array, segment, and representative
  mixed carriers. PAD raw-bit fixtures cover every type and adversarial floating/BFLOAT16 values.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [strides glossary](../../../../glossary.md#term-stride); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Output geometry is proved injective before preparation succeeds. Cold binding rejects read-only,
  undersized, misaligned, inaccessible, wrong-carrier, non-canonical external BOOL, and overlapping
  output/input representations before the first write.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Generated arbitrary ranges and worker boundaries cover each output logical position exactly
  once. Tile wrapping and composition segment transitions use primitive carry/reset state without
  hot division/modulo or per-element allocation.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
- Scalar and parallel-scalar are the only selected compute forms; vector preference falls back
  deterministically. Parallel ranges are disjoint and preserve existing worker/failure behavior.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- `CpuDataMovementIr`, lowering, and emitter remain family-specific and CPU-private. Existing
  pointwise and affine IR inventories and tests remain unchanged except for the sealed-family and
  schema extension.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [sealed glossary](../../../../glossary.md#term-sealed); [schema glossary](../../../../glossary.md#term-schema))</span>
- Schema advances exactly to 12; compatibility equality/difference tests prove inclusion of every
  code-shaping fact and exclusion of documented cold instance facts. Older schemas fail closed,
  optional persistence remains disabled, and exactly one artifact is realized.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [persistence glossary](../../../../glossary.md#term-persistence); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Shared Prepare, Runtime, Model, Compiler, Planning, Config, Trace, Engine, dependencies, Gradle,
  architecture, backend-conformance, integration, native routes, and public APIs remain unchanged.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary impact, and planning/status evidence in this overall change.
- [CPU](../../../../glossary.md#term-cpu) 0005A–0006 remain `Complete`; CPU 0006A becomes `Complete` only after every gate passes;
  0006A1, 0006A2, 0006B, and later tasks remain `Draft` without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest'
./gradlew :backends:cpu:test
```

The sole final [CPU](../../../../glossary.md#term-cpu) suite after executable stabilization records suites, tests, failures, errors,
skips, and Java 26 [toolchain](../../../../glossary.md#term-toolchain) [identity](../../../../glossary.md#term-identity-meanings). It must cover the acceptance [matrix](../../../../glossary.md#term-matrix-meanings), one/sixteen/repeated
composition inputs, checked [overflow](../../../../glossary.md#term-overflow), exact resource order, [scalar](../../../../glossary.md#term-scalar)/zero/ranged/parallel behavior,
all [carrier](../../../../glossary.md#term-storage-carrier) forms, [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), failure-before-write behavior, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) invalidation, and
unchanged [pointwise](../../../../glossary.md#term-pointwise)/affine regressions.

Documentation-focused validation after the implementation evidence:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate local Markdown [targets](../../../../glossary.md#term-target) and anchors, [canonical](../../../../glossary.md#term-canonical-form) headings, balanced fences, final
newlines, trailing whitespace, exact 38-path ceiling, package/type placement, schema 12, status
[synchronization](../../../../glossary.md#term-synchronization), absence of 0006A1-or-later detailed specs, and no forbidden layer/build/test-tree
changes. The documentation pass reuses successful Java evidence unless it changes executable
behavior or records a concrete stale-evidence risk.

Repository-wide validation is deferred to CPU 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci). Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance),
and integration tests are not [run](../../../../glossary.md#term-run) because this task changes no [dependency](../../../../glossary.md#term-dependency), shared contract,
composed [Engine](../../../../glossary.md#term-engine) path, or cross-backend conformance claim.

## Dependencies

- [CPU 0006](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete`.
- Completed CPU 0005A–0005J whole-partition, access, strategy, materialization, typed carrier,
  generated-artifact, cold-binding, and failure contracts.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [materialization glossary](../../../../glossary.md#materialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact); [binding glossary](../../../../glossary.md#term-binding))</span>
- Current Model PAD, TILE, CONCAT, and STACK semantic kinds, attributes, Tensor descriptors,
  exact provenance, static Shape, <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [PAD glossary](../../../../glossary.md#padding); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [Shape glossary](../../../../glossary.md#shape))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), and resolved `LayoutDescriptor` contracts.
- Current Planning logical-memory/maximal-partition contracts, staged Prepare declarations and
  assignments, and Runtime multiple-selection, cold-binding, validity, and resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006A1 (Draft): portable static window extraction for `UNFOLD_AXIS` and both `UNFOLD2D`
  attribute variants, reusing the bounded movement family and adding exact NCHW/window/padding
  geometry without fold accumulation.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [padding glossary](../../../../glossary.md#padding))</span>
- CPU 0006A2 (Draft): portable [`GATHER`](../../../../glossary.md#gather), `GATHER_ELEMENTS`, `GATHER_ND`, and `ONE_HOT`, with
  INT32/INT64 index carriers, complete execution-time prevalidation before output writes,
  deterministic first-invalid-index failures, indexed mapping, and no wrap/clamp/default behavior.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- CPU 0006B (Draft) follows 0006A2 for functional updates, scatter, and overlap-fold behavior.
- CPU 0008A remains the owner of general partition-DAG decomposition and bounded cross-family
  fusion. Dynamic/symbolic execution still requires an explicit shared exact-binding contract.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [binding glossary](../../../../glossary.md#term-binding))</span>

## Architecture impact

Expected impact: None.

The task uses existing [backend-owned lowering](../../../../glossary.md#backend-owned-lowering), exact pre-assignment requirements, post-assignment
[artifact](../../../../glossary.md#term-artifact) realization, [Runtime](../../../../glossary.md#term-runtime) multiple-selection/cold-[binding](../../../../glossary.md#term-binding), and per-[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings) contracts.
If those contracts cannot express unique multi-input declarations plus one materialized output,
stop and report the exact conflict rather than broadening shared layers.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository in a clean implementation context.

Read completely:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/backends/cpu/master-plan.md
- docs/planning/backends/cpu/tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md
- completed CPU 0005A–0006 contracts and the directly relevant current Model, Planning, Prepare,
  Runtime, CPU guide, glossary, and API boundaries named by the task

Implement CPU 0006A exactly as specified within its exact 38-path map. Preserve completed CPU
0006 and every architecture boundary. Do not implement unfold, gather, one-hot, later tasks,
general DAG fusion, shared contracts, public APIs, dependencies, Gradle, conformance/integration
work, commits, or pushes. Advance the private generator schema exactly from 11 to 12. Stop on an
architecture conflict, path-ceiling breach, or need for per-output-element prepared geometry.

After executable Java stabilizes, run the focused command and the sole final CPU suite, then hand
the exact diff and test evidence to a separate clean documentation-focused context. That pass
must follow docs/developer-guide/documentation-rules.md, finalize affected Javadocs, package
summaries, CPU guide, glossary, and the three planning/status records, run CPU Javadoc and
documentation validation, and update this task's evidence, notes, completion summary, and final
status. It must not rerun successful Java tests unless executable behavior changes or a concrete
risk is recorded. Mark the task Complete only after every gate passes.
```

## Local decisions

- The former broad row is split into 0006A static pad/tile/composition movement, 0006A1 window
  extraction, and 0006A2 gather/one-hot. This is a dependency split, not an architecture change.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- One node is the complete 0006A partition boundary. General decomposition and mixed-family
  chains remain with CPU 0008A.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [decomposition glossary](../../../../glossary.md#term-decomposition); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Composition admits at most sixteen semantic input occurrences. Repeated values retain repeated
  semantic roles but share one unique resource declaration and carrier.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Output layouts must be injective. Extending affine repeated-address equivalence to non-affine
  mappings is unnecessary for this capability and would materially enlarge proof scope.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective); [scope glossary](../../../../glossary.md#term-scope))</span>
- Compact cold geometry plus generated odometers is required; per-output-element prepared tables
  are rejected for memory and repeated-run cost.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [run glossary](../../../../glossary.md#term-run))</span>
- Every movement row is scalar-compute with deterministic parallel-scalar eligibility. Vector
  movement is neither implemented nor promised.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Index validation is deliberately absent here and fully owned by 0006A2, whose pre-write rule is
  already recorded so later implementation cannot expose partial results.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Known limitations

- Only one fully static, resolved-[layout](../../../../glossary.md#layout) [PAD](../../../../glossary.md#padding), TILE, CONCAT, or STACK [node](../../../../glossary.md#node) is executable.
- Composition supports at most sixteen input occurrences. Larger Model occurrences remain valid
  model state but are not CPU-capable in this slice.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [slice glossary](../../../../glossary.md#slice))</span>
- Output layouts must be injective and distinct from every input representation. Identity
  operations still materialize a separate result.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [Identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- There is no vector movement, window extraction, value-dependent indexing, scatter/fold,
  dynamic binding, general DAG execution, native route, tuning, benchmark, or performance claim.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [binding glossary](../../../../glossary.md#term-binding); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Ordinary Tensor construction leaves these result layouts unresolved; CPU capability applies
  only after the projected occurrence contains exact resolved descriptors. This task does not
  add the missing layout-selection or Engine composition lifecycle.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence); [descriptors glossary](../../../../glossary.md#term-descriptor); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context supplied the stabilized Java evidence:

- The required focused command passed 8 suites and 43 tests with zero failures, zero errors, and
  zero skipped tests.
- The sole final `./gradlew :backends:cpu:test` passed 32 suites and 153 tests with zero failures,
  zero errors, and one expected skipped opt-in persistence test.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- The [toolchain](../../../../glossary.md#term-toolchain) was Java 26.0.2, OpenJDK 64-Bit Server VM 26.0.2+10-55.
- The implementation changed 26 of the 38 allowlisted paths and advanced the generator schema
  exactly from 11 to 12. <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`git diff --check` passed before documentation finalization.

Documentation context `/root` independently read the complete repository, architecture,
documentation-profile, [planning](../../../../glossary.md#term-planning), task, [CPU](../../../../glossary.md#term-cpu) guide, glossary, relevant [API](../../../../glossary.md#term-application-programming-interface-api)/boundary, final source,
test, diff, and generated-[Javadoc](../../../../glossary.md#term-javadoc) evidence. It finalized affected Javadocs and package summaries,
the CPU guide, glossary, this record, master plan, and roadmap without changing executable Java
behavior. It therefore reused the successful Java suites instead of rerunning them.

- `./gradlew :backends:cpu:javadoc` passed on the finalized source documentation.
- Repository-local Markdown validation passed local links and explicit anchors, balanced fences,
  canonical headings, final newlines, and trailing-whitespace checks.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Scope validation passed the exact 38-path allowlist and ceiling, package/type inventory,
  generator schema 12, synchronized <span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>`Complete` status, and unchanged completed history.
- Inventory validation confirmed that 0006A1, 0006A2, 0006B, and every later CPU task remain
  `Draft` and that no detailed later task specification exists.
- Forbidden-change validation confirmed no edits to public Tensor/Compile/Runtime APIs, shared
  Model/Compiler/Planning/Prepare/Runtime contracts, architecture/ADR/tests, Gradle/dependencies,
  backend-conformance/integration trees, native routes, or later task specifications.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Final `git diff --check` passed.

## Implementation notes

- `CpuCapabilityProvider` now truthfully admits the exact fully static resolved-layout one-node
  PAD/TILE/CONCAT/STACK occurrence matrix while complete-partition lowering remains fail-closed.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [node glossary](../../../../glossary.md#node); [PAD glossary](../../../../glossary.md#padding); [occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- `CpuDataMovementIr` is the third sealed portable IR form. It retains family/rank/type,
  first-occurrence unique input accesses, ordered composition occurrence mapping, one output
  store, and exact PAD bits while excluding concrete instance geometry.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [rank glossary](../../../../glossary.md#term-rank))</span>
- `CpuNonAffineMovementLowering.Geometry` retains compact extents, offsets, strides, axes,
  padding widths, repeats, segment prefixes, and range-start state. No per-output-element table is
  prepared or retained.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [axes glossary](../../../../glossary.md#term-axis); [padding glossary](../../../../glossary.md#padding); [element glossary](../../../../glossary.md#term-element))</span>
- Generated and scalar-reference realizations copy represented bits for all six Model data types.
  Generated loops use carry/reset coordinate state and contain no per-element division or modulo.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [represented bits glossary](../../../../glossary.md#term-represented-bits); [Model glossary](../../../../glossary.md#term-model); [carry glossary](../../../../glossary.md#term-carry); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Analysis declares unique inputs followed by one distinct injective output. Cold binding rejects
  complete input/output-span overlap before execution. Movement selects scalar or parallel-scalar
  execution; vector preference falls back to scalar.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [binding glossary](../../../../glossary.md#term-binding); [vector glossary](../../../../glossary.md#term-vector))</span>
- Schema 12 invalidates schema-11 artifacts by recording the new movement structure in canonical
  generated identity. No migration reader was added.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Public API shapes and Tensor/Compile/Runtime contracts required no change. Existing Model
  meanings and the backend-owned analysis/finalization/cold-binding boundary already express the
  capability. Architecture, ADRs, architecture tests, Gradle/dependencies, conformance/integration
  tests, and native routes likewise required no change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shapes glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0006A is complete. The [CPU portable route](../../../../glossary.md#cpu-portable-route) now executes exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) [PAD](../../../../glossary.md#padding), TILE, CONCAT, or STACK [occurrence](../../../../glossary.md#term-occurrence) through compact [backend](../../../../glossary.md#backend)-private movement
[IR](../../../../glossary.md#term-intermediate-representation-ir) and geometry, unique [variadic](../../../../glossary.md#term-variadic) declarations, exact all-six-type [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) generation,
[scalar](../../../../glossary.md#term-scalar)/parallel-scalar orchestration, and one distinct [injective](../../../../glossary.md#term-injective) materialized output. The scalar
reference, focused [regression](../../../../glossary.md#term-regression) coverage, generated-[artifact](../../../../glossary.md#term-artifact) [schema](../../../../glossary.md#term-schema), Javadocs, package summaries,
CPU guide, glossary, master plan, and roadmap are synchronized. No unresolved issue remains in
task [scope](../../../../glossary.md#term-scope). CPU 0006A1, 0006A2, 0006B, and later work remain Draft without detailed specifications.
