# Model Master Plan

## Goal

Define Synaptik's backend-independent tensor semantics, immutable graph model, public tensor state, and host storage contracts.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Model capability baseline](capabilities.md)

## Scope

- data type, dimension, shape, and layout value models
- typed tensor, node, and value identifiers
- backend-independent operations and immutable attributes
- tensor descriptors, immutable graph values and nodes, and compiled graph state
- public `Tensor`, host storage abstractions, factories, and minimal provenance
- model-level representation and public expression construction for the selected legacy capability baseline

## Out of scope

- graph compilation and optimization
- backend ownership and partition planning
- prepared execution, runtime residency, and device buffers
- backend-specific storage, lowering, or kernels

## Module invariants

- `Tensor` is public mutable API state and is not an IR node.
- `Operation` owns semantics and never backend support.
- Compiled graph state is immutable.
- Host storage never represents runtime device residency.

## Allowed dependencies

- JDK standard library
- No project dependency until a focused task demonstrates an architecture-compliant need.

## Forbidden dependencies

- planning, compiler, runtime, prepare, and engine modules
- concrete backend modules
- kernel selection, device residency, runtime state, and prepared execution

## Package structure

The module root `io.github.pho001.synaptik.model` is a namespace boundary, not the default destination for new types. Model contracts are grouped by cohesive responsibility:

```text
io.github.pho001.synaptik.model.datatype
  Data type metadata, promotion, and host-independent bit conversion.

io.github.pho001.synaptik.model.shape
  Dimensions, immutable shapes, axes, and local broadcasting.

io.github.pho001.synaptik.model.layout
  Resolved logical layout geometry and layout classification.

io.github.pho001.synaptik.model.tensor
  Public Tensor state, TensorId, TensorDescriptor, TensorFactory, eager initialization helpers,
  and provenance.

io.github.pho001.synaptik.model.storage
  Host-visible storage contracts and implementations.

io.github.pho001.synaptik.model.operation
  Backend-independent operation semantics and immutable attributes.

io.github.pho001.synaptik.model.operation.elementwise.binary
  Typed parameterless semantic kinds for tensor-to-tensor elementwise arithmetic.

io.github.pho001.synaptik.model.graph
  NodeId, ValueId, graph values/nodes, graph phase, publication binding,
  and immutable compiled graph state.
```

Package dependencies remain acyclic. `datatype` and `shape` are foundational leaves; `layout` may depend on `shape`; `storage` may depend on `datatype`; `operation` may consume foundational value contracts but must not depend on public `Tensor` or compiled graph state; `tensor` may compose foundational values, host storage, and operation provenance; and `graph` may compose tensor descriptors and operation semantics. Package-private helpers live in the package whose contracts they implement.

Operation-family subpackages are introduced only when a focused operation task demonstrates a cohesive boundary. Generic `util`, `common`, `internal`, and `misc` packages are not part of the planned structure.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md) | Complete | - | Define data type categories, metadata, floating promotion, and BFLOAT16 conversion. |
| 0002 | [Shape and dimension model](tasks/0002-shape-and-dimension-model.md) | Complete | - | Define static and symbolic dimensions, immutable shapes, checked element counts, axes, and broadcasting. |
| 0003 | [Layout descriptor model](tasks/0003-layout-descriptor-model.md) | Complete | 0002 | Define resolved layout kinds, checked element strides, offset/span, and view metadata. |
| 0003A | [Data type package migration](tasks/0003a-data-type-package-migration.md) | Complete | 0001 | Move completed data type contracts into `model.datatype` without changing behavior. |
| 0003B | [Shape package migration](tasks/0003b-shape-package-migration.md) | Complete | - | Move completed dimension and shape contracts into `model.shape` without changing behavior. |
| 0003C | [Layout package migration](tasks/0003c-layout-package-migration.md) | Complete | 0003B | Move completed layout contracts into `model.layout` and preserve their shape imports. |
| 0004 | [Typed identifiers](tasks/0004-typed-identifiers.md) | Complete | 0003A–0003C | Define TensorId, NodeId, and ValueId in their owning domain packages. |
| 0005 | [Operation semantic foundation](tasks/0005-operation-semantic-foundation.md) | Complete | - | Define the minimal operation-kind and typed-attribute contracts without family-specific semantics. |
| 0006 | [Operation model](tasks/0006-operation-model.md) | Complete | 0005 | Define the minimal immutable backend-independent kind-and-attributes descriptor. |
| 0007 | [Tensor descriptor model](tasks/0007-tensor-descriptor-model.md) | Complete | 0001–0003, 0003A–0003C | Define data type, shape, explicit resolved/unresolved layout, and requires-grad descriptors. |
| 0008 | [Graph value and node model](tasks/0008-graph-value-and-node-model.md) | Complete | 0004, 0006, 0007 | Define immutable graph value and node records. |
| 0009 | [Compiled graph model](tasks/0009-compiled-graph-model.md) | Complete | 0008 | Define immutable graph container, forward/backward phase, and standalone publication binding. |
| 0010 | [Host storage abstraction](tasks/0010-host-storage-abstraction.md) | Complete | 0001, 0003A | Define exact-size borrowed Java 26 memory-segment host storage without device buffers. |
| 0011 | [Public Tensor skeleton](tasks/0011-public-tensor-skeleton.md) | Complete | 0004, 0007, 0010 | Define stable public Tensor identity/descriptor/label and synchronized optional host-storage state without graph or runtime state. |
| 0012 | [Tensor factory foundation](tasks/0012-tensor-factory.md) | Complete | 0010, 0011 | Expose descriptor-based public construction, optional borrowed storage attachment, and JVM-wide tensor-ID allocation without allocating memory. |
| 0012A | [JVM-managed heap host storage allocation](tasks/0012a-host-storage-allocation.md) | Complete | 0010, 0012 | Add exact-span typed primitive-array allocation through the existing borrowed heap-segment storage contract. |
| 0012B | [Flat typed tensor import](tasks/0012b-flat-typed-tensor-import.md) | Complete | 0012A | Import copied flat primitive arrays into dense-contiguous tensors with exact carrier and logical-count validation. |
| 0012C | [Nested typed tensor import](tasks/0012c-nested-typed-tensor-import.md) | Complete | 0012B | Infer exact type and static dense shape from validated rectangular primitive arrays, then flatten and delegate to typed flat import. |
| 0012D | [Constant tensor creation](tasks/0012d-constant-tensor-creation.md) | Complete | 0012B | Add exact typed rank-zero scalars and independent dense zeros, ones, zeros-like, and ones-like tensors. |
| 0012E | [Range and prefix population](tasks/0012e-range-and-prefix-population.md) | Complete | 0012B | Add typed integer ranges plus strict and cyclic exact-carrier prefix population under explicit validation. |
| 0012F | [Random tensor creation](tasks/0012f-random-tensor-creation.md) | Complete | 0012B | Add normally distributed floating tensors from an explicit caller-owned random source with bounded reproducibility. |
| 0012G | [Uniform random tensor creation](tasks/0012g-uniform-random-tensor-creation.md) | Complete | 0012F | Add continuous uniform floating tensors with explicit half-open bounds and the existing caller-owned source policy. |
| 0012H | [Integral random tensor creation](tasks/0012h-integral-random-tensor-creation.md) | Complete | 0012F | Add exact INT32/INT64 overloads with exclusive bounds and unbiased JDK bounded sampling. |
| 0012I | [Bernoulli random tensor creation](tasks/0012i-bernoulli-random-tensor-creation.md) | Complete | 0012F | Add BOOL tensors sampled from an explicit probability using the existing caller-owned source policy. |
| 0013 | [Tensor provenance skeleton](tasks/0013-tensor-provenance-skeleton.md) | Complete | 0006, 0011, 0012 | Attach immutable operation-and-input origin metadata to Tensor for future compiler-owned graph capture. |
| 0013A | [Full-value and identity-matrix tensor creation](tasks/0013a-full-value-and-identity-matrix-tensor-creation.md) | Complete | 0012B, 0012D | Add typed full-value tensors through canonical `full` and dense rectangular identity matrices, with `eye` exactly aliasing the canonical `identityMatrix` semantics. |
| 0014A | [Binary arithmetic semantic kinds](tasks/0014a-binary-arithmetic-semantic-kinds.md) | Complete | 0005, 0006 | Define typed parameterless ADD, SUB, MUL, DIV, MIN, MAX, and POW kinds. |
| 0014B | Binary arithmetic Tensor expressions | Draft | 0013, 0014A | Build locally validated broadcast-aware Tensor expressions with ordered provenance. |
| 0014C | Unary arithmetic and activation semantic kinds | Draft | 0005, 0006 | Define parameterless unary and activation kinds. |
| 0014D | Unary arithmetic and activation Tensor expressions | Draft | 0013, 0014C | Build unary and activation Tensor expressions with locally derived descriptors. |
| 0014E | Scalar arithmetic and clamp semantics | Draft | 0005, 0006 | Define typed scalar attributes and clamp semantic contracts. |
| 0014F | Scalar arithmetic and clamp Tensor expressions | Draft | 0013, 0014E | Build scalar arithmetic and clamp Tensor expressions without eager execution. |
| 0015 | Comparison, logical, selection, and cast operations | Draft | 0013 | Represent comparison, boolean, where, and explicit cast capabilities. |
| 0016 | Reduction and scan operations | Draft | 0013 | Represent numeric and boolean reductions, scans, softmax, and tie policies. |
| 0017 | Layout and view operations | Draft | 0002, 0003, 0013 | Represent reshape, view, slice, composition, pad, tile, unfold, and fold capabilities. |
| 0018 | Indexing and scatter operations | Draft | 0001, 0013 | Represent gather, take, select, and functional scatter capabilities. |
| 0019 | Linear algebra and attention operations | Draft | 0013 | Represent matmul, linear, and scaled dot-product attention capabilities. |
| 0020 | Convolution and pooling operations | Draft | 0013 | Represent NCHW convolution and two-dimensional pooling capabilities. |
| 0021 | Normalization operations | Draft | 0013 | Represent batch, layer, and RMS normalization capabilities. |
| 0022 | Loss operations | Draft | 0013 | Represent dense/index NLL and cross-entropy variants and reductions. |
| 0023 | Compiler-generated semantic operations | Draft | 0006, 0014A–0014F, 0015–0022 | Represent backend-neutral backward and compiler-generated operation descriptors without autograd rules. |
| 0024 | Model capability parity audit | Draft | 0001–0023 | Verify model representation and public expression construction against the selected legacy baseline. |

## Milestones

- Value foundations and package organization: tasks 0001–0004, including 0003A–0003C
- Operation and immutable graph model: tasks 0005–0009
- Public tensor and host storage: tasks 0010–0013 and factory follow-ups 0012A–0012I and 0013A
- Public operation capability families: tasks 0014A–0014F and 0015–0022
- Compiler-generated model semantics and model parity: tasks 0023–0024

## Current status

Draft, with the post-foundation checkpoint and task 0014A complete. Task 0014B remains the next
Draft planning frontier without a detailed specification.

The capability baseline is documented and the ordered task queue covers its model-level
responsibilities. Tasks 0001 through 0007 and package migrations 0003A–0003C are complete. Task
0008, graph value and node model, task 0009, compiled graph model, and task 0010, host storage
abstraction, are complete. Task 0011, public Tensor skeleton, and task 0012, the bounded Tensor
factory foundation, are also complete. Task 0012A, JVM-managed heap host storage allocation, is
complete. Task 0012B, flat typed tensor import, is also complete. Task 0012C, nested typed tensor
import, is complete. Task 0012D, constant tensor creation, and task 0012E, deterministic range and
prefix population, are also complete. Normal population task 0012F, uniform population task 0012G,
and integral population task 0012H are complete. Bernoulli task 0012I and provenance task 0013 are
also complete. Full-value and identity-matrix factory task 0013A is complete. The foundation
checkpoint selected continued sequential model operation-family work. Task 0014A, binary
arithmetic semantic kinds, is complete; task 0014B remains the next Draft planning frontier
without a detailed specification.

## Open questions

- Exact public overloads and operation-attribute record boundaries remain local to the applicable operation-family tasks.
- Reassess the next cross-module vertical-slice opportunity after binary semantic kinds and public
  expression construction provide one concrete capturable operation family.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The initial data type baseline is `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- Static dimensions use non-negative `long` sizes; dynamic dimensions use explicit canonical symbols rather than negative sentinels.
- Scalar shape is rank zero, and zero-sized static dimensions are supported.
- Local broadcasting is right-aligned and conservative for symbolic dimensions; graph-wide symbolic constraints remain a compiler responsibility.
- Numeric layout descriptors are resolved only for fully static shapes and use non-negative `long` element strides and offsets.
- Layout geometry distinguishes dense, offset-dense, strided, and broadcast views; planning remains responsible for materialization decisions.
- Model contracts are organized by `datatype`, `shape`, `layout`, `tensor`, `storage`, `operation`, and `graph` responsibilities; the module root is not a flat type container.
- Java 26 is the project baseline. Stable Java 26 APIs require no preview opt-in; preview and incubator features remain disabled unless a focused owning-module task explicitly configures and validates them.
- Typed identifiers live with their domains. The current plan includes `TensorId`, `NodeId`, and `ValueId`; `OperationId` is deferred unless a focused task demonstrates identity distinct from `NodeId`.
- Host storage contracts precede the public `Tensor`, and `Tensor` reuses `TensorDescriptor` rather than duplicating descriptor validation.
- `HostTensorStorage` is a sealed model boundary with one final identity-based
  `MemorySegmentStorage` implementation. The wrapper borrows an exact-size live segment, exposes
  raw segment/read-only/liveness facts, uses checked `long` capacity sizing, and owns no arena,
  allocation, close behavior, typed access, alignment, byte order, tensor geometry, or runtime
  residency policy.
- Operation-family table order coordinates delivery; dependencies record only real contract prerequisites rather than the preceding row.
- All selected legacy public operation capabilities must be representable without backend knowledge.
- Model capability parity and end-to-end executable parity are tracked separately.
- Fusion is not a model-level mathematical operation capability.
- The compiled graph container stores ordered values, topological nodes, explicit input/output
  boundaries, and an exact node-to-forward/backward-phase mapping. It stores no derived indexes.
- Publication binding remains a standalone `TensorId`-to-`ValueId` model DTO for a later
  compiler-owned publication plan; it is not part of `CompiledGraphModel`.
- The current graph-phase vocabulary is exactly forward and backward. Optimizer-update graph work
  remains a future architecture change, not a task-0009 phase.
- The task-0011 Tensor skeleton is one final public identity object with package-private
  construction. It retains one stable `TensorId`, immutable `TensorDescriptor`, and normalized
  optional label; task 0012 owns the public factory and ID allocation policy.
- Tensor's only mutable state is a synchronized optional borrowed `HostTensorStorage`
  association. Matching data type is always required; resolved layouts require capacity at least
  their referenced element span, while unresolved layouts do not invent physical geometry.
- Tensor accepts read-only storage, rejects storage already dead at attachment, continues to expose
  storage that dies later, owns no arena, retains object identity equality/hashing, and stores no
  graph-local ID, gradient/trainable/publication, runtime, or backend state. Task 0013 adds only
  final optional provenance metadata and does not change host storage's sole-mutation role.
- Task 0012 is a non-instantiable static `TensorFactory` with exactly descriptor-only and
  descriptor/optional-label/optional-storage creation methods. It delegates semantic label and
  storage validation to the package-private Tensor constructor and performs no storage allocation,
  import, population, descriptor construction, layout resolution, or provenance work.
- Factory-assigned tensor IDs are unique across factory calls in one JVM, including concurrent
  calls. A hidden `AtomicLong`/`AtomicBoolean` allocator issues non-negative candidates from zero
  through `Long.MAX_VALUE`, permits the final value once, never wraps or reuses a value, and then
  fails permanently. Numeric order and gaplessness are not public caller contracts.
- Factory argument-container null failures occur before ID allocation. Tensor label/storage
  failures occur after allocation and consume the candidate so the factory does not duplicate
  canonical validation or attempt unsafe concurrent rollback.
- The broad factory baseline is split into completed task 0012A for JVM-managed heap allocation,
  completed task 0012B for flat typed import, completed task 0012C for nested typed import,
  completed task 0012D for constant tensors, completed task 0012E for deterministic range/prefix
  population, completed task 0012F for normal random tensors, completed task 0012G for uniform
  random tensors, completed task 0012H for integral tensors, and completed task 0012I for
  Bernoulli tensors. These rows remain before completed provenance task 0013 and completed
  full-value/identity task 0013A. The completed model foundation checkpoint selected task 0014A
  as the next implementation frontier.
- Task 0012A adds only JVM-managed heap allocation to `TensorFactory`. It allocates one typed
  primitive array whose length is the resolved layout's referenced element span, wraps the
  `MemorySegment.ofArray(...)` result in the existing `MemorySegmentStorage`, and delegates to the
  existing `create(...)` path.
- Java 26 heap segments use an automatic scope that keeps the primitive-array heap base reachable
  and is always accessible from any thread. Task 0012A therefore adds no owning wrapper, arena,
  close behavior, external owner, or storage-contract change.
- Task 0012A requires resolved layout, rejects span above `Integer.MAX_VALUE`, and keeps allocation
  separate from the imports in completed tasks 0012B and 0012C, constant creation in completed task
  0012D, deterministic population in completed task 0012E, and random population in completed task
  0012F–0012I.
- Task 0012B adds six typed flat-array overloads for `double[]`, `float[]`, raw BFLOAT16 `short[]`,
  `int[]`, `long[]`, and BOOL `byte[]`. It accepts only resolved dense-contiguous layout, validates
  source length against logical element count, copies all input data, and normalizes BOOL bytes to
  canonical zero or one without retaining caller arrays.
- Offset, strided, and broadcast layouts are rejected by flat import because mapping independent
  row-major source values into aliased or sparse physical geometry is a distinct scatter/view
  policy. Task 0012B reuses task-0012A allocation and the existing factory identity path.
- Task 0012C accepts exactly rank-two-or-greater Java arrays whose ultimate component is one of the
  six primitive host carriers. One `Object` method is used because arbitrary array rank has no
  finite overload family; runtime class metadata must still prove declared rank and exact carrier.
- Nested import validates the full reachable structure for rectangular lengths and non-null
  subarrays, rejects empty non-final axes whose trailing extents are unobservable, accepts an empty
  final leaf axis, and flattens row-major into a fresh matching carrier. It synthesizes only a
  fully static dense-contiguous descriptor and delegates final creation to task 0012B.
- Task 0012D defines exact primitive-carrier rank-zero scalars, including an explicitly named
  BFLOAT16 conversion, plus all-data-type zeros and ones over fully static shapes. Zeros reuse
  default-zero allocation; scalars and ones reuse typed flat import.
- Constant `*Like` methods copy only template shape and data type. They take explicit label and
  gradient intent and create new dense-contiguous descriptor, storage, and identity without
  observing or preserving template layout, label, storage, liveness, or ID.
- Task 0012E keeps deterministic population type-exact: `int` and `long` ranges produce only
  `INT32` and `INT64`, while strict and cyclic prefixes use six primitive-carrier overloads with no
  implicit conversion. All results synthesize canonical dense layout and reuse flat import.
- Integer ranges are eager non-differentiable leaf data with inclusive start, exclusive end,
  positive or negative non-zero step, exact overflow-safe count, and no automatic label. Prefixes
  require fully static shape, copy source values, preserve raw BFLOAT16 bits, and reuse downstream
  BOOL normalization. Empty cyclic input is valid only for an empty result.
- Task 0012F uses one transient caller-owned `RandomGenerator` and stores no random service, source,
  seed, or algorithm. It consumes exactly one `nextGaussian()` per logical element, applies an
  explicit binary64 normal transformation, converts only to FLOAT64/FLOAT32/BFLOAT16, and delegates
  completed carriers to flat import.
- Random reproducibility is bounded to equivalent generator implementation/state and identical
  arguments without interfering use. No cross-algorithm/provider/Java-version promise, default
  source, synchronization, or seed-only convenience is introduced.
- User-approved random initialization expansion remained sequential: completed task 0012G adds
  floating uniform sampling, completed task 0012H adds typed bounded integral sampling, and
  completed task 0012I adds BOOL Bernoulli sampling. Each reuses the caller-owned source policy
  without changing task 0012F.
- Task 0012H uses two `randomInt` overloads: int bounds infer INT32 and long bounds infer INT64.
  Both use strict half-open bounds, false gradient intent, and the matching unbiased JDK bounded
  generator method without project-owned modulo arithmetic.
- Task 0012I uses one BOOL-only `randomBernoulli` method with finite probability in `[0,1]`. It
  consumes one unbounded binary64 draw per element even at probability endpoints and stores the
  strict `draw < probability` result as canonical zero/one bytes before BOOL flat import.
- Random factory methods and package-private helpers remain in `model.tensor`. A `randoms` package
  would break useful package-private collaboration or require a public implementation surface and
  is not justified without independent public random-domain types.
- Task 0013 adds one immutable `TensorProvenance` value containing a backend-independent
  `Operation` and an ordered immutable snapshot of input Tensor identities. Tensor retains it as
  optional final metadata; it receives no graph-local identity and does not become IR.
- Existing public factory paths remain provenance-free leaves. One package-private derived-
  construction seam reuses the existing TensorFactory allocator, attaches exact provenance, and
  creates no storage, inference, graph, compiler, runtime, or backend state.
- Task 0013A is model-owned eager tensor creation, not a training initializer, graph operation, or
  runtime/backend capability. `full` and `identityMatrix` are the canonical factory names, and
  `eye` is the exact convenience alias for `identityMatrix` semantics.
- Task 0013A added six primitive-carrier `full` methods, including explicitly converted
  `fullBFloat16`, plus one all-data-type rectangular `identityMatrix`. `eye` delegates only to the
  canonical method. All results are dense provenance-free leaves created through flat import.
- The post-foundation checkpoint selected continued model work instead of an immediate
  cross-module vertical slice because no production concrete OperationKind existed yet for
  compiler capture or backend capability work.
- The broad former task 0014 is decomposed into semantic-vocabulary and public-expression pairs:
  binary arithmetic 0014A–0014B, unary/activation 0014C–0014D, and scalar/clamp 0014E–0014F.
- Task 0014A implements one parameterless `BinaryArithmeticKind` enum in
  `model.operation.elementwise.binary` with exact constants ADD, SUB, MUL, DIV, MIN, MAX, and POW.
  Broadcast geometry, dtype rules, provenance, and Tensor methods remain in task 0014B.

## Risks

- Accidentally treating public `Tensor` as compiled IR.
- Leaking runtime storage or backend support into the model.
- Expanding the public API before value-model invariants are stable.
- Creating cycles between `operation`, `tensor`, and `graph` packages.
- Enabling preview or incubator features globally instead of containing them in the module that requires them.
- Treating the operation inventory as permission to move graph inference, autograd rules, fallback, or execution into model.
- Reproducing accidental legacy behavior instead of specifying and testing the intended contract.
- Letting a global identity counter wrap, collide under concurrency, or become a runtime service
  registry rather than remaining hidden model-only allocation state.
- Treating completed JVM heap allocation as import/population or native/runtime allocation parity
  before the applicable typed-population and deterministic-resource contracts exist.

## Notes

Execute tasks in table order, including package migrations 0003A through 0003C before task 0004. The operation-family rows are task groups, not permission for oversized implementations; replace the current frontier row with smaller sequential task rows before implementation when its detailed scope would exceed the limits in [the planning guide](../../planning-guide.md).

Package migrations 0003A–0003C and tasks 0004–0009 are complete. Task 0008 added the two local
immutable graph element records, and task 0009 added the structurally closed graph container,
forward/backward node phases, and standalone publication binding. Task 0010 added the sealed raw
host-storage boundary and exact-size borrowed Java 26 memory-segment wrapper. Task 0011 added the
completed public Tensor skeleton with stable metadata and a synchronized borrowed host-storage
association. Task 0012 completed public descriptor-based construction, optional borrowed storage
attachment, and JVM-scoped ID allocation only. Task 0012A completed exact-span typed primitive-
array heap allocation through automatic-scope segments without changing the borrowed storage
contract. Task 0012B completed copied flat typed import for all six data types with
dense-contiguous/count validation and BOOL normalization. Task 0012C completed validated
rectangular nested primitive-array import, exact carrier/static-shape inference, and row-major
delegation to flat import. Task 0012D completed exact typed scalars and independent dense zero/one
constants. Task 0012E completed deterministic typed range and strict/cyclic prefix population, and
task 0012F completed explicit-source normal-random population, task 0012G completed bounded
continuous-uniform floating population, task 0012H completed bounded-integral population, and task
0012I completed BOOL Bernoulli population, task 0013 completed immutable Tensor provenance and
the package-private derived-construction seam, and task 0013A completed type-safe full-value and
rectangular identity creation. The completed post-foundation checkpoint selected sequential model
operation-family work, and task 0014A completed the first concrete parameterless kind family.
Task 0014B and all later operation-family tasks remain Draft without detailed specifications.
The legacy branch must be consulted read-only for capability and test evidence when preparing each
applicable capability task.
