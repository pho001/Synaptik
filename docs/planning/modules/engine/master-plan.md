# Engine Master Plan

## Goal

Provide the public lifecycle facade and explicit composition root for compiler, prepare, runtime, and concrete backends.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- public compiled graph facade
- explicit backend registration
- deterministic standard composition of known built-in backends
- compile and prepare orchestration
- composition of runtime, validators, tracing, and backends
- explicit typed caller-input binding and published-result access
- explicit host materialization/publication needed by downstream persistence consumers

## Out of scope

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator and core reflective discovery
- Tensor-owned execution, backward methods, gradients, or runtime state

## Module invariants

- Engine is the outer composition root.
- Every usable backend is registered explicitly by Engine construction code. Task 0001 owns one
  explicit CPU adapter because it is the sole supported complete lifecycle adapter. A later
  standard factory may own a fixed ordered inventory of known built-ins only as their adapters and
  complete schedule-composition contracts become real.
- Backend eligibility and partition ownership are resolved before Runtime execution.
- Concrete backends never depend on engine.

## Allowed dependencies

- modules/model
- modules/planning
- modules/compiler
- modules/runtime
- modules/prepare
- modules/config
- modules/trace
- concrete backend modules

Engine declares dependencies directly for every public contract its source names. Task 0001's
advanced API names Model `Tensor` and `HostTensorStorage`, while its package-private composition
seam names Planning `BackendCapabilityProvider`; direct Model and Planning dependencies are
therefore required rather than inherited transitively through Compiler, Prepare, or CPU.

## Forbidden dependencies

- No inward module may depend on engine.

## Package structure

```text
io.github.pho001.synaptik.engine/
  Engine                      public ordinary standard-composition owner
  CompiledGraph               public ordinary owner-bound compile handle and input metadata
  PreparedExecution           public ordinary owner-bound prepared handle
  RunResult                   public ordinary publication lease and explicit materialization owner
  HostTensorValue             public immutable detached canonical host payload
  ScalarObjectiveBackwardResult
                               public detached objective and target-aligned gradient values
  AdvancedEngine              public advanced composition and lifecycle owner
  AdvancedCompiledGraph       public opaque owner-bound compile handle
  AdvancedPreparedExecution  public opaque owner-bound prepared handle
  AdvancedRunResult           public lifecycle-only result wrapper
  package-private composition and lifecycle machinery
```

The root package is a deliberate small facade. Task 0001 adds no public `spi` package or generic
backend abstraction: its package-private composition seam has the current CPU realization and a
focused fake test realization only. Task 0002 adds the distinct ordinary `Engine` owner with a
construction-and-close surface only. Complete task 0003 adds the three ordinary handle/result
types and lifecycle methods shown above without exposing the advanced representation-level
surface. Complete task 0004 adds the one host payload and method shown above without widening the
advanced surface.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md) | Advanced composition and representation-level lifecycle foundation | Complete | Compiler 0006B3; CPU 0010F; current Prepare 0003–0004 and Runtime 0010 contracts | Own exactly one supported CPU lifecycle adapter and expose opaque owner-bound compile/prepare handles plus a representation-level run seam for integration and tests. Reject zero/pass-through and mixed/multi-partition preparation; do not claim nonexistent schedule composition. |
| [0002](tasks/0002-standard-built-in-composition.md) | Standard built-in composition | Complete | 0001; supported built-in backend adapters | Added a distinct ordinary `Engine.standard()` construction path that owns one fresh CPU-only built-in composition without exposing `CpuBackendIntegration`, advanced handles, or inward integration contracts. Preserved `AdvancedEngine.takeOwnership(...)`; mixed-backend success still requires another concrete adapter and a justified complete schedule-composition contract. |
| [0003](tasks/0003-typed-logical-input-binding-and-published-result-access.md) | Typed logical input binding and published-result access | Complete | 0001–0002; [Compiler 0006B4](../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md); current publication/Prepare/Runtime contracts | Added the ordinary compile, prepare, TensorId-based host-input binding, synchronous run, and metadata-only publication/result lifecycle without exposing inward identities, coordinates, representations, or advanced handles. |
| [0004](tasks/0004-explicit-host-materialization-boundary.md) | Explicit host materialization boundary | Complete | 0003; Runtime 0015; CPU 0010G | Lazily copies one exact selected publication occurrence into a detached immutable bounded `HostTensorValue` with canonical row-major big-endian bytes. Serializes copying against result/Engine close and exposes no representation, backend storage, arena, `MemorySegment`, `ValueId`, or slot. |
| [0005](tasks/0005-one-shot-forward-convenience.md) | Engine-owned one-shot forward convenience | Complete | 0002–0004; Runtime 0015; CPU 0010G | Added exact single-output and ordered-output `Engine.forward(...)` overloads with explicit logical inputs and one aggregate canonical-byte limit. Each call freshly compiles, prepares, runs, materializes every forward publication in order, and cleans up under one Engine admission; it adds no Tensor execution or cache. |
| [0005A](tasks/0005a-automatic-input-discovery-and-compute-convenience.md) | Automatic input discovery and compute convenience | Complete | 0005; Compiler 0006B4 | Replaced the two explicit-input one-shot `forward(...)` methods with four `compute(...)` overloads. Inventories reachable provenance-free Tensor leaves transiently by object identity, then selects only matching Tensors in authoritative final `CompiledGraph.inputs()` order; adds no Compiler IR traversal, liveness inference, retained Tensor state, or cache. |
| [0006](tasks/0006-one-shot-scalar-objective-backward-convenience.md) | Engine-owned one-shot scalar-objective backward convenience | Complete | 0005A; Compiler 0006B5; Prepare 0005; CPU 0010H | Added one explicit-target `Engine.backward(...)` call returning a detached scalar objective plus immutable target-aligned gradients. Reuses 0005A's transient leaf-inventory/final-binding selection seam with no explicit input list and Compiler's absent scalar unit seed and ERROR policy through the completed source-only constant chain; retains the explicit-seed ordinary compile and advanced full-request paths. |
| 0007 | Optional model-autotuning composition | Draft | 0002; 0005; Config 0006A; tools/tuning 0001; operational representative execution facts | Explicitly map Config's request into tuning's generic collaboration before preparation, retaining deterministic untuned fallback and keeping all tuning out of Runtime. This does not yet implement tools/tuning 0002 graph/plan search. |
| 0008 | Engine lifecycle capability checkpoint | Draft | 0001–0006; Compiler 0006B; CPU 0008A | Validate standard and advanced composition, typed input/output ownership, host materialization, one-shot forward/backward lowering, cleanup, concurrency, architecture tests, documentation, and representative NCW Conv1d plus NCHW Conv2d and NCDHW Conv3d forward execution before persistence adapters or NN convolution integration depend on Engine. |


## Milestones

- Explicit backend composition
- Compile facade
- Prepare and run lifecycle facade

## Current status

Tasks 0001–0006 and the complete source-only constant chain through CPU 0010H are Complete.
Engine 0007–0008 remain Draft without detailed specifications.
Compiler 0006B3, Prepare 0003–0004, Runtime 0010 and its closure hardening,
and CPU 0010F supply the bounded CPU-only Engine lifecycle without shared-contract changes:
`GraphCompilationPort` supplies complete compile artifacts, `GraphPreparation` accepts explicit
positional preparation plus one complete assembler, `PreparedExecutionRunner` runs
representation-level inputs, and `CpuBackendIntegration` supplies all three CPU collaborations.

The first clean implementation attempt exposed one planning error before any implementation was
retained: the planned API directly names Model types and its package-private composition seam
directly names a Planning type, but the Engine build does not yet declare either project. Task
0001 now requires the exact direct `modules:model` and `modules:planning` dependencies, the
focused Engine dependency inventory test, and repository-wide validation appropriate to a module-
dependency change. This is a realization of the existing outer composition direction, not an
architecture decision change.

The corrected implementation then proved a second planning omission after Engine compilation,
six focused Engine tests, one focused architecture test, and whitespace validation passed. The
integration-test module depended only on Engine, whose deliberate `implementation` dependencies
are not exported. Its test directly names CPU, Config, and Model contracts and javac must also
resolve Compiler and Runtime types exposed by the advanced Engine signatures. Task 0001 therefore
retains the integration module's Engine `implementation` dependency and adds exact ordered
`testImplementation` dependencies on Compiler, Runtime, Config, Model, and CPU. It does not widen
Engine dependencies to `api` or add unrelated integration dependencies. This is test-fixture
dependency closure, not a production architecture decision change.

Task 0001 is deliberately an advanced/integration foundation. It uses the current standalone
`CompileMode`, `GraphOptimizationConfig`, `BackendIntent`, and `PartitionScoringConfig` values and
the existing representation-level Runtime caller input. Missing `CompileConfig`, `PrepareConfig`,
and `RunOptions` aggregates do not block that seam; their final convenience ownership remains in
Config. The normal standard composition is current from Engine 0002, typed binding and result
access are current from 0003, and host materialization is current from 0004. Therefore 0001 is not
documented as the completed end-user experience.

The critical seam audit does not support the earlier broad implication that several registered
backends can already contribute to one schedule. CPU is the only supported concrete lifecycle
adapter. Its assembler deliberately owns the complete schedule for exactly one non-empty maximal
CPU partition; Metal and CUDA contain placeholders only. Task 0001 therefore takes ownership of
exactly one caller-supplied CPU integration, keeps compiled and prepared delegates behind opaque
owner-bound handles, and preserves CPU's zero/pass-through and mixed/multi-partition rejection.
It adds no public hypothetical backend interface. A package-private composition collaboration is
justified by the current CPU realization and lifecycle failure-injection tests.

The current standard composition is compatible with the architecture and required no
authoritative decision task. “Registered explicitly” permits its fixed Engine-owned factory to
construct and register only known built-in adapters in deterministic order. It does not permit
classpath or annotation scanning, `ServiceLoader`, hidden service location, mutable process-global
Engine state, or Runtime backend selection. The compile/prepare lifecycle selects eligible
partition ownership from that fixed composition. Until another concrete adapter and a shared
complete-schedule contribution contract exist, the truthful standard inventory is CPU-only.
Ordinary users will not name or construct a CPU adapter.

The supported future user surface converges on Tensor expressions, standard or advanced Engine
composition, compile/prepare/run, typed input bindings, typed/host results, one-shot convenience,
and optional autotuning. Compiler, Prepare, Runtime, and CPU integration contracts are public SPI
only and must not appear in ordinary user-facade signatures; `.internal` stays private.

One-shot execution remains Engine-owned. Complete task 0005 historically introduced the ordinary
explicit-input `Engine.forward(...)` singleton and ordered-output overloads. Complete task 0005A
replaces both without compatibility aliases using the selected four `Engine.compute(...)`
overloads. It transiently inventories provenance-free leaves from immutable Tensor expression
provenance, then lets Compiler's final ordered bindings select the actual run inputs. This is not
Compiler IR traversal or Engine-owned liveness inference, and no Tensor reference survives the
synchronous call.
Complete task 0006 lowers backward convenience to Compiler's explicit functional gradient
request. Its outward API is restricted to a scalar objective with explicit gradient targets;
0003's ordinary explicit-seed compile overload and the advanced full-request path remain
available. Engine must not inspect the graph to guess targets or invent a no-argument backward
promise whose seed/target meaning is undefined.

Diagnosis context `01a0a570-06d4-7012-814b-ca71af8676e7` showed that the absent seed could survive
as a source-only published compile-time constant with unresolved layout, no producer partition,
and no consumer partition. At that point Prepare projected only partition-node-connected values,
CPU received no source or assignment, and even a fabricated assignment would have left Engine
host-copy preflight rejecting the unresolved layout. The completed architecture-owned repair is
Compiler 0006B5 -> Prepare 0005 -> CPU 0010H: Compiler closes the canonical logical descriptor,
CPU contributes exact physical geometry through Prepare's complete source-only handoff and shared
assignment, and CPU builds a fresh initialized representation for each Runtime run state. Complete
Engine 0005A was independent of that chain because Complete Compiler 0006B4 already supplies
authoritative final ordered input bindings. Engine 0006 uses both completed seams without
an inward API change. Runtime needs no new task because current initialization, validity, ordered
publication/alias, lease, isolated-state, and cleanup contracts are sufficient.

Config 0006A mapping and tools/tuning 0001 integration belong to optional Engine 0007, not Engine
0001 or the standard untuned path. Tools/tuning 0002 remains later bounded graph/plan tuning.

Model/training checkpoint persistence may consume the completed task 0004 host-value boundary,
but must not bypass it by reading backend storage from NN or Training. The boundary supplies
detached bytes, not a checkpoint format or training workflow.

The dimensional-convolution program adds no Engine operation switch or convolution-specific
binding API. Task 0008 will exercise representative rank-one composition and first-class rank-two
and rank-three convolution through the same typed logical-input, Prepare, Runtime, and publication
mapping established by tasks 0001–0006. This is the execution-readiness gate for the later NN
layer integration checkpoint; it does not move shape inference, lowering, or kernel selection into
Engine.

Detailed task 0001 records the exact advanced API, lifecycle gate, ownership transfer, failure
rollback, file ceiling, validation, and clean documentation handoff. Its final evidence includes
7/7 focused Engine tests, 2/2 integration tests, 1/1 focused architecture test, Engine Javadoc,
the distinct-package public fixture, exact `javap` inspection, and the repository checkpoint of
3,067 tests with zero failures or errors and 28 skipped. It also records the two historical
compile failures that corrected dependency scope; neither is a final failure.

Engine 0002 is Complete. Its implementation adds a distinct ordinary
`Engine` type with `standard()`, `isClosed()`, and `close()` only. The standard factory owns one
fresh CPU integration and delegates its lifetime to the already-proved advanced lifecycle; it
does not expose that delegate. Typed compile/prepare/run signatures were reserved for task 0003 so the
ordinary surface does not inherit Compiler, Runtime representation, or CPU integration types.
The implementation and documentation contexts passed the exact 12/12 Engine, 1/1 integration,
1/1 architecture, Javadoc, public-shape, fixture, bytecode, token, Markdown, scope, status,
unchanged-Gradle/advanced-source, and whitespace gates recorded by the task.

Detailed Engine 0003 is `Complete` after implementation context
`01a0a46c-8a9e-7862-bd81-0d3495092894` and clean documentation context
`01a0a488-b8dd-7bf1-a8fc-947f531f420f`. Complete Compiler 0006B4 exposes the immutable ordered
`TensorId`/final-`ValueId` association for every caller-bindable input. Existing publication facts
preserve forward and gradient Tensor identity, order, duplicates, and aliases; Prepare preserves
caller-input and forward-then-gradient publication occurrence order; Runtime supplies isolated
execution and a metadata-independent result lease; and CPU supplies the host-storage borrow seam.
The ordinary API binds arbitrary-order caller Tensors by ID, snapshots current caller-owned host
storage once per run, preserves every forward/gradient publication occurrence and alias role, and
keeps numerical values behind explicit per-occurrence materialization. The corrected real CPU examples use
only CONTIGUOUS over resolved FLOAT32 leaves, including `seedLeaf.contiguous()` as the repeated
explicit seed. The exact lifecycle, ownership, validation order, 19-path implementation ceiling,
tests, documentation handoff, and current limitations are recorded in the task. No architecture
or inward-contract change was required.

Independent Engine 0004 planning context `01a0a4b4-9b2f-7832-977e-ed984e291a4d` found two bounded
prerequisite gaps and correctly left Engine blocked. Runtime 0015 and CPU 0010G subsequently
closed them: Runtime now lends the exact indexed publication representation while its result lease
is open, and CPU now copies an exact supported representation plus resolved static descriptor into
fresh bounded canonical bytes for all six current types. Clean planning context
`01a0a4f6-d1f9-7b61-9090-27a25b034228` fixed the ordinary API, occurrence authentication,
validation, lifecycle/concurrency protocol, delegation, tests, documentation handoff, and 23-path
ceiling in detailed task 0004. Implementation context
`01a0a502-9546-7de3-b34f-882f918f8e97` completed the executable work, and documentation context
`01a0a510-07fe-7bd1-816b-8e7318b5387b` finalized the public explanation and validation evidence.
Engine 0004 is `Complete`. Detailed Engine 0005 is also `Complete` after clean planning context
`01a0a527-b8d0-78e1-9c56-f39f73a91e26`, implementation context
`01a0a533-6b58-7d70-9cb6-3bcc9b07ee6c`, scope-correction context
`01a0a53c-4ea1-72d2-88d4-79c03a92a585`, completion context
`01a0a540-aad0-7551-856b-b10109f56a68`, and clean documentation context
`01a0a545-5c95-7cc0-84b8-ba3dd02914a2`. It adds the two exact ordinary `forward` overloads,
explicit logical inputs, an aggregate returned-payload byte bound, ordered detached host values,
one-admission lifecycle, and cleanup/failure rules within the corrected fourteen-path ceiling.
Detailed Engine 0005A is `Complete` after implementation context
`01a0afef-3f95-78d1-b56e-d44aaeaf002c` and its mandatory clean documentation finalization. It
replaced the two explicit-input `forward(...)` overloads with four `compute(...)` overloads and
uses transient Model-expression leaf inventory plus final Compiler binding selection. It executed
before CPU 0010H under the recorded sequential ordering exception because it changed only
Engine's ordinary convenience surface. Detailed Engine 0006 is `Complete` after implementation
context `01a0b11d-cc89-7590-8836-0555b05e7001` and clean documentation context
`01a0b126-f60b-7a11-8d7c-621e532952f0`. Its exact outward `backward(...)` API has no explicit
input list, fixes Compiler's absent positive-one scalar seed and ERROR policy, and returns a
detached objective plus target-aligned gradients under one aggregate byte bound. The real scalar
CPU fixture proves the objective and positive-one gradient through the completed source-only
constant chain. Engine 0005 remains `Complete`. Engine 0007–0008 remain `Draft` without detailed
specifications.

## Open questions

- Expand the standard built-in inventory only after another supported lifecycle adapter and a
  complete multi-backend schedule-composition contract exist; task 0002 fixes the current
  inventory to CPU only.
- Define mixed-backend schedule contributions only after a second concrete lifecycle adapter
  establishes a non-hypothetical consumer need. The current complete CPU assembler cannot be
  combined with another complete assembler.
- Reassess the Draft Engine 0007 optional autotuning frontier separately; do not create or specify
  it as part of completed Engine 0006.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Engine 0004 adds one final immutable `HostTensorValue` in the ordinary Engine facade. It
  exposes `DataType dataType()`, `Shape shape()`, `long elementCount()`, `long byteSize()`, and a
  fresh read-only `ByteBuffer bytes()` view over detached canonical row-major bytes. Multi-byte
  values use big-endian order; BOOL uses one canonical `0` or `1` byte and BFLOAT16 retains its
  represented 16 bits. It exposes no descriptor layout or physical span.
- Ordinary materialization is an explicit lazy per-occurrence call
  `RunResult.materialize(RunResult.Publication publication, long maximumBytes)`. Selection uses
  the exact publication object from that result, not index, Tensor equality, `TensorId`, role, or
  descriptor equality. Each successful call returns a fresh independent snapshot; repeated or
  aliased occurrences are never cached, merged, or deduplicated.
- The caller-supplied non-negative byte limit applies to the checked canonical logical byte count
  before allocation or physical access. Static Shape, resolved layout, checked element/byte
  counts, the byte limit, and the JVM array-size ceiling are validated before copying. All six
  current data types and zero-element values are covered; binding-dependent Shape or unresolved
  layout fails closed at this first boundary.
- Materialization is valid only while the originating result is open. Result close and Engine
  close wait for an admitted synchronous materialization; concurrent materializations on one
  result are serialized. A completed `HostTensorValue` owns no closeable resource and remains
  readable after result or Engine closure. Failure publishes no partial value and preserves the
  original unchecked failure with distinct cleanup failures suppressed under the existing rules.
- Runtime 0015 owns only leased result-indexed `BufferRepresentation` access. CPU 0010G owns the
  descriptor-aware physical copy and canonical encoding. Engine owns publication selection,
  outward lifecycle coordination, byte-budget validation defense, and the ordinary payload. No
  Model, Prepare, Runtime, CPU, NN, Training, or Checkpoint contract is exposed through the
  ordinary signatures.
- Engine 0005 historically introduced `Engine.forward(Tensor, List<Tensor>, long)` and
  `Engine.forward(List<Tensor>, List<Tensor>, long)`; completed 0005A removes them without aliases.
- Engine 0005A replaces those methods rather than retaining aliases. It selects `compute(Tensor)`,
  `compute(Tensor, long)`, `compute(List<Tensor>)`, and `compute(List<Tensor>, long)`; no-limit
  forms delegate with `Long.MAX_VALUE`, while checked arithmetic and per-value JVM array ceilings
  remain. The implementation inventories reachable provenance-free leaves identity-safely and
  selects only matches in final `CompiledGraph.inputs()` order.
- Complete Engine 0006 selects one explicit-target `Engine.backward(...)` method and one immutable
  `ScalarObjectiveBackwardResult` separating the detached scalar objective from target-aligned
  gradients. Compiler supplies the absent positive-one scalar seed and ERROR policy; existing
  explicit-seed ordinary compilation and the advanced full request remain unchanged.

## Risks

- Becoming a service locator or absorbing backend implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
