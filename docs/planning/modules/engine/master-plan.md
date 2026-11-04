# [Engine](../../../glossary.md#term-engine) Master Plan

> **Editorial historical reading key.**
>
> Read this as a historical implementation map, not an independent API authority. Engine coordinates
> Compiler artifacts, shared Prepare, backend integrations, and outward handles; Model owns numerical
> semantics, backends own lowering and physical access, and Runtime owns the inward preparation and
> unique run-lease protocol. Caller input storage remains borrowed; publication metadata is not a
> detached value. Explicit materialization produces the independent host copy.
>
> The task rows preserve successive contracts: the inference session was introduced and later removed,
> numerical-profile propagation was later removed, and the original mixed-transfer domain was bounded.
> Consult the [current
> lifecycle](../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
> and [explicit composition
> contract](../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition)
> separately. Engine closes results, then preparations, then integrations; recorded acceptance
> evidence and conflicting frontier snapshots remain historical assertions, not new verification.

## Goal and authority

[Engine](../../../glossary.md#term-engine) is the public [lifecycle](../../../glossary.md#lifecycle) [facade](../../../glossary.md#term-facade) and explicit [composition root](../../../glossary.md#term-composition-root) for [compiler](../../../glossary.md#term-compiler), [Prepare](../../../glossary.md#prepare),
[Runtime](../../../glossary.md#term-runtime), tuning, and concrete [backends](../../../glossary.md#backend). This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`,
`modules/engine`, and the [compile](../../../glossary.md#compile)/prepare/[run](../../../glossary.md#term-run) lifecycle headings, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0015: Explicit Engine backend composition](../../../design/decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([backend composition glossary](../../../glossary.md#backend-composition))</span>
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](../../../design/decisions/0016-cpu-metal-mixed-owner-schedule.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [Metal glossary](../../../glossary.md#term-metal); [owner glossary](../../../glossary.md#term-owner); [prepared schedule glossary](../../../glossary.md#prepared-schedule--preparedschedule))</span>
- [ADR 0017: Reusable inference session facade](../../../design/decisions/0017-reusable-inference-session-facade.md)<span data-glossary-navigation="true"> ([inference glossary](../../../glossary.md#term-inference))</span>
- [ADR 0026: Profile-free numerical semantics](../../../design/decisions/0026-profile-free-numerical-semantics.md)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../glossary.md#profile-free-numerical-semantics))</span>

## [Lifecycle](../../../glossary.md#lifecycle) position

Current [API](../../../glossary.md#term-application-programming-interface-api) after Engine0021:

```text
Tensor outputs -> Engine compile handle -> Engine PreparedExecution
caller inputs  -> Engine.run or Engine.runValues -> leased RunResult
publication    -> explicit materialization -> detached HostTensorValue
```

[Engine](../../../glossary.md#term-engine) owns composition and outward handles. Its [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution) owns one inward preparation.
[Prepare](../../../glossary.md#prepare) constructs the recipe; [Runtime](../../../glossary.md#term-runtime) owns the inward PreparedExecution, its unique [run](../../../glossary.md#term-run) [lease](../../../glossary.md#term-lease),
and physical prepared-[resource lifetime](../../../glossary.md#ownership-borrowing-and-lease). One-shot compute/backward and tuning use the same boundary.

## [Scope](../../../glossary.md#term-scope) and non-goals

[Engine](../../../glossary.md#term-engine) owns standard and advanced composition, [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names)/preparation orchestration, [owner](../../../glossary.md#term-owner)-bound
prepared handles, typed input [binding](../../../glossary.md#term-binding) and [publication](../../../glossary.md#term-publication) [metadata](../../../glossary.md#term-metadata), explicit
[host](../../../glossary.md#term-host) [materialization](../../../glossary.md#materialization), one-shot compute/backward workflows, and bounded optional [model](../../../glossary.md#term-model)-[autotuning](../../../glossary.md#autotuning--model-autotuning)
composition.

Engine0021 delivered existing Engine [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution) as the sole ordinary public prepared owner;
Engine.[prepare](../../../glossary.md#prepare)/[run](../../../glossary.md#term-run)/runValues replaced the thin session factory/type. [Runtime](../../../glossary.md#term-runtime)'s prepared owner,
advanced [lifecycle](../../../glossary.md#lifecycle), [TrainingSession](../../../glossary.md#training-session--trainingsession) and [detached](../../../glossary.md#detached-model-results) results remain. No replacement [facade](../../../glossary.md#term-facade).

Engine does not implement [kernels](../../../glossary.md#kernel), [lowering](../../../glossary.md#lowering), [backend](../../../glossary.md#backend) storage, [graph](../../../glossary.md#graph) [optimizer](../../../glossary.md#term-optimizer) passes, Runtime
state, a [service locator](../../../glossary.md#term-service-locator), reflective discovery, [Tensor](../../../glossary.md#tensor)-owned execution/backward state, or hidden
backend selection during a run.

### Audit-driven [Metal](../../../glossary.md#term-metal)-only source [publication](../../../glossary.md#term-publication) — [Ready](../../../glossary.md#term-ready)

Actual complete-[signature](../../../glossary.md#term-signature) execution found [constant](../../../glossary.md#term-constant) cotangents and source-only publications
incorrectly requiring a [CPU](../../../glossary.md#term-cpu) adapter in an explicitly Metal-only [Engine](../../../glossary.md#term-engine). The
[existing source-publication protocol extension](../../../architecture/contracts/runtime-prepare-engine.md)
authorizes completing the same producerless-value geometry and initialization/import/export
path for a registered supporting Metal composition. Reuse existing [Prepare](../../../glossary.md#prepare)/[Runtime](../../../glossary.md#term-runtime) [ownership](../../../glossary.md#term-ownership-meanings);
source-[owner](../../../glossary.md#term-owner) selection and composite assembly must agree deterministically. Preserve existing
CPU behavior and caller/constant [authentication](../../../glossary.md#term-authentication), exact physical sizes and pre-finalization
[validation](../../../glossary.md#term-validation). No synthetic [operation](../../../glossary.md#operation), CPU fallback, secondary initializer, execution loop or
new public [facade](../../../glossary.md#term-facade) is in [scope](../../../glossary.md#term-scope). Actual Metal-only constant [gradients](../../../glossary.md#term-gradient), source-only raw values,
reuse and failure cleanup plus independent Class C review are acceptance gates.

## Stable invariants and [ownership](../../../glossary.md#term-ownership-meanings)

- Engine is the outer composition root. It explicitly constructs or takes ownership of concrete
  adapters; concrete backends and inward modules never depend on Engine.<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [composition root glossary](../../../glossary.md#term-composition-root); [backends glossary](../../../glossary.md#backend); [modules glossary](../../../glossary.md#term-module))</span>
- `Engine.standard()` owns one fresh fixed [CPU](../../../glossary.md#term-cpu) composition. `Engine.Builder` takes explicit
  ownership of opened CPU and/or Metal integrations and prepares CPU-only, Metal-only, or supported
  mixed-owner schedules through the same shared Prepare transaction. <span data-glossary-navigation="true"> ([Metal glossary](../../../glossary.md#term-metal); [owner glossary](../../../glossary.md#term-owner); [Prepare glossary](../../../glossary.md#prepare); [transaction glossary](../../../glossary.md#term-transaction))</span>`AdvancedEngine` remains a
  supported CPU-only lower-level integration boundary.
- Ordinary compile/prepare/run/runValues signatures expose Engine/Model
  values, not Compiler, Prepare, Runtime, or backend SPI identities. Builder exposes CPU/Metal
  integration identities only at the explicit construction boundary.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([compile glossary](../../../glossary.md#compile); [run glossary](../../../glossary.md#term-run); [signatures glossary](../../../glossary.md#term-signature); [Model glossary](../../../glossary.md#term-model); [Compiler glossary](../../../glossary.md#term-compiler); [Runtime glossary](../../../glossary.md#term-runtime); [SPI glossary](../../../glossary.md#term-service-provider-interface-spi))</span>
- Ordinary and advanced prepared handles expose direct closeable ownership. Each
  prepared owner controls exactly one inward Runtime execution; closing it unregisters and closes
  that execution once without closing an already returned result.
- Runtime remains the unique run-lease authority. Engine synchronizes outward admission and
  delegate ownership without adding a second resource lease or waiting close protocol. A run that
  already acquired its prepared lease may finish; prepared-owner close does not wait, and the final
  lease release performs deferred prepared-resource cleanup.<span data-glossary-navigation="true"> ([lease glossary](../../../glossary.md#term-lease); [admission glossary](../../../glossary.md#term-admission); [resource lease glossary](../../../glossary.md#ownership-borrowing-and-lease); [release glossary](../../../glossary.md#term-release))</span>
- Every one-shot, tuning trial, correctness run, loser, fallback rollback, and rejected
  preparation is closed exactly once. A selected standalone preparation has one published Engine
  owner.<span data-glossary-navigation="true"> ([rollback glossary](../../../glossary.md#term-rollback))</span>
- Engine shutdown waits for admitted operations, closes retained results in reverse
  run-publication order, then ordinary and advanced prepared handles in their shared reverse
  preparation-publication order, and finally integrations in reverse registration order.<span data-glossary-navigation="true"> ([publication glossary](../../../glossary.md#term-publication))</span>
- Materialization borrows one authenticated publication only while its result is open and returns
  a detached immutable value. Engine never exposes a physical representation or backend handle.<span data-glossary-navigation="true"> ([Materialization glossary](../../../glossary.md#materialization); [detached glossary](../../../glossary.md#detached-model-results); [immutable glossary](../../../glossary.md#term-immutable))</span>

## Dependencies

Current allowed direct dependencies are `modules/model`, `modules/planning`, `modules/compiler`,
`modules/runtime`, `modules/prepare`, `modules/config`, `modules/trace`, `modules/backend/cpu`,
`modules/backend/metal`, and `tools/tuning`. Each edge exists only for a contract named by
composition. No inward semantic role or [concrete backend](../../../glossary.md#concrete-backend) may depend on `modules/engine`.
A future physical [modules](../../../glossary.md#term-module)/training or modules/[checkpoint](../../../glossary.md#term-checkpoint) directory would not make that outward
[owner](../../../glossary.md#term-owner) an inward role; its separately approved [dependency](../../../glossary.md#term-dependency) contract, not a path prefix, must govern.
Engine0024 removes the unused [CUDA](../../../glossary.md#term-cuda) marker edge; the separate unsupported CUDA source remains.

## Package map

All types remain in `io.github.pho001.synaptik.engine`; the root is a deliberately small [facade](../../../glossary.md#term-facade),
not a catch-all service [registry](../../../glossary.md#term-registry).

| Surface | Types and role |
|---|---|
| Ordinary public [lifecycle](../../../glossary.md#lifecycle) | [`Engine`](../../../glossary.md#term-engine), nested single-use `Engine.Builder`, `CompiledGraph`, closeable [`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution), closeable [`RunResult`](../../../glossary.md#run-result--runresult), [detached](../../../glossary.md#detached-model-results) [`HostTensorValue`](../../../glossary.md#host-tensor-value--hosttensorvalue), and `ScalarObjectiveBackwardResult`. |
| Public [autotuning](../../../glossary.md#autotuning--model-autotuning) | `ModelAutotuningRequest` and `ModelAutotuningPreparation`, which publish one retained ordinary prepared handle plus outcome/evidence. |
| Advanced public lifecycle | `AdvancedEngine`, `AdvancedCompiledGraph`, closeable `AdvancedPreparedExecution`, and closeable `AdvancedRunResult`. |
| Package-private composition | `EngineBackendComposition`, [CPU](../../../glossary.md#term-cpu) and [Metal](../../../glossary.md#term-metal) realizations, the ordered [backend](../../../glossary.md#backend) registry, selected-adapter [preparation result](../../../glossary.md#model-autotuning-request--preparation-result), representative execution/correctness machinery, lifecycle registries, and cleanup arbitration. |

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md) | Advanced composition and representation-level [lifecycle](../../../glossary.md#lifecycle) foundation | Complete | [Compiler](../../../glossary.md#term-compiler) 0006B3; [CPU](../../../glossary.md#term-cpu) 0010F; current [Prepare](../../../glossary.md#prepare) 0003–0004 and [Runtime](../../../glossary.md#term-runtime) 0010 contracts | Own exactly one supported CPU lifecycle adapter and expose [opaque](../../../glossary.md#term-opaque) [owner](../../../glossary.md#term-owner)-bound [compile](../../../glossary.md#compile)/prepare handles plus a representation-level [run](../../../glossary.md#term-run) seam for integration and tests. Reject zero/pass-through and mixed/multi-[partition](../../../glossary.md#partition) preparation; do not claim nonexistent schedule composition. |
| [0002](tasks/0002-standard-built-in-composition.md) | Standard built-in composition | Complete | 0001; supported built-in [backend](../../../glossary.md#backend) adapters | Added a distinct ordinary `Engine.standard()` construction path that owns one fresh CPU-only built-in composition without exposing `CpuBackendIntegration`, advanced handles, or inward integration contracts. Preserved `AdvancedEngine.takeOwnership(...)`; mixed-backend success still requires another concrete adapter and a justified complete schedule-composition contract. |
| [0003](tasks/0003-typed-logical-input-binding-and-published-result-access.md) | Typed logical input [binding](../../../glossary.md#term-binding) and published-result access | Complete | 0001–0002; [Compiler 0006B4](../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md); current [publication](../../../glossary.md#term-publication)/Prepare/Runtime contracts | Added the ordinary compile, prepare, [TensorId](../../../glossary.md#tensorid)-based [host](../../../glossary.md#term-host)-input binding, [synchronous](../../../glossary.md#term-synchronous) run, and [metadata](../../../glossary.md#term-metadata)-only publication/result lifecycle without exposing inward [identities](../../../glossary.md#term-identity-meanings), coordinates, representations, or advanced handles. |
| [0004](tasks/0004-explicit-host-materialization-boundary.md) | Explicit host [materialization](../../../glossary.md#materialization) boundary | Complete | 0003; Runtime 0015; CPU 0010G | Lazily copies one exact selected [publication occurrence](../../../glossary.md#publication-occurrence) into a [detached](../../../glossary.md#detached-model-results) [immutable](../../../glossary.md#term-immutable) bounded [`HostTensorValue`](../../../glossary.md#host-tensor-value--hosttensorvalue) with [canonical](../../../glossary.md#term-canonical-form) [row-major](../../../glossary.md#term-row-major) [big-endian](../../../glossary.md#text-encoding-and-file-boundaries) bytes. Serializes copying against result/[Engine](../../../glossary.md#term-engine) close and exposes no representation, backend storage, [arena](../../../glossary.md#term-arena), [`MemorySegment`](../../../glossary.md#term-memorysegment), [`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot), or slot. |
| [0005](tasks/0005-one-shot-forward-convenience.md) | Engine-owned one-shot forward convenience | Complete | 0002–0004; Runtime 0015; CPU 0010G | Added exact single-output and ordered-output `Engine.forward(...)` overloads with explicit logical inputs and one aggregate canonical-byte limit. Each call freshly compiles, prepares, runs, materializes every forward publication in order, and cleans up under one Engine [admission](../../../glossary.md#term-admission); it adds no [Tensor](../../../glossary.md#tensor) execution or [cache](../../../glossary.md#term-cache). |
| [0005A](tasks/0005a-automatic-input-discovery-and-compute-convenience.md) | Automatic input discovery and compute convenience | Complete | 0005; Compiler 0006B4 | Replaced the two explicit-input one-shot `forward(...)` methods with four `compute(...)` overloads. Inventories reachable [provenance](../../../glossary.md#provenance)-free Tensor [leaves](../../../glossary.md#term-leaf) transiently by object [identity](../../../glossary.md#term-reference-identity), then selects only matching Tensors in authoritative final `CompiledGraph.inputs()` order; adds no Compiler [IR](../../../glossary.md#term-intermediate-representation-ir) traversal, [liveness](../../../glossary.md#term-liveness) [inference](../../../glossary.md#term-inference), retained Tensor state, or cache. |
| [0006](tasks/0006-one-shot-scalar-objective-backward-convenience.md) | Engine-owned one-shot [scalar](../../../glossary.md#term-scalar)-[objective](../../../glossary.md#term-objective) backward convenience | Complete | 0005A; Compiler 0006B5; Prepare 0005; CPU 0010H | Added one explicit-[target](../../../glossary.md#term-target) `Engine.backward(...)` call returning a detached [scalar objective](../../../glossary.md#loss) plus immutable target-aligned [gradients](../../../glossary.md#term-gradient). Reuses 0005A's transient leaf-inventory/final-binding selection seam with no explicit input list and Compiler's absent scalar unit [seed](../../../glossary.md#term-seed) and ERROR policy through the completed source-only [constant](../../../glossary.md#term-constant) chain; retains the explicit-seed ordinary compile and advanced full-request paths. |
| [0006A](tasks/0006a-representative-tuning-execution-and-safe-fallback.md) | Representative tuning execution and safe fallback foundation | Complete | 0003; 0006; Runtime 0010/0015; Prepare 0004; CPU 0010I; reviewed Config 0006A and tools/tuning 0001 contracts | Added package-private representative-input binding, synchronous complete trial execution, cleanup, failure isolation, fresh selected preparation, and deterministic strict/allowed fallback. Adds no public [API](../../../glossary.md#term-application-programming-interface-api), tuning algorithm, cache work, or tools/tuning [dependency](../../../glossary.md#term-dependency). |
| [0007](tasks/0007-optional-model-autotuning-composition.md) | Optional [model](../../../glossary.md#term-model)-[autotuning](../../../glossary.md#autotuning--model-autotuning) composition | Complete | 0002; 0005; [0006A](tasks/0006a-representative-tuning-execution-and-safe-fallback.md); Config 0006A; tools/tuning 0001; [CPU 0010I](../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md) | Added one CPU-only public representative request with caller-defined model identity; maps the sole [handoff](../../../glossary.md#term-handoff) to [occurrence](../../../glossary.md#term-occurrence) 0/partition 0/[weight](../../../glossary.md#term-weight) 1, invokes cache-first tuning, and returns fresh selected preparation plus translated evidence or explicit safe fallback outside Runtime. |
| [0008](tasks/0008-engine-lifecycle-capability-checkpoint.md) | Engine lifecycle capability [checkpoint](../../../glossary.md#term-checkpoint) | Complete | 0001–0007; Compiler 0006B3–0006B6; Prepare 0003/0003A/0004/0005; Runtime 0010/0012/0014/0015; CPU 0008/0008A/0010F–0010I; Config 0006A; tools/tuning 0001 | Consolidated repository-wide evidence and current documentation for the standard, advanced, typed, materialized, one-shot forward/backward, bounded tuning/fallback, cleanup, failure, [concurrency](../../../glossary.md#term-concurrency), dependency, and public dimensional-[convolution](../../../glossary.md#term-channels-first-convolution-layers) execution boundaries. |
| [0008A](tasks/0008a-representative-complete-plan-correctness-oracle.md) | Representative complete-plan correctness [oracle](../../../glossary.md#term-test-oracle) | Complete | 0004, 0006A–0008; Runtime 0015; CPU 0010G and 0010J | Added the smallest package-private Engine oracle that [captures](../../../glossary.md#term-capture-meanings) all ordered representative publications as bounded canonical bytes and compares later complete recipes by exact [represented-bit](../../../glossary.md#storage-mapping-contiguity-and-injectivity) equality. It owns no timing, tuning algorithm, cache, public request, selected preparation, or fallback policy. |
| [0009](tasks/0009-public-complete-plan-autotuning-composition.md) | Public complete-plan autotuning composition | Complete | 0006A–0008A; Config 0006B; tools/tuning 0002; CPU 0010I–0010J; Prepare 0004; Runtime 0010/0015 | Composed the authenticated Phase-1 decision into CPU's complete-plan [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters), adapted exact correctness and fresh execution to generic Phase 2, freshly prepared the authenticated winner, and extended existing public evidence without a new request, [operation](../../../glossary.md#operation), or top-level public type. |
| [0010](tasks/0010-prepared-handle-ownership-and-closure.md) | Prepared-handle [ownership](../../../glossary.md#term-ownership-meanings) and closure | Complete | Runtime 0016; Prepare 0006; 0001–0009 | Made ordinary and advanced prepared handles explicit closeable owners, closed every one-shot/trial/loser/[rollback](../../../glossary.md#term-rollback) preparation exactly once, and closed retained preparations after results and before backend integration shutdown. |
| [0011](tasks/0011-compile-artifact-projection-boundary-reconciliation.md) | Compile-[artifact](../../../glossary.md#term-artifact) projection boundary reconciliation | Complete | 0010; Prepare 0003A/0005/0006; CPU 0010F/0010H–0010J | Restored Engine/Prepare ownership of `CompileArtifacts` orchestration, removed CPU's production Compiler edge, and preserved CPU preparation/tuning behavior through stable projections. |
| [0012](tasks/0012-runtime-prepare-engine-authority-reconciliation.md) | Runtime, Prepare, and Engine authority reconciliation | Complete | 0011; Compiler 0006B8; current Engine/Prepare/Runtime APIs | Reconciled and independently reviewed the root, scoped contract, and focused explanations against the current owner-bound CPU Engine lifecycle and ordered Runtime schedule-step model. |
| [0013](tasks/0013-prepared-execution-guide-status-reconciliation.md) | [Prepared execution](../../../glossary.md#prepared-execution--preparedexecution) guide status reconciliation | Complete | 0012; Model 0025M; current Engine lifecycle and autotuning APIs | Reconciled and independently reviewed current bounded two-phase tuning, prepared-handle closure, and detached output materialization explanations. |
| [0014](tasks/0014-explicit-backend-composition-architecture.md) | Explicit [backend composition](../../../glossary.md#backend-composition) architecture | Complete | 0013; [Metal](../../../glossary.md#term-metal) 0004; current Compiler/Prepare/Runtime contracts | Selected the public Engine builder, concrete ownership overloads, fixed capability/availability inventory, construction and [close lifecycle](../../../glossary.md#ownership-borrowing-and-lease), single-owner cold routing, Metal configuration ownership, and [fail-closed](../../../glossary.md#term-fail-closed) mixed-owner boundary without changing production behavior. |
| [0015](tasks/0015-cpu-metal-single-owner-composition.md) | CPU/Metal single-owner Engine composition | Complete | 0014; Metal 0004; current Compiler/Prepare/Runtime contracts | Added the concrete builder ownership lifecycle, fixed [registry](../../../glossary.md#term-registry), single-owner cold routing, direct adapter ingress/materialization, public Metal lifecycle integration, CPU-only tuning gate, and real CPU/Metal integration coverage. |
| [0016](tasks/0016-cpu-metal-mixed-owner-schedule.md) | CPU/Metal mixed-owner schedule and transfer | Complete | 0015; Prepare 0006; Runtime 0016; Metal 0004 | Added independently reviewed owner-indexed shared representations, Compiler-ordered caller inputs, complete-set Prepare routing, explicit bounded bidirectional [F32](../../../glossary.md#floating-point-precision-and-rounding) CPU/Metal transfer, ordered mixed scheduling, direct per-occurrence handle [capture](../../../glossary.md#term-capture-meanings), rollback, and real public lifecycle evidence. |
| [0017](tasks/0017-reusable-inference-session-api.md) | Reusable inference session API | Complete | 0016 | Added the independently reviewed thin public session that prepares one owner-bound [graph](../../../glossary.md#graph) once and delegates repeated/concurrent input binding, run, result, materialization, and close behavior to the existing lifecycle without another compiler, scheduler, runner, or result. |
| [0018](tasks/0018-explicit-numerical-profile-propagation-spine.md) | Explicit numerical-profile propagation spine | Complete | Model 0027; Config 0006; current [Planning](../../../glossary.md#term-planning)/Compiler/Prepare/Engine contracts; CPU and Metal fail-closed migration points | Delivered the independently approved [atomic](../../../glossary.md#resource-transaction-and-rollback) cutover across profile-qualified Planning queries, Compiler artifacts, Prepare projection, Engine/AdvancedEngine capture, every caller, and profile-keyed backend plans/caches while keeping Runtime/[Trace](../../../glossary.md#trace) profile-free and relaxed capability disabled. |
| [0019](tasks/0019-lifecycle-and-mixed-owner-test-repair.md) | Lifecycle and mixed-owner [regression](../../../glossary.md#term-regression) repair | Complete | 0016–0018; Metal 0077; project review | Make copy/close tests deterministic, restore real [heterogeneous](../../../glossary.md#term-heterogeneous) transfer fixtures, and correct public tuning [Javadoc](../../../glossary.md#term-javadoc). |
| [0020](tasks/0020-backend-native-layout-relocation.md) | Mechanical backend/native layout cutover | Complete | CP0005D Complete; user-approved layout | Delivered parent/child mappings, coordinated layout/consumers and native package/[device](../../../glossary.md#term-device) checks; independent review and combined test/Javadoc passed, identities/packages/[kernels](../../../glossary.md#kernel)/[ABI](../../../glossary.md#term-application-binary-interface-abi) preserved. |
| [0021](tasks/0021-prepared-execution-fresh-value-binding.md) | Sole ordinary prepared owner and [fresh-value binding](../../../glossary.md#fresh-value-binding--enginerunvalues) | Complete | Complete 0017/0019/0020; reviewed P2 and approved simplification | Removed session API/delegates, migrated consumers, added [Engine.runValues](../../../glossary.md#fresh-value-binding--enginerunvalues); independent approval and native-enabled test/Javadoc gate passed. |
| [0022](tasks/0022-result-publication-and-closed-payload-cleanup.md#result) | Result publication rollback and closed-[payload](../../../glossary.md#artifact-envelope-and-atomic-publication) cleanup | Complete | Engine0021/Runtime0016; actual before evidence; independent final approval | Frozen publication regression failed-before/passed-after; 417 affected tests passed, 2 opt-in timing skips; public F32/[F64](../../../glossary.md#floating-point-precision-and-rounding) bounded-[heap](../../../glossary.md#term-heap)/[lifetime](../../../glossary.md#term-lifetime)/[race](../../../glossary.md#term-race) proof passed. Existing owners/counts/metadata/caller storage preserved. |
| [0023](tasks/0023-reentrant-close-and-lease-admission.md#result) | [Reentrant](../../../glossary.md#memory-management-atomic-counters-and-complexity) close and failure-atomic [lease](../../../glossary.md#term-lease) admission | Complete | Engine0022/Runtime0016; actual real-[GPU](../../../glossary.md#term-gpu) self-wait and bounded-heap lease-allocation failures | Exact reviewed nine-file correction; affected728pass/12existing skips, public Metal34pass and unchanged real self-close/[OOM](../../../glossary.md#term-out-of-memory-oom) consumers pass; final independent source/evidence/docs APPROVE0. No native-memory/performance/[release](../../../glossary.md#term-release) claim. |
| [0024](tasks/0024-versioned-local-distribution.md#result) | Versioned local production [distribution](../../../glossary.md#term-distribution) | Complete | Complete0023/Metal0080/81/82; exact product closure; final independent source/evidence/docs APPROVE0 | Actual eighteen-product archives, uncached extracted-source compile and external [Maven](../../../glossary.md#term-maven)/[Gradle](../../../glossary.md#term-gradle) CPU/Metal train/save/load/new-train consumers pass. Final source freeze/legal grant/release acceptance remain separate. |

## [Model](../../../glossary.md#term-model)-execution program and immediate frontier

Engine0023 corrective frontier: None after independent final source/evidence/docs APPROVE0.
Its exact reviewed source and actual repaired consumers are accepted. The first failed public-
consumer [compile](../../../glossary.md#compile) stays FAILED; the real [Trace](../../../glossary.md#trace) [API](../../../glossary.md#term-application-programming-interface-api) export repair passed separately.
Engine0022 remains Complete for [publication](../../../glossary.md#term-publication) [rollback](../../../glossary.md#term-rollback) and closed-[payload](../../../glossary.md#artifact-envelope-and-atomic-publication) retention with its
independent reviews and actual proof; no complete production [release](../../../glossary.md#term-release) is implied.

Engine0024 is the independent build/[distribution](../../../glossary.md#term-distribution) [Ready](../../../glossary.md#term-ready) frontier under the user-authorized
production goal. Its coordinated [dependency](../../../glossary.md#term-dependency) clarification and bounded contract are independently
approved. This technical work does not claim a final source freeze or public redistribution grant.

[Prepare0015 symbolic-length Scan CPU vertical](../prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu))</span>
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning [validation](../../../glossary.md#term-validation),
the sole bounded probe and configured root test/[Javadoc](../../../glossary.md#term-javadoc) PASS. Its [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) named-B/[runtime](../../../glossary.md#term-runtime)-length
forward and actual reverse use existing runValues/[admission](../../../glossary.md#term-admission)/publication [owners](../../../glossary.md#term-owner), without a new
[memo](../../../glossary.md#cache-and-memoization)/session/type/[lease](../../../glossary.md#term-lease). Timing limitations, including [nonstationary](../../../glossary.md#term-nonstationary) reverse gaps, are in the Result;
no performance parity or successor launch follows.

The [P2 program](../../model-execution-refactor.md) incorporates the five accepted product choices.
Engine0021 removed [InferenceSession](../../../glossary.md#former-inference-session--inferencesession)/[Engine](../../../glossary.md#term-engine).session and delivered
[Engine.runValues](../../../glossary.md#fresh-value-binding--enginerunvalues)(prepared, values) in compiled order without mutating [prototypes](../../../glossary.md#term-prototype). Existing
Engine.[run](../../../glossary.md#term-run) retains exact-ID/arbitrary-order matching. Both adopt former session-strength admission:
Engine first, prepared null/[owner](../../../glossary.md#term-owner)/closed gate, then inputs; direct run's error precedence changes.
No new public type; Engine prepared wrapper versus Runtime unique lease boundary stays intentional.

At the completed program [checkpoint](../../../glossary.md#term-checkpoint) the frontier was None; Engine0022's separately admitted repair
is above. Prepare0015 is Complete. Engine0021 is Complete after independent Class C approval and
the coordinator's native-enabled integrated test/Javadoc gate. Its brief owns evidence. Training,
benchmark, reflective hooks and integration consumers use direct prepared [ownership](../../../glossary.md#term-ownership-meanings). No generic
Training/Checkpoint relocation or cold [codec](../../../glossary.md#term-codec) was added by Engine0021. Training0007 is Complete
after final independent APPROVE and the coordinator gate. Authorized Training frontier: None.
Compiler0008 and the selected model-execution program are Complete; the former Draft-candidate
claim is obsolete. [Prepare0014 symbolic Scan vertical](../prepare/tasks/0014-symbolic-scan-cpu-vertical.md)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../../glossary.md#symbolic-scan))</span>
is Complete (PlanningAuthorized=true; ImplementationAuthorized=true record historical admission).
It qualifies scoped numeric admission/actual publication through existing prepared ownership and
run/runValues, not a memo/session/second lease. Independent code/docs APPROVE, 0 open P1/P2;
configured source-matched root gate PASS, MAIN accepted 2026-10-25. Engine source frontier None;
user “jdi na to” approved source, MAIN verified Ready 2026-10-25 17:00 UTC. MAIN completed
serial integration/shared docs and the configured vertical join; Parallel group None.
Shared-document owner:
model-execution coordinator; parallel group None.

Engine0020 is Complete on 2026-10-16: [backend](../../../glossary.md#backend)/native [layout](../../../glossary.md#layout) integrated with parent-and-child
projectDir mappings, independent review, native package/[device](../../../glossary.md#term-device) checks and combined test/Javadoc.
The :backends:* [identities](../../../glossary.md#term-identity-meanings), packages, [kernels](../../../glossary.md#kernel), [ABI](../../../glossary.md#term-application-binary-interface-abi) and [dependency direction](../../../glossary.md#java-platform-and-build-tools) remain unchanged;
Training0007 now relocates Training only; Checkpoint has not moved. Generic [modules](../../../glossary.md#term-module) require exact role-aware architecture
checks, not blanket directory exemptions. Completed evidence is not rerun for P2 [planning](../../../glossary.md#term-planning).

## Milestones and current frontier

- Engine0023 corrective source frontier: None. Its bounded acceptance and Engine0022/Prepare0015 evidence remain separate from wider native-memory and final [release](../../../glossary.md#term-release) gates.
- Explicit [CPU](../../../glossary.md#term-cpu) composition and advanced [lifecycle](../../../glossary.md#lifecycle) are Complete through 0002.
- Ordinary typed compile/prepare/run, publication, host materialization, compute, and backward are
  Complete through 0006.<span data-glossary-navigation="true"> ([compile glossary](../../../glossary.md#compile); [prepare glossary](../../../glossary.md#prepare); [run glossary](../../../glossary.md#term-run); [publication glossary](../../../glossary.md#term-publication); [host glossary](../../../glossary.md#term-host); [materialization glossary](../../../glossary.md#materialization))</span>
- Representative and complete-plan autotuning composition plus its capability checkpoint are
  Complete through 0009.<span data-glossary-navigation="true"> ([autotuning glossary](../../../glossary.md#autotuning--model-autotuning); [checkpoint glossary](../../../glossary.md#term-checkpoint))</span>
- Prepared-handle ownership, closure, and compile-artifact projection reconciliation are Complete
  through 0011.<span data-glossary-navigation="true"> ([artifact glossary](../../../glossary.md#term-artifact))</span>
- Explicit composition architecture and its independently reviewed CPU/Metal single-owner and
  bounded mixed-owner implementations are Complete through 0016.<span data-glossary-navigation="true"> ([Metal glossary](../../../glossary.md#term-metal); [owner glossary](../../../glossary.md#term-owner))</span>
- The reusable [inference](../../../glossary.md#term-inference)-session [facade](../../../glossary.md#term-facade) from exact base `4fc4fd3d` is Complete as 0017 after the
  required Class C and narrow documentation review sequence.
- Historical numerical-profile spine 0018 is Complete; Model 0032 removed its profile API and
  propagation. Current lifecycle is profile-free. Audit repair 0019 is Complete.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
- Engine0021 is Complete: prepared-only ordinary [ownership](../../../glossary.md#term-ownership-meanings), positional fresh inputs and migrated consumers.

## Live risks and gates

- Preserve 0016's exact owner-indexed representation assignment, explicit transfer preflight,
  shared Prepare transaction, and immutable direct-reference schedule. Registration of CPU and
  Metal must not imply fallback, discovery, conversion, or an unsupported transfer path.<span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner); [preflight glossary](../../../glossary.md#term-preflight); [Prepare glossary](../../../glossary.md#prepare); [transaction glossary](../../../glossary.md#term-transaction); [immutable glossary](../../../glossary.md#term-immutable); [CPU glossary](../../../glossary.md#term-cpu); [Metal glossary](../../../glossary.md#term-metal))</span>
- Preserve one outward registered prepared handle for every prepared execution. Close all
  temporary, losing, and rollback preparations exactly once. Engine shutdown must retain the
  results-first, preparations-second, integrations-last order above.
  Never duplicate Runtime's lease protocol.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../glossary.md#prepared-execution--preparedexecution); [rollback glossary](../../../glossary.md#term-rollback); [Engine glossary](../../../glossary.md#term-engine); [Runtime glossary](../../../glossary.md#term-runtime); [lease glossary](../../../glossary.md#term-lease))</span>
- Preserve Engine0021's removal of the thin InferenceSession without replacing it. Reuse prepared metadata/adapters;
  no second Runtime lease or compile/prepare/provider/registry work per run. TrainingSession stays
  for optimizer/transaction state; mechanical migration preserves prepared-before-seed cleanup.<span data-glossary-navigation="true"> ([InferenceSession glossary](../../../glossary.md#former-inference-session--inferencesession); [metadata glossary](../../../glossary.md#term-metadata); [compile glossary](../../../glossary.md#compile); [registry glossary](../../../glossary.md#term-registry); [run glossary](../../../glossary.md#term-run); [TrainingSession glossary](../../../glossary.md#training-session--trainingsession); [optimizer glossary](../../../glossary.md#term-optimizer); [seed glossary](../../../glossary.md#term-seed))</span>
- Keep public materialization detached and explicit; do not expose Runtime representations,
  backend storage, or inward SPI types through ordinary signatures.<span data-glossary-navigation="true"> ([materialization glossary](../../../glossary.md#materialization); [detached glossary](../../../glossary.md#detached-model-results); [backend glossary](../../../glossary.md#backend); [SPI glossary](../../../glossary.md#term-service-provider-interface-spi); [signatures glossary](../../../glossary.md#term-signature))</span>
- Metal 0004's route candidates and cache compatibility remain backend-private; explicit
  composition does not expose or interpret them.<span data-glossary-navigation="true"> ([cache glossary](../../../glossary.md#term-cache))</span>
- Preserve 0011's repaired boundary: Engine/Prepare own Compiler aggregates and the single
  projection, while concrete CPU production remains Compiler-free.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler))</span>
- Numerical semantics belong to Model; Engine has no numerical-profile selector after Model
  0032. Backend route/version/type compatibility remains cold and backend-owned. Historical Metal
  0051 is not the current Engine successor or prerequisite.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [Backend route glossary](../../../glossary.md#backend-route))</span>
- Metal configuration, native open, and partial-open rollback remain Metal-owned; Engine may take
  an opened integration but must not duplicate or interpret Metal policy.

## Authorized drift-remediation exception

The user's requested repair order authorizes sequential drift remediation ahead of the normal
post-[Metal](../../../glossary.md#term-metal) frontier reassessment: [Engine](../../../glossary.md#term-engine) 0011 repaired the [CPU](../../../glossary.md#term-cpu) boundary, [Compiler](../../../glossary.md#term-compiler) 0006B8 repaired
[autograd](../../../glossary.md#autograd) documentation, and Engine 0012 repaired the first confirmed remaining-drift cluster.
Their implementations and required reviews are complete through Engine 0013. The separately
owned tuning package [Javadoc](../../../glossary.md#term-javadoc) repair is also Complete as tools/tuning 0004; do not reopen it as an
undetailed successor.

Engine 0014 is a separate user-authorized contract-first workstream from exact base `83f2800`.
The shared-document integration [owner](../../../glossary.md#term-owner), not this branch, owns the later global-roadmap update.

Engine 0015 is the completed single-owner implementation workstream from exact base `6f5be27c`.

Engine 0016 is the completed and independently reviewed mixed-owner workstream from exact base
`0fe35a845a18fb9c70cdf87081b6a6c688fece47`.

Engine 0017 is the completed and independently reviewed reusable [inference](../../../glossary.md#term-inference)-session workstream from
exact base `4fc4fd3d1e44613cdbdc44051e0bc637992d0de4`.

Engine 0018 is the completed and independently reviewed numerical-profile propagation workstream
at implementation `ce7a7dfa67c0dce5cd7b41b2a5bc90ae305393c3` plus documentation/Javadoc
remediation `07a01b9c13c2ceea0922b9af82da3e6e08375306`.

## Status [normalization](../../../glossary.md#term-normalization)

[Engine](../../../glossary.md#term-engine)'s local capability queue is Complete through 0018, with [Model](../../../glossary.md#term-model) 0032's later profile-free
cutover applied. Project-review test/documentation repair 0019 is Complete.

## History and update policy

Detailed results, [validation](../../../glossary.md#term-validation) commands, context identifiers, audits, and past ordering exceptions
remain in linked task files and Git history. Update this map only when [ownership](../../../glossary.md#term-ownership-meanings), dependencies,
task order/status, a milestone, or a live gate changes. Put executable [scope](../../../glossary.md#term-scope) and evidence in the
task brief and follow the [planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
