# CPU Backend Master Plan

Current source: `modules/backend/cpu`; Gradle identity remains `:backends:cpu`. Planning tasks and
completed reports retain their original locations; see the
[path migration table](../../../developer-guide/repository-layout.md#backend-path-migration-and-historical-navigation).

## Goal

Maintain a truthful CPU plan. Preparation owns lowering, route/representation choice, resources,
and finalization; portable generation is baseline, and native peers require exact evidence.

The operating model is:

```text
Planning selects owner=CPU
  -> CPU analysis selects a complete legal route and declares resources
  -> shared Prepare assigns slots
  -> CPU finalization constructs the executable
  -> Runtime cold-binds and executes the fixed schedule
```

## Authority and contracts

This plan coordinates work; it does not define architecture. The authoritative contract is
[ARCHITECTURE.md](../../../../ARCHITECTURE.md), especially these exact headings:

- [`modules/runtime`](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime) — execution, run state, binding,
  and hot-path exclusions.
- [`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare) — analysis, requirements, slots,
  and finalization.
- [Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules) — ownership,
  dependencies, capability, and candidates.
- [Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)
  — benchmark, tuning, cache, and selection ownership.
- [CPU backend routes](../../../architecture/contracts/backend-execution.md#cpu-backend-routes) — one CPU identity and
  backend-private portable, native, specialized, and fused routes.
- [Prepare lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) and
  [Run lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) — phase placement.

The [module-boundary](../../../architecture/module-boundaries.md),
[dependency](../../../architecture/dependency-rules.md), and
[performance/tuning](../../../architecture/performance-evidence-and-tuning.md) guides explain the
contract; the [planning guide](../../planning-guide.md) controls task format/status.

## Scope and non-goals

Scope covers truthful capability; whole-partition lowering and bounded fusion; portable scalar,
Vector API, and Class-File execution; CPU storage, workspace, scheduling, tracing, artifacts;
joint route/representation choice; qualified OpenBLAS and future proved peers; and typed tuning
candidates.

CPU does not own Tensor semantics, global compilation, shared slots, Runtime policy, tuning
measurement/cache mutation, Engine composition, or Metal/CUDA routes. It creates no separate CPU
backends, inferred relaxed numerics, or platform-label-selected vendor routes.

The [first-release product target](../../production-release-goal.md#java-cpu-target) requires
equal F32/F64 support and a temporary lock on Java CPU BF16/FP16 floating computation, retaining
the existing implementation without expanding it. Shared types, storage, raw transfers,
conversions and discrete semantics remain. CPU0078's reviewed CPU-only authority is accepted;
implementation/qualification follow its Ready packet. Later unlocking needs an explicit decision.

The production installation check also repaired unconditional Vector API species queries in
scalar preparation and MATMUL discovery. A plain-JVM consumer now computes actual FLOAT32/
FLOAT64 forward values and gradients without resolving the incubator module, and the public
runtime-length Model trains repeatedly with fresh lengths in both types. Explicit vector
requests retain their policy; no kernel, numerical or specialization identity changed. See
the [runtime prerequisites](../../../user-guide/running-models.md#jvm-prerequisites).

The [reviewed first-release matrix](../../../backend-guide/production-support.md) records static
F32/F64 scope and actual-closure evidence. CNN1d/2d and now Conv3d training passed both types;
the coupled [Compiler0006C](../../modules/compiler/master-plan.md) extension adds generated 3D
windows without widening ordinary named-B convolution admission. CPU0077 is Complete: 45 passing
tests/Javadocs and actual public F32/F64 softmax/backward proof. CPU0078 low lock and CPU0079 raw
reshape are Complete. Conv3d shared-source reviews APPROVE0 and actual every-role numerical,
three-step training/save/load/new-training proof passed. MAIN accepted root test/build/distribution
and fresh Maven/no-Git source consumers after independent reviews.
Raw UNFOLD3D preserves F32/F64/BF16/F16 values; arithmetic FOLD3D is F32/F64 only. General Scan
composes complete Conv3d forward/reverse bodies in both types, modes and traversal directions.
Exact legacy carrier boundaries and the low-arithmetic lock remain unchanged.

The authorized **Pool3d training completion is implemented and locally runtime-qualified**,
superseding its Ready frontier. CPU FLOAT32/FLOAT64 matching-gradient MaxPool3d/AveragePool3d
reuse existing forward kernels, Compiler pooling adjoints and 3D windows; no standalone backward
kernel or fallback was added. Actual public consumers passed both pools' forward and explicit
cotangent gradients against independent spatial scatter and per-output finite differences,
including batch/channel overlap, anisotropic geometry and literal ceil mode. Conv3d → either
pool → flatten → linear passed every parameter role against independent parameter finite
differences, three SGD steps, and save/close/load with bit-exact state and fresh configured SGD.
The same actual consumer scope passed Metal-only FLOAT32; shared Metal Pool2d/Pool3d overlap
uses coordinated identity 40 while FLOAT64/low non-overlap boundaries remain.

CPU Scan qualification covers FLOAT32/FLOAT64 × max/average × all-valid/runtime-length ×
forward/reverse: independent forward and actual backward input-role finite differences plus
save/close/load/fresh execution passed. This uses qualified existing `lowerBatch` closure, not
wider ordinary named-B admission or Metal Scan support. CPU BF16/F16 arithmetic remains locked.
These are local implementation/runtime results, not a joined release gate or newly published
candidate acceptance. The requested comprehensive audit has locally qualified the CPU closure
below; Metal integration and the joined release qualification remain separate work.

### Audit-driven CPU coverage completion — locally qualified

The user's subsequent comprehensive CPU/Metal audit authorizes completing the confirmed
Model-valid CPU gaps: promoted pointwise operand tuples, signed-step materializing SLICE and
its existing SLICE_UPDATE pullback, variadic CONCAT/STACK beyond the old sixteen-input guard,
and sound injectivity proofs for supported large physical outputs. Reuse the existing typed
IR, scalar/conversion/comparison emitters and movement/layout owners; no second evaluator,
operation registry, runtime fallback or implicit low arithmetic is authorized.

Implementation must retain exact mixed stored-value comparisons, Model promotion and original
cotangent carrier conversion, checked signed logical coordinates without negative storage
strides, and checked classfile/resource limits. Extend conservative physical proofs soundly;
never accept a possibly overlapping output. Ordinary named-B remains independently bounded.
The CPU0078 low-computation lock and Compiler's three deferred fixed-recurrent BPTT signatures
remain unchanged. Actual before/after public execution, every affected derivative role, all
seven applicable raw movement carriers, inactive Scan rows and independent Class C review
are acceptance gates; provider answers alone are not completion.

Existing-contract conformance repairs also cover literal attention EXP normalization, first-product
seeds for its nonempty weighted-value contractions, and fixed-recurrent projections. They do not
change Model formulas or authorize fixed-recurrent BPTT. Generated attention classes require a
selective compatibility identity; controlled single-output F64 emission, inactive NaNs and
negative-zero publications require actual execution alongside the ordinary finite fixtures.

Implementation and independent source review passed. The final CPU catalog covers all 137
ordinary signatures across 115 kinds: 804 forward and 736 actual selected-AD runs completed,
with no execution-stage failures. Remaining selection rejections are confined to low-storage
projections: rejected forward kinds were classified under CPU0078; selected low VJPs may also
require locked arithmetic. Execution hashes are not numerical oracles. Separately, 178 independent
forward/backward cases passed, including the sixteen Pool3d Scan cases. Nine public
promoted/signed/variadic witnesses and both-type
composite signed-zero witnesses passed; controlled single-output attention also passed both
types/directions with inactive NaNs. Selective affine projection 73 and attention projection 74
leave generator envelope 71 unchanged. This is local qualification, not release acceptance.


## Stable invariants

- Capability is exact and fail-closed across semantics, type, shape, layout, determinism,
  resources, and execution. A carrier or installed library is not capability evidence.
- Planning selects the single `BackendId("cpu")`. CPU analysis owns lowering, legal fusion,
  route/configuration and representation choice, and exact requirements; shared Prepare remains
  CPU-blind. Finalization follows slot assignment without changing the route or requirements.
- Runtime receives a fixed executable. One route-independent `CpuKernelIr` contains typed values,
  ordered semantics, access plans, loops, and stores; route, workers, species, artifact location,
  graph/Runtime identities, and bindings remain outside it.
- Portable strategies are scalar, vector, parallel-scalar, and parallel-vector. Cold binding
  resolves typed arrays/segments/offsets; start/end kernels and hot loops contain no discovery,
  reflection, maps, strings, graph/operation dispatch, route choice, boxing, or object allocation.
- Optimal direct Java for the same specialization is the generated-code oracle. Semantic,
  Class-File/decompilation, hidden-call/allocation, and performance evidence is family-specific.
- Generated identity covers every bytecode/compatibility fact but excludes compatible instance
  facts. Verified process-local reuse is required; trusted-root persistence remains optional and
  disabled. Generated-class storage and the persistent tuning cache remain separate.
- The seven Model types retain exact represented carriers, but storage never implies arithmetic,
  vector, native, or numerical support; [0009](tasks/0009-portable-generated-coverage-closure-checkpoint.md)
  owns the checked inventory. Run-owned buffers are aligned native off-heap; borrowed inputs are
  non-owning, and selected materializations are explicit, declared, costed, and reusable.
- Route/storage selection compares complete transition, resource, reuse, and concurrency cost.
  Fusion legality precedes profitability; safe heuristics work without tuning, and Runtime never
  searches or revises a prepared choice.
- Current profile-free Model semantics are hard eligibility constraints. Hardware, provider
  presence, workload size, or measurement cannot redefine numerical permission.
- Providers are ABI/lifetime leaves. CPU owns discovery, qualification, capability, fallback,
  thread/lifetime coordination, and candidates. Prepared resources and isolated per-run storage
  obey Prepare/Runtime ownership, concurrency, rollback, transfer, and cleanup contracts.

## Dependencies and package ownership

Allowed direct production dependencies are `modules/model`,
`modules/config`, `modules/planning`, `modules/runtime`, `modules/prepare`,
`modules/backend-contract`, `modules/trace`, and `modules/backend/openblas-provider`. `modules/engine` is
forbidden. Engine 0011 removed the historical production Compiler edge; Compiler is test-only.
CPU does not depend on tuning tools; its collaborations expose opaque candidates.

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider       supported capability
  CpuBackendIntegration       supported lifecycle SPI
  CpuLocalWorkloadTuning      supported local tuning
  CpuCompletePlanTuning       supported plan tuning
  internal/
    memory/                    storage and binding
    prepare/                   analysis/finalization
    lowering/                 units, geometry, fusion
    ir/                       typed IR/access plans
    codegen/emit/             Class-File emission
    route/portable/           portable realization
    route/nativeblas/         OpenBLAS; future BLAS leaves
    route/nativeops/          future vendor-operation leaves
    cache/                    generated compatibility
    executable/               execution/workers
    reference/                conformance oracles
```

Only those four root types are supported; `.internal` contracts are unsupported. Vendor leaves
consume common analysis and never add another backend identity, graph interpreter, registry,
service locator, or provider-owned lowering.

Completed CPU0028 added internal `CpuRecurrentScanLowering` (nested Facts/Geometry) in lowering,
`CpuRecurrentScanPreparedExecutable` (private Invocation) in executable, and `CpuRecurrentScanKernel`
in route.portable. Existing lowered/unit records gain mutually exclusive direct recurrent facts;
no new public facade, generic hierarchy, shared lifecycle type or package move.

Completed CPU0029 retains FLOAT32 `CpuRecurrentScanKernel` and adds three internal concrete types:
`CpuRecurrentScanFloat64Kernel`, `CpuRecurrentScanBFloat16Kernel`, `CpuRecurrentScanFloat16Kernel`
in existing `io.github.pho001.synaptik.backend.cpu.internal.route.portable`.
Addressing stays in existing lowering/binding types; no new addressing class or generic execution
protocol was added. Oracle/timing types
are test-only in `internal.route.portable`; no new package or unrelated abstraction is authorized.

Completed [CPU0030](tasks/0030-general-scan-f32-pointwise-vertical.md) adds only four family-specific
internal production types: `lowering.CpuGeneralScanLowering` (nested Facts/Geometry),
`ir.CpuGeneralScanIr` (derived code-shaping facts using existing CpuKernelIr, not another semantic
body), `codegen.emit.CpuGeneralScanEmitter`, and `executable.CpuGeneralScanPreparedExecutable`
(private Invocation). Existing prepare/unit/cache/binding types retain their responsibilities.
It implements the specified invocation/step ABI, canonical plus generated scalar-zero read closure,
temporal carry-bank/resources and schema 71 compatibility within the bounded F32 domain.

Completed [CPU0031](tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md) reuses these four seams
for F32 broadcast/SUM/layout forward and actual reverse closure. Its only new top-level
production type is `CpuGeneralScanReductionEmitter` in existing
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`: step-bound SUM generation, not
a general execution layer or the ordinary aggregate entry's fixed two-boundary/scratch ABI.
Shape/liveness/resource and generation-time binding facts stay in the existing scan types.

Completed [CPU0032](tasks/0032-general-scan-f32-matmul-and-reverse.md#result) adds only generation-only
`CpuGeneralScanMatmulEmitter` in existing `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`.
Existing Stage/Stages, lowering, carrier/MATMUL emission and cache seams gain explicit two-operand
step facts, not another executor or AD layer. No other production type/package was added.

Completed [CPU0033 F64 domain closure](tasks/0033-general-scan-f64-domain-closure.md) reuses these
lowering/IR/emitter/executable, prepare and cache packages; no new top-level production type or
duplicate F64 executor was added. Full declared algebra and actual reverse are qualified together.

CPU0070's [replacement package/ownership plan](existing-emitter-composition-plan.md#first-foundation-and-stable-immediate-successor)
proposes no added/moved production type, package or public signature. Completed CPU0071 separates
setup/bound MATMUL emission in existing lowering/IR/carrier/loop/emitter owners; completed CPU0072
closes current pointwise/SUM reuse via approved helpers in existing `lowering.CpuPartitionLowering`,
`lowering.CpuAggregateLowering`, `codegen.emit.CpuClassFileKernelGenerator` and
`codegen.emit.CpuAggregateEmitter`; no added/moved type or public signature. The
[brief](tasks/0072-common-pointwise-sum-composition.md#files-current-package-map-and-proposed-helper-seams)
indexes reviewed seams and the frozen implementation. Ordinary scalar/SIMD/tiled math and workers
retain their family/executable owners. Both Scan math subemitters were retired after migration and
actual-AD qualification, preserving temporal bindings/proofs and v9. No second slot manager or body.
New named types require exact source justification and user approval, not the archived four-type
mandate. Scan SIMD and parallel execution remain separately gated; no PhaseState/protocol selected.

User-approved 2026-10-30 [CPU0073 package/seam design](row-map-multi-reduction-fusion-plan.md#minimal-proposed-production-map-and-signatures):
three CPU-internal production types/two nested records and exact common seams; ordinary lowered
unit and Scan-emitted region share one proof/emitter, not new math or a Runtime owner. Boole
APPROVE/MAIN authority acceptance preceded implementation; final independent reviews and MAIN
integration accepted 2026-10-30. These types/seams are implemented; no extra type/seam implied.

[CPU0049 implemented package map](tasks/0049-producer-reduction-fusion.md#files-types-and-composition-seams)
adds only `internal.ir.CpuAggregateFusionIr`: existing aggregate plus pointwise producer/tail facts
with explicit bindings for their distinct domains. Existing lowering/emitter/binder/cache owners
remain; no row-region/represented-SUM reuse for exact ordinary SUM, new math owner or executor.
Bounded static F32/F64 SUM/MEAN + ADD/RELU implementation/review/gates are Complete, MAIN accepted
2026-10-31; no authority delta. The brief owns exact results and diagnostic limitations.

[CPU0050 completed package map](tasks/0050-average-pool2d-backward-broadcast-fold.md#files-types-and-reuse-seams):
no new/moved production type or package. Existing `internal.ir.CpuFoldIr`,
`internal.lowering.CpuFoldLowering.Geometry` gain explicit compact-source mapping facts;
existing `CpuFoldEmitter` owns inverse-coordinate gather and shared typed addition/carrier helpers.
Existing decomposition/prepare/binding/cache owners retain their responsibilities; no parallel
pooling emitter, semantic operation or Compiler recipe change. Extra types require MAIN approval.
Selected domain is F32/F64 only; low chains retain their existing path/artifact, not a new
conformance claim. The brief records the confirmed pre-existing low-fold rounding defect separately.

[CPU0051 proposed package map](tasks/0051-softmax-logsoftmax-backward-fusion.md#files-symbols-and-proposed-types):
only new `internal.ir.CpuSoftmaxBackwardIr` (fixed two-pass topology/bindings) and
`internal.codegen.emit.CpuSoftmaxBackwardEmitter` (direct loops composing existing exact SUM/scalar/carrier
emitters). Recognition stays in `CpuAggregateLowering`; decomposition, resources, binding and cache
stay with existing owners. The reduced-domain ADD/RELU tail in `CpuAggregateFusionIr` cannot express
this full-domain reread. No semantic operation, general framework, executor or authority change.

## Ordered task list

The table owns order and status; linked tasks own detailed evidence.

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [CPU capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md) | Superseded | Stable planning, runtime, prepare, backend-contract, and trace contracts | Replaced by 0005A; historical evidence retained. |
| 0002 | [Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md) | Superseded | 0001; generated JVM-bytecode CPU-kernel architecture contract; Java 26 Class-File and Vector API toolchain | Replaced by 0005A; historical evidence retained. |
| 0003 | [Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md) | Superseded | 0002; stable CPU finalization and artifact compatibility; explicit trusted local root | Replaced by 0005A; historical evidence retained. |
| 0004 | [Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md) | Superseded | 0001–0003 | Replaced by 0005A; historical evidence retained. |
| 0005 | [Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md) | Superseded | 0002–0004 | Replaced by 0005A; historical evidence retained. |
| 0005A | [Atomic partition-kernel architecture reset](tasks/0005a-atomic-partition-kernel-architecture-reset.md) | Complete | 0001–0005; current shared Prepare contracts | Established whole-partition portable execution. |
| 0005B | [Universal access plans and right-aligned broadcasting](tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md) | Complete | 0005A | Delivered. |
| 0005C | [Vector and parallel portable strategies](tasks/0005c-vector-and-parallel-portable-strategies.md) | Complete | 0005B | Delivered. |
| 0005D | [Materialization, specialization, and persistence evidence gate](tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md) | Complete | 0005C | Delivered. |
| 0005E | [Portable pointwise types, carriers, and semantic-family expansion](tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md) | Complete | 0005D | Delivered. |
| 0005F | [Floating division and exact scalar-power realization](tasks/0005f-floating-division-and-exact-scalar-power-realization.md) | Complete | 0005E | Delivered. |
| 0005G | [Extrema, clamp, Tensor power, and logical coverage](tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md) | Complete | 0005F; Model 0018T/0018U/0025A | Delivered. |
| 0005H | [Portable unary, transcendental, and activation closure](tasks/0005h-portable-unary-transcendental-and-activation-closure.md) | Complete | 0005G; Model 0018P/0018T1/0019A semantics; Java 26 math/Vector contracts | Delivered. |
| 0005I | [FLOAT32 vector parity and vector-emission boundary](tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md) | Complete | 0005H; Java 26 `FloatVector`/`DoubleVector`; completed pointwise access, specialization, and numerical contracts | Delivered. |
| 0005J | [Bounded pointwise coverage and parity hardening](tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md) | Complete | 0005I; current Model pointwise semantics; Java 26 Byte/Int/Long/Float/Double Vector API contracts | Delivered. |
| 0006 | [Portable static affine views and boundary materialization](tasks/0006-portable-static-affine-views-and-boundary-materialization.md) | Complete | 0005J | Delivered. |
| 0006A | [Portable pad, tile, and tensor-composition movement](tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md) | Complete | 0006 | Delivered. |
| 0006A1 | [Portable static window extraction](tasks/0006a1-portable-static-window-extraction.md) | Complete | 0006A | Delivered. |
| 0006A2 | [Portable gather and one-hot indexing](tasks/0006a2-portable-gather-and-one-hot-indexing.md) | Complete | 0006A1 | Delivered. |
| 0006B | [Portable functional slice update](tasks/0006b-portable-functional-slice-update.md) | Complete | 0006A2 | Delivered. |
| 0006B1 | [Portable functional scatter](tasks/0006b1-portable-functional-scatter.md) | Complete | 0006B | Delivered. |
| 0006B2 | [Portable overlap fold](tasks/0006b2-portable-overlap-fold.md) | Complete | 0006B1 | Delivered. |
| 0006C | [Portable stable ordering and selection coverage](tasks/0006c-portable-stable-ordering-and-selection.md) | Complete | 0006B2 | Delivered. |
| 0006D | [Portable explicit-state RNG and dropout coverage](tasks/0006d-portable-explicit-state-rng-and-dropout.md) | Complete | 0006C | Delivered. |
| 0007 | [Portable cumulative scan coverage](tasks/0007-portable-cumulative-scan-coverage.md) | Complete | 0006D; Model 0023E; completed 0005A–0005J | Delivered. |
| 0007A | [Portable ordinary extrema and boolean reductions](tasks/0007a-portable-ordinary-extrema-and-boolean-reductions.md) | Complete | 0007; current Model aggregate-reduction contracts | Delivered. |
| 0007A0 | [Generated hot-path parity correction](tasks/0007a0-generated-hot-path-parity-correction.md) | Complete | 0005A/0005B/0005C/0005I; 0007; 0007A; completed local bytecode/performance audit | Delivered. |
| 0007A0A | [Affine and movement generated-loop parity](tasks/0007a0a-affine-and-movement-generated-loop-parity.md) | Complete | 0006/0006A/0006A1/0006B; 0007A0 | Delivered. |
| 0007A0B | [Indexing generated-loop parity](tasks/0007a0b-indexing-generated-loop-parity.md) | Complete | 0007A0A; completed 0006A2 semantics | Delivered. |
| 0007A0C | [Scatter generated-loop parity](tasks/0007a0c-scatter-generated-loop-parity.md) | Complete | 0007A0B; completed 0006B1 semantics | Delivered. |
| 0007A0D | [Fold generated-loop parity](tasks/0007a0d-fold-generated-loop-parity.md) | Complete | 0007A0C; completed 0006B2 semantics | Delivered. |
| 0007A0E | [Ordering generated-loop parity](tasks/0007a0e-ordering-generated-loop-parity.md) | Complete | 0007A0D; completed 0006C semantics | Delivered. |
| 0007A0F | [Random and dropout generated-loop parity](tasks/0007a0f-random-and-dropout-generated-loop-parity.md) | Complete | 0007A0E; completed 0006D semantics | Delivered. |
| 0007A1 | [Portable ordinary numerical aggregate reductions](tasks/0007a1-portable-ordinary-numerical-aggregate-reductions.md) | Complete | 0007A; 0007A0F | Delivered. |
| 0007A1A | [Generated scalar-body self-containment](tasks/0007a1a-generated-scalar-body-self-containment.md) | Complete | 0007A1; approved schema-29 generated-code audit | Delivered. |
| 0007A1B | [Scatter algorithmic parity](tasks/0007a1b-scatter-algorithmic-parity.md) | Complete | 0007A1A; completed 0006B1/0007A0C semantics | Delivered. |
| 0007A1C | [Generated/direct evidence closure](tasks/0007a1c-generated-direct-evidence-closure.md) | Complete | 0007A1B; completed generated-family inventory through 0007A1 | Closed semantics, structure, and twenty five-fork parity rows. |
| 0007A1D | [Native-order segment layout hoisting](tasks/0007a1d-native-order-segment-layout-hoisting.md) | Superseded | 0007A1C first-fork evidence | Retained schema-32 hoisting passed semantics/structure but failed all 13 isolated performance targets; later A1E–A1O/A1C work replaced the unfinished local gate without retroactive acceptance. |
| 0007A1E | [Movement general-address-loop parity](tasks/0007a1e-movement-general-address-loop-parity.md) | Complete | 0007A1D stable schema-32 prerequisite and failed fork | Delivered. |
| 0007A1F | [BOOL movement and aggregate residual parity](tasks/0007a1f-bool-movement-and-aggregate-residual-parity.md) | Complete | 0007A1E | Delivered. |
| 0007A1G | [Fold and dropout residual parity](tasks/0007a1g-fold-and-dropout-residual-parity.md) | Complete | 0007A1F | Delivered. |
| 0007A1H | [Numerical aggregate residual parity](tasks/0007a1h-numerical-aggregate-residual-parity.md) | Complete | 0007A1G | Delivered. |
| 0007A1I | [Indexing residual parity](tasks/0007a1i-indexing-residual-parity.md) | Complete | 0007A1H | Delivered. |
| 0007A1J | [Cumulative scan residual parity](tasks/0007a1j-cumulative-scan-residual-parity.md) | Complete | 0007A1I | Delivered. |
| 0007A1K | [Affine-copy residual parity](tasks/0007a1k-affine-copy-residual-parity.md) | Complete | 0007A1J | Delivered. |
| 0007A1L | [Pointwise general-loop residual parity](tasks/0007a1l-pointwise-general-loop-residual-parity.md) | Complete | 0007A1K | Delivered. |
| 0007A1M | [Scatter MIN residual parity](tasks/0007a1m-scatter-min-residual-parity.md) | Complete | 0007A1L | Delivered. |
| 0007A1N | [Multi-axis MIN residual parity](tasks/0007a1n-multi-axis-min-residual-parity.md) | Complete | 0007A1M | Delivered. |
| 0007A1O | [Pointwise ledger evidence reconciliation](tasks/0007a1o-pointwise-ledger-evidence-reconciliation.md) | Complete | 0007A1C evidence; 0007A1A; 0007A1L; 0007A1N | Delivered. |
| 0007A2 | [Portable binding-aware sum-to-Shape reduction](tasks/0007a2-portable-binding-aware-sum-to-shape-reduction.md) | Complete | Complete 0007A1C; accumulated schema-42 evidence through 0007A1O; Model 0023A; Compiler 0005B | Delivered. |
| 0007B | [Portable arg-extrema coverage](tasks/0007b-portable-arg-extrema-coverage.md) | Complete | 0007A2; Model 0018U1; Compiler 0005B | Delivered. |
| 0007C | [Portable masked reduction coverage](tasks/0007c-portable-masked-reduction-coverage.md) | Complete | 0007A1; 0007B; Model 0018Q; Compiler 0005B | Delivered. |
| 0007D | [Portable logarithmic, statistical, and norm reduction coverage](tasks/0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md) | Complete | 0007A1; Model 0018V; Compiler 0005B | Delivered. |
| 0007E | [Portable stable softmax and log-softmax coverage](tasks/0007e-portable-stable-softmax-and-log-softmax-coverage.md) | Complete | 0007D; Model 0016I/0016J; Compiler 0005B | Delivered. |
| 0007F | [Portable layer and RMS normalization coverage](tasks/0007f-portable-layer-and-rms-normalization-coverage.md) | Complete | 0007E; Model 0021/0021A; Compiler 0005B | Delivered. |
| 0007F1 | [Portable batch-normalization inference coverage](tasks/0007f1-portable-batch-normalization-inference-coverage.md) | Complete | 0007F; Model 0021B; Compiler 0005B | Delivered. |
| 0007F2 | [Portable batch-normalization training and statistic-transition coverage](tasks/0007f2-portable-batch-normalization-training-and-statistic-transition-coverage.md) | Complete | 0007F1; Model 0021C; Compiler 0005B | Delivered. |
| 0008 | [Portable grouped NCHW Conv2d execution foundation](tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md) | Complete | 0002–0007F2; Model 0020; Model 0025G; Model 0025H; Compiler 0006B | Delivered. |
| 0008A | [Portable channels-first dimensional convolution closure](tasks/0008a-portable-channels-first-dimensional-convolution-closure.md) | Complete | 0008; Model 0025G–0025H; Compiler 0006B | Delivered. |
| 0008B | [General partition-DAG computation-unit decomposition and bounded fusion](tasks/0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md) | Complete | 0006–0008A | Delivered. |
| 0008C | [Typed specialized-subgraph and epilogue recognition](tasks/0008c-typed-specialized-subgraph-and-epilogue-recognition.md) | Complete | 0007F2–0008B | Delivered. |
| 0008D | [Bounded fusion profitability and typed decision facts](tasks/0008d-bounded-fusion-profitability-and-typed-decision-facts.md) | Complete | 0008B–0008C | Delivered. |
| 0008E | [Bounded multi-input materialization and representation reuse](tasks/0008e-bounded-multi-input-materialization-and-representation-reuse.md) | Complete | 0008D | Delivered. |
| 0008E1 | [Shared partition-DAG adoption and reconstruction removal](tasks/0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) | Complete | Prepare 0003A; 0008E | Delivered. |
| 0008F | [Portable MATMUL execution and bounded linear epilogues](tasks/0008f-portable-matmul-execution-and-bounded-linear-epilogues.md) | Complete | 0008E1; Model 0019/0019D; Compiler 0005D | Delivered. |
| 0008G | [Portable max/average Pool2d execution](tasks/0008g-portable-max-average-pool2d-execution.md) | Complete | 0008F; Model 0020A–0020A1; Compiler 0005D | Delivered. |
| 0008G1 | [Portable Pool1d composition validation and Pool3d generated execution](tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md) | Complete | 0008G; Model 0025I–0025K; Compiler 0006B1–0006B2 | Delivered. |
| 0008H | [Portable scaled-dot-product attention execution](tasks/0008h-portable-scaled-dot-product-attention-execution.md) | Complete | 0008G1; Model 0019E/0023F; Compiler 0005D | Delivered. |
| 0008I | [Portable loss-family execution](tasks/0008i-portable-loss-family-execution.md) | Complete (performance NON_PASSING) | 0008H; Model 0022–0022B; Compiler 0005D | Semantics and 792-class structure pass; fork 0 had 19/792 ratios above 1.15x, later forks were waived, and parity is not claimed. |
| 0008J | [BFLOAT16 scalar pointwise closure](tasks/0008j-bfloat16-scalar-pointwise-closure.md) | Complete | 0008I; current Model pointwise contracts | Delivered. |
| 0008K | [Cross-type CAST execution](tasks/0008k-cross-type-cast-execution.md) | Complete | 0008J; completed [Model 0025L](../../modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md) | Delivered. |
| 0008L | [Pointwise SIMD mask/output closure](tasks/0008l-pointwise-simd-mask-output-closure.md) | Complete | 0008K; Java 26 Vector API mask support | Delivered. |
| 0008M | [Vector MSE `NONE`](tasks/0008m-vector-mse-none.md) | Complete | 0008L; 0008I | Delivered. |
| 0008N | [Measured profitable FLOAT32/FLOAT64 Conv2d/Conv3d SIMD accumulation](tasks/0008n-measured-profitable-float32-float64-conv2d-conv3d-simd-accumulation.md) | Incomplete (bounded scalar stop) | 0008M; bounded implementation-time axis/profitability spike | Retained scalar production after generated/direct parity failed; 0008N1 later supplied the accepted replacement. |
| 0008N1 | [Generated Conv nested width-block loop/dataflow parity re-spike](tasks/0008n1-generated-conv-nested-width-block-loop-dataflow-parity.md) | Complete | 0008N bounded stop | Delivered the accepted generated Conv vector replacement. |
| 0008O | [Stable-reduction vector numerical spike](tasks/0008o-stable-reduction-vector-numerical-spike.md) | Cancelled | 0008N1 | Cancelled after fork-0 `KEEP_SCALAR`; the five-fork protocol is incomplete and no SIMD route was enabled. |
| 0008P | [Deterministic modular partial-reduction parallelism](tasks/0008p-deterministic-partial-reduction-parallelism.md) | Complete | 0008O (Cancelled; retained evidence recorded); 0007A1 | Delivered partial-reduction machinery; selection remains fail-closed at `KEEP_WHOLE_CELL`. |
| 0008Q | [Finite scalar-immediate and clamp generated-code equivalence](tasks/0008q-scalar-immediate-clamp-generated-equivalence.md) | Complete | 0005F, 0005G, 0005J, 0008J, 0008L; current schema-63 pointwise lowering/preparation/generation | Delivered. |
| 0008Q1A | [Vector scalar-power hot-path self-containment](tasks/0008q1a-vector-scalar-power-hot-path-self-containment.md) | Complete | 0005F, 0005I, 0007A1A, 0008L, 0008Q; independent 0008Q1 structural failure | Delivered. |
| 0008Q1 | [Finite scalar-immediate and clamp matrix](tasks/0008q1-finite-scalar-immediate-clamp-matrix.md) | Complete | 0005F, 0005G, 0005J, 0008J, 0008L, 0008Q; complete 0008Q1A and its documentation pass; current schema-64 pointwise lowering/preparation/generation | Delivered. |
| 0009 | [Portable CPU coverage and implementation-closure checkpoint](tasks/0009-portable-generated-coverage-closure-checkpoint.md) | Complete | 0001–0008P; complete 0008Q1 and its documentation pass; completed 0008Q1A and its documentation pass; current checked semantic inventory | Closed support/inventory without a universal performance claim. |
| 0009A | [Scalar-immediate and clamp generated support and clean-Java semantic closure](tasks/0009a-scalar-immediate-clamp-clean-java-structural-equivalence.md) | Complete | 0009; completed 0008Q1 finite semantic matrix | Delivered. |
| 0009B | [Ordinary pointwise and CAST structural-oracle closure](tasks/0009b-ordinary-pointwise-and-cast-structural-oracle-closure.md) | Complete | 0009A | Delivered. |
| 0009C1 | [CONCAT and STACK generated-entry hot-path hygiene](tasks/0009c1-concat-stack-generated-entry-hot-path-hygiene.md) | Complete | 0009B | Delivered. |
| 0009C | [Affine, movement, indexing, scatter, and random structural-oracle closure](tasks/0009c-affine-movement-indexing-scatter-random-structural-oracle-closure.md) | Complete | 0009C1 | Delivered. |
| 0009D | [Aggregate, scan, ordering, and fold route-closure parent](tasks/0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md) | Complete | 0009C; completed 0009D1A | Retained generated route. |
| 0009D1 | [Aggregate generated-route retention decision](tasks/0009d1-aggregate-structural-oracle-closure.md) | Complete | 0009C; 0009D; completed 0009D1A | Retained generated route. |
| 0009D1A | [Aggregate segment-layout prologue hygiene](tasks/0009d1a-aggregate-segment-layout-prologue-hygiene.md) | Complete | 0009C; 0009D; D1 review finding | Retained generated route. |
| 0009D2 | [Scan route decision and possible direct-Java migration](tasks/0009d2-scan-route-decision-and-possible-direct-java-migration.md) | Complete | 0009D1 | Retained generated route. |
| 0009D3 | [Ordering, index, and scratch direct-Java route decision](tasks/0009d3-ordering-index-scratch-direct-java-migration.md) | Complete | 0009D2 | Retained generated route. |
| 0009D4 | [Fold and window route decision and possible direct-Java migration](tasks/0009d4-fold-window-route-decision-and-possible-direct-java-migration.md) | Complete | 0009D3 | Retained generated route. |
| 0009E | Reductions, normalizations, and loss direct-Java migration parent | Draft | 0009D4 | Master-only grouping; child decisions are complete, but this row remains Draft. |
| 0009E1 | [Partial integral reduction direct-Java migration](tasks/0009e1-partial-integral-reduction-direct-java-migration.md) | Complete | 0009D4 | Retained generated route. |
| 0009E1A | [Masked reduction route decision](tasks/0009e1a-masked-reduction-route-decision.md) | Complete | 0009E1 | Retained generated route. |
| 0009E1B | Advanced reduction route-decision parent | Complete | 0009E1A | Master-only parent; all three child route decisions are complete. |
| 0009E1B1 | [Log-sum-exp route decision](tasks/0009e1b1-log-sum-exp-route-decision.md) | Complete | 0009E1A | Retained generated route. |
| 0009E1B2 | [Statistical reduction route decision](tasks/0009e1b2-statistical-reduction-route-decision.md) | Complete | 0009E1B1 | Retained generated route. |
| 0009E1B3 | [Norm reduction route decision](tasks/0009e1b3-norm-reduction-route-decision.md) | Complete | 0009E1B2 | Retained generated route. |
| 0009E1C | [Softmax-style reduction route decision](tasks/0009e1c-softmax-style-reduction-route-decision.md) | Complete | 0009E1B3 | Retained generated route. |
| 0009E2 | [Normalization route decision and possible direct-Java migration](tasks/0009e2-normalization-route-decision-and-possible-direct-java-migration.md) | Complete | 0009E1C | Retained generated route. |
| 0009E3 | [Loss route decision and possible direct-Java migration](tasks/0009e3-loss-route-decision-and-possible-direct-java-migration.md) | Complete | 0009E2; CPU 0008I fork-0 NON_PASSING | Retained bounded loss routes; 0008I remains non-passing evidence, not a performance claim. |
| 0009F | [Hybrid route decisions for specialized compute families](tasks/0009f-hybrid-route-decisions-for-specialized-compute-families.md) | Complete | 0009E3 | Retained generated route. |
| 0009F1 | [MATMUL and convolution route decisions](tasks/0009f1-matmul-and-convolution-route-decisions.md) | Complete | 0009F | Retained generated route. |
| 0009F2 | [Pooling route decisions](tasks/0009f2-pooling-route-decisions.md) | Complete | 0009F1 | Retained generated route. |
| 0009F3 | [Attention and BatchNorm route decisions](tasks/0009f3-attention-and-batch-normalization-route-decisions.md) | Complete | 0009F2 | Retained generated route. |
| 0009G | [Final support, correctness, hygiene, and inventory checkpoint](tasks/0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md) | Complete | 0009F3 | Delivered. |
| 0009G1 | [Scalar-strategy evidence correction](tasks/0009g1-scalar-strategy-evidence-correction.md) | Complete | 0009G | Delivered. |
| 0010 | [Narrow OpenBLAS BLAS-compatible native route](tasks/0010-narrow-openblas-blas-compatible-native-route.md) | Complete | 0005A; 0009G1; completed OpenBLAS provider | Delivered the narrow qualified FLOAT32/FLOAT64 route. |
| 0010A | [Automatic OpenBLAS discovery and internal composition foundation](tasks/0010a-automatic-openblas-discovery-and-internal-composition-foundation.md) | Complete | 0010; completed OpenBLAS provider | Delivered. |
| 0010B | [Bounded OpenBLAS MATMUL representation expansion](tasks/0010b-bounded-openblas-matmul-representation-expansion.md) | Complete | 0010A; 0008E | Delivered. |
| 0010C | [Coordinated OpenBLAS thread candidates and shared CPU thread budget](tasks/0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md) | Complete | 0010B; stable CPU worker orchestration | Delivered. |
| 0010D | [Installed OpenBLAS qualification and target fingerprinting](tasks/0010d-installed-openblas-qualification-and-target-fingerprinting.md) | Complete | 0010C | Delivered. |
| 0010D1 | Qualified direct BFLOAT16-output OpenBLAS MATMUL route | Blocked (deferred optional side branch) | 0010D; OpenBLAS provider 0004 | Resume only after provider 0004 proves an exported direct BFLOAT16-output ABI and one final narrowing. |
| 0010E | [FLOAT32/FLOAT64 OpenBLAS tuning candidates and compatible decisions](tasks/0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md) | Complete | 0010D | Delivered typed candidates and compatible-decision consumption; CPU performs no measurement or persistence. |
| 0010F | [Supported CPU lifecycle integration adapter](tasks/0010f-supported-cpu-lifecycle-integration-adapter.md) | Complete | 0010E; Prepare 0003–0004; Runtime 0010; Compiler 0006B3 | Delivered the supported CPU lifecycle adapter. |
| 0010G | [Canonical caller-owned host snapshot export](tasks/0010g-canonical-caller-owned-host-snapshot-export.md) | Complete | 0010F; Runtime 0015; Engine 0003 | Delivered bounded caller-owned canonical host export. |
| 0010H | [Source-only published-constant CPU materialization](tasks/0010h-source-only-published-constant-cpu-materialization.md) | Complete | Compiler 0006B5; Prepare 0005; 0010F–0010G | Delivered source-only constant materialization for the sole non-empty CPU composition. |
| 0010I | [Supported CPU local-workload tuning composition adapter](tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md) | Complete | 0010E–0010H; Prepare 0004; tools/tuning 0001 consumer contract | Delivered supported local-workload tuning composition. |
| 0010J | [Supported complete-plan candidate and decision producer](tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md) | Complete | 0008D–0008F; 0010E–0010I; Prepare 0004; tools/tuning 0001 | Delivered supported complete-plan candidate production. |
| 0010K | [CPU backend guide status reconciliation](tasks/0010k-cpu-backend-guide-status-reconciliation.md) | Complete | 0010J; Engine 0009/0013; tuning 0001–0004; Prepare 0007 | Reconciled current CPU Engine, Config, Prepare, tuning, and public-facade status. |
| 0010L | [CPU execution and public lifecycle status reconciliation](tasks/0010l-cpu-execution-and-public-lifecycle-status-reconciliation.md) | Complete | 0010K; Compiler 0006B10; Model 0025M; Engine 0013 | Reconciled bounded CPU execution, public Engine/training status, and the capability-provider complete-partition inventory without changing behavior or authority. |
| 0010M | [Evidence-backed CPU production defaults](tasks/0010m-evidence-backed-production-defaults.md) | Complete | 0009G1; 0010L; Engine 0017; benchmarks 0001 | Evidence-backed no-change: retained scalar/1/1/1 production after broad vector evidence failed scope, reproducibility, and target gates; hardened the report-only protocol. |
| 0011 | Intel oneMKL BLAS and VML peer routes | Blocked | 0010E; 0005A; 0009; concrete Intel CPU use case and supported oneMKL ABI evidence | Blocked until a concrete Intel workload and supported oneMKL ABI evidence both exist. |
| 0012 | Intel oneDNN partition peer routes | Draft | 0005A; 0009; stable common CPU lowering; concrete DNN/ML use case and supported oneDNN ABI evidence | Planned only after a concrete oneDNN use case and supported ABI evidence. |
| 0013 | Apple Accelerate peer routes | Draft | 0005A; 0009; concrete Apple CPU use case and supported Accelerate ABI evidence | Planned only after a concrete Apple CPU use case and supported Accelerate ABI evidence. |
| 0014 | AMD AOCL-BLAS and AOCL-LibM peer routes | Draft | 0005A; 0009; concrete AMD CPU use case and supported AOCL ABI evidence | Planned only after a concrete AMD CPU use case and supported AOCL ABI evidence. |
| 0015 | Optional AMD ZenDNN partition peer routes | Draft | 0014; 0005A; 0009; stable common CPU lowering; concrete ZenDNN use case and integration evidence | Optional ZenDNN route waits for 0014 plus a concrete use case and integration evidence. |
| 0016 | Cross-route CPU tuning-cache integration | Draft | 0010E; Prepare 0004; tools/tuning 0001; 0011–0015 as implemented | Generalize the proved tuning contract only across vendor peers that are actually implemented. |
| 0017 | [Explicit ACCELERATOR numerical-profile realization](tasks/0017-explicit-accelerator-numerical-profile-realization.md) | Complete | Model 0027; Config 0006; Engine 0018; current exact portable/OpenBLAS routes and profile-keyed plans/caches | Delivered identical exact CPU semantics and routes under both profiles with profile-separated plans, generated artifacts, OpenBLAS workloads, and tuning/cache identities. |
| 0018 | [Shared external-read recognition validation](tasks/0018-shared-external-read-recognition-validation.md) | Complete | CPU 0008C/0008D/0008F/0017; Engine 0018 (all Complete); user chose authenticated `ValueId` binding | DAG-authenticated virtual-output provenance preserves legal shared-weight MATMUL+RELU fusion and rejects forged positions; focused and full CPU tests passed, with independent Class C approval. |
| 0019 | [CPU partition size independent of fusion-unit size](tasks/0019-unbounded-cpu-partition-bounded-fusion-units.md) | Complete | CPU 0008B/0008D/0008E/0008F/0017; Engine 0018 (Complete); independent of 0018 | Long maximal CPU partitions now execute as bounded local units; 1,039 CPU tests and focused Engine execution passed. |
| 0020 | [Bare MATMUL realization and strategy alignment](tasks/0020-bare-matmul-realization-strategy-alignment.md) | Complete | CPU 0008F, 0010M, 0018, 0019 (all Complete) | Default scalar and explicit vector MATMUL realization now agree with prepared strategy and species; public INT32 execution, CPU tests, and independent Class C review passed. |
| 0021 | [Composed validation and loss binding repair](tasks/0021-composed-value-validation-and-loss-bindings.md) | Complete | 0019–0020; project review | Fix F16 softmax reading, value checks after producers, and occurrence-level loss bindings. |
| 0022 | [Activation extremes and low working precision](tasks/0022-activation-extremes-and-low-working-precision.md) | Complete | Reviewed/integrated 0021; current Model semantics | Fix ERF/GELU large-finite saturation and low activation F32 working sites; serialize generated identity work. |
| 0023 | [Vector ERF direct-Java shape](tasks/0023-vector-erf-direct-java-shape.md) | Draft | 0022 Complete; explicit selection | Follow up the bounded vector ERF timing gap and preexisting coefficient-loop shape; no broad benchmark campaign or parity claim. |
| 0024 | [Default segment route heap carrier binding](tasks/0024-default-segment-route-heap-carrier-binding.md) | Complete | 0022 Complete; ordinary heap Engine reproduction; explicit user approval | Exact heap segment binding and mixed-carrier overlap repair passed focused, public Engine, independent re-review, and combined validation. |
| 0025 | [Scatter validation coordinate rank repair](tasks/0025-scatter-validation-coordinate-rank.md) | Complete | 0024; 0006B1/0007A0C/0007A1B Complete | Independent APPROVE, zero findings; main's unchanged-snapshot native root gate passed at NN0029 join (2026-10-19). No NN dependency. |
| 0026 | [Stacked LSTM cold preparation diagnostic](tasks/0026-stacked-lstm-cold-preparation-diagnostic.md) | Complete | CPU0025; Compiler0009; NN0029 Complete | Repeated finish/topological scans dominate sampled combined CPU analysis; prepare censored at 180-second JVM deadline. Pasteur APPROVE, zero findings; main accepted 2026-10-20. |
| 0027 | [Unit topology adjacency ordering](tasks/0027-unit-topology-adjacency-ordering.md) | Complete | CPU0026 Complete | Invocation-local adjacency and stable Kahn ordering preserve exact dependencies/order; CPU, conformance, Fluent integration and bounded LSTM probe passed; independent APPROVE, zero findings. |
| 0028 | [Fixed recurrent scan forward execution](tasks/0028-fixed-recurrent-scan-forward-execution.md) | Complete | NN0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 Complete | Canonical FLOAT32 scalar Java RNN/GRU/LSTM accepted 2026-10-20 after independent APPROVE and main's root test/Javadoc gate; evidence in brief. |
| 0029 | [Recurrent scan scalar type and layout coverage](tasks/0029-recurrent-scan-scalar-type-and-layout-coverage.md) | Complete | CPU0028; CPU0022 Complete | Four scalar floating types and resolved nonnegative-stride input layouts accepted 2026-10-21 after independent APPROVE, zero P1/P2, and main's source-matched native root test/Javadoc PASS; timing parity unproven. |
| 0030 | [F32 pointwise general-scan vertical milestone](tasks/0030-general-scan-f32-pointwise-vertical.md) | Complete | Model0034/0035; Planning0008; Compiler0011; Prepare0010; CPU0022/0027/0028/0029 Complete | Bounded F32 pointwise forward/actual automatic reverse accepted by main 2026-10-22 after independent review and focused checkpoint PASS; remaining declared algebra/closure + F64 stays gated. |
| 0031 | [F32 broadcast/SUM/layout scan closure](tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) | Complete | CPU0030 and accepted predecessor baseline Complete | Main accepted 2026-10-22, 14:38+ UTC after final independent code/docs APPROVE (0 P1/P2), module and named checkpoint PASS; MATMUL/F64 remain Draft. |
| 0032 | [F32 MATMUL and actual reverse in emitted scan](tasks/0032-general-scan-f32-matmul-and-reverse.md#result) | Complete | CPU0031; accepted foundation baseline Complete | Main accepted 2026-10-22, 17:09+ UTC after independent code/docs and test-addendum APPROVE, CPU module/Javadoc and named checkpoint PASS; F64 remains Draft. |
| 0033 | [General-scan F64 domain closure](tasks/0033-general-scan-f64-domain-closure.md#result) | Complete | Model0034/0035; Planning0008; Compiler0011; Prepare0010; CPU0030–0032 Complete | Full homogeneous F64 algebra + actual reverse accepted by main after independent code/docs approval and native root PASS; unexplained timing limitation retained. |
| 0034 | [General-scan coordinate loops and carry transition](tasks/0034-general-scan-coordinate-loops-and-carry-transition.md#result) | Complete | CPU0033 + low-level join; accepted Compiler0012 baseline Complete | Main accepted 2026-10-23 after independent code/docs and recovery approval, CPU/Javadoc and targeted checkpoint PASS; 900 paired samples audited, bounded generated/direct ratios 0.970/0.990/0.969, no universal parity or successor advancement. |
| 0035 | [General-scan emission cleanup and specialized oracle](tasks/0035-general-scan-emission-cleanup-and-specialized-oracle.md#result) | Complete | CPU0034 Complete, accepted baseline retained | Bounded cleanup accepted 2026-10-23 after independent APPROVE and final gates; 900-pair matched static-B4/H8/runtime-long campaign shows ~3× gap, not parity; separate narrow performance follow-up required, not launched. |
| 0036 | [General-scan javac loop control](tasks/0036-general-scan-javac-loop-control.md#result) | Complete | CPU0035 and accepted foundation Complete | Main accepted 2026-10-23 after independent approval and gates; bounded structural canonicalization with narrow near-reference timings, supplementary guard qualification insufficient; no successor. |
| 0037 | [General-scan dense actual read views](tasks/0037-general-scan-dense-read-views.md#result) | Complete | CPU0036 and accepted foundation/low-level join Complete | Main accepted 2026-10-24 after Newton final APPROVE and CPU/Javadoc, named and configured-root gates; no NN migration or new performance claim. |
| 0038 | [Symbolic Scan timing methodology](tasks/0038-symbolic-scan-timing-methodology.md#result) | Complete | Prepare0014; CPU0037 Complete | MAIN accepted 2026-10-25 after Sagan final APPROVE and single bounded probe; method reuse/oracle fairness repaired, production unchanged, no parity or successor claim. |
| 0039 | [Symbolic Scan fixed-geometry specialization](tasks/0039-symbolic-scan-fixed-geometry-specialization.md#result) | Complete | CPU0038; Prepare0014; CPU0037 Complete | MAIN accepted bounded structural/correctness/verification after independent APPROVE and final gates/probe; original performance gap remains open (largest observed ratio 2.351869×), no successor or extra measurement authorized. |
| 0040 | [Symbolic Scan phase and optimized-code diagnostic](tasks/0040-symbolic-scan-phase-code-diagnostic.md#result) | Complete | CPU0039; Prepare0014; CPU0037 Complete | MAIN accepted bounded diagnostic after final independent APPROVE; reverse code-quality differences observed, performance issue unresolved; production frozen. |
| 0041 | [Symbolic Scan reverse carrier specialization](tasks/0041-symbolic-scan-reverse-carrier-specialization.md#result) | Complete | CPU0040; CPU0039; CPU0037; Prepare0014 Complete | MAIN accepted 2026-10-26 after Peirce final APPROVE and all gates; immutable port-backed entry receivers resolve the measured large-case gap with bounded near-Java phase timings, not universal parity; CPU frontier None. |
| 0042 | [Length-aware general Scan execution](tasks/0042-length-aware-general-scan-execution.md#result) | Complete | Model0037/0038; Compiler0013; Prepare0011/0014; CPU0033/0037/0041 Complete | MAIN accepted 2026-10-26 after independent Class C code/docs APPROVE and recorded gates; fixed-geometry F32/F64 forward/actual reverse, nonstationary probe limits, CPU frontier None. |
| 0043 | [Symbolic-length actual reverse diagnostic](tasks/0043-symbolic-length-reverse-diagnostic.md#result) | Complete | Prepare0015; CPU0041; CPU0042 Complete | MAIN accepted frozen source/evidence 2026-10-27 after independent APPROVE and module/preservation gates; compilation-phase confounds observed, historical cause/residual overhead unknown, no production repair or third probe. |
| 0044 | [Symbolic-length Scan code-shape optimization](tasks/0044-symbolic-length-scan-code-shape.md#result) | Complete | CPU0043; Prepare0015; CPU0041; CPU0042 Complete | MAIN accepted bounded four-fix v4 repair and sole comparison, 2026-10-27; lower elapsed time than v3 in both measured cases, Java parity unresolved. Independent reviews/gates/docs complete; all four bodies consumed, no successor/retry. |
| 0045 | [Equivalent symbolic-length Scan clean Java](tasks/0045-symbolic-length-scan-equivalent-java.md#result) | Complete | CPU0044; Prepare0015; CPU0041; CPU0042 Complete | MAIN accepted source/evidence/docs 2026-10-28; equivalent reference and bounded v5 offset sharing delivered, no demonstrated speedup or Java parity. Budget exhausted. |
| 0046 | [Collected generated-kernel cache entry cleanup](tasks/0046-collected-kernel-cache-entry-cleanup.md#result) | Complete | Compiler0016; CPU0045 Complete | MAIN accepted 2026-10-28: cold exact-token cleanup, live/concurrent reuse, focused checks and Javadoc PASS. Full run's legacy unconditional timing-gate failure remains explicit. |
| 0047 | [Private PERMUTE input absorption into MATMUL](tasks/0047-private-permute-matmul-absorption.md#result) | Complete | 0046 | MAIN accepted 2026-10-28: backing/stride reads remove private transpose resources; publication barriers, epilogues and native-copy eligibility preserved. Filtered CPU/Javadoc, conformance and Engine integration passed; no timing claim. |
| 0048 | [Proven static typed-one count folding](tasks/0048-static-typed-one-count-folding.md#result) | Complete | 0047; accepted 0076 baseline | MAIN accepted 2026-10-31 after source/docs APPROVE, filtered CPU/Javadoc and named join PASS; exact count/resource removal with original source retention, no timing claim. |
| 0049 | [Producer-reduction fusion and executable epilogues](tasks/0049-producer-reduction-fusion.md#result) | Complete | 0048; Complete 0073–0076 baseline | MAIN accepted 2026-10-31: independent FINAL APPROVE, filtered CPU/Javadoc and named join PASS; exact static F32/F64 scalar SUM/MEAN + ADD/RELU, diagnostic closed without steady-profitability claim. |
| 0050 | [AveragePool2d backward compact-source fold](tasks/0050-average-pool2d-backward-broadcast-fold.md#result) | Complete | 0049 Complete | MAIN accepted 2026-10-31: F32/F64 static SCALAR/single-range compact fold, existing IR/emitter, preserved DIV boundary; Dirac FINAL APPROVE 0 P1/P2, named join and filtered CPU/Javadoc PASS. Sole diagnostic retained; parity/cause unresolved, low fix separate; frontier None. |
| 0050A | [Low fold FP32 accumulation repair](tasks/0050a-low-fold-fp32-accumulation.md#result) | Complete | 0050 Complete | MAIN accepted after Raman FINAL APPROVE and qualified targeted/Javadoc closure; original full gate remains FAILED, unchanged successes reused. Generic-loop efficiency unqualified; frontier None, no timing/successor. |
| 0051 | [Softmax and LogSoftmax backward fusion](tasks/0051-softmax-logsoftmax-backward-fusion.md#result) | Complete | 0050A Complete | MAIN accepted: bounded two-pass fusion, independent FINAL APPROVE, CPU/Javadoc and qualified consumer closure. Sole synthetic diagnostic did not establish parity; CPU0052 held. |
| 0052 | Private recurrent-gate SLICE read mapping | Draft | 0051 | Read projection ranges directly in ordinary pointwise consumers, without changing NN recipes or observable/saved values. |
| 0053 | [Fresh Scan/Java continuation measurement](tasks/0053-scan-java-continuation-measurement.md#result) | Cancelled | 0045; 0047 | User stopped after untimed historical-class-pin mismatch; launcher preparation/review complete, 0/3 measured slots used, no new performance result or production change. |
| 0054 | [Scan ClassFile and ABI verification](tasks/0054-scan-classfile-and-abi-verification.md#result) | Complete | Compiler0016; 0047; retained 0053 evidence | MAIN accepted 2026-10-28 after independent APPROVE: name-only v5 ClassFile change causally confirmed, exact ABI/raw-output checks passed. No benchmark or production change. |
| 0055 | [Verified Scan/Java measurement](tasks/0055-verified-scan-java-measurement.md#result) | Complete | 0045; Compiler0016; 0047; 0054 | MAIN accepted 2026-10-28 after independent actual-result APPROVE: three serial uncensored runs, exact schedule/raw checks, preserved evidence and unresolved narrow Java gap. No production change. |
| 0056 | [Scan invocation binding](tasks/0056-scan-invocation-binding.md#result) | Complete | 0055; Prepare0015 | Independently reviewed rejection: negligible hot gain, repeat binding regression; candidate withdrawn exactly, original invocation retained, Java gap open. |
| 0057 | [Scan body/Java offline audit](tasks/0057-scan-body-java-offline-audit.md#result) | Complete | 0056 | Independently reviewed stage/count/JIT comparison; zero-base and inactive-cursor differences identified, native cause unproved; no source change or new measurement. |
| 0058 | [Symbolic-length zero scratch base](tasks/0058-symbolic-length-scan-zero-scratch-base.md#result) | Complete | 0057 | Independently approved all-B scratch proof and baked v6; typed semantic/cache/bytecode and CPU/Javadoc/conformance/Engine gates pass, no timing or publication. |
| 0059 | [Scan v6/v5/Java entry comparison](tasks/0059-scan-v6-v5-java-measurement.md#result) | Complete | 0058 | MAIN accepted independently approved bounded observations: repaired preflight/three JVMs pass, no consistent v6 speedup, local Java gap and full-path JIT limits retained. No source change. |
| 0060 | [Symbolic-length Scan Java loop shape](tasks/0060-symbolic-length-scan-java-loop-shape.md#result) | Complete | 0058; 0059; Prepare0015 | Independently approved combined F32/F64 loop/init/address simplification and baked v7; owning/named gates pass, sole diagnostic shows modest improvement with Java/JIT limits retained. |
| 0061 | [Scan feature-contraction affine addresses](tasks/0061-scan-feature-contraction-addresses.md#result) | Cancelled | 0060; Prepare0015 | Performance-only experiment completed; candidate failed its frozen retention gate and was restored exactly to v7. No functional suites or retained capability. |
| 0062 | [Decompiled Scan Java lab](tasks/0062-decompiled-scan-java-lab.md#result) | Complete | 0060; Prepare0015; 0061 restored | Two reviewed off-tree runs pass raw checks; no prototype beats G7. Java gap remains, no generator port or functional suites. |
| 0063 | [Clean Scan Java with generated ABI](tasks/0063-clean-scan-java-generated-abi.md) | Complete | 0062; 0060; Prepare0015 | Reviewed off-tree J4 whole-loop alignment; separate port authorized as 0064. |
| 0064 | [Port measured clean-Java Scan shape](tasks/0064-scan-clean-java-generator-port.md#result) | Complete | 0063 measured/reviewed; 0060; Prepare0015 | Combined baked v9; matched timing benefit, independent review, CPU/Javadoc and named conformance/integration gates PASS. |
| 0065 | [Vector MATMUL TANH epilogue](tasks/0065-vector-matmul-tanh-epilogue.md#result) | Complete | 0064 stable base; existing MATMUL/vector TANH | Ordinary F32/F64 direct/tiled vector TANH; independent review and CPU/Javadoc/conformance PASS, no Scan route or parity claim. |
| 0066 | [Shared ordinary/Scan emission plan](tasks/0066-shared-scan-body-emission-plan.md) | Complete | 0064/0065 Complete | Source-grounded three-stage plan; independent APPROVE, no blocking findings; MAIN accepted 2026-10-28. Planning only; no source/capability grant. |
| 0067 | [Common scalar body lowering/emission](tasks/0067-common-scalar-body-lowering-and-emission.md) | Superseded | Historical 0066 | Replaced by CPU0070/0071/0072; old four-type mandate and diagnostic budget retired. |
| 0068 | [Qualified shared feature-vector bodies](tasks/0068-qualified-shared-body-vector-emission.md) | Superseded | Historical 0067 | Replaced by shared existing math plus separately approved bounded Scan SIMD qualification. |
| 0069 | Worker integration ([archived outline](shared-body-emission-proposal.md#vector-realization-then-worker-integration)) | Superseded outline | Historical 0068 | Replaced by the unresolved parallel-composition design/authority gate; no phase protocol selected. |
| 0070 | [Existing-emitter composition replan](tasks/0070-existing-emitter-composition-replan.md) | Complete | CPU0064/0065 accepted; user direction 2026-10-29 | Independent APPROVE, 0 P1/P2; MAIN accepted replacement planning delivery 2026-10-29; no source or measurement grant. |
| 0071 | [Common MATMUL binding/setup foundation](tasks/0071-common-matmul-binding-foundation.md#result) | Complete | Accepted 0070, retained 0064/0065, Compiler0015/0016/Prepare0015 and USER gates | Reviewed common MATMUL/setup, exact two static cutovers, module/named joins PASS; sole diagnostic closed, no speed/parity claim. |
| 0072 | [Common pointwise/SUM composition](tasks/0072-common-pointwise-sum-composition.md#result) | Complete | Seven Complete predecessors; verified MAIN Ready/GO 2026-10-29 | Common normalization/composition and represented SUM; independent APPROVE/P2 closure, combined module/Javadoc and MAIN named gates accepted; literal retention, zero measurements. |
| 0073 | [Row-map/multiple-terminal fusion](tasks/0073-row-map-multi-reduction-fusion.md#result) | Complete | 0064/0065/0070/0071/0072; Compiler0015/0016/Prepare0015 Complete | Common bounded ordinary/Scan core, independent final APPROVE, accepted targeted closure/evidence transfer, MAIN named join and docs closure; performance unmeasured. |
| 0074 | [Row-fusion SIMD and parallel realizations](tasks/0074-row-fusion-simd-parallel.md#result) | Complete | CPU0073 + retained Complete baseline | MAIN accepted functional capability, final CPU/Javadoc, named gates and collected bounded comparison; steady-state profitability unqualified, concrete JIT-pressure follow-up separate, defaults unchanged. |
| 0075 | [Scalar row-fusion load/address reuse](tasks/0075-row-fusion-scalar-load-address-reuse.md#result) | Complete | CPU0073/0074 Complete; explicit selective-cutover approval | Independent final APPROVE, CPU/Javadoc PASS; ordinary 32→20 / Scan 27→22 load sites. Sole comparison closed: ordinary ratios JIT-qualified, no material Scan speedup; Java gap and C1 failures remain. |
| 0076 | [Generic row-invariant load staging](tasks/0076-row-invariant-load-staging.md#result) | Complete | CPU0073/0074/0075 Complete; explicit user approval | Independent review, complete-lifetime fallback and affected-only identity; CPU/Javadoc/named Engine PASS, bounded B256 improvement/B32 JIT caveat; defaults unchanged. |
| 0077 | [F64 finite-softmax admission parity](tasks/0077-f64-finite-softmax-admission-parity.md#result) | Complete | NN0033; reviewed production scope; accepted bounded authority | Independent APPROVE, 45 passing targeted tests/Javadocs and actual public both-type/kind backward proof; no arithmetic/identity change. |
| 0078 | [Temporary low-arithmetic lock](tasks/0078-temporary-low-arithmetic-lock.md#result) | Complete | CPU0077; final production/fixture/resource/evidence/docs APPROVE0; accepted CPU-only authority/ADR0060 | MAIN's affected gate1716passed64skipped; actual both-low rejection/retention and F32-master restore/resume pass. Implementation and old qualified identities retained; no low expansion. |
| 0079 | [Materializing RESHAPE admission](tasks/0079-materializing-reshape-admission.md#result) | Complete | CPU0078/Compiler0019 Complete; final independent source/evidence/docs APPROVE0 and exact integration | Full CPU 1,473 passed/60 existing skips, conformance 83 passed; unchanged actual seven-type raw reshape passed. Existing affine copy, low lock and old identities retained; joined release gate remains separate. |
| 0080 | [Shared generation-time kernel method emission](tasks/0080-shared-kernel-method-emission.md#result) | Complete | Complete0078/0079; final independent source/evidence/docs APPROVE0 | Exact source integrated; CPU1,474pass/60existing skips, all85retained files exact, actual prefixed ordinary/contiguous-and-strided-loss dispatches and84actual VJPs pass. Original coverage overclaim corrected; no new capability. |
| 0081 | [General Scan ordinary-unit composition](tasks/0081-ordinary-unit-general-scan-composition.md) | Ready | Complete0080/0078/0079, Model0046/0047/0048 and Compiler0017A/B; accepted frozen coupled0017 interface and both independent authority APPROVE0 reviews | MAIN admits exact dirty-source/disjoint contexts for the complete all-valid/length typed selected-backward cutover; physical/actual qualification pending, no whitelist substitute or implemented capability claim. |

## Active dependency DAG and integration

CPU0078 and CPU0079 are Complete after final independent source/evidence/documentation APPROVE0,
exact-byte integration and their recorded full module/conformance/public-consumer proofs.
CPU0080 parity factoring is Complete after final independent acceptance. Model0048 source/
forward-coordinate semantics and Compiler0017B ordinary rules remain independently qualified.
No broader Scan body or low computation is granted. MAIN owns shared integration and the final
source-matched production checkpoint; no old performance campaign is reopened.

[CPU0051 Complete](tasks/0051-softmax-logsoftmax-backward-fusion.md#result),
MAIN accepted 2026-10-31 after independent FINAL APPROVE, filtered CPU/Javadoc and qualified consumer
closure. Its Result preserves the failed first join and test-only targeted correction with unchanged
success reuse, not a relabeled full-join PASS. The sole synthetic diagnostic did not establish parity,
profitability or a JIT cause. Verified launch/base, seven integration fields, matrix and exact evidence
remain in the brief. Parallel group None; MAIN owns shared integration. CPU0052 remains Draft/held;
Scan budgets stay closed, with no automatic successor, further timing or Git publication.

[CPU0050A Complete](tasks/0050a-low-fold-fp32-accumulation.md#result),
MAIN accepted 2026-10-31 after Raman FINAL APPROVE, 0 open P1/P2, and qualified gate closure.
Complete CPU0050 → Complete CPU0050A → Complete CPU0051 → held Draft CPU0052.
CPU0050A repairs existing low fold numerics under current Model authority, not CPU0050 compact-low
admission. Existing emitter/reference/IR identity owners only; no added/moved production type/package.
Its brief records all seven integration fields and verified actual-before manifest
`/private/tmp/synaptik-cpu0050a.yqY321/before.json` (2596 files, matching HEAD `07a281ed`, empty index),
plus `before-source.tar.gz`; all hashes matched before planning edits. Historical Ready verification
and MAIN-LAUNCH proof remain in the brief. No parallel repo writers; MAIN owns shared integration.
Actual order: focused evidence/pre-final review → named join once PASS → failed CPU gate → approved
metadata/environment recovery → targeted recovery → metadata/Javadoc closure → final review/MAIN acceptance.
The Result preserves the original full FAILED gate and targeted closure with unchanged-outcome reuse,
not a full-repeat/full-green claim. Recovery changes only test-derived hashes/provenance, not new math.
Arithmetic core aligned; generic traversal retained under explicit technical deviation evidence, not
whole-loop/dataflow/performance parity. Known geometry hoisting grants no task/campaign; CPU0051 admission is above.
No timing budget, architecture change, global schema bump or Git publication.
[CPU0050](tasks/0050-average-pool2d-backward-broadcast-fold.md#result)
Complete, MAIN accepted 2026-10-31 after Dirac FINAL APPROVE and all gates PASS; worker closed.
CPU0051 is Complete above; CPU0052 remains held. [CPU0048](tasks/0048-static-typed-one-count-folding.md#result) is
Complete, MAIN accepted 2026-10-31 after independent source/docs APPROVE, the final filtered
CPU/Javadoc gate, named conformance/Engine join and protection checks passed.
Closed DAG: Complete CPU0047 + accepted CPU0076 → CPU0048 implementation/review
→ final worker gate → MAIN join. The brief owns verified
base/contracts, seven integration fields and the single new `internal.ir.CpuConstantWriteIr` package seam.
MAIN owns shared documents; no concurrent writer or Git publication grant.

[CPU0049 Complete](tasks/0049-producer-reduction-fusion.md#result), MAIN accepted 2026-10-31:
Complete CPU0048/CPU0073–0076 → verified Ready/user instruction → clean implementation and
independent FINAL APPROVE → filtered CPU/Javadoc and MAIN named join PASS. The Result owns
recovered gate evidence, preservation and the sole nonstationary diagnostic, not steady profitability.
Closed DAG: Complete CPU0049 → CPU0050 freeze/independent review → MAIN named join once → sole
diagnostic → final filtered CPU/Javadoc → MAIN acceptance, 2026-10-31. CPU0050's Result owns seven
integration fields, exact commands/evidence and limits; base `07a281edd3086efde2853211b71b48288db027e1`
plus accepted dirty actual-before retained. Parallel group None; MAIN owns shared integration.
F32/F64 static SCALAR/single-range only, existing fold owners/no new production type; source/guide freeze
and ordinary literal artifacts retained. BF16/F16 remain unchanged compatibility controls, not numerical
qualification; its separate F32-accumulator/final-narrowing repair is CPU0050A Complete above.
Bounded diagnostic does not establish parity/cause, steady-state/C2 or speedup. CPU0050 frontier closed;
CPU0051 has its separate bounded admission above; CPU0052 held, Scan budgets paused/closed.
No historical campaign, automatic successor or stage/commit/push. CPU0073's row-region exclusion is unchanged.

At CPU0076 acceptance the source/Ready frontier was None. Complete CPU0073/0074/0075 → accepted authority/Ready
packet → clean CPU0076 implementation/untimed freeze → independent source/protocol review → MAIN
sole finite release/result acceptance → final CPU/Javadoc/named Engine gates and independent docs,
all complete. The [CPU0076 Result](tasks/0076-row-invariant-load-staging.md#result) owns evidence,
qualifications, seven integration fields and frozen base. Parallel group None; MAIN owns shared
integration with exclusive final documentation delegation. That completion itself opened no successor/measurement grant;
the later CPU0048-only admission is above; defaults unchanged, no rollout or stage/commit/push.

Complete CPU0073/0074 → CPU0075 implementation, sole bounded
comparison, final CPU/Javadoc PASS and independent review are closed. The
[CPU0075 Result](tasks/0075-row-fusion-scalar-load-address-reuse.md#result) owns evidence and
limits; its seven integration fields remain in the brief. Parallel group None; MAIN owns shared
docs. That completion itself authorized no successor, new timing budget, rollout or stage/commit/push. At that checkpoint CPU0048–0052
remained held/inactive; the remaining Scan/Java gap and C1 failures are disclosed limitations.

CPU0074 is Complete, MAIN accepted 2026-10-30; its source/measurement frontier is closed.
Complete CPU0073 + retained baseline → accepted authority/Ready → implementation/independent
reviews → named joins → final CPU/Javadoc and sole bounded comparison → MAIN acceptance.
The [Brief Result](tasks/0074-row-fusion-simd-parallel.md#result) centralizes final receipts and limits:
functional support is accepted, steady-state profitability is not qualified. Concrete JIT-pressure
investigation is recommended separately, not a new task/frontier/campaign. The single budget is
exhausted; no retry or old campaign opens. All seven integration fields and launch receipt remain
in the brief; Parallel group None, CPU0048–0052 held/inactive at that acceptance, no automatic successor.
Guarded ordinary-only `ROW_REGION_STATE` remains cold metadata, not Scan tuning, pointwise repair
or scoring change. CPU0010M scalar/max1 defaults remain unchanged; no rollout or commit/push follows.
The completed-frontier statements below describe their earlier checkpoints, not CPU0074 admission.

CPU0070 planning delivery is Complete after independent APPROVE and MAIN acceptance on 2026-10-29.
MAIN's admission/preservation receipt remains in the brief. CPU0071 and CPU0072 are Complete;
their source writers are closed. CPU0072's [Result](tasks/0072-common-pointwise-sum-composition.md#result)
records independent Class C APPROVE/P2 closure, combined module/Javadoc evidence and MAIN's named
join acceptance on 2026-10-29. Their frontiers closed; CPU0073 completion is below.
CPU0067–0069 stay superseded. The subsequent backward-versus-optimized-Java code comparison
motivated CPU0073, not repeated timing.
The bounded comparison identifies a cross-operation staging gap; the user approved
[CPU0073's exact design/contract packet](row-map-multi-reduction-fusion-plan.md) and starting
implementation on 2026-10-30. DAG: Complete CPU0072 + retained baseline → Boole authority APPROVE
(0 P1/P2)/MAIN acceptance → verified MAIN Ready receipt → CPU0073 clean implementation/review
→ MAIN named join and independent documentation closure, all Complete on 2026-10-30.
Source and measurement frontiers: None; no automatic successor. Its [Result](tasks/0073-row-map-multi-reduction-fusion.md#result)
owns validation and limits, including the failed full-module run, passing repairs and approved
unchanged-outcome transfer. MAIN owns shared integration; parallel group None. The recorded
dirty base is preserved; no measurement, commit or push is authorized.
The [narrow compatibility exception](../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)
is implemented. Independent corrected source/Javadoc review approved; MAIN accepted source and
module/named gates on 2026-10-29. The [Result](tasks/0071-common-matmul-binding-foundation.md#result)
owns validation counts, finite-corpus/species limits and unsteady diagnostic evidence; budget closed.

CPU0045, [Compiler0016](../../modules/compiler/tasks/0016-matmul-gradient-noop-reduction-elision.md#result),
CPU0046 and [CPU0047](tasks/0047-private-permute-matmul-absorption.md#result) are Complete.
CPU0066 planning delivery remains [Complete](tasks/0066-shared-scan-body-emission-plan.md), independently
approved and MAIN accepted 2026-10-28; its direction is historical, not the active DAG.
Replacement DAG: CPU0064/0065 + retained Compiler0015/0016/Prepare0015 → CPU0070 acceptance
→ CPU0071 → CPU0072, all Complete. No old budget reuse or automatic successor launch.
CPU0071's completed review, implementation, module/Javadoc, named conformance/Engine joins and
sole authorized diagnostic are recorded in its Result. No speedup/parity gate or new capability
follows: timing differences remain an unresolved performance question, not a correctness failure.
CPU0071/0072 use existing owners and CodeBuilder allocation. New named types require
demonstrated source need and user approval. Bounded Scan SIMD authority/whole-binder qualification
and a genuine parallel-composition design/authority decision remain separate gates, not detailed
task cascades. Ordinary workers and per-submission Jobs are unchanged; inline math does not provide
their orchestration. The foundation is complete; [Job reuse remains optional, not a dependency or blocker](existing-emitter-composition-plan.md#deferred-job-reuse-evidence).
Future ordinary synchronous worker composition needs narrow approved authority and budget-ownership
qualification, not zero scheduler metadata allocation or a mandated new cohort/phase/barrier.
Parallel group None; conflicts include common emission, Scan, geometry/cache/resources and queued
CPU0048–0052. Common base remains `07a281edd3086efde2853211b71b48288db027e1` plus accepted
dirty work, verified before launch. MAIN completed the DAG's named conformance/Engine checkpoint;
clean Class C implementation and independent targeted review are complete. MAIN owns integration
and delegated the final documentation/status pass exclusively. CPU0070 grants no Java, authority,
build or measurement change.
CPU0072 retains literal accepted-current generated bytes/identities and unchanged resources/ABI/
binding; MAIN accepted the [four-group conditional transfer table](existing-emitter-composition-plan.md#cpu0072-evidence-transfer-table)
with all limits on 2026-10-29, not a new certification ledger or historical-parity gate. A necessary
shape/namespace change or new type/handoff requires a narrow user approval, not CPU0071's exception.
The target remains general composition of existing ordinary emitters, not all-operation admission;
forward, actual backward and parallel each qualify separately, without a speculative task campaign.
Next prioritize [complete backward-versus-optimal-Java code comparison](existing-emitter-composition-plan.md#post-integration-actual-scan-evaluation)
and avoidable-work analysis before optional Job reuse. No timing budget/protocol is inherited.
CPU0062 Complete → [CPU0063 Complete](tasks/0063-clean-scan-java-generated-abi.md) → CPU0064 Complete → CPU0065 Complete.
CPU0060 + Prepare0015 Complete + CPU0061 restored → [CPU0062 Complete](tasks/0062-decompiled-scan-java-lab.md#result).
CPU0060 + Prepare0015 Complete → [CPU0061 Cancelled](tasks/0061-scan-feature-contraction-addresses.md#result).
The performance-only experiment is closed: its sole candidate failed the retention gate and was
restored exactly to v7, with no functional suite. No retry, retained capability or successor follows.
CPU0058 + CPU0059 + Prepare0015 Complete → [CPU0060 Complete](tasks/0060-symbolic-length-scan-java-loop-shape.md#result).
MAIN accepted serial clean implementation, independent code/docs/prelaunch/result reviews and
owning/named gates on 2026-10-28. The sole diagnostic slot is consumed; Java gap and full-path JIT
limits remain. No CPU0048–0052 launch, automatic successor or old measurement-budget reopening.
[CPU0059](tasks/0059-scan-v6-v5-java-measurement.md#result) is Complete, MAIN accepted after
independent actual-result APPROVE (0 P1/P2). Repaired preflight and three measured JVMs passed;
no consistent v6 speedup, local Java gap and full-path JIT limits remain. All slots consumed;
original failure/evidence unchanged. That completed grant authorizes no automatic successor or further measurement.
CPU0058 Complete → CPU0059 is the user's distinct finite measurement grant; old budgets stay closed.
CPU0057 → [CPU0058 Complete](tasks/0058-symbolic-length-scan-zero-scratch-base.md#result): MAIN accepted
independent code/docs APPROVE and the CPU/Javadoc/conformance/Engine gates on 2026-10-28.
Only coefficient-proved nonbanked scratch bases and combined baked v6 changed; no timing or publication.
CPU0056 → CPU0057 Complete: MAIN accepted parallel read-only source/ClassFile and JIT analyses,
serial independent report review and documentation/preservation checkpoint on 2026-10-28.
The report identifies bounded code-shape opportunities, not an authorized implementation or timing cause.
CPU0055 + accepted Prepare0015 → CPU0056 Complete: independently reviewed hot/cold comparison
rejected per-run handle insertion; exact withdrawal retains the original invocation. MAIN accepted
the result on 2026-10-28. All measurement grants are closed; CPU0057 grants no further execution.
CPU0047 records its bounded read-mapping/native-route
scope, filtered owning validation and public Engine integration. Only CPU0047 is published as
`07a281ed`. The user stopped CPU0053 after its untimed historical-pin mismatch; that task stays
Cancelled without a retry. CPU0054's separately requested untimed verification is Complete after independent
APPROVE: the ClassFile difference is confined to its generated name, with ABI and tested raw
results unchanged. The subsequent user continuation delivered CPU0055, now Complete after
independent actual-result APPROVE: one retained preflight and three serial measured JVMs,
unchanged Java and preserved historical inputs/results. The wrapper filename false-negative and
reviewed evidence-only continuation remain explicit in its Result. Its slots stay closed.
CPU0048/CPU0049/CPU0050/CPU0050A/CPU0051 are Complete after MAIN acceptance; no CPU frontier is open;
CPU0052 remains Draft/held.
The user approved structural repairs before returning to Scan performance analysis:
Compiler0016 → CPU0046 → CPU0047 → CPU0048 → CPU0049 → CPU0050 → CPU0050A → CPU0051 → CPU0052.
CPU0049 includes both producer-to-reduction and recognized reduction-epilogue realization; all
other requested repairs retain separate rows. Each Draft needs its compact verified packet before
execution. Parallel group None; MAIN owns serial integration and shared documents under the
explicit user-approved context exception for the completed repairs. The user separately authorized
publication of only CPU0047 and CPU0053's finite measurement before CPU0048–0052; those remaining
repairs grant no other Git mutation or benchmark. Completed measurement budgets remain exhausted;
CPU0053's isolated campaign is now Cancelled at the user's request, with no measured slots used;
it did not reopen historical selectors or evidence and authorizes no continuation.
Completed CPU0043 + Prepare0015 + CPU0041 + CPU0042 → [CPU0044 Complete](tasks/0044-symbolic-length-scan-code-shape.md#result):
MAIN accepted the bounded four-fix v4 repair and sole comparison on 2026-10-27 after independent
source/evidence APPROVE, owning gates and accepted documentation receipts. Java parity remains
unresolved, not a new authorized task. All four bodies consumed; no fifth/retry, rollback or successor.
Seven integration fields, 2487-file repair-request base, protocols/evidence/limits remain in the brief.
Parallel group None; MAIN owns final serial integration/shared documents.
Historical completed diagnostic: Prepare0015 + CPU0041 + CPU0042 →
[CPU0043 Complete](tasks/0043-symbolic-length-reverse-diagnostic.md#result), a bounded diagnostic with
no production change and a bounded-unknown historical cause/residual overhead, not parity.
Its brief owns seven integration fields, accepted dirty base, evidence and launcher limits.
Independent Class C review approved; MAIN accepted frozen source/evidence and retains final
shared-document integration/receipt acceptance. Its acceptance granted no automatic successor or further timing;
CPU0044's separately authorized bounded repair is Complete above.

[Prepare0015](../../modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning validation,
the sole bounded probe and configured root test/Javadoc PASS. It delivers one named B plus runtime
lengths, fixed T/features, homogeneous F32/F64 forward/actual reverse and bounded outer/control/ROW(1)
through existing owners, retaining old identities/bytes. Separate combined/typed identities, no new
type/memo/prerequisite chain. The short probe has large nonstationary reverse gaps; it is not parity
or a JIT-cause proof. CPU0043's diagnostic and CPU0044's bounded repair are Complete.
Historical v3 performance acceptance stays withheld; the accepted v4 comparison does not resolve
the broader Java-parity target. All four bodies consumed; no rollback, new work or measurement authorized.

At Model0043 acceptance, planning/source/authority frontiers were None. [Model0043](../../modules/model/tasks/0043-symbolic-length-scan-integration-design.md)
is Complete for [combined symbolic-B/runtime-length F32/F64 Scan](../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
DESIGN DELIVERY ONLY; Hume the 4th final Class C APPROVE, 0 P1/P2/no unresolved findings;
MAIN accepted 2026-10-26, not new semantics, authority or implementation.
Its future vertical first qualifies bounded typed outer/control and ROW(1) constant plumbing,
including proposed 34-limb F64 exact SUM/MEAN, then combined generated forward/actual reverse.
At that design checkpoint CPU source/test and authority frontiers were None; completed CPU0043 is above.
MAIN's serial review/integration is closed; Parallel group None; PlanningAuthorized=true is historical.

Complete Model0037/0038 + Compiler0013 + Prepare0011/0014 + CPU0033/0037/0041 →
[CPU0042 Complete](tasks/0042-length-aware-general-scan-execution.md#result), MAIN accepted
2026-10-26 after independent Class C code/docs APPROVE and the recorded CPU/Javadoc, named joined
checkpoint and sole bounded probe. Scope is fixed T/B/features length-aware F32/F64 forward and
actual Compiler reverse, not dynamic-B lengths, cold/v5 or NN0032 migration. Existing CPU owners
implement mixed INT64 control, producer-ordered preflight and activity/shared contribution loops;
no new production type/package, supported API or Runtime handoff. Probe timing is nonstationary,
not steady-state parity or a diagnosed compiler cause. The brief owns base, seven integration
fields, evidence and limits. Parallel group None; MAIN owns integration and delegated the exclusive
final docs pass after source freeze. Source/test frontier None; other frontiers unchanged,
no automatic successor, further measurement or commit/push.

Complete Prepare0014 + CPU0037 → CPU0038 Complete →
[CPU0039](tasks/0039-symbolic-scan-fixed-geometry-specialization.md#result) Complete, MAIN accepted
2026-10-25 for bounded fixed-geometry structural/correctness/verification after independent final
code/docs APPROVE and module/Javadoc, named checkpoint and single bounded probe PASS.
At CPU0039 acceptance the original performance gap remained unresolved; that implementation/reviewer closed.
Complete CPU0039 + Prepare0014 + CPU0037 →
[CPU0040](tasks/0040-symbolic-scan-phase-code-diagnostic.md#result) Complete for bounded diagnostic only.
Complete CPU0040 + CPU0039 + CPU0037 + Prepare0014 →
[CPU0041](tasks/0041-symbolic-scan-reverse-carrier-specialization.md#result) Complete, MAIN accepted
2026-10-26 after Peirce final source/docs and evidence APPROVE and all gates. At CPU0041 acceptance,
CPU source/test frontier was None; CPU0042's subsequent completion is above, other areas remain held.
The controlled source change binds demanded port-backed cuts directly to immutable entry receivers
in existing `internal.codegen.emit.CpuGeneralScanEmitter`, with symbolic v3 in `internal.ir.CpuGeneralScanIr`;
`CpuGeneralScanMatmulEmitter`, arithmetic/resources and checked FFM are unchanged. The measured large
case now has four-way guard-free reverse backedges and near-Java timings, not universal parity or
an internal C2 phi proof. The brief owns results, limits and all seven integration fields.
Historical admission verified Complete predecessors/reviews/gates, sole writer/inactive conflicts,
2475-file pre-plan snapshot, empty index and HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
plus accepted dirty work. Parallel group None; MAIN-owned serial implementation/review, final CPU
gate, named join and one DEV/one FINAL capture are closed. Full 2476-file freeze remained unchanged
through gates; canonical/family/static v4/fixed/ordinary identities and four oracle/timing files retained.
CPU0038 remains historical Complete; its acceptance itself authorized no production successor.

CPU0022 + CPU0028 → CPU0029, all Complete. CPU0030 bounded vertical milestone is Complete.
Before CPU0038, CPU frontier was None. [CPU0037](tasks/0037-general-scan-dense-read-views.md#result) is Complete,
accepted by main 2026-10-24 after Newton final independent APPROVE and final gates. Pre-code authority
was accepted at 05:44 UTC, launch verified at 05:47:51 UTC; no successor authorized.
CPU0036 + accepted foundation/low-level join → CPU0037, all Complete. Conflicts: scan admission/resources/binding/
IR/identity/shared docs (inactive); Parallel group None; base HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
plus accepted dirty work. Main owns serial integration/shared docs and the brief's final checkpoint.
Package map: existing `internal.lowering.CpuGeneralScanLowering` only; no new production type/package.

[Prepare0014 symbolic Scan CPU vertical](../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md)
is Complete, MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured
source-matched root gate (PlanningAuthorized=true; ImplementationAuthorized=true record admission), joining Complete
Compiler0014/Prepare0013/Prepare0011 and the recorded CPU0037/resource/Engine/source predecessors.
It coordinates existing lowering/IR/preparer/emitter/executable owners, exact run sizing and a
distinct symbolic identity while freezing static schema71/v4 keys/bytes. No new type/signature
selected or added; any future addition requires explicit approval. At that acceptance CPU source frontier was None.
Planning APPROVE, 0 P1/P2; user “jdi na to” source approval and MAIN Ready verification 2026-10-25 17:00 UTC.
Parallel group None; MAIN completed serial integration/shared docs and the configured vertical join.
The brief records bounded CPU/Engine measurements and their limits, not performance parity.

CPU0033 + low-level join and accepted Compiler0012 baseline →
[CPU0034](tasks/0034-general-scan-coordinate-loops-and-carry-transition.md#result), all Complete.
Main accepted CPU0034's bounded coordinate-loop/carry optimization on 2026-10-23 after independent
review, runtime gates and exact-data/JIT audit. Existing emission/IR/cache owners remain; the brief
owns integration metadata, results and limits. CPU0034 → CPU0035 are Complete; Parallel group None,
main-owned serial implementation/review/checkpoint/campaign closed. CPU0035 Result owns evidence
and limits. CPU0035 → CPU0036 are Complete; CPU0036 Result owns evidence and disclosed limits.
Its brief retains verified base/dependencies/conflicts and integration metadata: Parallel group None,
serial clean implementation/review, main-owned shared documents/checkpoints/single campaign closed.
Checkpoint/NN/other gates stay held; no additional campaign or performance qualification implied.
CPU0032 accepted Complete 2026-10-22, 17:09+ UTC;
its [Result](tasks/0032-general-scan-f32-matmul-and-reverse.md#result) owns final evidence and limits.
CPU0023 and later domain gates stay Draft; [CPU0033 full homogeneous F64 closure](tasks/0033-general-scan-f64-domain-closure.md)
is Complete after main accepted independent code/docs approval and the native root join.
NN0022A/B remain Complete compatibility baselines, not new CPU dependencies.

[General scan program](../../general-differentiable-scan.md) / [Model0034 decision](../../modules/model/tasks/0034-general-differentiable-scan-decision.md)
records Model0034 authority decision/cutover Complete, accepted by main 2026-10-21 after final
independent Class C APPROVE, zero blockers; the brief owns result evidence. Compiler0010 fixed-adjoint
direction is Superseded. Serial DAG: completed reviewed authority cutover
→ Model source/IR → Planning body query → Compiler body AD → Prepare projection → bounded
CPU0030 F32 pointwise forward/automatic-reverse Complete → CPU0031 F32 broadcast/SUM/layout
Complete → CPU0032 rank-two MATMUL/actual F32 reverse Complete → CPU0033 full homogeneous F64 algebra/closure Complete
→ low-level root/native join Complete.
Base main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`; Parallel group None, main owns
shared documents/integration. CPU0030 is Complete. Actual user approval (“ok, tak tedy pokracuj”)
covers the concrete body/reverse-scan/residual proposal and authority cutover, not CPU implementation.
The brief records historical Complete predecessors/base/inactive conflicts verified on 2026-10-21.
Authority/Model frontiers None; [Model0035 source/IR](../../modules/model/tasks/0035-general-scan-source-and-compiled-body.md)
is Complete, accepted by main 2026-10-22 after independent approval and its compatibility checkpoint.
Planning0008 body query is Complete, accepted by main 2026-10-22 after final independent APPROVE and
compatibility gates; Planning frontier None. [Compiler0011](../../modules/compiler/tasks/0011-general-scan-capture-and-first-order-ad.md)
is Complete, accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint.
[Prepare0010](../../modules/prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand validation and resource handoff, main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint.
At that completed milestone CPU/Prepare/Compiler/Planning/Model/NN frontiers were None;
  CPU0034–CPU0045 are Complete; the current structural-repair queue is recorded above.
Bounded CPU F32 pointwise forward + actual
automatic reverse [CPU0030](tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete, accepted
by main 2026-10-22 after independent review and focused checkpoint PASS; its brief owns evidence.
Current base retains accepted uncommitted predecessors and CPU0030–0033 work.
[CPU0031 Result](tasks/0031-general-scan-f32-broadcast-sum-layout-closure.md#result) owns final
review/module/checkpoint evidence and main acceptance; its seven integration fields and historical
launch verification remain recorded. User-selected CPU0032 is Complete;
full F64 closure is CPU0033, Complete. Production ownership remains CPU-only. The low-level
root/native join is Complete; padding/NN, persistence and BF16/FP16 qualification stay separately gated Draft.

CPU0032's brief retains all seven integration fields, historical dependency/base verification and
main launch authorization. Serial integration, independent code/docs and two-file test-addendum
approval, final CPU test/Javadoc, named conformance/Engine and targeted architecture checkpoint,
and main acceptance are complete. Main owns shared documents; no implementation context remains.
[CPU0033 Result](tasks/0033-general-scan-f64-domain-closure.md#result) owns main acceptance, independent
code/docs and identity-test addendum approval, final CPU/Javadoc and source-matched native root evidence.
Its seven fields preserve actual HEAD plus accepted uncommitted foundations. Main owns shared documents;
at that acceptance CPU frontier was None, low-level join Complete, no successor launch. The shaped-forward timing gap remains
unexplained, reverse samples nonstationary; no JIT-cause or performance-parity claim.

Completed CPU0029 integration metadata (historical, not CPU0030 launch authority):

- Depends on: CPU0028 and CPU0022, verified Complete with accepted review/validation on 2026-10-21.
- Conflicts with: scan admission/lowering/kernel/binding/resources, numerical policy, current
  capability ledger and shared explanatory/planning edits; serialize these scopes.
- Parallel group: None; sole serial workstream, no concurrent writer authorized.
- Common base revision: `f918d7d7dab3fff0b144747f56d629733bd443c7` plus CPU0029 planning packet.
- Integration order: main launch verification → clean CPU executor → independent Class C code/docs
  review/corrections → main checkpoint → closure, all complete; no active executor.
- Integration validation: brief's focused conformance/public Engine commands, then main's single
  source-matched native-enabled root `test javadoc`, passed; exact evidence is in the brief.
- Shared-document integration owner: main; serial explanation/Javadoc drafting/review, main-owned
  planning/closure. User-owned dirty notice/cast-oracle files remain untouched.

Existing authority suffices: no public API, Engine orchestration, NN migration, generated/native/
vector/parallel scan route or tuning change. Low arithmetic uses FLOAT32 work/accumulation with
final narrowing per published value and exact stored recurrent state. Timing is a narrow bounded
diagnostic, not speed qualification or a performance claim.

## Milestones and current frontier

- CPU0048/CPU0049 are Complete after independent approval, recorded gates and MAIN acceptance.
  [CPU0049 Result](tasks/0049-producer-reduction-fusion.md#result) owns bounded scope/evidence/limits;
  [CPU0050](tasks/0050-average-pool2d-backward-broadcast-fold.md#result) Complete, MAIN accepted
  2026-10-31 after Dirac FINAL APPROVE and all gates PASS; no new production types, bounded existing-owner
  fold only. Separate [CPU0050A](tasks/0050a-low-fold-fp32-accumulation.md#result) Complete after FINAL
  approval and qualified gate closure; original full gate remains FAILED, not full green.
  Frontier was None after CPU0051 acceptance; CPU0077's separately admitted release repair is
  indexed above. CPU0052 Draft/held, historical Scan timing paused; no automatic successor.
- CPU0076 is Complete after independent source/protocol/result/documentation approval and MAIN's
  CPU/Javadoc/named Engine PASS. B256's bounded improvement and B32's JIT caveat stay in its Result;
  no universal speed/parity/C1 repair or automatic successor. Its frontier/budget closed; CPU0048 admission is above.
- CPU0074 is Complete and MAIN accepted; the brief records final validation and the sole comparison.
  Steady-state profitability is unqualified; concrete JIT-pressure investigation is separate.
  Scalar CPU0073/no-region Scan/defaults remain; no rollout or automatic successor is authorized.
- CPU0066 completed planning result is retained; CPU0067/0068 and CPU0069 outline are superseded.
  CPU0070 replacement, CPU0071 MATMUL foundation and CPU0072 pointwise/SUM composition are Complete
  after independent review and recorded MAIN gates on 2026-10-29. Their frontiers are closed;
  the subsequent backward-code comparison motivated CPU0073. Literal-transfer limits remain; CPU0072
  measured nothing and CPU0071's sole diagnostic stays spent/closed, with no parity claim.
- CPU0073's [exact fusion design](row-map-multi-reduction-fusion-plan.md) and implementation start
  were user-approved 2026-10-30. Implementation, independent final reviews, MAIN named join and
  documentation closure are Complete; source/measurement frontiers None, no automatic successor.
  Performance is unmeasured; no commit/push. CPU0048–0052 stayed queued at that checkpoint.
- CPU0065 Complete delivers ordinary F32/F64 vector MATMUL TANH after independent review and recorded gates; frontier None. CPU0063/0064 Complete: reviewed J4 alignment ported as combined baked v9, with scoped timing benefit and CPU/Javadoc/named integration gates; no broader Scan parity or vector/parallel capability. CPU0062 slots closed; CPU0061 Cancelled; CPU0048–0052 queued, CPU0053 Cancelled.
  [CPU0042](tasks/0042-length-aware-general-scan-execution.md#result)
  is Complete, MAIN accepted 2026-10-26 after independent code/docs APPROVE and recorded gates.
  [CPU0041](tasks/0041-symbolic-scan-reverse-carrier-specialization.md#result) remains Complete after
  MAIN acceptance/all gates; bounded measured-case gap resolved, no automatic launch.
- Completed DAG: NN0021A + Model0025E/0025F + Compiler0006A + CPU0027 + Engine0021 + CP0010/CP0011
  → [CPU0028](tasks/0028-fixed-recurrent-scan-forward-execution.md)
  → [NN0021B](../../extensions/nn/tasks/0021b-fixed-recurrent-scan-implementation-program.md), all Complete.
  At that checkpoint CPU frontier was None. Main accepted independent approval and the final root gate
  on 2026-10-20; Parallel group None, shared-document owner main. Briefs own evidence/base metadata.
  Scalar FLOAT32 canonical heap/native/mixed forward execution is delivered. CPU0029 is Complete
  for scalar four-type/input-layout coverage; vector/parallel/native scan routes,
  further Data/NN migration, BPTT and performance claims remain outside its scope.
- Completed diagnostic DAG: CPU0025 + Compiler0009 + NN0029 → CPU0026 → [CPU0027](tasks/0027-unit-topology-adjacency-ordering.md), all Complete. Main accepted the CPU-only fix after independent APPROVE and focused integration (2026-10-20); CPU0023 remains Draft. Parallel group None; main owns shared-document closure. Evidence and diagnostic limits remain in the brief; no architecture, route, fusion-policy or resource-contract change.
- Portable coverage and closure are Complete through 0009G1; OpenBLAS and supported lifecycle/
  tuning collaborations are Complete through 0010J.
- Evidence-backed production-default task 0010M is `Complete`: ordinary production remains
  scalar with configured/available parallelism `1`/`1` and minimum elements per worker `1`; the
  invalid broad vector candidate was reverted and its reports removed.
- 0007A1D is Superseded after its failed isolated performance gate; 0010D1 and 0011 are
  Blocked; 0012–0016 are Draft.
- [0019](tasks/0019-unbounded-cpu-partition-bounded-fusion-units.md) is Complete by explicit
  user-authorized exception to table order. [0018](tasks/0018-shared-external-read-recognition-validation.md)
  is also Complete: CPU-private DAG-derived `ValueId` binding closes the shared-weight/virtual-
  output validation hole while retaining legal MATMUL+RELU fusion. Focused and full CPU validation
  and independent Class C re-review passed. [0020](tasks/0020-bare-matmul-realization-strategy-alignment.md)
  is Complete from clean `f4e35b45`, with portable public execution and independent review
  passing. Audit repairs 0021 and 0022 are Complete after independent review and the combined
  checkpoint; 0023 is a proposed Draft follow-up. CPU 0024 is Complete, with the heap-carrier
  binding and mixed-carrier overlap repair accepted. CPU 0007A1D's historical failed performance
  gate is closed as Superseded, not accepted as passing.
- Documentation-only 0010K and 0010L and profile realization
  [0017](tasks/0017-explicit-accelerator-numerical-profile-realization.md) are Complete. CPU 0017
  landed at `372a8b98`; its independent Class C review and the combined numerical-profile backend
  integration checkpoint passed with zero findings or failures.

## Current numerical identity

Model 0032 removed the selectors and profile fields introduced by historical CPU 0017. The
review baseline uses CPU generated schema 69 and explicit dtype/shape/strategy compatibility.
FLOAT16 is an active Model type with occurrence-specific CPU support, not a future Model 0026
prerequisite. Completed CPU 0022 advanced the generated envelope to schema 70 for
activation-extreme and low-working-site corrections. CPU0030 advances the current envelope and
general-scan family identity to 71; schemas 70 and earlier are stale.
Historical profile integration evidence remains in its completed task records.
CPU0034/CPU0035's historical staged v2/v3 payloads were superseded by CPU0036's
`scan-shaped-v4` and `scan-matmul-v4` in both specialization streams; real v1/v2/v3 artifacts reject. Envelope/family 71,
geometry-free canonical v1 projections and unrelated pointwise/fixed keys/bytes stay frozen.
CPU0041 advances only the symbolic baked payload to v3 in both streams, rejecting symbolic v1/v2;
canonical symbolic v1, family/schema 71 and retained static/ordinary identities and bytes are unchanged.
CPU0042 adds only length canonical v1 and baked `|scan-length-aware-v1=` in both streams;
runtime L values are excluded, and retained all-valid/fixed/ordinary keys and generated bytes stay unchanged.
The implemented [CPU0071 exception](../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)
narrowly qualifies those historical retention promises. Its Result records the two static MATMUL
cutovers and exact retained controls, not universal byte-preservation or performance proof.

The [CPU0073 accepted mechanism](../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)
uses only selected ordinary projection 72 and ordinary/Scan row-fusion-v1 appendages.
Envelope 71, Scan projection 71 and noneligible keys/ClassFiles/resources stay unchanged;
bounded execution is implemented, independently reviewed and MAIN accepted 2026-10-30.

[CPU0076's implemented exception](../../../architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
permits new affected-only scalar staging markers in both streams; no-staging cases, canonical
keys, envelope71/projections72/71, SIMD/nonregional bytes and historical resources remain unchanged.
Complete-body/rank/lifetime-budget fallback preserves existing capability; the Result owns bounded
evidence rather than a blanket generation or performance guarantee.

## Live gates and decisions

- **CPU0031–0033 and low-level join Complete:** Compiler0010 is Superseded. All-valid canonical/static T first;
  F32 is the initial working case, first completed declared CPU domain includes F32 AND F64.
  Emit outer time loop + admitted typed body emitters at generation time, only in finalization
  after exact resource declaration/assigned slots. Hoist invariants and prefer direct time offsets;
  justify necessary staging/carry/temporaries, retaining live residuals/MATMUL storage. Changing
  addresses/geometry/carry belong to RunState, never shared prepared state. Existing entry-ABI
  emitters need explicit bindings; no runtime interpreter/per-op lookup/per-step callback or allocation
  or T-fold growth. Follow program's clean-Java/Class-File/short warmed performance gate; no no-copy,
  vector/parallel, all-dtype, JIT or CPU0029 parity promise. Concrete proposal/cutover approval is recorded;
  Model0034 authority and Model0035 source/IR are Complete; their frontiers None. CPU0033 and the
  low-level native root join and CPU0034–CPU0045 are Complete; the current repair queue is above.
  Planning0008 transport and Compiler0011 AD/capture are Complete;
  main accepted Compiler0011 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint.
  Prepare0010 scoped projection/resource handoff is Complete; other scan frontiers None.
  [CPU0030](tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete for bounded F32 pointwise
  forward/actual automatic reverse, accepted 2026-10-22. CPU0031 is Complete for bounded F32 broadcast/SUM/layout
  forward/actual reverse, including parameter-shaped carries and nonempty SUM from empty steps.
  CPU0032 closes MATMUL's fixed entry-slot/geometry seam and actual F32 automatic reverse;
  accepted evidence and limitations stay in its Result. F64's typed/byte-width/numerical
  qualification is [CPU0033](tasks/0033-general-scan-f64-domain-closure.md#result), Complete after independent
  code/docs approval and native root PASS; no new AD layer. No padded training upgrade or extra timing campaign. Old four-dtype/layout fixed routes remain.
- CPU 0024 is Complete after exact heap-segment binding, the P1 mixed-carrier overlap fix,
  clean independent re-review, and direct Engine/Checkpoint acceptance. Its combined
  source-matched `test javadoc` checkpoint passed. CPU0025 is Complete after independent APPROVE,
  zero findings; main's final native root gate passed on the unchanged reviewed NN0029 snapshot.
  CPU0026 diagnostic, CPU0027 adjacency/stable-Kahn fix and CPU0028 bounded scan execution are
  Complete after independent reviews and configured validation. CPU0023 stays Draft; CPU0029 is
  Complete after final independent APPROVE and main's source-matched root gate. CPU0030–CPU0036 and the low-level join are Complete;
  CPU0037–CPU0045 are Complete; the current repair queue is above. CPU0040's diagnostic
  left the issue unresolved at that checkpoint; CPU0041 closes the narrow fix with bounded
  observed near-Java performance in the measured case, not universal parity or a wider campaign.
  CPU0036 owns reused bounded loop-control evidence/limits; CPU0034/0035 timing stays historical.
  Existing architecture, other routes, fusion policy and resource contracts are unchanged.
- CPU 0023 is a proposed bounded vector ERF shape follow-up, not an authorized successor. The
  audit diagnostic shows a current timing gap, but does not establish its cause or a regression.
- CPU 0018 closed the virtualized-output/shared-weight recognition validation hole with
  CPU-private DAG-derived `ValueId` provenance. The separate >8-node CPU graph-size choice was
  fixed by 0019: retain maximal same-owner partitions and bound only local computation units.
- CPU 0020 aligned vector-eligible bare MATMUL realization with the configured scalar/vector
  preparation strategy before finalization; it makes no new route or speed claim.
- [OpenBLAS provider 0004](../openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)
  and CPU 0010D1 resume only after proving an exported direct BFLOAT16-input/output ABI and full
  FLOAT32 contraction with one final narrowing. Existing portable BFLOAT16 and FLOAT32/FLOAT64
  OpenBLAS routes are unaffected.
- CPU 0011 requires a concrete Intel workload plus supported oneMKL ABI/lifecycle/thread/numerical
  evidence. Other vendor rows require their stated use case and ABI/integration evidence. No
  vendor priority is implied.
- OpenBLAS remains a narrow FLOAT32/FLOAT64 MATMUL peer. Discovery, qualification, binary/session
  identity, thread coordination, and whole-plan representation cost precede eligibility.
- Completed [Prepare 0004](../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
  transports opaque candidates; [tuning 0001](../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
  owns local measurement/cache mutation; [tuning 0002](../../tools/tuning/tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md)
  owns complete-plan measurement. CPU 0010I/0010J are producers. Safe heuristics remain correct
  without tuning; versioned compatibility and corrupt evidence fail closed.
- 0007A1D failed 13 final-fork targets; 0008I has non-passing fork-0 evidence; 0008N retained scalar
  until 0008N1; 0008O is Cancelled at `KEEP_SCALAR`; 0008P remains `KEEP_WHOLE_CELL`. None is a
  universal generated-code or performance success.
- The portable generated route remains the semantic fallback. New SIMD or route migrations need
  family-specific direct-Java evidence. Generic movement/indexing/pooling SIMD and automatic
  decomposed loss/softmax recognition are not authorized by the current audit repairs.
- CPU 0016 extends tuning only across implemented vendor peers and adds no CPU measurement or
  generated-class persistence. Current plans, specializations and tuning compatibility are
  profile-free; retained dtype, source, target and schema identities remain mandatory.

- CPU 0010M retains five machine-local scalar reports and a minimum future-comparison acceptance
  framework under `backends/cpu/evidence/cpu-0010m-defaults/`. The reports validate the hardened
  timing floor and metadata but explicitly lack selected-plan facts and cannot authorize a
  production change.
  No compatible cross-target scalar/vector evidence covered the broad eligible policy, so
  parallelism, vector preference, materialization, partial reductions, native thresholds, tiles,
  fusion/decomposition, and specialization budgets all remain unchanged.

## Risks

- Leaking route choice into Planning/Runtime, provider policy into ABI leaves, or Engine/tuning
  ownership into CPU; treating platform/library/carrier/benchmark facts as capability.
- Ignoring transition, resource, or concurrency cost; weakening evidence gates; or narrowing
  general layouts/carriers while optimizing dense forms.
- Allowing hidden hot-path dispatch/allocation/boxing/reflection/synchronization, or confusing
  generated-class and tuning-cache compatibility with authentication or JIT persistence.

## Planning and history policy

New work uses the [compact brief](../../planning-guide.md); completed tasks are not rewritten.
Logs, benchmarks, inventories, context IDs, audits, old frontiers, and completed narratives stay
in task files/Git, outside default executor input. Code, tests, contracts, this table, and linked
structured Status/Result override older prose.
