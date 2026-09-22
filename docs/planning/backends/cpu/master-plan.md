# CPU Backend Master Plan

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
- The six Model types retain exact represented carriers, but storage never implies arithmetic,
  vector, native, or numerical support; [0009](tasks/0009-portable-generated-coverage-closure-checkpoint.md)
  owns the checked inventory. Run-owned buffers are aligned native off-heap; borrowed inputs are
  non-owning, and selected materializations are explicit, declared, costed, and reusable.
- Route/storage selection compares complete transition, resource, reuse, and concurrency cost.
  Fusion legality precedes profitability; safe heuristics work without tuning, and Runtime never
  searches or revises a prepared choice.
- Exact/default numerics and determinism are hard filters. Relaxed candidates require explicit
  permission; hardware, provider presence, workload size, or measurement cannot grant it.
- Providers are ABI/lifetime leaves. CPU owns discovery, qualification, capability, fallback,
  thread/lifetime coordination, and candidates. Prepared resources and isolated per-run storage
  obey Prepare/Runtime ownership, concurrency, rollback, transfer, and cleanup contracts.

## Dependencies and package ownership

Allowed direct dependencies are `modules/compiler` only for 0010F, `modules/model`,
`modules/config`, `modules/planning`, `modules/runtime`, `modules/prepare`,
`modules/backend-contract`, `modules/trace`, and `backends/openblas-provider`. `modules/engine` is
forbidden. CPU does not depend on tuning tools; its collaborations expose opaque candidates.

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
| 0007A1D | [Native-order segment layout hoisting](tasks/0007a1d-native-order-segment-layout-hoisting.md) | Review needed | 0007A1C first-fork evidence | Schema-32 semantics and structure pass; all 13 final-fork performance targets failed, with forks 2–5 and aggregates open. |
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
| 0011 | Intel oneMKL BLAS and VML peer routes | Blocked | 0010E; 0005A; 0009; concrete Intel CPU use case and supported oneMKL ABI evidence | Blocked until a concrete Intel workload and supported oneMKL ABI evidence both exist. |
| 0012 | Intel oneDNN partition peer routes | Draft | 0005A; 0009; stable common CPU lowering; concrete DNN/ML use case and supported oneDNN ABI evidence | Planned only after a concrete oneDNN use case and supported ABI evidence. |
| 0013 | Apple Accelerate peer routes | Draft | 0005A; 0009; concrete Apple CPU use case and supported Accelerate ABI evidence | Planned only after a concrete Apple CPU use case and supported Accelerate ABI evidence. |
| 0014 | AMD AOCL-BLAS and AOCL-LibM peer routes | Draft | 0005A; 0009; concrete AMD CPU use case and supported AOCL ABI evidence | Planned only after a concrete AMD CPU use case and supported AOCL ABI evidence. |
| 0015 | Optional AMD ZenDNN partition peer routes | Draft | 0014; 0005A; 0009; stable common CPU lowering; concrete ZenDNN use case and integration evidence | Optional ZenDNN route waits for 0014 plus a concrete use case and integration evidence. |
| 0016 | Cross-route CPU tuning-cache integration | Draft | 0010E; Prepare 0004; tools/tuning 0001; 0011–0015 as implemented | Generalize the proved tuning contract only across vendor peers that are actually implemented. |
| 0017 | Explicit relaxed numerical candidate consumption | Draft | Config 0006; 0005F; stable exact portable and implemented peer-route consumers | Admit relaxed candidates only with explicit Config permission and compatibility identity. |

## Milestones and current frontier

- Portable coverage and closure are Complete through 0009G1; OpenBLAS and supported lifecycle/
  tuning collaborations are Complete through 0010J.
- 0007A1D is Review needed; 0010D1 and 0011 are Blocked; 0012–0017 are Draft.
- No CPU task is Ready or In progress. The [roadmap](../../roadmap.md) owns the repository frontier.

## Live gates and decisions

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
  family-specific direct-Java evidence. Generic movement/indexing/pooling SIMD, FLOAT16 before
  Model 0026, and automatic decomposed loss/softmax recognition are not authorized.
- CPU 0016 extends tuning only across implemented vendor peers and adds no CPU measurement or
  generated-class persistence. CPU 0017 requires explicit Config numerical permission and keys
  numerical mode without hot-path policy lookup.

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
