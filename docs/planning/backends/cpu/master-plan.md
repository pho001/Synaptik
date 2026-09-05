# CPU Backend Master Plan

## Goal

Implement truthful CPU capability reporting, backend-owned preparation, portable correctness and
SIMD routes, optional native routes, storage, workspace, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CPU capability provider
- partition lowering, specialization, and fusion
- computation-oriented partition execution units and a CPU-private loop-oriented kernel IR
- portable generated scalar and JDK Vector API elementwise/reduction strategies
- standard JDK Class-File API generation for portable scalar and Vector API computation kernels
- optional OpenBLAS routes for supported BLAS-compatible linear algebra
- distinct Intel oneMKL BLAS/VML and oneDNN integrations
- Apple Accelerate BLAS, vDSP, and vForce integrations
- distinct AMD AOCL-BLAS/AOCL-LibM and later optional ZenDNN integrations
- other specialized and fused routes only when a concrete capability justifies them
- CPU executables, storage, workspace, scheduling, and tracing
- deterministic generated-class identity, verification, and process-local compatible reuse, with
  optional trusted-root filesystem persistence as a cold-path policy
- joint route and physical-representation selection across relevant CPU dataflow and partition uses
- typed, version-controlled, tested route candidate generators and compatible workload-cache
  lookup during prepare

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- separate CPU route backends

## Module invariants

- Planning selects CPU ownership; CPU prepare selects the route.
- All CPU routes remain inside one concrete backend.
- CPU backend never depends on engine.
- The portable route is the bytecode-first Java Class-File API plus Vector API production baseline
  and the always-available semantic fallback for every occurrence it supports. The scalar
  reference is for conformance and fail-closed checking, not a Runtime operation/IR interpreter.
- The next implementation task atomically replaces the provisional per-node portable pipeline.
  CPU analysis lowers one complete partition into computation-oriented execution units, performs
  safe fusion before exact resource declaration, selects a realization route, and finalizes one
  partition-level executable. One `OperationKind` must not imply one lowering, emitter,
  invocation, or kernel.
- `LogicalMemoryPlan` remains logical and complete for graph values. A graph value receives a
  physical Runtime slot only when backend analysis emits a `PreparationResourceRequirement.Buffer`.
  Same-unit fused intermediates remain graph and IR values without declarations or slots.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected current implementation direction for
  every portable CPU computation kernel. This is non-authoritative planning: the architecture
  permits generated JVM-bytecode CPU computation kernels without making that builder or another
  generation library an invariant.
- Default portable classes use shape-polymorphic primitive `start`/`end` loops. Concrete compatible
  extents and element count are cold-bound facts and do not enter class/cache identity unless an
  explicit evidence-selected fixed-shape specialization consumes a bounded specialization budget.
- One route-independent `CpuKernelIr` records typed boundary/virtual values, ordered semantics,
  access-plan form, universal loop model, and stores. Route, thread count, vector species,
  artifact root, Runtime/graph identities, generator versions, and instance bindings are separate.
- Every currently selected executable Model operation semantic must gain truthful portable
  generated coverage before the portable capability milestone closes. Metadata-only or zero-work
  view occurrences need no generated computation. Unsupported executable semantics fail closed
  and are not advertised until their generated coverage exists.
- OpenBLAS is a narrow cross-platform FLOAT32/FLOAT64 native fallback for supported BLAS-compatible
  linear algebra. It is neither a universal fallback nor the preferred CPU route by identity.
  BFLOAT16 requires a separately verified version, instruction-set architecture (ISA), and
  operation route. FLOAT16 support is neither broad nor baseline by assumption.
- CPU never offloads internally to MPSGraph or a custom Metal kernel. Planning must first select
  Metal ownership, after which the separate Metal backend owns those routes.
- CPU owns capability truth, provider coordination, route selection, fallback, thread/lifetime
  coordination, and tuning. Native provider layers remain low-level ABI/lifetime leaves.
- Common whole-partition lowering, fusion legality/profitability, `CpuKernelIr`, access plans,
  materialization accounting, numerical/determinism filtering, and representation planning are
  route-independent. Native provider adapters do not interpret graphs, plan broadcasting or
  fusion, or own shared resource lifetimes.
- Fusion legality is a fail-closed correctness decision made before profitability. Profitability
  ranks only complete legal CPU candidates and may select a split plan even when a fused plan is
  legal. Safe bounded heuristics are the default; optional model autotuning may later measure
  eligible complete candidates before Runtime, but Runtime never searches or revises the choice.
- Planning selects one CPU backend owner and one `BackendId("cpu")`. CPU Prepare chooses among the
  portable baseline, OpenBLAS where narrowly eligible, and vendor/platform-specialized peer routes
  by exact capabilities and whole-plan cost. There is no fixed vendor priority.
- ARM selection is capability-first. Apple Silicon may select Accelerate; another ARM target uses
  portable code generation unless a later task adds an explicitly verified provider.
- CPU candidate generators return complete valid route-specific configurations; shared tuning
  sees them opaquely.
- Safe CPU heuristics remain correct when tuning is disabled or a compatible cache entry is
  absent.
- Candidate eligibility is filtered for exact operation semantics and required determinism before
  any performance comparison. Current exact/default semantics do not permit vendor fast- or
  relaxed-math routines.
- Internal portable and vendor routes may implement fast exponential, hyperbolic-tangent, or
  similar algorithms, but an exact/default candidate is eligible only when it satisfies the
  ordinary operation's conformance contract. A genuinely relaxed approximation requires explicit
  caller permission; CPU must not infer it from hardware, provider availability, workload size,
  tuning objectives, or benchmark results.
- Each eligible operation family fixes its logical input, accumulation or other intermediate, and
  output types before CPU route selection. FLOAT32 accumulation is the expected default for
  numerically sensitive 16-bit work; exceptions require an explicit Model semantic contract.
- `DataType` availability and a matching storage carrier never advertise or select a CPU route.
  Every 16-bit candidate requires exact ABI, ISA/hardware, operation, layout, numerical,
  determinism, and resource filtering before route/workload benchmarking, safe heuristics, or
  compatible tuning evidence can compare it.
- Memory-segment storage and execution route are orthogonal. Scalar Java, JDK Vector API, and
  Foreign Function and Memory (FFM) native-provider calls may use the same native-backed
  `MemorySegment` without copying merely because the route changes.
- Run-owned internal CPU buffers use aligned native off-heap `MemorySegment` storage as the
  canonical interoperable CPU representation. Complete task 0001 implements its exact alignment,
  arena/lifetime ownership, access, cleanup, zero-size, and allocation contracts.
- Borrowed caller inputs are handled per value and use. Compatible heap-backed inputs may remain
  heap-backed. CPU preparation introduces at most one necessary native materialization for an
  exact selected downstream FFM route and reuses it across compatible consumers.
- Ordered carrier access form is an explicit code-shaping structural specialization. Each portable
  generated class has one direct static entry signature for its actual ordered primitive-array or
  `MemorySegment` pattern; that pattern participates in compatibility/class identity. Exact
  carrier objects and byte offsets remain cold bindings. CPU analysis receives and validates the
  backend-owned prepared pattern, finalization emits or loads its one artifact, and Runtime binding
  only accepts matching concrete carriers. This avoids emitting every possible carrier combination
  in each class while preserving direct hot code without Runtime generation, storage discovery, or
  dispatch.
- `CpuPartitionAnalysisInputs.DEFAULT` disables lowering-manifest retention and uses an empty
  explicit carrier list as the policy "one exact `MEMORY_SEGMENT` form per lowering-derived
  boundary." Explicit CPU analysis inputs may immutably select an ordered typed heap/segment
  pattern whose count and order must equal the derived boundary list. Boundary cardinality is not
  fixed by the analysis-input contract.
- Portable `MemorySegment` storage is representation-only and accepts the logical data type. A
  two-byte representation does not imply executable arithmetic or Vector support. The current
  generated specialization admits Java Vector lanes only for FLOAT64, FLOAT32, INT32, and INT64;
  BFLOAT16 and future FLOAT16 need separately established routes.
- Portable execution has exactly four strategies: scalar, vector, parallel-scalar, and
  parallel-vector. Scalar/vector is the compute axis; single-thread/parallel is the orchestration
  axis. Every generated kernel accepts `start` and `end`; workers dispatch chunks outside it.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace
- backends/openblas-provider

## Forbidden dependencies

- modules/engine

## Package structure

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider       sole public truthful fail-closed CPU capability provider
  package-info.java           public package boundary and current status
  internal/
    memory/                   representations and cold binding
    prepare/                  analysis and finalization lifecycle
    lowering/                 whole-partition unit formation, fusion, and focused family geometry
    ir/                       canonical IR, normalized access plans, and typed family identity
    codegen/emit/             portable Class-File generation and direct family/loop emission
    route/portable/           portable route selection/realization plan
    cache/                    structural identity and optional persistence
    executable/               prepared partition execution, worker orchestration, and the sole
                              bounded Conv2d two-unit composite
    reference/                conformance/fail-closed scalar reference
```

Task 0005A adopts this structure atomically. Java subpackages are separate access domains, not
friends, so only the minimal cross-package contracts are technically public below `.internal` and
are explicitly unsupported API. `CpuCapabilityProvider` remains the sole supported public CPU
type. No JPMS export, service locator, registry, broad facade, compatibility bridge, or retained
flat pipeline is permitted.

Later tasks add only their concrete leaves: `route/nativeblas/{openblas,accelerate,mkl,aocl}` for
BLAS-compatible calls and `route/nativeops/{accelerate,mkl,onednn,aocl,zendnn}` for vDSP/vForce,
VML, oneDNN, AOCL-LibM, and ZenDNN. These target locations are not placeholder packages and are not
created by 0005A. All consume the common analysis above; none creates another backend identity.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [CPU capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md) | Superseded | Stable planning, runtime, prepare, backend-contract, and trace contracts | Historical foundation replaced atomically by 0005A; retained as validation and design evidence. |
| 0002 | [Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md) | Superseded | 0001; generated JVM-bytecode CPU-kernel architecture contract; Java 26 Class-File and Vector API toolchain | Historical generator foundation replaced atomically by 0005A; retained as validation and design evidence. |
| 0003 | [Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md) | Superseded | 0002; stable CPU finalization and artifact compatibility; explicit trusted local root | Historical mandatory durable-store design replaced by 0005A's optional persistence policy. |
| 0004 | [Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md) | Superseded | 0001–0003 | Historical per-node candidate/finalization architecture replaced atomically by 0005A. |
| 0005 | [Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md) | Superseded | 0002–0004 | Historical per-node dense ADD route replaced atomically by 0005A's fused partition kernel. |
| 0005A | [Atomic partition-kernel architecture reset](tasks/0005a-atomic-partition-kernel-architecture-reset.md) | Complete | 0001–0005; current shared Prepare contracts | Adopted structured internals and only the portable route leaf; replaced the per-node path with whole-partition lowering, route-independent IR, universal start/end Class-File generation, exact declarations, and one partition executable; proved shape-polymorphic FLOAT64 ADD-to-GELU-to-MUL fusion. |
| 0005B | [Universal access plans and right-aligned broadcasting](tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md) | Complete | 0005A | Delivered one normalized per-value access system over current ShapeBroadcast/LayoutDescriptor semantics, complete static scalar/rank-expanded/multi-axis/zero/strided/heap/segment/mixed-carrier support, one direct entry per ordered carrier-pattern specialization, and five distinct dense-to-general-odometer scalar state machines. |
| 0005C | [Vector and parallel portable strategies](tasks/0005c-vector-and-parallel-portable-strategies.md) | Complete | 0005B | Added preferred-species FLOAT64 vector, parallel-scalar, and parallel-vector realization over universal start/end kernels; direct contiguous runs vectorize without gather, and explicit caller-owned CPU-private workers execute deterministic disjoint chunks. |
| 0005D | [Materialization, specialization, and persistence evidence gate](tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md) | Complete | 0005C | Added at most one CPU-internal contiguous materialization before assignment, enforced four-candidate/one-artifact/zero-shape/zero-unroll budgets, and recorded a `KEEP_DISABLED` opt-in persistence-evidence verdict. |
| 0005E | [Portable pointwise types, carriers, and semantic-family expansion](tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md) | Complete | 0005D | Delivered the first bounded five-type core pointwise increment through one nineteen-opcode family pipeline, derived-boundary typed carriers, and the completed unit/IR/access/materialization/route/artifact/executable architecture; unsupported rows remain fail-closed. |
| 0005F | [Floating division and exact scalar-power realization](tasks/0005f-floating-division-and-exact-scalar-power-realization.md) | Complete | 0005E | Added exact/default same-typed FLOAT32/FLOAT64 binary and scalar DIV plus direct scalar `POW`; retained semantic `POW` while selecting only the proved positive-one, identity, one-multiply square, or one-division reciprocal realizations. All three family opcodes preserve the completed pointwise boundaries, and reciprocal power remains semantically distinct from DIV. |
| 0005G | [Extrema, clamp, Tensor power, and logical coverage](tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md) | Complete | 0005F; Model 0018T/0018U/0025A | Added exact same-typed binary/scalar MIN/MAX, first-class floating CLAMP, direct floating Tensor/Tensor POW, and canonical-BOOL AND/OR/NOT through the existing family pipeline. Every new row is scalar or parallel-scalar; one-instruction CLAMP, completed budgets, and fail-closed cross-type CAST/unary boundaries remain preserved. |
| 0005H | [Portable unary, transcendental, and activation closure](tasks/0005h-portable-unary-transcendental-and-activation-closure.md) | Complete | 0005G; Model 0018P/0018T1/0019A semantics; Java 26 math/Vector contracts | Closed all nineteen FLOAT32/FLOAT64 unary kinds through the existing family pipeline, preserved completed NEG and classification rows, extended/corrected GELU, selected an explicit scalar/vector algorithm, special-value, and tolerance matrix, advanced to schema 8, and retained cross-type CAST fail-closed. |
| 0005I | [FLOAT32 vector parity and vector-emission boundary](tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md) | Complete | 0005H; Java 26 `FloatVector`/`DoubleVector`; completed pointwise access, specialization, and numerical contracts | Added preferred-species FLOAT32 parity for the exact twenty-one current vector-eligible pointwise opcodes, kept every unsupported row on scalar fallback, split operation-level vector bytecode emission from pure vector math, and made the selected ERF/GELU coefficient provenance and binary32 derivation auditable. |
| 0005J | [Bounded pointwise coverage and parity hardening](tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md) | Complete | 0005I; current Model pointwise semantics; Java 26 Byte/Int/Long/Float/Double Vector API contracts | Added exact floating extrema/clamp/ReLU/sign/cast, signed-integral arithmetic/extrema/cast, canonical-BOOL logic/cast, and virtual floating-mask-to-WHERE preferred-species parity; schema 10, deterministic scalar fallback, and all completed budgets remain explicit. |
| 0006 | [Portable static affine views and boundary materialization](tasks/0006-portable-static-affine-views-and-boundary-materialization.md) | Complete | 0005J | Folds bounded straight-line static RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, SELECT, and normalized SLICE mappings without computation or slots when they remain internal; otherwise generates one exact portable scalar or parallel-scalar boundary copy, including CONTIGUOUS, for all six Model data types through the resulting seven carrier forms. The new `SHORT_ARRAY` form copies BFLOAT16 represented bits only and does not advertise BFLOAT16 arithmetic or numerical support. |
| 0006A | [Portable pad, tile, and tensor-composition movement](tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md) | Complete | 0006 | Added one fully static resolved-layout PAD, TILE, CONCAT, or STACK node through compact CPU-private movement IR, unique multi-input declarations, exact represented-bit scalar/parallel-scalar generation, and one materialized injective output; schema 12 records movement identity. |
| 0006A1 | [Portable static window extraction](tasks/0006a1-portable-static-window-extraction.md) | Complete | 0006A | Added one-node fully static resolved-layout UNFOLD_AXIS for all six current types and both ordered floating-only UNFOLD2D variants through the completed movement foundation, with exact axis/NCHW mapping, dilation, floor/ceil grids, conceptual-zero and typed-padding represented bits, compact unequal-rank geometry, and schema 13; no fold accumulation. |
| 0006A2 | [Portable gather and one-hot indexing](tasks/0006a2-portable-gather-and-one-hot-indexing.md) | Complete | 0006A1 | Added GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT with INT32/INT64 carriers, a complete pre-write execution-time index-validation pass, deterministic first-invalid-index failure, scalar or parallel-scalar writes, and schema 14; no wrap, clamp, default selection, partial Gather output, or all-false invalid one-hot row. |
| 0006B | [Portable functional slice update](tasks/0006b-portable-functional-slice-update.md) | Complete | 0006A2 | Added one fully static resolved-layout SLICE_UPDATE occurrence for both current `SliceAttrs` signed finite-coordinate placement and `CropToShapeAttrs` target-relative placement through the existing represented-bit movement pipeline, with functional base/update selection, injective disjoint output, compact geometry, scalar or parallel-scalar generation, and schema 15. |
| 0006B1 | [Portable functional scatter](tasks/0006b1-portable-functional-scatter.md) | Complete | 0006B | Added SCATTER_ELEMENTS, Gather-compatible SCATTER_ADD, and SCATTER_ND with exact base participation, INT32/INT64 bounds, complete pre-write bounds/NONE-duplicate validation, represented-value reductions, deterministic disjoint output ranges, declared exact floating-product scratch, and schema 16. |
| 0006B2 | [Portable overlap fold](tasks/0006b2-portable-overlap-fold.md) | Complete | 0006B1 | Added FOLD_AXIS and FOLD2D with represented positive-zero initialization, deterministic row-major overlap accumulation, exact 2D padding exclusion, type-specific addition, disjoint output ranges, scalar fallback, zero workspace, and schema 17. |
| 0006C | [Portable stable ordering and selection coverage](tasks/0006c-portable-stable-ordering-and-selection.md) | Complete | 0006B2 | Added one-node fully static resolved-layout stable SORT and ARGSORT plus two-output TOP_K for all six current types through deterministic scalar/slice-parallel generated execution, with exact NaN-last, signed-zero, logical-tie, empty/K, output-order, bounded per-range scratch, overlap, schema-18, and multi-store behavior. |
| 0006D | [Portable explicit-state RNG and dropout coverage](tasks/0006d-portable-explicit-state-rng-and-dropout.md) | Complete | 0006C | Materialized zero-input INITIAL_STATE and executes FLOAT64/FLOAT32 three-output DROPOUT with the versioned CPU-private `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` mapping, exact uniform/threshold/scaling rules, canonical BOOL mask, modulo state advancement, deterministic scalar/parallel replay, zero workspace, complete overlap rejection, and schema 19. BFLOAT16 dropout remains truthfully fail-closed. |
| 0007 | [Portable cumulative scan coverage](tasks/0007-portable-cumulative-scan-coverage.md) | Complete | 0006D; Model 0023E; completed 0005A–0005J | Executes one static resolved-layout CUM_SUM/CUM_PROD occurrence across five numeric types and all four modes with sequential typed accumulation, whole-slice scalar/parallel-scalar execution, complete overlap rejection, zero workspace/materialization, bounded injectivity validation, and schema 20. |
| 0007A | [Portable ordinary extrema and boolean reductions](tasks/0007a-portable-ordinary-extrema-and-boolean-reductions.md) | Complete | 0007; current Model aggregate-reduction contracts | Executes one-node full, single-axis, and multi-axis MIN/MAX/ALL/ANY across exact numeric/BOOL types with empty identities, logical row-major deterministic folds, output-cell-only parallelism, arbitrary supported layouts/carriers, zero workspace, complete overlap rejection, and schema 21. |
| 0007A0 | [Generated hot-path parity correction](tasks/0007a0-generated-hot-path-parity-correction.md) | Complete | 0005A/0005B/0005C/0005I; 0007; 0007A; completed local bytecode/performance audit | Corrected every measured dense generated-code defect atomically: cold-proved int scalar/address loops, one-bound Vector API loops, typed generated scan/aggregate hot bodies with general fallbacks, schema 22 compatibility, stable code-shape/semantic tests, and an isolated five-fork `<= 1.15x` direct-Java near-parity gate. |
| 0007A0A | [Affine and movement generated-loop parity](tasks/0007a0a-affine-and-movement-generated-loop-parity.md) | Complete | 0006/0006A/0006A1/0006B; 0007A0 | Added schema-23 cold-proved integer affine and seven-family movement bodies, corrected multidimensional TILE carry/wrap semantics, retained typed general-long forms, and passed isolated affine/TILE/SLICE_UPDATE five-fork per-case `<= 1.15x` gates. |
| 0007A0B | [Indexing generated-loop parity](tasks/0007a0b-indexing-generated-loop-parity.md) | Complete | 0007A0A; completed 0006A2 semantics | Replaced bridge-only GATHER/GATHER_ELEMENTS/GATHER_ND/ONE_HOT hot work with carrier- and type-specialized generated loops, advanced schema 23 to 24, and passed dense FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT independently at `1.084234x` and `0.895060x` direct-loop parity while retaining complete pre-write index validation and general layouts. |
| 0007A0C | [Scatter generated-loop parity](tasks/0007a0c-scatter-generated-loop-parity.md) | Complete | 0007A0B; completed 0006B1 semantics | Embedded typed scatter output/contribution/reduction and exact-product bodies, advanced schema 24 to 25, preserved complete cold validation and general layouts, and passed dense unique SCATTER_ELEMENTS and duplicate-index FLOAT32 SCATTER_ADD independently at `0.979533x` and `0.983230x`. |
| 0007A0D | [Fold generated-loop parity](tasks/0007a0d-fold-generated-loop-parity.md) | Complete | 0007A0C; completed 0006B2 semantics | Embedded schema-26 carrier-, type-, family-, access-, mapping-, and addition-specialized FOLD_AXIS/FOLD2D loops with dense integer and typed general-long forms; the fixed overlapping FLOAT32 FOLD_AXIS gate passed every fork and aggregate at `0.926451x` while preserving CPU 0006B2 semantics. |
| 0007A0E | [Ordering generated-loop parity](tasks/0007a0e-ordering-generated-loop-parity.md) | Complete | 0007A0D; completed 0006C semantics | Embedded schema-27 carrier-, represented-type-, family-, direction-, output-, and access-specialized SORT/ARGSORT/TOP_K loops with dense integer and typed general-long forms; stable FLOAT32 SORT and two-output TOP_K passed every fork and aggregate at `1.090165540x` and `0.965951020x` while preserving exact scratch and CPU 0006C semantics. |
| 0007A0F | [Random and dropout generated-loop parity](tasks/0007a0f-random-and-dropout-generated-loop-parity.md) | Complete | 0007A0E; completed 0006D semantics | Embedded schema-28 typed INITIAL_STATE and FLOAT64/FLOAT32 DROPOUT state/mapping/value/mask bodies with dense integer and general long-address forms; every fixed dense FLOAT64/FLOAT32 fork and aggregate passed `<= 1.15x` while preserving replay, counter advancement, three-output binding, and general carriers. |
| 0007A1 | [Portable ordinary numerical aggregate reductions](tasks/0007a1-portable-ordinary-numerical-aggregate-reductions.md) | Complete | 0007A; 0007A0F | Delivered the closed SUM/MEAN/PROD inventory with schema-29 direct dense/general bodies, exact floating state and one result-format rounding, modular integral arithmetic, predeclared per-range scratch, independent oracles, passing CPU tests/Javadoc/Class-File/documentation gates, and all 13 five-fork performance cases at `<= 1.15x`. |
| 0007A1A | [Generated scalar-body self-containment](tasks/0007a1a-generated-scalar-body-self-containment.md) | Complete | 0007A1; approved schema-29 generated-code audit | Embedded the five scalar activation formulas, BFLOAT16 scan conversion/arithmetic, and aggregate extrema/Boolean combination directly in scalar generated bytecode; retained only the typed chunk-level `CpuVectorMath` vector boundary, advanced current-only compatibility to schema 30, and passed the scoped raw-bit/member-reference plus five-fork direct-Java gates. |
| 0007A1B | [Scatter algorithmic parity](tasks/0007a1b-scatter-algorithmic-parity.md) | Complete | 0007A1A; completed 0006B1/0007A0C semantics | Replaced scratch-free output-per-update grouping with range-owned copy-then-update and `O(output + workers * updates)` total work; retained the output-owned exact floating-product safe split and unchanged scratch, advanced schema 30 to 31, and passed all six five-fork direct-Java gates plus the five scratch-free `>= 4x` improvement gates. |
| 0007A1C | [Generated/direct evidence closure](tasks/0007a1c-generated-direct-evidence-closure.md) | Complete | 0007A1B; completed generated-family inventory through 0007A1 | Accumulated schema-42 evidence closes exact semantics, all twenty five-fork performance rows, generated/decompiled structural gates, and every required axis; A1O's tested replacement closes the original ledger's final pointwise structural-only accounting gap without rewriting historical evidence. |
| 0007A1D | [Native-order segment layout hoisting](tasks/0007a1d-native-order-segment-layout-hoisting.md) | Review needed | 0007A1C first-fork evidence | Retains schema-32 invocation-local typed segment layouts and passing semantic/Java/Class-File evidence, but all 13 required performance targets failed the final fork; forks 2–5 and aggregates remain open. |
| 0007A1E | [Movement general-address-loop parity](tasks/0007a1e-movement-general-address-loop-parity.md) | Complete | 0007A1D stable schema-32 prerequisite and failed fork | Added schema-33 cold-proved bounded primitive geometry/cursor loops for PAD, CONCAT, UNFOLD_AXIS, and UNFOLD2D with typed general-long fallback; all four targets and three controls passed every fork at `<= 1.15x`. |
| 0007A1F | [BOOL movement and aggregate residual parity](tasks/0007a1f-bool-movement-and-aggregate-residual-parity.md) | Complete | 0007A1E | Added schema-34 cold-proved occurrence-major canonical-BOOL STACK copies and full-visit zero-stride ANY folds with typed general-long fallbacks; both targets and all three controls passed every fork at `<= 1.15x`. |
| 0007A1G | [Fold and dropout residual parity](tasks/0007a1g-fold-and-dropout-residual-parity.md) | Complete | 0007A1F | Added schema-35 guarded bounded forms for the frozen mixed-carrier padded/dilated FLOAT32 FOLD2D and rank-one FLOAT32 dropout shapes with exact clean-Java algorithm/dataflow equivalence and typed general-long fallbacks; both targets and all three controls passed every fork at `<= 1.15x`. |
| 0007A1H | [Numerical aggregate residual parity](tasks/0007a1h-numerical-aggregate-residual-parity.md) | Complete | 0007A1G | Added schema-36 guarded exact-state forms for frozen FLOAT32 axis-one MEAN and BFLOAT16 axes-zero/two PROD with typed fallbacks; both targets and all three controls passed every fork and median at `<= 1.15x`. |
| 0007A1I | [Indexing residual parity](tasks/0007a1i-indexing-residual-parity.md) | Complete | 0007A1H | Added schema-37 guarded primitive cursor forms for frozen mixed-carrier FLOAT64 GATHER and FLOAT32 GATHER_ND, including one fixed 16-element full-range suffix body inside the existing artifact; both targets and all controls passed every fork and median at `<= 1.15x`. |
| 0007A1J | [Cumulative scan residual parity](tasks/0007a1j-cumulative-scan-residual-parity.md) | Complete | 0007A1I | Added the schema-38 completely guarded fixed reverse exclusive INT64 product segment-cursor body with arbitrary legal complete-slice ranges and typed fallback; target and controls passed every accepted fork and median at `<= 1.15x`. |
| 0007A1K | [Affine-copy residual parity](tasks/0007a1k-affine-copy-residual-parity.md) | Complete | 0007A1J | Added the schema-39 completely guarded raw-BFLOAT16 `A-GENERAL` body through sole owner `CpuAffineCopyEmitter`, preserving composed PERMUTE/SLICE mapping, arbitrary legal ranges, zero workspace, and typed fallback; target and controls passed every accepted fork and median at `<= 1.15x`. |
| 0007A1L | [Pointwise general-loop residual parity](tasks/0007a1l-pointwise-general-loop-residual-parity.md) | Complete | 0007A1K | Added the schema-40 completely guarded frozen FLOAT32 mixed-carrier `P-SCALAR-GENERAL` ordinal loop with arbitrary legal ranges and unchanged typed fallback; exact semantics and all target/control five-fork gates passed. |
| 0007A1M | [Scatter MIN residual parity](tasks/0007a1m-scatter-min-residual-parity.md) | Complete | 0007A1L | Added the schema-41 completely guarded frozen INT64 `SCATTER_ND + MIN` direct copy and tuple/suffix loops through sole owner `CpuScatterEmitter`; target and controls passed every fork and median, leaving only `X-MIN-MULTI`. |
| 0007A1N | [Multi-axis MIN residual parity](tasks/0007a1n-multi-axis-min-residual-parity.md) | Complete | 0007A1M | Added the schema-42 completely guarded frozen BFLOAT16 multi-axis MIN primitive traversal through sole owner `CpuAggregateEmitter`; exact semantics and all twenty rows passed five accepted forks and medians, with one unrelated whole sample retained and rejected. |
| 0007A1O | [Pointwise ledger evidence reconciliation](tasks/0007a1o-pointwise-ledger-evidence-reconciliation.md) | Complete | 0007A1C evidence; 0007A1A; 0007A1L; 0007A1N | Preserved the original 79-line ledger, replaced 40 unauthorized pointwise structural-only labels with verified generated/direct equivalence categories, added a stable repository ledger test, and completed the final A1C re-audit without changing production bytes or schema 42. |
| 0007A2 | [Portable binding-aware sum-to-Shape reduction](tasks/0007a2-portable-binding-aware-sum-to-shape-reduction.md) | Complete | Complete 0007A1C; accumulated schema-42 evidence through 0007A1O; Model 0023A; Compiler 0005B | Added `SUM` with exact `SumToShapeAttrs`, right-aligned bound-Shape validation, leading/aligned reduction geometry, all five numeric types, truthful resources, direct generated loops, and schema 43 without dynamic unresolved execution. Its corrected 37-path scope permits the A1O ledger integration test to distinguish historical schema-42 evidence from exact current schema 43 while preserving the ledger resource unchanged. |
| 0007B | [Portable arg-extrema coverage](tasks/0007b-portable-arg-extrema-coverage.md) | Complete | 0007A2; Model 0018U1; Compiler 0005B | Added one fully static resolved-layout one-axis ARG_MIN/ARG_MAX occurrence with exact five-type ordering, FIRST/LAST logical ties, INT64 indices, zero resources, focused private owners, direct generated loops, deterministic scalar/parallel output-cell ranges, schema 44, and passing five-fork optimal-direct-Java parity. |
| 0007C | [Portable masked reduction coverage](tasks/0007c-portable-masked-reduction-coverage.md) | Complete | 0007A1; 0007B; Model 0018Q; Compiler 0005B | Added exactly one fully static axis-removing FLOAT64/FLOAT32/BFLOAT16 masked SUM/MEAN occurrence with a canonical BOOL mask, directional right-aligned broadcast exactly to the data Shape, pre-classification false exclusion, exact selected-count semantics, existing exact floating state, focused private owners, direct typed bytecode, schema 45, and passing optimal-direct-Java evidence. |
| 0007D | [Portable logarithmic, statistical, and norm reduction coverage](tasks/0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md) | Complete | 0007A1; Model 0018V; Compiler 0005B | Added fully static FLOAT64/FLOAT32/BFLOAT16 LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, and L2_NORM through shared output-cell geometry, three focused direct emitters, exact special values, stable finite algorithms, schema 46, and passing optimal-direct-Java evidence. |
| 0007E | [Portable stable softmax and log-softmax coverage](tasks/0007e-portable-stable-softmax-and-log-softmax-coverage.md) | Complete | 0007D; Model 0016I/0016J; Compiler 0005B | Added first-class one-axis SOFTMAX/LOG_SOFTMAX over the CPU-private finite, positive-width admitted subset through direct stable max-shift generated loops, zero workspace, schema 47, and passing `<= 1.15x` evidence gates; decomposed graphs are never inferred as this family. |
| 0007F | [Portable layer and RMS normalization coverage](tasks/0007f-portable-layer-and-rms-normalization-coverage.md) | Complete | 0007E; Model 0021/0021A; Compiler 0005B | Added all four first-class Layer/RMS forms through shared static trailing-slice geometry, separate direct numerical emitters, exact ordered floating promotion, deterministic complete-slice ranges, Layer-only existing exact-state scratch, RMS zero workspace, schema 48, and passing optimal-clean-Java evidence; decomposed graphs are never inferred as normalization. |
| 0007F1 | [Portable batch-normalization inference coverage](tasks/0007f1-portable-batch-normalization-inference-coverage.md) | Complete | 0007F; Model 0021B; Compiler 0005B | Added stateless coordinatewise five-input/one-output batch inference with arbitrary normalized channel axis, exact ordered promotion/epsilon, channel-hoisted running-statistic work, deterministic channel/non-channel ranges, zero workspace, direct generated execution, and schema 49. |
| 0007F2 | [Portable batch-normalization training and statistic-transition coverage](tasks/0007f2-portable-batch-normalization-training-and-statistic-transition-coverage.md) | Complete | 0007F1; Model 0021C; Compiler 0005B | Added first-class five-input/five-output static training execution through complete-channel scalar/parallel-scalar ranges, exact-sum scratch, corrected biased and unbiased variances, typed momentum/epsilon transitions, saved statistics, mixed output Shapes, direct five-output publication, complete overlap validation, and schema 50. |
| 0008 | [Portable grouped NCHW Conv2d execution foundation](tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md) | Complete | 0002–0007F2; Model 0020; Model 0025G; Model 0025H; Compiler 0006B | Added direct grouped NCHW Conv2d with intrinsic optional bias, groups/depthwise, explicit padding/stride/dilation, scalar/parallel-scalar complete-output-cell generated code, schema 51, and passing optimal-clean-Java evidence. Legal external ADD and ADD-plus-RELU remain direct; the sole tagged two-unit CPU-private composite materializes one independently supported suffix through one ordinary intermediate buffer and one atomic Runtime executable boundary. |
| 0008A | [Portable channels-first dimensional convolution closure](tasks/0008a-portable-channels-first-dimensional-convolution-closure.md) | Complete | 0008; Model 0025G–0025H; Compiler 0006B | Validated exact NCW Conv1d through its visible virtual-singleton `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition and added direct grouped NCDHW Conv3d with intrinsic optional bias, resolved arrays/segments/mixed carriers, zero workspace/materialization, scalar/parallel-scalar complete-output-cell generated execution, schema 52, and passing structural/performance evidence. External Conv3d epilogues and general DAG handling remain fail-closed. |
| 0008B | [General partition-DAG computation-unit decomposition and bounded fusion](tasks/0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md) | Complete | 0006–0008A | Added deterministic one-to-eight-unit partition-DAG decomposition, bounded vertical/horizontal ordinary-pointwise fusion, exact materialized split buffers, final-index unit workspaces, general atomic sequential finalization, and direct multi-store Class-File/performance evidence. The final CPU suite passed 99 suites/512 tests with three expected skips; all five accepted generated/direct forks and their aggregate passed `<= 1.15x`, while all six rejected samples remain retained. |
| 0008C | [Typed specialized-subgraph and epilogue recognition](tasks/0008c-typed-specialized-subgraph-and-epilogue-recognition.md) | Complete | 0007F2–0008B | Added recognition-only CPU-private typed facts for exact MATMUL, Conv1d/Conv2d/Conv3d, selected floating-reduction epilogues, and already first-class softmax/normalization kernels. The uniform suffix is optional external ADD plus at most one exact activation/CLAMP. Only CPU 0008's existing Conv2d ADD/ADD-RELU form is already specialized; MATMUL stays unsupported until 0008F and every other recognized epilogue retains the exact 0008B split. Schema 52, artifact identity, capability, generated code, and public/shared contracts remain unchanged; exact baseline and no-leakage evidence passed. |
| 0008D | [Bounded fusion profitability and typed decision facts](tasks/0008d-bounded-fusion-profitability-and-typed-decision-facts.md) | Complete | 0008B–0008C | Implemented the complete admitted bounded set, deterministic checked integer ranking, best-only tie fallback, graph-identity-free typed facts, shared typed contraction outcomes, exact retained-recognition overlap validation, and independent publication/write role recomputation. The authoritative CPU suite passed 536 tests in 103 suites with 3 skips and no failures/errors; both retained runs kept all 45 samples per comparison, schema 52 and generated forms remain unchanged, and the clean documentation pass finalized Javadocs, guide, glossary, and planning evidence. |
| 0008E | [Bounded multi-input materialization and representation reuse](tasks/0008e-bounded-multi-input-materialization-and-representation-reuse.md) | Complete | 0008D | Preserved complete bounded direct/single/disjoint-pair candidates, `CO_CONSUMED_PAIR`, resources, generated copy units, reuse, schema 53, and execution-equivalence evidence while making every materialized form candidate-only for ordinary preparation. Ordinary preparation selects CPU 0008D direct; future Prepare 0004/CPU 0016/Tuning 0001–0002 own compatible end-to-end promotion, and Runtime never selects. The corrective implementation passed 115 focused tests and the final 544-test/104-suite CPU run with 3 expected skips and no failures/errors; clean documentation context `01a04317-b784-76e3-a93b-ff35106284b9` finalized Javadocs, guide, glossary, and planning evidence. |
| 0008E1 | [Shared partition-DAG adoption and reconstruction removal](tasks/0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md) | Complete | Prepare 0003A; 0008E | Adopted Prepare's immutable partition-local DAG across CPU decomposition, pointwise and affine lowering, recognition, and profitability boundary accounting while preserving CPU-owned unit/candidate/IR facts. The focused six-suite matrix passed 63 tests; the authoritative CPU rerun passed 547 tests across 104 suites with 3 expected skips and no failures/errors. Schema 53 and generated/cache/executable/finalizer production paths remain unchanged. |
| 0008F | [Portable MATMUL execution and bounded linear epilogues](tasks/0008f-portable-matmul-execution-and-bounded-linear-epilogues.md) | Complete | 0008E1; Model 0019/0019D; Compiler 0005D | Added complete static vector/matrix/batched/right-broadcast MATMUL across all current non-BOOL numeric promotions through four bounded portable generated realizations, a complete scalar fallback, full-K accumulation, independent output-work-unit parallelism, explicit candidate-only 0008E materializations, and exact 0008C bias/one-terminal fusion or safe split. The 182-test focused baseline, 576-test CPU checkpoint, Class-File scans, and five-fork `<= 1.15x` evidence passed at schema 54. No native route, K splitting, panel packing, or hot-path policy was added. |
| 0008G | [Portable max/average Pool2d execution](tasks/0008g-portable-max-average-pool2d-execution.md) | Complete | 0008F; Model 0020A–0020A1; Compiler 0005D | Added first-class NCHW max and fixed-count average pooling through one generated scalar/parallel-scalar form with exact literal floor/ceil geometry, extrema/divisor/accumulator/special-value rules, static resolved layouts, array/segment carriers, disjoint output-cell ranges, zero workspace, schema-55 identity, and an optimal clean Java oracle. Focused and 596-test CPU validation, structural/schema scans, and five-fork `<= 1.15x` evidence passed; no pooling fusion or materialization was added. |
| 0008G1 | [Portable Pool1d composition validation and Pool3d generated execution](tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md) | Complete | 0008G; Model 0025I–0025K; Compiler 0006B1–0006B2 | Recognizes only the exact visible NCW Pool1d singleton-height composition, keeps its rank edits virtual, and reuses byte-identical schema-55 Pool2d code without a Pool1d capability or artifact. Adds direct NCDHW max/average Pool3d scalar/caller-parallel complete-output-cell execution for static resolved BFLOAT16/FLOAT32/FLOAT64 carriers with zero workspace/materialization and schema 56. Focused 181-test and 618-test CPU validation, 24-family structural scans, and all five-fork ratios passed. |
| 0008H | [Portable scaled-dot-product attention execution](tasks/0008h-portable-scaled-dot-product-attention-execution.md) | Complete | 0008G1; Model 0019E/0023F; Compiler 0005D | Implemented the exact static one-/two-output attention subset through schema-57 direct generated execution: optional right-broadcast BOOL mask, top-left causal eligibility, frozen stable normalization, exact per-range score/weight workspace, atomic partition-DAG boundaries, and publication. All 11,880 specializations passed exhaustive structural inspection; the normalized equivalence map supports the 992-row five-fork performance matrix, whose ratios passed; documentation/Javadoc validation passed; and the final CPU checkpoint passed 127 suites/628 tests with 0 failures/errors and 7 expected skips. |
| 0008I | [Portable loss-family execution](tasks/0008i-portable-loss-family-execution.md) | Complete | 0008H; Model 0022–0022B; Compiler 0005D | Added schema-58 scalar loss execution and passed semantic plus complete 792-Class-File structural evidence. One old-protocol full fork failed 2/792 rows, and the corrected 792-class by five-fork gate was waived/closed by project decision rather than passed; that missing evidence transfers explicitly to CPU 0009. |
| 0008J | [BFLOAT16 scalar pointwise closure](tasks/0008j-bfloat16-scalar-pointwise-closure.md) | Complete | 0008I; current Model pointwise contracts | Closed exactly 44 current BFLOAT16 pointwise forms through scalar/caller-parallel scalar arrays, segments, mixed carriers, and all five current access regimes. Raw BFLOAT16 virtual locals preserve one encode boundary per producing logical node; WHERE preserves selected bits and predicates emit canonical BOOL. Schema-59, semantic/Class-File, 132-suite/692-test CPU, and five-fork representative `<= 1.15x` evidence passed. No SIMD, CAST, mixed promotion, materialization policy, native route, or fallback was added. |
| 0008K | [Cross-type CAST execution](tasks/0008k-cross-type-cast-execution.md) | Complete | 0008J; completed [Model 0025L](../../modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md) | Implemented all 36 Model-defined F64/F32/BF16/I64/I32/BOOL pairs through generated scalar/caller-parallel scalar arrays, segments, and mixed carriers over contiguous/offset-dense, positive-strided, and rank-zero layouts. Exact semantics, 576-class structural evidence, 405-sample five-fork performance evidence, and the post-correction 708-test CPU suite passed at schema 60. Cross-type CAST remains scalar-only; negative storage strides, SIMD CAST, native routes, and automatic materialization remain excluded. |
| 0008L | Pointwise SIMD mask/output closure | Draft | 0008K; Java 26 Vector API mask support | Close FLOAT32/FLOAT64 dense-contiguous vector and parallel-vector predicate output: materialize canonical BOOL bytes and reload them as `VectorMask` for WHERE on arrays and segments, with scalar tails. Retain virtual masks. FLOOR/CEIL and SIGMOID/SILU/GELU/general POW SIMD remain excluded. |
| 0008M | Vector MSE `NONE` | Draft | 0008L; 0008I | Add FLOAT32/FLOAT64 contiguous array/segment vector and parallel-vector MSE `NONE` with scalar tail. SUM/MEAN remain scalar because a horizontal vector reduction changes the defined accumulation order. |
| 0008N | Conv FLOAT32/FLOAT64 SIMD accumulation | Draft | 0008M; benchmark axis spike | Vectorize only the measured profitable Conv2d/Conv3d inner accumulation axis for FLOAT32/FLOAT64, retaining scalar tails and existing output-cell parallelism. Conv1d remains the existing Conv2d composition rather than gaining an emitter; BFLOAT16 is deferred. |
| 0008O | Stable-reduction vector numerical spike | Draft | 0008N | Establish a CPU-local numerical eligibility contract and benchmark evidence before any SIMD inner loop for softmax, categorical loss, or attention. It must preserve existing Model semantics and stop for a Model decision if those semantics are insufficient. |
| 0008P | Deterministic partial-reduction parallelism | Draft | 0008O | Add the separately scoped architecture of partial IR body, per-worker workspace, parallel partial ranges, deterministic combine kernel, and final publication for selected large reductions; do not treat it as an ordinary emitter extension. |
| 0009 | Portable generated-coverage closure checkpoint | Draft | 0001–0008P, explicitly including 0005A–0005J, corrective 0007A0A–0007A1O and any later residual corrections selected before 0007A2, 0007A–0007F2 including 0007A1/0007A2 and 0007F/0007F1/0007F2, Prepare 0003A, CPU 0008E1, CPU 0008G1, and the ordered 0008A–0008P sequence; complete current selected Model semantic inventory | Prove the bytecode/Vector portable route is the truthful supported semantic baseline and fallback, including completed per-family generated/direct corrective evidence, dimensional-convolution execution before general DAG work, safe general DAG decomposition, bounded recognition/fusion, deterministic split fallback, bounded single/dual-input materialization with representation reuse, shared partition-DAG adoption without reconstruction drift, and typed cold decision evidence. Inventory current generated families/forms and current Class-File hashes; map each current form to retained generated-versus-optimal-clean-Java performance evidence; reuse retained evidence only when the timed generated Class-Files/forms remain byte-identical and the protocol/scope applies; and classify evidence as current, representative-only, stale, or missing. CPU 0008I's corrected full 792-class by five-fork evidence enters this inventory explicitly as missing/deferred, not passed. Rerun fixed five-fork evidence only for missing, stale, or insufficiently representative hot paths, not indiscriminately for every historical task. Require exhaustive matrices only where specialization materially changes generated code and the bounded inventory justifies them; otherwise require a justified representative matrix covering types, carriers, loop shapes, and algorithm branches. Do not make a whole-backend performance-parity claim until these gaps pass. Detailed 0009 planning follows only after the ordered 0008I–0008P sequence completes. Classify metadata-only work, prove unsupported work fails closed, and close capability/conformance before native peer-route expansion. |
| 0010 | Narrow OpenBLAS BLAS-compatible native route | Draft | 0005A; 0009; completed OpenBLAS provider | Add only `route.nativeblas.openblas` for eligible BLAS-compatible linear algebra, preserving portable alternatives and using shared lowering, representations, exact filtering, materialization accounting, and whole-plan transition cost; never treat OpenBLAS as universal or preferred. |
| 0011 | Intel oneMKL BLAS and VML peer routes | Draft | 0005A; 0009; concrete Intel CPU use case and supported oneMKL ABI evidence | Add distinct `route.nativeblas.mkl` BLAS and `route.nativeops.mkl` VML leaves over shared analysis, without duplicating graph interpretation, fusion, access planning, or lifecycle ownership. |
| 0012 | Intel oneDNN partition peer routes | Draft | 0005A; 0009; stable common CPU lowering; concrete DNN/ML use case and supported oneDNN ABI evidence | Add `route.nativeops.onednn` as a distinct eligible partition route over common lowering/IR and whole-plan cost, without collapsing it into oneMKL or portable code generation. |
| 0013 | Apple Accelerate peer routes | Draft | 0005A; 0009; concrete Apple CPU use case and supported Accelerate ABI evidence | Add `route.nativeblas.accelerate` for BLAS and `route.nativeops.accelerate` for vDSP/vForce over shared analysis; Apple Silicon is capability-selected, while MPSGraph and Metal kernels remain outside CPU. |
| 0014 | AMD AOCL-BLAS and AOCL-LibM peer routes | Draft | 0005A; 0009; concrete AMD CPU use case and supported AOCL ABI evidence | Add distinct `route.nativeblas.aocl` and `route.nativeops.aocl` leaves over shared analysis and whole-plan cost, preserving the portable fallback and avoiding provider-owned lowering. |
| 0015 | Optional AMD ZenDNN partition peer routes | Draft | 0014; 0005A; 0009; stable common CPU lowering; concrete ZenDNN use case and integration evidence | Add `route.nativeops.zendnn` only for verified eligible DNN partitions, distinct from AOCL and portable generation and without another backend identity. |
| 0016 | Compatible CPU workload-tuning-cache selection | Draft | 0004; 0010–0015 as implemented; deferred Prepare opaque-candidate handoff; stable tuning-artifact compatibility | Reuse only compatible persistent selected-route evidence while retaining exact filtering and safe heuristic fallback; keep selected-route evidence distinct from the generated-class artifact store and add no measurement or tuning-cache mutation to CPU prepare. |
| 0017 | Explicit relaxed numerical candidate consumption | Draft | Config 0006; 0005F; stable exact portable and implemented peer-route consumers | Admit and compare relaxed portable or vendor candidates only under explicit caller permission, keep common analysis authoritative for eligibility and selected realization plans, and include numerical mode in compatibility/manifests without hot-path policy lookup. |

## Post-0008I execution strategy

This is an implementation-order and evidence plan, not an architecture change. The established
portable bytecode generator remains the CPU baseline; it is not broadly rewritten into a pure-Java
backend. Generated code remains the preferred leaf realization for bounded fused pointwise chains,
dense MATMUL/Conv inner loops, and the selected MSE `NONE` vector loop. Java continues to own
lowering and preparation, cold binding, access/materialization decisions, worker orchestration,
reference kernels, and irregular or memory-bound family orchestration.

Existing generated emitters are retained. In particular, current reductions already have
geometry-aware generated forms and must not be stopped or rewritten as part of this sequence. A
future migration of one family to Java requires an isolated, same-algorithm optimized-Java versus
generated spike; the exact-arithmetic/BigInteger reference oracle is not a production candidate.
The evidence must cover steady-state performance as well as preparation/class-cache and memory
cost before removing an emitter.

Do not add generic SIMD for movement, gather, scatter, or pooling without family-specific
benchmark evidence. Do not add FP16 before Model 0026, or automatic recognition of decomposed
loss/softmax graphs. These exclusions do not remove any existing realization.


## Milestones

- CPU representation, binding, generator, and durable artifact-store foundation
- Portable generated family coverage and closure
- Optional native routes and complete candidate selection
- Tuning-cache integration and conformance

## Current status

CPU 0005A is Complete. It atomically replaced the provisional work from CPU 0001–0005, whose
task records are now Superseded but remain preserved as historical evidence. The current module
has exactly one detailed implemented partition-kernel slice and one route leaf,
`internal.route.portable`. Detailed CPU 0005B and detailed
[CPU 0005C Vector and parallel portable strategies](tasks/0005c-vector-and-parallel-portable-strategies.md)
are Complete. Detailed
[CPU 0005D Materialization, specialization, and persistence evidence gate](tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md)
is Complete; detailed
[CPU 0005E Portable pointwise types, carriers, and semantic-family expansion](tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
is Complete. Detailed
[CPU 0005F Floating division and exact scalar-power realization](tasks/0005f-floating-division-and-exact-scalar-power-realization.md)
is Complete. Detailed
[CPU 0005G Extrema, clamp, Tensor power, and logical coverage](tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
is Complete. Detailed
[CPU 0005H Portable unary, transcendental, and activation closure](tasks/0005h-portable-unary-transcendental-and-activation-closure.md)
is Complete. Detailed
[CPU 0005I FLOAT32 vector parity and vector-emission boundary](tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)
is Complete. Detailed
[CPU 0005J Bounded pointwise coverage and parity hardening](tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md)
is Complete. Detailed
[CPU 0006 Portable static affine views and boundary materialization](tasks/0006-portable-static-affine-views-and-boundary-materialization.md)
is Complete. The fresh operation inventory first split the former broad row into this static
resolved-layout affine slice and later non-affine/index, functional scatter/fold, ordering, and
explicit-state random work. Planning for the next frontier found that the remaining
non-affine/index row still combined three distinct dependencies. Detailed
[CPU 0006A Portable pad, tile, and tensor-composition movement](tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md)
is now `Complete`; detailed
[CPU 0006A1 Portable static window extraction](tasks/0006a1-portable-static-window-extraction.md)
is `Complete`, and detailed
[CPU 0006A2 Portable gather and one-hot indexing](tasks/0006a2-portable-gather-and-one-hot-indexing.md)
is `Complete`. It delivers Gather/one-hot index loading plus complete pre-write execution-time
validation. Detailed
[CPU 0006B Portable functional slice update](tasks/0006b-portable-functional-slice-update.md)
is `Complete`. Detailed
[CPU 0006B1 Portable functional scatter](tasks/0006b1-portable-functional-scatter.md) is
`Complete`. Detailed
[CPU 0006B2 Portable overlap fold](tasks/0006b2-portable-overlap-fold.md) is `Complete`. Detailed
[CPU 0006C Portable stable ordering and selection](tasks/0006c-portable-stable-ordering-and-selection.md)
is `Complete` and depends on CPU 0006B2. Detailed
[CPU 0006D Portable explicit-state RNG and dropout coverage](tasks/0006d-portable-explicit-state-rng-and-dropout.md)
is `Complete`. Detailed
[CPU 0007 Portable cumulative scan coverage](tasks/0007-portable-cumulative-scan-coverage.md)
is `Complete`. Detailed
[CPU 0007A Portable ordinary extrema and boolean reductions](tasks/0007a-portable-ordinary-extrema-and-boolean-reductions.md)
is `Complete`. Detailed
[CPU 0007A0 Generated hot-path parity correction](tasks/0007a0-generated-hot-path-parity-correction.md)
is `Complete`. Detailed
[CPU 0007A0A Affine and movement generated-loop parity](tasks/0007a0a-affine-and-movement-generated-loop-parity.md)
is `Complete`. Detailed
[CPU 0007A0B Indexing generated-loop parity](tasks/0007a0b-indexing-generated-loop-parity.md)
is `Complete`. Detailed
[CPU 0007A0C Scatter generated-loop parity](tasks/0007a0c-scatter-generated-loop-parity.md)
is `Complete`. Detailed
[CPU 0007A0D Fold generated-loop parity](tasks/0007a0d-fold-generated-loop-parity.md) is
`Complete`. Detailed CPU 0007A0E and CPU 0007A0F are `Complete`. Detailed
[CPU 0007A1](tasks/0007a1-portable-ordinary-numerical-aggregate-reductions.md) is `Complete`
after its Java, Class-File, performance, and independent documentation gates. The approved
schema-29 generated-code audit inserts detailed
[CPU 0007A1A Generated scalar-body self-containment](tasks/0007a1a-generated-scalar-body-self-containment.md)
is `Complete`. Detailed
[CPU 0007A1B Scatter algorithmic parity](tasks/0007a1b-scatter-algorithmic-parity.md) is
`Complete`. Detailed CPU 0007A1C and CPU 0007A1O are `Complete`: A1O preserved the immutable
original ledger, added the versioned tested replacement, supplied the pointwise equivalence and
five-fork evidence, and closed A1C's final original gate. Detailed CPU
0007A1D also remains
`Review needed`: schema 32 and its Java/semantic/Class-File gates are stable, but all 13 required
performance targets failed the final fork. Detailed CPU 0007A1E through CPU 0007A1N are
`Complete`. Detailed
[CPU 0007A2 Portable binding-aware sum-to-Shape reduction](tasks/0007a2-portable-binding-aware-sum-to-shape-reduction.md)
is `Complete`. Detailed
[CPU 0007B Portable arg-extrema coverage](tasks/0007b-portable-arg-extrema-coverage.md) is
`Complete`; detailed [CPU 0007C Portable masked reduction coverage](tasks/0007c-portable-masked-reduction-coverage.md)
is `Complete`. Detailed [CPU 0007D Portable logarithmic, statistical, and norm reduction coverage](tasks/0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md)
is `Complete`. Detailed
[CPU 0007E Portable stable softmax and log-softmax coverage](tasks/0007e-portable-stable-softmax-and-log-softmax-coverage.md)
is `Complete`; detailed
[CPU 0007F Portable layer and RMS normalization coverage](tasks/0007f-portable-layer-and-rms-normalization-coverage.md)
is `Complete`. Detailed
[CPU 0007F1 Portable batch-normalization inference coverage](tasks/0007f1-portable-batch-normalization-inference-coverage.md)
is `Complete`; detailed
[CPU 0007F2 Portable batch-normalization training and statistic-transition coverage](tasks/0007f2-portable-batch-normalization-training-and-statistic-transition-coverage.md)
is `Complete`. Detailed CPU 0008 is `Complete` and establishes only the grouped
NCHW Conv2d foundation plus its bounded family-local epilogue/split contract. MATMUL is implemented
by detailed Complete [CPU 0008F](tasks/0008f-portable-matmul-execution-and-bounded-linear-epilogues.md) following Complete
[CPU 0008E1](tasks/0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md), which adopted the shared partition DAG. Detailed
[CPU 0008G](tasks/0008g-portable-max-average-pool2d-execution.md) is `Complete`; detailed
[CPU 0008H](tasks/0008h-portable-scaled-dot-product-attention-execution.md) is `Complete`; CPU
0008I is `Complete` with its corrected full performance gate explicitly waived/closed by project
decision rather than passed. Detailed CPU 0008J, Model 0025L, and CPU 0008K are `Complete`; CPU
0008L is the next `Draft` CPU frontier.
CPU 0008A is `Complete`: it
validates Conv1d through the explicit Conv2d
composition and adds Conv3d execution before the general DAG. CPU 0008B, CPU 0008C, and CPU 0008D
are `Complete`; detailed [CPU 0008E](tasks/0008e-bounded-multi-input-materialization-and-representation-reuse.md)
is `Complete`. It closes the profitability and materialization sequence in CPU
0008B–0008E. The existing diff corrected affine-copy generated/oracle parity and retains the
same-instruction pair failure as `CO_CONSUMED_PAIR`. An honest production-selectable FLOAT64 heap
one-copy row then passed generated/oracle at `1.000326680` but failed complete-plan
selected/direct at `2.605250934`. The task is Complete with every materialized form retained as
a complete executable candidate but ordinary preparation selecting CPU 0008D's direct topology
with a typed unproved-materialization reason. Cross-unit reuse remains candidate metadata rather
than promotion proof; future opaque Prepare/tuning/cache owners may promote compatible plans from
end-to-end evidence. This is a private selection correction, not a timing constant, blacklist,
new generated realization, route, or architecture change. The first full corrective CPU rerun
executed 544 tests with 3 skips and 9 failures in exactly seven stale selected-copy test owners.
That run is discovery evidence, not success. The cohesive task authorized only those seven
owners to obtain retained candidates explicitly and use existing private finalization/execution
machinery where required; implementation must stop if another production owner is necessary.
Those test edits are complete: the focused matrix passed 115 tests and the final full CPU rerun
passed 544 tests across 104 suites with 3 expected skips and no failures/errors. Final clean
documentation context `01a04317-b784-76e3-a93b-ff35106284b9` corrected the stale CPU guide and
three named glossary entries and finalized Javadocs. The stopped documentation
context `01a04303-9552-72a0-a5dc-c95d20413779` made no edits because the prior task scope did not
authorize the guide. Replacement documentation context `01a0430b-de37-7de3-bfa1-9b21f2cd21d3`
also made no edits because its audit demonstrated that the current CPU portable preparation plan,
CPU contiguous materialization plan, and Materialization glossary entries were stale while the
task prohibited glossary edits. The corrected maximum is 43 paths (14 production/Javadoc, 24
test/evidence, and 5 documentation/planning), with exactly 36 actual paths after the guide and
glossary were added. User-authorized Prepare 0003A, detailed CPU 0008E1, and CPU 0008F are
`Complete`. Detailed CPU 0008G, CPU 0008G1, CPU 0008H, and CPU 0008I are `Complete`; detailed CPU
0008J and Model 0025L are `Complete`; detailed CPU 0008K is `Complete`; CPU 0008L is the next `Draft` frontier.
The inserted pooling order is Model 0025I -> Model 0025J -> Model 0025K -> Compiler
0006B1 -> Compiler 0006B2 -> CPU 0008G1 -> CPU 0008H. The existing order through Model 0025G,
Model 0025H, Compiler 0006B, and CPU 0008–0008G remains unchanged; Compiler 0006C remains a
separate gradient-closure task that does not block CPU forward execution. CPU
0006D is one bounded
one-node family task because
INITIAL_STATE and DROPOUT share the same explicit-state execution, generated-emitter,
multi-output-binding, replay, and schema boundary.

Prepare 0003A completion changes no CPU production, topology identity, generated code, schema 53,
fusion/materialization policy, performance evidence, or Runtime behavior. Its sole CPU source
change relocates one malformed duplicate-producer test expectation to the shared Prepare
construction boundary. The implementation-owned CPU run passed 104 suites and 544 tests with 3
expected skips and no failures or errors; the one required shared checkpoint reported the same
CPU counts. Clean documentation context `01a043d7-113c-7ee2-8257-42678c1a7be4` finalized the
shared public contract and evidence. CPU consumption and reconstruction removal are now detailed
Complete CPU 0008E1, detailed Complete CPU 0008F, detailed Complete CPU 0008G, and Complete
[CPU 0008G1](tasks/0008g1-portable-pool1d-composition-validation-and-pool3d-generated-execution.md)
close the pooling execution frontier; detailed
[CPU 0008H](tasks/0008h-portable-scaled-dot-product-attention-execution.md) is `Complete`; CPU
0008I is `Complete` with the recorded performance validation exception, detailed CPU 0008J is
`Complete`; Model 0025L and detailed CPU 0008K are `Complete`; CPU 0008L is the next `Draft`
implementation frontier.

CPU 0008G1 retains the visible three-occurrence Pool1d graph while recognizing only the exact
private singleton-height topology for schema-55 Pool2d reuse. Its first-class Pool3d implementation
adds schema 56 and 24 emitted kind/type/carrier families with direct depth-height-width scalar
loops, array/segment carriers, caller-owned output-cell ranges, and zero workspace/materialization.
Implementation context evidence passed 20 focused suites/181 tests and the final 120-suite/618-test
CPU checkpoint with five intentional opt-in skips and no failures or errors. The accepted fresh
five-fork root `/tmp/synaptik-cpu-0008g1-pool3d-evidence-20260830c` has manifest digest
`80457c6c181e0ae80e6c33c264ddbc02202c79bd3f67e76cf017172d540cf85c`; the worst individual
generated/direct ratio was `1.002680589x`, and all six aggregate ratios passed `<= 1.15x`.
Two earlier development roots failed FLOAT32 maximum before the direct oracle was corrected and
are retained only as pre-fix failure evidence. The independent documentation pass finalized the
CPU guide, Tensor/Compile API boundaries, glossary, affected Javadocs, and planning/status records
without changing executable behavior or repeating the successful Java suites.

CPU 0008F completes the static portable MATMUL family at schema 54. Its four bounded generated
realizations cover all thirteen ordered non-BOOL numeric promotions, vector/matrix/batched and
right-aligned broadcast geometry, full-K accumulation, disjoint output work units, exact 0008C
bias/one-terminal fusion or canonical split, and direct plus candidate-only one-input
materialization forms. The implementation context passed the 17-suite/182-test focused baseline,
the sole 109-suite/576-test CPU checkpoint, CPU Javadoc, complete Class-File/forbidden-reference
inspection, and five immutable generated/direct performance forks. All kernel samples and
aggregate medians passed `<= 1.15x`; the worst individual ratio was `1.002573`. Attempt-16's
212-file manifest verifies with digest
`a88806a9118c1a967ecc77eaf6da9582d15afa959e8368348a0d2ba2a47d4b61`. The two retained
materialization companions pass their bounded measurements but do not promote automatic policy;
ordinary preparation remains direct. Clean documentation context `/root` finalized Javadocs, the
CPU guide, the existing MATMUL glossary entry, and planning/status evidence without executable
Java changes. No public/shared API, dependency, architecture, Gradle, conformance, integration,
or native-route boundary changed.

CPU 0007A1A is Complete at schema 30. Its retained 28-class audit covers ten scalar activation,
four vector activation, two BFLOAT16 scan, twelve MIN/MAX/ALL/ANY aggregate artifacts, and their
scalar tails. Scalar/scan/aggregate artifacts have no Synaptik-owned calls; vector artifacts
reference only typed `CpuVectorMath` chunk methods; and none contains a method handle,
`invokedynamic`, dynamic constant, or bootstrap method. This is not an all-generated-kernel claim.
The fixed 24-case direct-primitive-Java comparison passed all 120 fork results and 24 aggregate
ratios at `<= 1.15x`; the maximum fork ratio was `1.117147x` and maximum aggregate ratio was
`1.111438x`, both BOOL ANY. The focused command passed 64 tests, and the authoritative CPU suite
passed 340 cases with one expected assumption skip. Clean documentation context
`01a01b11-b773-7e33-b4d1-94fa62aeeb2b` verified the read-only evidence checksums, finalized
Javadocs/guide/planning, and reused the stable Java evidence. Schema-29 artifacts are current-only
safe misses. Detailed CPU 0007A1B is Complete at schema 31. Its five scratch-free forms use
range-owned copy-then-update, while floating exact MUL retains output-owned grouping and unchanged
per-range scratch. All 30 fork ratios and all six aggregate ratios passed `<= 1.15`; scratch-free
generated medians improved by `1795.9x`, `1966.0x`, `1925.0x`, `1388.1x`, and `6652.7x` over the
preserved schema-30 classes, and exact MUL improved by `2.09x`. The focused seven-owner matrix and
53-suite/343-test CPU run passed with one existing opt-in skip. Documentation context
`01a01b5c-abf0-7f81-902f-e6d47d585503` finalized schema-31 Javadocs, guide, evidence, and planning.
CPU 0007A1C froze and retained its 20-row corpus. Exact semantics passed all 20 rows, but fork 1
failed 17 `<= 1.15x` gates; only `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` passed.
The explicit more-than-two-code-shaping-owners stop fired before production edits, forks two
through five, aggregate ratios, exhaustive member reports, Java suites, or Javadoc. At that A1C
stop, source remained schema 31, while malformed `schema-after.txt` was not accepted as evidence. CPU
0007A1C was therefore `Review needed` at that checkpoint. CPU 0007A1D retains its schema-32 invocation-local segment-
layout implementation but is `Review needed` and Incomplete: all 13 target rows failed fork 1,
while all three controls passed; forks two through five and an aggregate do not exist. A
diagnostic version with native-order/with-order construction removed entirely still left ten
targets above `1.15x`. CPU 0007A1E is Complete at schema 33: frozen semantics passed all 20 rows,
and its four movement targets plus three controls passed every one of five forks at `<= 1.15x`.
The unchanged full probe still exited nonzero because deferred rows failed, so A1C remained open
at that checkpoint.
CPU 0007A1F is Complete at schema 34: frozen semantics remained `VERIFIED,20`, both targets and
three controls passed all five forks, and the full probe remained nonzero only for deferred rows.
CPU 0007A1G is Complete at schema 35: frozen semantics remained `VERIFIED,20`, its fold/dropout
targets and three controls passed all five forks, and the full probe stayed nonzero only for nine
or ten deferred diagnostic rows. CPU 0007A1H is Complete at schema 36: exact semantics remained
`VERIFIED,20`; its guarded FLOAT32 axis-one MEAN and BFLOAT16 axes-zero/two PROD targets plus all
three controls passed every fork and median at `<= 1.15x`; and the full probe stayed nonzero only
for seven explicitly deferred rows. CPU 0007A1I is Complete at schema 37: both indexing targets
and all controls passed five forks, leaving exactly five persistent rows. CPU 0007A1J is Complete
at schema 38: `C-SCAN-GENERAL` and all controls passed every accepted fork and median, leaving
four rows. CPU 0007A1K is Complete at schema 39: `A-GENERAL` and all controls passed every
accepted fork and median. CPU 0007A1L is Complete at schema 40: `P-SCALAR-GENERAL` and all
controls passed every accepted fork and median. CPU 0007A1M is Complete at schema 41: frozen
semantics remain `VERIFIED,20`; `S-GENERAL-MIN` passed five accepted forks at `0.984900063x`,
`0.988888234x`, `0.983823803x`, `0.978065816x`, and `0.992400680x`, with median
`0.984900063x`, and all controls passed. CPU 0007A1N is Complete at schema 42: frozen semantics
remain `VERIFIED,20`; all twenty rows and their medians passed five accepted forks at `<= 1.15x`;
`X-MIN-MULTI` measured median `0.811182115x`; and one unrelated `M-CONCAT` whole sample was
retained and rejected. CPU 0007A1O subsequently closes the ledger gap with a tested versioned
replacement and fresh evidence for all 40 formerly structural-only pointwise rows. CPU 0007A1C
and CPU 0007A1O are Complete. CPU 0007A2, detailed CPU 0007B, and detailed CPU 0007C are
`Complete`. Detailed CPU 0007D, CPU 0007E, and CPU 0007F are `Complete`. Detailed CPU 0007F1 is
`Complete`; detailed CPU 0007F2 is `Complete`, while later work remains Draft without detailed specifications.

CPU 0007B is Complete at schema 44. Its six deterministic, constructor-free generated classes
cover ARG_MIN and ARG_MAX with FIRST and LAST ties across the frozen corpus. Retained exact
semantics report `VERIFIED,6`; the final CPU XML reports 384 tests with zero failures or errors
and one expected skip. All 30 individual fork ratios and all six aggregate ratios passed
`<= 1.15x`, with maximum observed fork ratio `1.038305072x`. The 40-path final scope contains
exactly three new CPU-private production types, leaves the existing aggregate owners unchanged,
and adds no public API, architecture, dependency, build, conformance, or integration change.

CPU 0007C is Complete at schema 45. Its six retained FLOAT64/FLOAT32/BFLOAT16 SUM/MEAN classes
prove directional right-aligned mask mapping, mask-before-data exclusion, exact selected counts,
three buffers, per-range exact-state workspace, and complete output-cell scalar/parallel-scalar
ownership without materialization or partial/combine state. The final uncached CPU suite reports
398 tests, one expected skip, and no failures or errors. All five isolated forks passed all six
generated-versus-optimal-direct-Java cases at `<= 1.15x`; the worst ratio was `0.980505473x`.
Retained schema-42 ledger evidence remains explicitly historical. The final 42-path scope contains
exactly three new CPU-private production types and changes no public/shared/build/architecture/
conformance/integration boundary.

CPU 0007D is Complete at schema 46. Its fifteen retained generated classes cover
`LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` across FLOAT64,
FLOAT32, and BFLOAT16, plus three unchanged-family controls. The final uncached CPU suite reports
416 tests, one expected opt-in persistence skip, and no failures or errors. All targets and
controls passed five isolated optimal-direct-Java forks at `<= 1.15x`; the worst fork ratio was
`1.099406784x` and worst aggregate ratio was `1.087934030x`, both FLOAT64 log-sum-exp. Complete
Class-File and member inspection found no forbidden generated member. The final 45-path scope has
27 production/Javadoc paths, 13 test paths, the five named documentation/planning paths, and
exactly six new CPU-private production types. It changes no public/shared/build/architecture/
conformance/integration boundary. CPU 0007A1D remains historical Review needed.

CPU 0007E is Complete at schema 47. It executes only explicit first-class `SOFTMAX` and
`LOG_SOFTMAX` over fully static resolved-layout FLOAT64/FLOAT32/BFLOAT16 inputs within the
CPU-private finite, positive-selected-width subset. Heap, native-order segment, and mixed carriers
support arbitrary legal layouts; complete-slice scalar or parallel-scalar ranges require zero
workspace. Validation completes before output mutation or worker submission, rejects all overlap,
and treats a non-selected zero extent as an empty no-op. SOFTMAX uses maximum, compensated shifted-
exponential sum, and exponential/division passes without `Math.log`; LOG_SOFTMAX computes its one
required logarithm per slice. The final uncached CPU suite passed 70 suites/428 tests with one
expected skip and no failures or errors, and CPU Javadoc passed with two incubator warnings. The
eight targets and three controls passed all 55 isolated-fork and 11 aggregate `<= 1.15x` gates;
the worst fork was `1.119471916x` and the worst aggregate was `1.113919290x`, both BFLOAT16
LOG_SOFTMAX heap. The 163-file retained bundle verifies all 162 manifest entries and contains the
70 XML reports. Complete generated-member inspection found no forbidden hit. The final 47-path
scope has 29 production/Javadoc paths, 13 test paths, five documentation/planning paths, and
exactly five new CPU-private production types. It changes no public/shared/build/architecture/
conformance/integration boundary. CPU 0007A1D remains Review needed; detailed CPU 0007F is
Complete. Detailed CPU 0007F1 is Complete; detailed CPU 0007F2 is Complete, while later work remains
Draft without detailed specifications.

CPU 0007F is Complete at schema 48. It executes the four explicit first-class Layer/RMS forms for
fully static resolved-layout BFLOAT16/FLOAT32/FLOAT64 operands with ordered promotion and exact
typed epsilon. Complete leading-slice scalar/parallel-scalar ranges are deterministic; Layer uses
three passes and one existing exact-state scratch slice per simultaneous range, while RMS uses a
scaled-square two-pass body and zero workspace. Static typed native-order segment layouts remove
invocation-local layout construction. The final uncached CPU suite passed 76 suites/449 tests and
the focused post-fix run passed 10 suites/119 tests, both with zero failures or errors. All 85
per-fork and 17 aggregate ratios passed `<= 1.15x`; worst fork was `1.136183168x` and aggregate
worst was `1.113266704x`, both BFLOAT16 RMS. All 17 generated classes passed complete Class-File
and member scans; the 241-file retained manifest verifies with digest
`a7ca999336d73dbf2fee3d2414ff31f0339a5048ec7e2b5ab804b1c5186829c9`. The final scope has 35
paths: 16 production/Javadoc, 14 tests, and exactly five documentation/planning paths, with exactly
five new CPU-private production types. No public/shared/build/architecture/conformance/integration
boundary changes. CPU 0007A1D remains Review needed; detailed CPU 0007F1 is Complete; detailed CPU
0007F2 is Complete, while all later work remains Draft without detailed specifications.

CPU 0007F1 is Complete at schema 49. It executes one explicit five-input/one-output
`BATCH_NORM_INFERENCE` occurrence over fully static resolved layouts with arbitrary channel axis,
ordered BFLOAT16/FLOAT32/FLOAT64 promotion, exact typed epsilon, direct running variance, and
zero workspace/materialization. Preparation selects deterministic channel or flattened
non-channel ranges; generated entries use range-entry decode, odometer/incremental addressing,
and FLOAT32 or FLOAT64 locals at the required computation boundary. Focused validation passed 112
tests; the final uncached CPU XML reports 462 tests, zero failures, zero errors, and two expected
skips. All five accepted forks and medians passed the frozen `<= 1.15x` matrix; the tightest batch
fork was BN-MIX-F32 at `1.149753751x`. The 281-file retained bundle verifies against manifest
digest `185ecb1b1da84d20774b5f21979bbfc8cedb765cf03dc05610fa354bd7555029`. The final scope remains
within 46 paths and contains exactly 14 production paths, 13 tests, and four new CPU-private
production types. No public/shared/build/architecture/conformance/integration boundary changed.
CPU 0007A1D remains Review needed; CPU 0007F2 completion evidence follows.

CPU 0007F2 is Complete at schema 50. It executes exactly one explicit five-input/five-output
`BATCH_NORM_TRAINING` occurrence over fully static resolved layouts. Complete-channel scalar and
parallel-scalar ranges reuse one exact-state slice per active range, compute biased saved
statistics and a separately divided unbiased running-variance transition, and write all five
ordinary outputs only after complete cold carrier, layout, workspace, and overlap validation.
Focused validation passed 10 suites/114 tests with zero failures, errors, or skips. The final
uncached CPU suite from the implementation context passed 86 suites/472 tests with zero failures
or errors and two expected skips. Eight deterministic classes passed complete member and
operation-count scans; each has direct odometers, one square-root site, four semantic division
sites, and no forbidden helper, allocation, boxing, reflection, dispatch, or layout-construction
reference. Five accepted isolated Java 26.0.1 fixed-heap forks passed all 60 per-fork and 12
aggregate `<= 1.15x` gates; the worst fork was `1.123421082x` and worst aggregate was
`1.110678870x`, both `BNT-REPEAT`. One whole environment/classpath sample was rejected before
measurement and no ratio sample was discarded. The retained bundle verifies with digest
`d1138b75924cea2b1bbce6ba127213eb2181de7156884b857ce1e475b9b95edb`. The dirty worktree contains
47 paths: 25 production/Javadoc, 15 tests, five task documentation paths, and two preserved
unrelated planning paths; the bounded 0007F2 scope is 45 paths and exactly four new CPU-private
production types. No public/shared/build/architecture/conformance/integration boundary changed.
CPU 0007A1D remains Review needed; detailed CPU 0008, CPU 0008A, and CPU 0008B are Complete.
CPU 0008C, CPU 0008D, CPU 0008E, Prepare 0003A, and detailed CPU 0008E1 are Complete,
CPU 0008F is Complete, and later rows remain Draft. The
Conv1d/Conv3d-before-general-DAG
ordering correction remains preserved.

CPU 0007A1O implementation context `01a032f9-66c9-73f1-8960-8c39c97c830d`, initial audit
context `01a032f1-a956-7ca0-9a18-4ce3f585208b`, and the mandatory final documentation pass
verified evidence root `/private/tmp/synaptik-cpu-0007a1o-IPbQzuJi`. The original ledger retains
79 physical lines and the required SHA-256; the replacement contains 48 pointwise and 30
non-pointwise rows, exactly 40 fresh pointwise representatives, and only `INITIAL_STATE` as
structural-only. Exact semantics report `VERIFIED,44`; all 44 cases passed five fixed-heap forks
and medians at `<= 1.15x`, with worst fork `1.071655171x` and worst median `1.056510985x`, both
`P-IS_NAN`. All 44 classes are deterministic, four controls are byte-identical to A1N, complete
`javap -c/-v` is retained, the audit reports `AUDITED,44`, and the 384-entry manifest verifies.
The focused command passed 23 tests; final CPU XML has 55 suites and 361 tests, with 360 passes,
one expected skip, and no failures/errors. Schema remains 42 and production bytes are unchanged.
This accumulated A1C closure does not retroactively satisfy A1D's failed local performance gate.

CPU 0007A0 is a corrective insertion based on the completed local audit under
`/tmp/synaptik-bytecode-benchmark`. The audit measured the large dense generated scalar ADD at
3.63x direct Java, generated Vector API ADD at 1.84x, generated fused ADD -> GELU_EXACT -> MUL at
1.32x, bridge-only full MIN at 31.91x, and bridge-only CUM_SUM at 5.45x. The completed task
atomically corrected the shared generated loop/carrier boundary and current scan/aggregate
emitters, preserved all semantic/general forms, and advanced generated compatibility to schema
22. The final isolated five-fork probe under `/tmp/synaptik-bytecode-benchmark-0007a0` measured
generated/direct median-of-fork-medians ratios of 0.819 for scalar ADD, 0.852 for Vector ADD,
0.854 for fused ADD -> GELU_EXACT -> MUL, 1.001 for full MIN, and 0.998 for CUM_SUM on Java
26.0.1, macOS 26.5.2, aarch64, with 128-bit preferred `DoubleVector` species. It changes no
architecture, dependency, public API, route selection, or Runtime tuning boundary. Completed
tasks remain historically Complete; the correction supersedes only their former performance
implementation assumptions.

The post-0007A0 audit extends that evidence boundary across every remaining generated family.
`CpuAffineCopyEmitter` and `CpuDataMovementEmitter` already contained generated typed loops, and
CPU 0007A0A has now applied the same cold-proved integer loop/address boundary to their dense
heap-array forms. The generated entry remains `long start, long end`; narrowing and invariant
geometry loading occur once before the hot loop. Typed general-long forms remain for segments,
mixed carriers, arbitrary layouts, and unproved ranges. Independent review corrected TILE so
each source coordinate wraps by its own input extent and outer source axes advance only on the
matching output-axis carry.
CPU 0007A0F completed the final bridge-only random/dropout correction. The ordering and random
corrections stay separate because their algorithms and
resource contracts require different fair baselines:

| Task | Current generated-bytecode/source evidence | Required independent parity cases |
|---|---|---|
| 0007A0A | Complete: schema-23 dense integer affine/movement bodies, corrected TILE carry/wrap semantics, and preserved typed general-long fallbacks. | FLOAT64 CONTIGUOUS, FLOAT32 TILE, INT32 SLICE_UPDATE all passed. |
| 0007A0B | Completed schema-24 classes embed typed GATHER/GATHER_ELEMENTS/GATHER_ND/ONE_HOT bodies. Proved dense heap arrays use integer loop/address state; general layouts and segment or mixed carriers retain typed long-address forms without an `Object` bridge. | FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT passed at `1.084234x` and `0.895060x`. |
| 0007A0C | Complete: schema-25 classes embed typed SCATTER_ELEMENTS, Gather-compatible SCATTER_ADD, and SCATTER_ND output/contribution/reduction loops plus exact-product scratch state. Dense heap forms use integer state; general forms retain typed long addressing without an `Object` bridge. | Unique SCATTER_ELEMENTS replacement and duplicate-index FLOAT32 SCATTER_ADD passed at `0.979533x` and `0.983230x`. |
| 0007A0D | Complete: schema-26 classes embed typed FOLD_AXIS/FOLD2D mapping and sequential-addition bodies. Dense heap arrays use integer state; general layouts, segments, and mixed carriers retain typed long addressing without an `Object` bridge. | Fixed overlapping `[127,16] -> [1024]`, window 16, step 8, dense FLOAT32 FOLD_AXIS passed at `0.926451x`; this is the only performance-parity claim. |
| 0007A0E | Complete: schema-27 classes embed typed stable merge, comparison, selected-pair ordering, and represented output stores. Dense heap forms use integer state; general layouts, segments, and mixed carriers retain typed long addressing without an `Object` bridge. | Stable dense FLOAT32 SORT and two-output TOP_K passed at `1.090165540x` and `0.965951020x`. |
| 0007A0F | Complete: schema-28 classes embed typed initializer/dropout state, mapping, threshold, represented value, canonical mask, and loop bodies. Dense heap forms use integer state; arbitrary layouts, segments, and mixed carriers retain typed long addressing without an `Object` bridge. | Dense FLOAT64 and FLOAT32 DROPOUT passed every fork and aggregate at `0.241451022x` and `1.058174348x`. |

Each case uses an isolated reproducible minimum five-fork probe with at least five warmup batches,
nine randomized measurement rounds, adaptive batches of at least 25 ms, fixed heap, deterministic
inputs, exact pre/post verification, and a per-case generated/direct median-of-fork-medians gate of
`<= 1.15x`. Timing remains outside JUnit and never selects production behavior. Stable automated
tests instead lock generated bytecode/dispatch shape. CPU 0007A0A's baseline ratios were
`4.581565x`, `1.928399x`, and `2.181383x` for affine, TILE, and SLICE_UPDATE. The corrected
five-fork ratios are `0.868672x`, `1.107360x`, and `1.131942x`; all pass `<= 1.15x` on Java
26.0.1, macOS 26.5.2 aarch64 under the retained fixed-heap randomized protocol. CPU 0007A0B's
accepted corrected ratios are `1.084234x` and `0.895060x`. CPU 0007A0C's accepted ratios are
`0.979533x` and `0.983230x`. CPU 0007A0D's five fork ratios are `0.929879x`, `0.923513x`,
`0.925827x`, `0.927054x`, and `0.926274x`, with aggregate `0.926451x`. CPU 0007A0E's accepted
SORT ratios are `1.092722590x`, `1.077727976x`, `1.090165540x`, `1.049603676x`, and
`1.089690637x`, aggregate `1.090165540x`; its TOP_K ratios are `0.962180009x`, `0.969660236x`,
`0.936075945x`, `1.018861174x`, and `1.020688132x`, aggregate `0.965951020x`. CPU 0007A0F's
FLOAT64 ratios are `0.242432568x`, `0.240238314x`, `0.245355311x`, `0.242999122x`, and
`0.241854555x`, aggregate `0.241451022x`; its FLOAT32 ratios are `1.046430134x`, `1.063729192x`,
`1.052403558x`, `1.060950642x`, and `1.053413773x`, aggregate `1.058174348x`. These observations
apply only to the fixed cases and recorded host/JVM, not to broader layouts, states, parallel
orchestration, machines, or a general speedup claim.

CPU 0007A0F implementation context `01a0194d-ff2e-78d3-9815-f0830065185b` and independent review
orchestration context `01a0196a-0930-7521-972f-84021b85519b` with isolated review context
`/root/cpu_0007a0f_review` produced the accepted schema-28 result. The focused five-owner command
passed 50 tests, and the sole authoritative CPU suite passed 53 suites/323 tests with zero
failures/errors and one expected skip. Review changed tests only, passed 4 direct-generator tests
and 15 combined generator/random tests, strengthened exact descriptors and all 32 carrier-pattern
member-reference gates, and regenerated all six representative classes byte-identically. Clean
documentation context `01a01973-5ecf-75b0-ae50-09aebdd3ddf7` reused those stabilized results,
finalized Javadocs, the CPU guide, glossary no-change evidence, and planning state, and changed no
executable Java or tests.

CPU 0007A0E implementation context `01a018ec-ea40-7f21-b2be-1ec07564d0b8` and independent
review/fix context `01a01905-b866-70f3-89c7-181adcabe2c5` produced the final schema-27 bytes and
passed the focused five-owner command plus the single final CPU suite of 320 tests, zero
failures/errors, and one expected skip. Clean documentation context
`01a01914-1a96-7cc3-9215-c69a7531f30e` reused the retained evidence under
`/private/tmp/synaptik-cpu-0007a0e-implementation.dMNFPD`, finalized the CPU guide and planning
records, confirmed that the glossary needed no new term, and passed CPU Javadoc, rendered-page,
Markdown, exact-scope/status/order, unstaged-index, and whitespace validation without rerunning
Java tests or timing.

Implementation context `019ffff6-4acb-75b3-9e4d-b71359d8a6ed` and corrective review context
`01a0000e-d32a-7fd3-aaa3-4c7d32b7f5af` produced the final schema-23 implementation and TILE
regressions. The corrected focused movement run passed 14 tests, and the authoritative CPU suite
passed 53 suites/309 tests with zero failures/errors and one existing expected skip. Clean
documentation context `01a00019-c0d3-7f42-8bc3-adb24b115aa6` changed no executable Java or test,
reused the stabilized Java/probe evidence, and finalized the affected Javadocs/package summaries,
guide, glossary, and planning records.

CPU 0007A0B planning context `01a0002e-0a9f-7133-ba6f-80d2a1515d78`, implementation context
`01a00039-8814-7e62-a0cd-45df202c6a13`, and clean documentation context
`01a00074-37e3-7c33-af1c-18233eae156e` produced the schema-24 indexing correction. The focused
matrix passed 11 suites/100 tests, the authoritative CPU suite passed 53 suites/310 tests with one
existing opt-in skip, and both retained baseline/final-pass checksum manifests verify. Dense
FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT passed the independent gates at `1.084234x` and
`0.895060x`; complete pre-write validation and general typed long-address forms remain intact.

CPU 0007A0C planning context `01a00085-1280-7bf1-be61-f6e7b1cf1e4f`, implementation contexts
`01a00096-947b-7490-9ba7-d70b6d61fd5b` and `01a000b0-48c0-7e91-adeb-4067d6949747`, audit
contexts `01a000bb-5342-7aa0-8cdf-3e0ca7f8570e` and
`01a000bb-9755-7092-8c85-977119b355e9`, and clean documentation context
`01a000c9-bb65-7d52-82ba-29ce0c26157a` produced the schema-25 scatter correction. The focused
matrix passed 8 suites/105 tests, and the authoritative CPU suite passed 53 suites/312 tests with
zero failures/errors and one existing skip. Retained five-fork evidence passed both dense cases at
`0.979533x` and `0.983230x`; general layouts and exact-product entries remain semantic/Class-File
evidence rather than universally timed claims.

CPU 0007A0D implementation context `01a000f9-8118-73a0-bcc6-2f43e4f271fc` and audit/fix
context `01a00197-8794-7ca0-8849-ba3e1de1a0c0` produced the stabilized schema-26 fold
correction. The final focused five-suite command passed 50 tests, and the authoritative CPU suite
passed 53 suites/317 tests with zero failures or errors and one existing expected skip. Retained
Class-File evidence proves direct typed dense/general FOLD_AXIS and FOLD2D bodies; the fixed dense
FLOAT32 FOLD_AXIS fork ratios were `0.929879x`, `0.923513x`, `0.925827x`, `0.927054x`, and
`0.926274x`, with aggregate `0.926451x`. General, segment, mixed-carrier, BFLOAT16, integral, and
FOLD2D forms remain semantic/Class-File evidence rather than broad performance claims. Clean
documentation context `01a001a3-2387-7c21-b04b-de96079c5959` reused the stabilized executable
and timing evidence and finalized the affected Javadocs, guide, glossary, and planning records.

CPU 0007A executes one fully static resolved-layout ordinary MIN, MAX, ALL, or ANY occurrence.
It supports exact full, single-axis, and multi-axis forms, five represented numeric extrema types,
canonical BOOL folds, selected-domain identities, first-logical-NaN represented bits, signed zero,
arbitrary supported layouts/carriers, injective output, complete pre-mutation overlap rejection,
whole-output-cell scalar/parallel-scalar execution, zero workspace/materialization/partial/combine
state, an independent reference oracle, and schema 21. Implementation context
`019ffea4-7930-70c2-9773-ef9c76fefc17` passed the focused 10-test aggregate run, broader 50-test
integration/cache/preparation run, and final CPU suite. Clean documentation context
`019ffeb8-d37c-7c31-9c89-26a0264258d8` reused and independently recounted the preserved final
JUnit XML as 53 suites/303 tests, zero failures/errors, and one existing expected opt-in
persistence-evidence skip; changed no executable Java or tests; finalized Javadocs/package
summaries, guide, glossary, and planning; and passed final Javadoc, rendered-page, Markdown,
exact 37-path, schema/status/package, concurrent-scope-preservation, and whitespace gates.

CPU 0007 executes one fully static resolved-layout CUM_SUM or CUM_PROD occurrence across FLOAT64,
FLOAT32, BFLOAT16, INT32, and INT64 in every inclusive/exclusive and forward/reverse mode. Each
slice retains sequential typed accumulation, including BFLOAT16 rounding after every operation;
scalar or parallel-scalar execution partitions only whole slices. The plan declares input and
output only, rejects complete overlap before mutation, uses no workspace or materialization, and
advances generated compatibility to schema 20. Corrective implementation/audit context
`019ffcea-9e5e-7d60-9c45-f5664c8c4d4c` recorded a 1-suite/3-test regression, the final focused
11-suite/109-test matrix, and the sole latest authoritative 50-suite/292-test CPU run with one
existing skip and no failures or errors. Clean documentation context
`019ffcf5-ac50-7920-b528-1ea57c175e96` changed no executable Java or tests, finalized the
Javadocs/package summaries, guide, glossary, and planning records, and passed CPU Javadoc plus
rendered-page, Markdown, exact 39-path, schema/status, concurrent-scope-preservation, and
whitespace gates.

CPU 0006D materializes one zero-input INITIAL_STATE or executes one FLOAT64/FLOAT32 DROPOUT with
explicit state, a canonical BOOL mask, modulo counter advancement, deterministic scalar/parallel-
scalar replay, zero workspace, complete pre-mutation overlap rejection, and schema 19. The sole
authoritative final CPU suite passed 47 suites and 275 tests with 0 failures, 0 errors, and 1 skip;
post-review focused runs passed 71, 9, and 46 tests with no failures. Clean documentation context
`019ffcab-2c42-7d62-be4a-4b2815654c89` reused that stabilized executable evidence, changed no
executable behavior or tests, finalized affected Javadocs/package summaries, the CPU guide,
glossary, and planning records, and passed Javadoc, rendered-page, Markdown, exact 43-path,
schema/vector/status/package-placement, concurrent-scope-preservation, and whitespace checks.

CPU 0006B1 now executes exactly one fully static resolved-layout SCATTER_ELEMENTS,
Gather-compatible SCATTER_ADD, or SCATTER_ND occurrence through scalar or parallel-scalar
disjoint output ownership, complete bounds-before-duplicates validation, represented reductions,
and optional declared exact floating-product scratch. Implementation context
`019ff230-109c-73a3-933f-611ee7f6143d` and independent audit/fix context
`019ff248-a9e4-7150-8fbb-db2730d7cc1b` produced the stabilized 12-suite/103-test focused and
38-suite/230-test CPU evidence, with one expected existing skip in the latter and no failures or
errors. The mandatory clean documentation context had no available context ID; it reused those
tests, finalized Javadocs/package summaries, the CPU guide, glossary, and planning records, and
passed CPU Javadoc plus final documentation/scope/whitespace gates. Schema 16 was current at the
CPU 0006B1 checkpoint.

CPU 0006B2 now executes exactly one fully static resolved-layout FOLD_AXIS or FOLD2D occurrence.
It creates a fresh represented-positive-zero output and adds logical input positions in canonical
row-major order, with FLOAT64/FLOAT32/BFLOAT16 in both families, modular INT32/INT64 in axis fold,
BFLOAT16 rounding after each addition, and exact FOLD2D padding and ceil-tail exclusion. Arbitrary
supported layouts and carriers retain a distinct injective non-overlapping output; execution is
direct scalar or parallel-scalar over disjoint output ranges, with no workspace, materialization,
atomics, partials, or merge. Schema 17 was current at the CPU 0006B2 checkpoint. Implementation context
`019ff549-d477-7140-921d-8404d10a2c7e` recorded focused 12-suite/98-test and final CPU
41-suite/245-test evidence with respectively zero and one skips and no failures or errors on
OpenJDK 26.0.1+8-34 and Gradle 9.6.1. Clean documentation context
`019ff565-3991-72a1-a47a-78f63ae600ec` reused that evidence, finalized Javadocs, package-summary
review, guide, glossary, and planning records, and passed CPU Javadoc plus final documentation,
scope, synchronization, preservation, and whitespace gates without changing executable Java.

CPU 0006C now executes exactly one fully static resolved-layout stable SORT, ARGSORT, or TOP_K
occurrence across FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. It preserves NaN-last and
directional signed-zero order, increasing logical-index stability, exact represented value bits,
zero-based INT64 axis indices, deterministic increasing-index unsorted TOP_K output, and ordered
two-output binding. Scalar and complete-slice parallel-scalar execution share one schema-18
artifact and use exact disjoint two-region INT64 scratch per selected range. Cold binding rejects
every input/output and TOP_K output/output overlap before mutation or worker submission. The
original implementation pass recorded 257 tests before clean documentation context
`019ffbb7-f877-73f3-afc3-5cbd8b6f593d` finalized the affected documentation. A later coordinator
review found that one-output SORT/ARGSORT overlap validation could range complete bindings by slice
ordinals and miss overlap outside that prefix. Fix context
`019ffbc8-fb51-7a73-9504-9a3929cefe58` restored complete-boundary overlap checks, added the
multidimensional regression, and ran the authoritative final `./gradlew :backends:cpu:test`: 258
tests with zero failures, zero errors, and one skip. Clean documentation re-review context
`019ffbcb-4c30-7e53-8e4d-9474f5cda235` reused that result, changed no executable Java or tests,
confirmed the existing Javadocs, package summaries, CPU guide, and glossary remained accurate,
and synchronized final planning evidence.

Superseded task 0001 remains preserved through its sole detailed CPU
[capability, representation, binding, and parallel foundation](tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md)
specification. It introduces one truthful provider with stable `cpu` identity and no advertised
operation semantics, package-private aligned native representation and typed cold-binding
infrastructure, and a package-private bounded worker/range foundation. It does not add a CPU
preparer, executable semantic coverage, a route, generated code, OpenBLAS integration, or another
module change. Superseded
[task 0002 Portable Class-File API generator foundation](tasks/0002-portable-class-file-api-generator-foundation.md)
now adds the backend-private deterministic generator foundation and synthetic probes.
`CpuPortableExecutionMode.emit(CodeBuilder, CpuKernelSpecialization,
CpuFamilyKernelEmitter)` owns only structural scalar-versus-Vector emitter construction and
dispatch, while `CpuClassFileKernelGenerator` delegates without that switch. The trusted
`CpuLoweringFingerprint.fromDigest(byte[])` factory retains only an exact defensively copied
32-byte derived digest without rehashing. Scalar modes require `Tail.NONE`; Vector modes permit
`NONE`, `SCALAR`, and `MASKED`. Baked primitive-array byte offsets require data-type-width
alignment, exact segments retain baked zero offset, and dynamic array offsets remain cold typed
invocation values. Equal requests produce identical bytes and distinct hidden artifacts because
the direct generator path remains fresh. At the CPU-0002 checkpoint, CPU advertised and executed
no Model operation.
Superseded
[task 0003 Durable generated-kernel artifact store and cold loading](tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md)
adds one CPU-private durable reuse source: a
model-independent filesystem store under an explicit caller/composition-supplied trusted local
root. Deterministic content-addressed paths retain exact compatibility metadata and verified
class bytes in a self-contained envelope published by forced temporary-file write and atomic
move. Stored hits are accepted only after full structural metadata, checksum, class-file, and
entry-shape validation. Process-local equal requests single-flight, while loaded artifacts are
interned only weakly with stale-key cleanup; no strong completed LRU, expiry, or age-based
invalidation exists. The generator now separates deterministic byte emission from revalidated
stored-byte definition, and exact class-shape validation remains generator-owned. No production
operation-family source uses the store yet. Superseded CPU 0004 historically owned the CPU-private
integration, explicit root seam, and prepared-executable strong retention. Detailed
[task 0004 Typed portable analysis, specialization, and finalization](tasks/0004-typed-portable-analysis-specialization-and-finalization.md)
is Superseded. It added only the CPU-private staged lifecycle foundation: deterministic selection
from a direct typed candidate source, exact declarations before assignment, artifact-store use
only during finalization afterward, immutable generated-kernel/direct-handle retention, and
family-owned direct typed cold binding. CPU analysis preserves backend-declared byte geometry;
shared Prepare validates that geometry against assigned slots, and no dense-layout or
materialization policy was introduced. Its tests remain bounded and synthetic,
`CpuCapabilityProvider` remains fail-closed, and no real Model operation family is implemented or
advertised. CPU task 0005 is Superseded through its sole detailed
[Dense ADD and partition-sequence execution](tasks/0005-dense-add-and-partition-sequence-execution.md)
specification. It solves maximal-partition truth with an ordered node-kernel sequence while
limiting first executable coverage to static canonical dense FLOAT64/FLOAT32/INT32/INT64 ADD
through scalar single-thread native segments. Implementation context
`019fd19a-4262-7580-9674-226595356fbc` recorded the sole final CPU 31-test pass; the clean
documentation pass reused that evidence because it changed no executable Java behavior.

Detailed
[task 0005A Atomic partition-kernel architecture reset](tasks/0005a-atomic-partition-kernel-architecture-reset.md)
is Complete. It delivered the intentional atomic capability reset before family expansion:
structured internal packages with only `route.portable`, a route-neutral selected-plan
seam, whole-partition lowering, pre-declaration fusion, route-independent IR, universal primitive
start/end loops, normal in-memory Class-File generation, optional cold persistence, and one
partition executable. Its exact proving chain is canonical-dense
FLOAT64 ADD -> exact GELU -> MUL in one unit, with two logical graph/IR intermediates and no
physical slots for either. The same generated bytes and loaded compatibility identity must serve
two compatible extents. Detailed
[task 0005B Universal access plans and right-aligned broadcasting](tasks/0005b-universal-access-plans-and-right-aligned-broadcasting.md)
is Complete. It adds right-aligned static broadcast/layout normalization, exact per-boundary
declaration and accessed-range spans, complete write-injectivity proof, five generated scalar
state machines, and all sixteen ordered heap/segment carrier specializations. Detailed CPU 0005C
is Complete. CPU 0005D, detailed CPU 0005E, detailed CPU 0005F, and detailed CPU 0005G are also
Complete. Detailed CPU 0005H, detailed CPU 0005I, and detailed CPU 0005J are `Complete`;
task 0006 and detailed task 0006A are `Complete`, detailed task 0006A1 is `Complete`, detailed
task 0006A2 is `Complete`, detailed task 0006B is `Complete`, detailed task 0006B1 is `Complete`,
detailed task 0006B2, detailed task 0006C, and detailed task 0006D are `Complete`; detailed task
0007 and detailed 0007A are `Complete`; corrective tasks 0007A0, 0007A0A, and 0007A0B are
`Complete`; detailed 0007A0C, detailed 0007A0D, and detailed 0007A0E are `Complete`; detailed
0007A0F is `Complete`; detailed CPU 0007A1, CPU 0007A1A, and CPU 0007A1B are `Complete`.
CPU 0007A1C is `Complete`; CPU 0007A1D remains `Review needed`; detailed CPU 0007A1E through CPU
0007A1O, CPU 0007A2, detailed CPU 0007B, and detailed CPU 0007C are `Complete`. Detailed CPU 0007D
and detailed CPU 0007E and CPU 0007F are `Complete`, while detailed CPU 0007F1 is `Complete`;
detailed CPU 0007F2, detailed CPU 0008, detailed CPU 0008A, and detailed CPU 0008B are `Complete`;
CPU 0008C, detailed CPU 0008D, detailed CPU 0008E, and detailed CPU 0008E1 are `Complete`;
CPU 0008F, detailed CPU 0008G, detailed CPU 0008G1, detailed CPU 0008H, and CPU 0008I are
`Complete`; detailed CPU 0008J, Model 0025L, and CPU 0008K are `Complete`; CPU 0008L is the next
`Draft` frontier, CPU 0008L through 0008P remain ordered Draft follow-ups, and CPU 0009 through
0017 remain Draft. Prepare
0003A is Complete.
CPU 0005C preserves that exact slice and implements cold selection among all four portable
strategies. It uses the preferred Java 26 FLOAT64 species only for direct contiguous runs and
scalar broadcasts, scalar tails and general-odometer fallback, configured/available parallelism
bounds, deterministic disjoint chunks, and one explicit caller-owned CPU-private worker group
borrowed by finalization/execution. Strategy/species alter generated identity; extents, chunk
configuration, and worker identity remain cold facts. It adds no gather, masked tail,
materialization, shared lifecycle, Config surface, or performance claim.

The CPU 0005C implementation context's final corrected `./gradlew :backends:cpu:test` run passed
18 suites and 49 tests with zero failures, errors, or skips. Clean documentation context
`/root/cpu_0005c_docs` reused that evidence because no executable Java or test changed afterward,
finalized affected Javadocs, package summaries, the CPU guide, glossary, and planning records, and
passed CPU Javadoc plus the recorded Markdown, exact-scope, status, forbidden-vocabulary, and
whitespace gates.

Detailed CPU 0005D is Complete. CPU analysis can now compare direct access with at most one
eligible contiguous input copy, append exact workspace ID `0` before assignment, and retain the
original source separately from the adjusted generated consumer pattern. Finalization resolves
every assignment before one artifact lookup; execution copies once before consumer work. The
specialization budget is four complete candidates, one realized artifact, zero additional
planner-visible fixed-shape variants, and zero additional planner-visible unrolled variants.
Generation-time fixed-trip code within that one guarded artifact does not create another variant.
Optional explicit-root persistence uses one bounded verified
current-schema envelope. The recorded six-fixture Oracle JDK 26.0.1 evidence verdict is
`KEEP_DISABLED`, so default persistence remains off. Class bytes, JVM JIT machine code/profile,
and the future workload tuning cache remain distinct.

Implementation evidence includes the final 11-suite/40-test focused matrix, focused 8-, 13-, and
7-test runs, the sole final ordinary 21-suite/62-test CPU pass with one skipped evidence method,
and the sole successful explicit 1-test evidence run. Clean documentation context
`/root/cpu_0005d_docs` reused those stabilized results, finalized the exact 21 production/package
paths and five Markdown records, and passed CPU Javadoc plus Markdown, exact 38-path scope,
status, terminology, structure/order, persistence, and whitespace gates.

Detailed CPU 0005E is Complete. One through eight fully static connected straight-line pointwise
occurrences lower into one nineteen-opcode CPU-private sequence, one canonical IR, one artifact,
and one partition executable. The admitted matrix uses FLOAT64, FLOAT32, INT32, INT64, and BOOL
with exact typed array or segment carriers; boundaries are derived deterministically, internal
single-use results stay virtual, and exactly one final store is materialized. Scalar and
parallel-scalar generated execution cover every admitted row, while vector compute remains
FLOAT64 numeric-only. Same-type CAST preserves represented values; cross-type CAST and excluded
pointwise or later semantic families remain fail-closed. The generator schema is 5, default
persistence remains disabled, and no public API, common lowering, shared module, architecture,
dependency, build, conformance, or integration contract changed. Implementation evidence includes
the required focused matrix, a broader focused regression batch, and the sole final 24-suite/
73-test CPU pass with zero failures/errors and one opt-in evidence skip. Clean documentation
context `/root/cpu_0005e_docs` reused that evidence, finalized affected Javadocs/package summaries
and five Markdown records, and passed CPU Javadoc plus Markdown, exact 38-path scope, semantic,
status, and whitespace gates.

Detailed CPU 0005F is Complete. It extends the family pipeline to twenty-two opcodes with exact
same-typed FLOAT32/FLOAT64 binary DIV, scalar DIV, and scalar POW. Common lowering classifies exact
exponent bits once as direct, positive one, identity, square, or reciprocal; reciprocal remains
semantic scalar POW. FLOAT64 DIV and special power retain existing conditional vector eligibility,
while direct power and FLOAT32 use scalar compute with optional parallel orchestration. Schema 6,
canonical IR, specialization, and the cold manifest retain realization compatibility. The
implementation context passed compile-test, focused generated-kernel validation, a final 16-test
IR/lowering/generated run, and the sole final 25-suite/102-test CPU run with one opt-in timing
skip and no failures or errors. Clean documentation context `/root` reused that Java evidence and
passed CPU Javadoc, Markdown, exact 31-path scope, semantic/status, and whitespace gates.

Detailed CPU 0005G is Complete. It extends the closed family pipeline to 31 opcodes with exact
same-typed FLOAT32/FLOAT64/INT32/INT64 binary and scalar extrema, first-class FLOAT32/FLOAT64
range CLAMP, direct same-typed FLOAT32/FLOAT64 Tensor power, and canonical-BOOL AND/OR/NOT.
CLAMP remains one occurrence and one IR instruction with exact ordered raw bounds. All nine new
opcodes use scalar or parallel-scalar compute, while existing vector coverage and all fusion,
access, materialization, specialization, artifact, and lifecycle budgets remain unchanged.
Generator schema 7 rejects every older envelope without migration. Implementation context
`/root/cpu_0005g_impl` passed the focused 9-suite/41-test command and exactly one final
25-suite/106-test CPU command with zero failures/errors and one existing opt-in persistence-timing
skip. Clean documentation context `/root/cpu_0005g_docs` reused that evidence, finalized affected
Javadocs and five Markdown records, and passed CPU Javadoc, Markdown, exact authorized-scope,
semantic/status, and whitespace gates without changing executable Java.

Detailed CPU 0005H is Complete. It extends the closed family pipeline to exactly 48 opcodes and
schema 8, with one distinct opcode for each of the nineteen same-typed FLOAT32/FLOAT64 unary
semantics and the existing three floating classifications unchanged. Scalar and parallel-scalar
cover every unary row; the selected FLOAT64 lane-operator and ERF/GELU rows also admit vector and
parallel-vector compute, while vector-ineligible chains select scalar compute. GELU now covers
both precisions and maps negative infinity to negative zero; stable sigmoid, tanh-approximation
GELU, and SiLU preserve their fixed exceptional-value contracts. One-through-eight fusion, five
access regimes, one-copy/four-candidate/one-artifact materialization budgets, zero fixed-shape and
unroll budgets, same-type CAST, and fail-closed cross-type CAST/BFLOAT16 remain unchanged.

Implementation context `/root/cpu_0005h_impl` passed CPU compilation, test compilation, the
required focused nine-class command with 43 tests, and exactly one new final 25-suite/108-test CPU
command with zero failures/errors and one expected opt-in persistence-timing skip after correcting
FLOAT32 RSQRT to perform its square root and reciprocal before one final narrowing. This supersedes
the earlier 42-test/107-test evidence. Clean documentation context `/root/cpu_0005h_docs` reopened,
reused the restabilized Java evidence, finalized the
affected Javadocs/package summaries, CPU guide, glossary, and planning records, and passed CPU
Javadoc, Markdown/link/anchor/formatting, exact authorized-scope, semantic/status, and whitespace
gates without rerunning Java tests.

Detailed CPU 0005I is Complete. It closes the intentional FLOAT64-only vector realization before
family expansion with preferred-species FLOAT32 parity for the exact existing twenty-one eligible
opcodes. Package-private final `CpuVectorInstructionEmitter` owns the closed family switch, while
package-private final `CpuVectorMath` owns pure `FloatVector`/`DoubleVector` formulas. The retained
Cephes binary64 ERF family now has exact source-stable binary32 rounded tables, independent-oracle
and scalar-differential evidence, and explicit FLOAT32 bounds. Schema 9 rejects every older
envelope. Scalar fallback and tails, FLOAT64-only materialization, one-through-eight fusion, the
four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, optional persistence policy,
capability, public API, modules, dependencies, and architecture remain unchanged.

The implementation pass's revised focused eleven-class command passed 11 suites/62 tests. The
final CPU command passed 26 suites/117 tests with zero failures/errors and one expected opt-in
persistence-evidence skip. Clean documentation context `/root` reused that evidence, finalized
the affected Javadocs, CPU guide, glossary, and planning records, and passed CPU Javadoc plus
local Markdown, official-link, exact-scope/status/inventory, formatting, and whitespace gates
without rerunning Java tests. CPU 0005J and detailed CPU 0006 are Complete. CPU 0006A's later
completion is recorded below; detailed CPU 0006A1 and CPU 0006A2 are `Complete`, detailed CPU
0006B is `Complete`, and detailed CPU 0006B1, CPU 0006B2, and CPU 0006C are `Complete`; detailed
CPU 0006D and detailed CPU 0007 are `Complete`; later tasks remain
`Draft` without detailed specifications.

Detailed CPU 0005J is Complete. It preserves the exact forty-eight-opcode semantic inventory and
adds preferred-species FLOAT32/FLOAT64 extrema, clamp, ReLU, sign, and same-type cast;
INT32/INT64 modular arithmetic, signed extrema, and same-type cast; canonical BOOL logic and
same-type cast; and virtual floating comparison/classification masks through logical masks into
floating WHERE. A scalar/all-zero external BOOL condition may become an all-true or all-false
mask, while materialized masks and non-scalar external conditions remain scalar. Schema 10 keys
the exact typed species and mask topology. General odometers, too-short runs, unsupported
opcode/type pairs, direct scalar power, and unsafe mask boundaries deterministically select scalar
or parallel-scalar compute. The implementation preserves one-through-eight straight-line fusion,
one final store, scalar tails, FLOAT64-only one-input materialization, four candidates, one
artifact, and zero fixed-shape/unroll variants. The implementation pass compiled production and
tests; its required twelve-class matrix passed 12 suites/76 tests; additional focused generated-
kernel, vector-math, specialization, and preparation runs passed; and its sole final CPU suite
passed 26 suites/125 tests with zero failures or errors and one skipped opt-in persistence-
evidence test. The clean documentation pass reused that stable Java evidence, finalized the
authorized Javadocs and documentation, and passed CPU Javadoc, Markdown, exact-scope, semantic,
status, unchanged-layer, and whitespace checks without rerunning Java tests.

Detailed CPU 0006 is Complete. It composes one-through-eight connected static resolved-layout
affine view occurrences into one source-to-result mapping, keeps same-unit internal view values
virtual without declarations or Runtime slots, and emits one exact scalar or parallel-scalar
represented-bit boundary copy. The result retains its resolved offsets and positive or zero
strides; non-injective layouts use deterministic distinct-address writes only after proving that
repeated logical coordinates select the same source value. All six Model data types copy across
array, segment, and mixed carriers, with the seventh `SHORT_ARRAY` form limited to raw BFLOAT16
payloads. Schema 11 distinguishes affine structure and carrier compatibility. The implementation
pass's focused command and sole final CPU command passed; the final suite recorded 29 suites and
140 tests with zero failures, zero errors, and one skipped opt-in persistence-evidence test on
OpenJDK 26.0.1+8-34. Clean documentation context `/root/cpu_0006_docs` reused that stabilized Java
evidence, finalized Javadocs/package summaries, the CPU guide, glossary, and planning records,
and passed CPU Javadoc plus Markdown, exact-scope/status, and whitespace validation without
rerunning Java tests.

Detailed CPU 0006A is Complete. It adds one exact fully static resolved-layout PAD, TILE, CONCAT,
or STACK occurrence with one-through-sixteen ordered semantic inputs, first-occurrence unique
resource declarations, compact cold movement geometry, all-six-type represented-bit scalar or
parallel-scalar generation, and one distinct injective output. Generator schema 12 records family,
rank, occurrence mapping, and exact PAD bits. The focused command passed 8 suites/43 tests and the
sole final CPU command passed 32 suites/153 tests with zero failures or errors and one skipped
opt-in persistence test on Java 26.0.2, HotSpot 26.0.2+10-55. Clean documentation context `/root`
reused that stabilized evidence, finalized Javadocs/package summaries, the CPU guide, glossary,
and planning records, and passed CPU Javadoc plus Markdown, exact-scope/status/inventory, forbidden-
change, and whitespace validation without rerunning Java tests. Detailed CPU 0006A1 is `Complete`;
detailed CPU 0006A2 is `Complete`, detailed CPU 0006B is `Complete`, detailed CPU 0006B1 and CPU
0006B2, detailed CPU 0006C, detailed CPU 0006D, and detailed CPU 0007 are `Complete`,
and detailed CPU 0007A is `Complete`; corrective CPU 0007A0, CPU 0007A0A, and CPU 0007A0B are
`Complete`; detailed CPU 0007A0C, CPU 0007A0D, and CPU 0007A0E are `Complete`; detailed CPU
0007A0F is `Complete`. Detailed CPU 0007A1, CPU 0007A1A, and CPU 0007A1B are `Complete`.
CPU 0007A1C is `Complete`, CPU 0007A1D remains `Review needed`, and detailed CPU 0007A1E through
CPU 0007A1O, CPU 0007A2, detailed CPU 0007B, and detailed CPU 0007C are `Complete`; detailed CPU
0007D, detailed CPU 0007E, and detailed CPU 0007F are `Complete`; detailed CPU 0007F1 is `Complete`;
detailed CPU 0007F2 is `Complete`, and later work remains `Draft` without detailed specifications.

Detailed CPU 0006A1 is Complete. It extends the same movement pipeline with one fully static,
resolved-layout UNFOLD_AXIS occurrence for all six represented types or one floating UNFOLD2D
occurrence using direct positive-zero or exact matching typed padding. The implementation keeps
axis/NCHW mapping, dilation, floor/ceil grids, unequal-rank strides, and arbitrary-range start
state in compact cold geometry; emitted loops use carry/reset odometers and schema 13 records only
code-shaping identity. The focused matrix passed 10 suites/63 tests, and the sole final CPU suite
passed 32 suites/163 tests with one expected persistence skip on OpenJDK 26.0.1+8-34. Clean
documentation context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4` reused that stable evidence,
finalized Javadocs, the CPU guide,
glossary, and planning records, and passed CPU Javadoc plus scope, schema, status, Markdown, and
whitespace gates without changing executable Java.

Detailed CPU 0006A2 is Complete. It adds one fully static resolved-layout GATHER,
GATHER_ELEMENTS, GATHER_ND, or ONE_HOT occurrence through distinct mixed-type indexing IR,
compact geometry, deterministic complete run-bound validation, and one schema-14 output-writing
artifact. Unique inputs plus one output are declared with no materialization or workspace; valid
zero-output execution makes no generated call or worker submission. Implementation context
`019ff098-313c-7b53-8a03-df9f31fcf71f` established the implementation. Mandatory clean audit/fix
context `019ff0bc-5996-7c30-903e-6f32d1b53a36` then found no production defect, expanded the
allowlisted regression matrix, and recorded the final focused 12-suite/71-test pass plus final CPU
35-suite/183-test pass with one expected persistence-evidence skip and zero failures/errors on
Oracle OpenJDK 26.0.1, Runtime/VM 26.0.1+8-34. Mandatory clean documentation context
`019ff0c9-83cf-7d82-8f3c-e61be7a30269` independently finalized Javadocs, package summaries, the CPU
guide, glossary, and planning/status records and reused that stable executable evidence without
rerunning Java tests.

Detailed CPU 0006B is Complete. It adds one fully static resolved-layout SLICE_UPDATE occurrence
for both signed finite-coordinate and target-relative placement. The generated movement body
walks the output once, selecting base outside the region and update at selected positions for all
six represented types. It supports positive, negative, and non-unit steps, scalar and empty
cases, arbitrary scalar/parallel-scalar ranges, heap/segment/mixed carriers, deduplicated
`[0, 0]` input occurrences, one unit/artifact, strict output/input non-overlap, and schema 15.
Implementation context `019ff156-d822-7442-b8a6-20f7e5e55d4d` passed the exact focused
matrix before clean documentation context `019ff15f-a57f-7b62-aadc-40d3335d4a96` finalized
Javadocs, package summaries, the CPU guide, glossary, and planning records. Later mandatory
audit/fix context `019ff188-85cb-7d50-be34-c5c3038b5634` found no production defect and changed
tests only. It added direct evidence for generated arbitrary layouts/ranges/multiple axes;
parallel mixed, all-segment, and all-heap BOOL execution; binding, overlap, and input-alias rules;
one unit with exact deduplicated declarations, scalar/parallel and zero-output choices, no
workspace/materialization, exact final assignments, and one artifact; exact/invalid bounds and
output injectivity; and both reference forms across all ten authorized test owners.
Its exact focused command passed 10 suites and 91 tests, and its final CPU command passed 35
suites and 198 tests with one expected skip and no failures/errors on Oracle OpenJDK
26.0.1+8-34. Because no production Java contract changed after the prior Javadoc pass, the final
documentation synchronization updated evidence only and did not rerun Java tests or Javadoc.

The reset was a working-tree replacement, not deletion of history. CPU 0001–0005 are Superseded
with all recorded evidence preserved; the repository contains no old/new dual pipeline.

Future Model task 0026 is required before any CPU route advertises FLOAT16. That dependency does
not block tasks 0003–0004 or later current-type work: generated-artifact storage, portable
analysis/finalization, and current-type family coverage may proceed while remaining fail-closed for
FLOAT16.

The separately authorized architecture synchronization remains complete: the authoritative
contract and focused explanations permit implementation-neutral generated JVM-bytecode CPU
computation kernels while preserving CPU backend and Prepare ownership. Implementation context
`019fc815-42aa-7de2-8970-a2fcab3a390e` recorded the final focused 3-suite/18-test pass and the
sole final CPU 9-suite/34-test pass, both with zero failures, errors, or skips. The clean
documentation pass reused that executable evidence and finalized CPU Javadoc, guide, glossary,
and planning validation. The Class-File API and Vector API remain selected CPU-internal
implementation choices rather than architecture invariants.

Implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded CPU 0003's focused
3-suite/30-test pass and sole final CPU 10-suite/48-test pass, both with zero failures, errors, or
skips. The mandatory clean documentation pass reused that evidence, finalized the authorized
Javadocs, guide, glossary, and planning records, and changed no executable Java.

The CPU 0004 implementation pass recorded a final focused three-suite/24-test pass and the sole
stabilized CPU module 13-suite/72-test pass, both with zero failures, errors, or skips. Mandatory
clean documentation context `/root/cpu_0004_docs` reused that evidence, finalized all affected
production/package Javadocs, the CPU guide, glossary, task/master/roadmap records, and passed CPU
Javadoc plus exact-scope, Markdown, surface, status, excluded-path, and whitespace checks without
changing executable Java.

## Pointwise follow-up notes

CPU 0008K is `Complete`. CPU 0008L is the next `Draft` frontier and must not receive a detailed
specification until it becomes current. These notes do
not alter the ordered task rows or completed earlier CPU families.

- **CPU 0008L pointwise mask closure** — materialized FLOAT32/FLOAT64
  comparison/classification results and dense external BOOL conditions for WHERE are the next
  bounded vector work after CAST. Virtual masks remain the fusion form. Current Java 26 Vector API
  has no suitable FLOOR/CEIL operators, so those operations remain scalar; SIGMOID, SILU,
  GELU-TANH approximation, and general POW require a separate numerical and cost proof.
- **CPU 0008K cross-type CAST** — detailed and `Complete`
  [Model 0025L](../../modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md) now defines
  all 36 ordered current-type pairs and observable direct rounding,
  truncation/saturation, overflow/underflow, deterministic NaN, infinity, signed-zero, integral,
  BFLOAT16, and BOOL behavior. Java primitive or Vector API conversion availability is not that
  semantic authority.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented CPU routes and the shared opaque orchestration consumer.
- Vendor ABI/lifetime layers remain inside the CPU backend unless a later explicit architecture
  decision authorizes another provider module and dependency edge. This plan does not add such a
  module.
- Exact oneDNN partition/fusion coverage and ZenDNN coverage wait for concrete integration use
  cases. Apple Basic Neural Network Subroutines (BNNS) are deferred without a task row until a
  concrete integration spike or use case establishes scope and advantage.
- A future explicit numerical-policy contract must define any relaxed/fast-math permission before
  such a vendor candidate can become eligible. Current exact/default semantics admit none.
- Exact/default floating division and scalar-power realization are independent of that future
  permission. Complete 0005F adds primitive same-typed FLOAT32/FLOAT64 binary and scalar DIV and
  selects only its proved positive-one, identity, one-multiply square, and one-division reciprocal
  power realizations; every other admitted exponent retains direct power. Reciprocal power remains
  semantic `SCALAR_POW`, not DIV. Draft 0017 later consumes Config 0006 permission for genuinely
  relaxed candidates and therefore does not block exact work.
- Historical task 0001 implements the exact native `MemorySegment` representation, shared-arena
  ownership, zero-size/alignment/allocation/cleanup rules, borrowed heap/native cold binding,
  direct typed invocation seam, and worker evidence. Task 0005A retains the representation and
  binding contracts but removes unused worker types; completed task 0005C implements the minimum
  caller-owned worker group required by its parallel strategies. Route-specific
  representations and materializations still wait for their ordered tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The former broad 0006A row was dependency-split before implementation. Complete 0006A owns one-node
  static PAD/TILE/CONCAT/STACK movement and unique multi-input binding; detailed Complete 0006A1
  owns window extraction; detailed Complete 0006A2 owns Gather/one-hot plus complete pre-write
  index validation. All three slices have detailed task specifications.
- The former broad 0006B row is dependency-split. Detailed Complete 0006B owns only value-blind
  functional SLICE_UPDATE through the existing movement path. Detailed Complete 0006B1 owns
  index-valued functional scatter, complete pre-write bounds/NONE-duplicate validation, exact
  base/reduction semantics, deterministic output ranges, and declared floating-product scratch.
  Detailed Complete 0006B2 owns zero-initialized overlap fold and padding exclusion; detailed
  Complete 0006C depends on 0006B2 and owns stable ordering/selection execution. Detailed Complete
  0006D owns only one-node INITIAL_STATE and FLOAT64/FLOAT32 explicit-state DROPOUT through one
  CPU-private versioned counter mapping. The former broad CPU 0007 row is now split by current
  Model semantic-family and algorithm/resource boundary: detailed Complete CPU 0007 owns only
  CUM_SUM/CUM_PROD scans. The former ordinary-reduction row is further split at numerical/resource
  and binding-geometry boundaries: detailed Complete 0007A owns full/single-/multi-axis MIN/MAX/ALL/
  ANY with zero workspace; Complete 0007A1 delivers ordinary SUM/MEAN/PROD and its
  explicit accumulator design. Complete corrective 0007A1A–0007A1C, historical Review-needed
  0007A1D, and Complete corrective CPU 0007A1E–0007A1O precede Complete CPU 0007A2,
  which owns binding-aware
  target-Shape SUM. Detailed Complete 0007B owns arg
  extrema; detailed Complete 0007C owns masked reductions; detailed Complete 0007D owns logarithmic/statistical/norm
  reductions; detailed Complete 0007E owns stable softmax/log-softmax; Complete 0007F owns Layer/RMS
  normalization; detailed Complete 0007F1 owns batch inference; and detailed Complete 0007F2 owns
  batch training/statistic transition. CPU 0007A1O, detailed CPU 0007A2, detailed CPU 0007B, and detailed CPU 0007C are
  `Complete`; detailed CPU 0007D, CPU 0007E, and CPU 0007F are `Complete`. Detailed CPU 0007F1 is
  `Complete`; CPU 0007F2, detailed CPU 0008, CPU 0008A, CPU 0008B, CPU 0008C, and CPU 0008D are
  `Complete`; detailed CPU 0008E, Prepare 0003A, and detailed CPU 0008E1 are `Complete`,
  CPU 0008F is `Complete`, and later tasks remain Draft.
- CPU 0007 is first because the closed cumulative-scan family is independently executable and
  has no aggregate-combination dependency. Partitioning only across complete logical scan slices
  preserves one sequential typed accumulation order, requires no partial/combine workspace, and
  establishes the next generated range/body family without pre-deciding later reduction or
  normalization algorithms.
- CPU 0007A followed because extrema and boolean folds share one ordinary full/single-/multi-axis
  output/domain geometry, exact identity/selection rules, output-cell-only parallelism, and zero
  workspace. CPU 0007A1 is separate because ordinary floating SUM/MEAN require exact-real
  summation followed by result-format rounding rather than scan's per-step typed rounding, and
  PROD has its own product special-value/resource policy. CPU 0007A2 is separate because
  `SumToShapeAttrs` adds right-aligned binding-aware geometry. The ordinary numerical task remains
  behind the complete generated-family parity sequence.
- Completed CPU 0007A0 is inserted after semantic completion and before CPU 0007A1 because a completed
  bytecode/performance audit found material generated-hot-path defects in the shared pointwise
  loop shape and the bridge-only implementations delivered by CPU 0007/0007A. It is one atomic
  compatibility correction: dense heap arrays gain cold-proved int scalar/address loops, Vector
  API dense loops gain one precomputed bound and scalar tail, and every currently supported scan/
  aggregate form gains a typed generated hot body plus correct general fallback. The isolated
  acceptance gate passed at `<= 1.15x` equivalent direct Java for each of the five audited
  1,048,576-element cases across five fresh forks. Historical semantic completion remains intact.
- CPU 0007A0A–0007A0F extend corrective parity coverage across every generated family that existed
  before ordinary numerical aggregates. Completed CPU 0007A0A groups affine plus movement because
  their typed loops were already embedded and needed the same dense integer address correction.
  Completed CPU 0007A0B embeds typed indexing bodies while preserving the separate complete
  validation pass. Scatter, fold, ordering, and random/dropout remain separate because each has a
  distinct mapping, resource, state, comparison, or accumulation baseline. Detailed CPU 0007A0C,
  CPU 0007A0D, CPU 0007A0E, and CPU 0007A0F are `Complete`. Detailed CPU 0007A1 is `Complete`
  after passing its executable, Class-File, performance, Javadoc, and documentation gates.
  The completed schema-29 audit first inserted CPU 0007A1A–0007A1C before CPU 0007A2. Complete
  0007A1A removes all covered per-element scalar project-helper references while retaining the
  evidence-backed chunk-level `CpuVectorMath` boundary; Complete 0007A1B corrects scratch-free
  scatter complexity after an adverse-ratio baseline while retaining the exact-product safe split.
  CPU 0007A1C froze bounded general-layout, `MemorySegment`, family, and variant evidence without
  runtime tuning; exact semantics passed all 20 rows, but its first fork failed 17 ratios and
  triggered the explicit more-than-two-owner stop. No production owner changed. Evidence-driven
  CPU 0007A1D retained schema-32 invocation-local segment layouts but failed all 13 target ratios;
  it remains incomplete. Complete CPU 0007A1E owns the four-row movement general-address-loop
  cluster. Complete CPU 0007A1F owns the BOOL movement/aggregate group, Complete CPU 0007A1G owns
  the fold/dropout group, Complete CPU 0007A1H owns the numerical aggregate group, Complete CPU
  0007A1I owns the indexing group, Complete CPU 0007A1J owns the scan row, Complete CPU 0007A1K
  owns the affine-copy row, Complete CPU 0007A1L owns the pointwise general row, Complete CPU
  0007A1M owns the scatter-MIN row, and Complete CPU 0007A1N closes the final evidence-supported
  multi-axis MIN residual. Complete CPU 0007A1O preserves the original ledger and closes the final
  pointwise-ledger evidence gap with a versioned tested replacement and compatible direct-Java
  evidence for every pointwise category. The final five accepted twenty-row samples and A1O's
  five accepted 44-case forks satisfy every original A1C gate. This insertion is ordered
  corrective work and does not change architecture authority or waive A1D's historical failed
  local criterion. Detailed CPU 0007A2, detailed CPU 0007B, and detailed CPU 0007C are `Complete`.
  Detailed CPU 0007D, detailed CPU 0007E, and detailed CPU 0007F are `Complete`; detailed CPU
  0007F1 is `Complete`; CPU 0007F2 is `Complete`, and later rows remain Draft without detailed specifications.
- CPU 0006D selects `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1`: `mix64` uses shifts 30/27/31 and
  multipliers `0xbf58476d1ce4e5b9`/`0x94d049bb133111eb` after key bias
  `0x9e3779b97f4a7c15`; each draw is `mix64(counter + logicalIndex +
  mix64(key + keyBias))`, and its uniform is the top 53 bits times `0x1.0p-53`. This is a
  CPU-private artifact/configuration contract, not a Model or cross-backend bitstream promise.
- CPU 0006D fixes canonical one-byte BOOL mask storage, one binary64 probability comparison,
  FLOAT64 division and FLOAT32 widen/divide/narrow scaling, a single generated state prologue,
  zero workspace, and complete input/output plus output/output overlap rejection. BFLOAT16
  dropout remains fail-closed until a conforming direct scaling/conversion rule is established.
- CPU 0006A declares each distinct composition input `ValueId` once in first-occurrence order and
  retains a CPU-private ordered occurrence-to-boundary map. Repeated semantic inputs do not force
  duplicate shared requirements. The one output is separate, injective, and materialized.
- Value-dependent index work begins only in 0006A2. That slice must validate all INT32/INT64 index
  values before any output write and deterministically reject the first invalid logical index;
  invalid indices never wrap, clamp, select defaults, expose partial Gather output, or become an
  all-false one-hot row.
- Task 0005A replaces the provisional per-node implementation atomically. The permanent model is
  partition -> computation-oriented execution units -> canonical loop IR -> route realization ->
  one partition executable. There is no compatibility layer between the two designs.
- CPU lowering operates over the complete owned partition and supports pointwise/broadcast fusion,
  metadata/view folding, reduction/scan and stable multi-pass units, matrix/convolution epilogues,
  and explicit semantic kernels as their Draft tasks land. Fusion legality and profitability are
  separate; neither may cross alias/state/random/publication/fan-out/partition or numerical-order
  boundaries. Decomposed softmax is not silently recognized as stable `SOFTMAX`.
- Complete CPU 0008 is narrowed from the former five-family Draft row to one grouped NCHW
  Conv2d capability. The source-backed reason is task granularity: Conv2d, MATMUL, pooling,
  attention, and loss have distinct geometry, algorithms, numerical/resource/publication
  contracts, generated-loop shapes, and evidence matrices. CPU 0008 retains intrinsic optional
  bias plus only a bounded Conv2d-led compatible external ADD and at-most-one exact
  activation/clamp epilogue with a deterministic family-local split. Complete CPU 0008E1
  adopted the shared Prepare partition DAG without changing CPU behavior; detailed Complete CPU 0008F
  delivers bounded portable MATMUL execution; detailed Complete CPU 0008G delivers Pool2d, and
  detailed Complete CPU 0008G1 delivers Pool1d composition recognition plus direct Pool3d, and
  detailed Complete CPU 0008H delivers attention execution; CPU 0008I is Complete with its
  corrected full performance evidence missing/deferred to CPU 0009, detailed CPU 0008J is
  `Complete`; Model 0025L is `Complete`, and detailed CPU 0008K is `Complete`; CPU 0008L is the next `Draft` frontier. This planning
  refinement changes neither architecture nor the required 0008 -> 0008A -> 0008B -> 0008C ->
  0008D -> 0008E -> Prepare 0003A -> CPU 0008E1 -> CPU 0008F order.
- CPU 0008 resolves its in-progress split contradiction locally. `CpuPartitionPreparationPlan`
  keeps the one-unit invariant for every existing family, direct Conv2d, and legal fused Conv2d
  plan, but admits exactly two complete per-unit fact sets under one closed Conv2d materialized-
  suffix tag. Analysis declares the connecting Conv2d result once as an ordinary buffer before
  shared assignment. Finalization realizes both independently keyed artifacts and returns one
  CPU-private composite executable, so unchanged Runtime performs one ordinary atomic access-
  validity transition and the final publication suffix remains unchanged. Cold binding validates
  both units and cross-unit overlap before either writes; execution completes and joins the first
  unit before starting the second. This exception adds no general decomposition, recognition,
  candidate, profitability, or representation model and did not move work from the then-Ready
  0008B or Draft 0008C–0008E.
- Complete 0008A closes dimensional convolution immediately after CPU 0008: it validates visible
  Conv1d composition and adds direct Conv3d execution without depending on general fusion,
  profitability, or multi-input materialization. Complete 0008B generalizes the former
  straight-line unit boundary to bounded partition-DAG decomposition with deterministic
  materialized split fallback. Its detailed specification fixes stable topology, hard legality
  budgets, exact DAG-edge buffers, per-unit resources, and one atomic sequential composite; it
  performs deterministic maximal legal pointwise fusion without profitability ranking. Complete
  [0008C](tasks/0008c-typed-specialized-subgraph-and-epilogue-recognition.md) owns only a closed
  typed CPU-private recognition set. It adds no execution or generated form: MATMUL facts remain
  unsupported until 0008F, only the already implemented Conv2d ADD/ADD-RELU form maps to existing
  specialized execution, and recognized Conv1d/Conv3d/reduction epilogues keep the exact 0008B
  split. Its explicit semantic-kernel facts require the actual first-class SOFTMAX, LOG_SOFTMAX,
  Layer/RMS, or batch-normalization kind and never infer decomposed mathematics. CPU 0008D owns
  profitability ranking and cold
  decision facts. Legality rejects a candidate that cannot preserve semantics, fan-out/barrier
  rules, or hard resource budgets; profitability may reject an otherwise legal candidate because
  of estimated code size, live-value pressure, materialization, route eligibility, or complete-
  plan cost.
  Complete 0008E subsequently extends portable pointwise computation-unit representation planning:
  it retains complete candidates with at most two distinct external read-boundary copies and
  copy-once compatible reuse. A pair is rejected as `CO_CONSUMED_PAIR` when one represented
  instruction consumes both sources; both singles and eligible disjoint-consumer pairs remain
  candidate identities. The retained one-copy failure proves current static costs and cross-unit
  reuse do not establish promotion across the admitted domain. Ordinary preparation therefore
  selects CPU 0008D's direct topology; materialized forms remain executable candidate-only
  alternatives for future compatible end-to-end tuning. It does not decide fusion legality,
  invent a fused/split topology, create
  materialized DAG splits, or perform provider-specific packing or reorder. It enriches already
  legal topologies with bounded representation variants and retains 0008D's complete-plan facts;
  representation costs remain diagnostic and cannot promote a materialized ordinary plan. Each
  complete variant has a stable typed CPU-private identity carrying its materialized boundaries,
  layout/access regimes, reuse,
  workspace, execution strategy, and fused/split topology. Ordinary preparation retains but does
  not promote materialized candidates; the identities remain reproducible for the later opaque
  Prepare 0004 and Tuning 0001–0002 handoff. Direct access is the safe ordinary selection. Complete
  0008E performs no measurement,
  tuning-cache lookup or mutation, public tuning configuration, or shared handoff implementation.
- Detailed complete [0008D](tasks/0008d-bounded-fusion-profitability-and-typed-decision-facts.md)
  ranks the complete admitted set of legal existing fused/split forms with exact integer
  structural heuristics and closed typed legality, profitability, and selection facts. It retains
  the exact 0008B comparison baseline and immutable 0008C recognition snapshots, while canonical
  split wins ties, uncertainty, or incomplete bounded enumeration. It adds no generated/hot form,
  Trace payload, public registry, measurement, cache, or Runtime choice. Later
  Config 0006A supplies declarative model-autotuning inputs, Prepare 0004 carries candidates and
  compatible decisions opaquely, CPU 0016 consumes compatible workload-cache selections, and
  Tuning 0001–0002 may measure eligible complete candidates when the user explicitly requests
  model autotuning. Autotuning is not the default fusion-profitability mechanism, and no search,
  cache mutation, or choice occurs in Runtime. CPU 0008D's typed cold decision facts remain
  CPU-owned; later Trace backend payloads and tuning inspection may translate or consume them
  without making Trace or tuning the decision owner.
- One route-independent `CpuKernelIr` contains typed boundary/virtual values, ordered semantics,
  structural access-plan form, universal start/end loop model, and stores. Selected route, thread
  count, vector species, cache root, slots/segments, graph identities, generator versions, and
  instance bindings remain outside it.
- One normalized per-value CPU access plan eventually covers current right-aligned
  `ShapeBroadcast` and `LayoutDescriptor` semantics: scalars, rank/singleton/multi-axis/zero
  broadcast, multi-input masks and parameters, differing fused patterns, offsets, positive and
  zero strides, and heap/segment/mixed carriers. CPU 0005B covers only fully static shapes because
  current Prepare has no exact dynamic binding; a future explicit binding contract is required for
  dynamic or symbolic dimensions. It does not duplicate elementwise, `WHERE`, or fusion planners,
  and broadcast gradients remain `SUM_TO_SHAPE` plus later reduction coverage.
- CPU 0005B implements five ordered scalar regimes: dense linear, all-zero/scalar, last-axis bias,
  block/outer with a contiguous inner loop, and the complete general positive/zero-strided
  odometer. Each emits its own primitive state machine without hot cursor objects, virtual calls,
  or per-element division/modulo. Completed 0005C vectorizes only direct contiguous runs and scalar
  broadcasts, uses scalar fallback for general odometers, and promises no gather. Complete 0005D
  owns optional materialization; full semantics do not promise universal vectorization.
- Before shared assignment, CPU analysis may compare direct access with internal contiguous
  materialization using copy cost, kernel benefit, reuse/fan-out, vendor eligibility, memory, and
  repeated-run expectation. A selected copy resource is declared exactly before assignment and
  lowered without changing the Model graph; shared Prepare remains CPU-blind.
- Task 0005A replaces the flat execution package with cohesive `.internal` root contracts and
  `memory`, `prepare`, `lowering`, `ir`, `codegen.emit`, `route.portable`, `cache`, `executable`,
  and `reference` packages. It creates no native/provider leaf. Minimal cross-package contracts are
  technically public but unsupported; only `CpuCapabilityProvider` is supported public API.
- Physical CPU buffers are declared only for post-fusion unit/partition boundaries. Planning's
  logical requirement for every graph value does not force a buffer declaration or slot; a
  same-unit virtual intermediate remains represented in graph and IR only.
- Shared Prepare adds only fail-closed cross-planned-partition completeness: a producer, when
  present, and every distinct external consumer must each declare the crossing value. Shared code
  does not inspect CPU units or choose fusion/materialization.
- CPU task 0001 exposes only `CpuCapabilityProvider`. Its stable `BackendId("cpu")` is identity,
  not availability or readiness, and `supports` remains unconditionally false until later tasks
  deliver and test exact executable semantic coverage.
- Historical task 0001 placed physical representations, typed cold arguments, specialization, and
  worker/range coordination in one flat execution package. Task 0005A preserves its validated
  representation/lifetime evidence while replacing that package and removing foundations with no
  current consumer. Run-owned buffers retain exact aligned native allocation and borrowed Model
  storage remains non-owning.
- Task 0001 binds each selected representation independently. The six Model data types map to
  their exact observable primitive-array carriers with retained carrier-relative byte offsets.
  When the matching carrier is unavailable, `CpuBufferArgument.Segment` retains the exact segment
  or slice without asserting native provenance; this covers genuine native segments and JDK 26
  read-only heap segments whose `heapBase()` is empty. Route-specific bound invocations must copy
  direct typed fields out of cold arrays and perform no storage discovery in the hot call.
- Historical task 0001 proved bounded worker and range behavior. Task 0005A removes those unused
  production types while retaining their evidence. Completed 0005C implements one explicit
  caller-owned CPU-private worker group, borrowed by finalized executables, and reintroduces no
  public executor facade, shared lifecycle, or general task system.
- Java 26 `java.lang.classfile.CodeBuilder` is the selected primary generation mechanism for all
  portable CPU computation kernels. This remains a current planning choice rather than an
  architecture invariant. Native vendor providers remain separate optional routes; the generator
  does not replace OpenBLAS, oneMKL, oneDNN, Accelerate, AOCL, or ZenDNN.
- Portable capability is truthful and fail-closed. Every selected executable Model semantic must
  have generated coverage before the portable milestone closes; metadata-only or zero-work view
  occurrences require no generated computation and unsupported executable semantics are not
  advertised.
- A default generated class is specialized to one canonical computation topology plus structural
  access regime, fusion form, types, selected route/strategy/configuration, numerical/determinism
  policy, and compatibility facts. Compatible concrete extents, element count, offsets, carriers,
  addresses, slots, graph identities, and run identity are cold-bound and excluded. Additional
  fixed-shape or unrolled preparation variants require explicit later evidence and consume a
  bounded specialization budget; fixed-trip code emitted inside one guarded artifact is not a
  separate variant.
- Cold binding resolves observable heap primitive carriers, exact `MemorySegment` values whose
  matching carrier is unavailable, and mixed signatures into direct typed entry-point arguments. Generated
  hot code performs no heap-base discovery, generic type check, route choice, cache lookup, or
  storage-kind, data-type, layout, vector, parallel, broadcast, or operation switch.
- The portable execution matrix is exactly scalar, vector, parallel-scalar, and parallel-vector.
  Generated units accept primitive start/end bounds; orchestration owns chunks outside the inner
  loop. Family-specific lowering may produce range, tile, partial-reduction, multi-pass, and combine
  bodies without changing those orthogonal compute/orchestration axes.
- Operation semantics are lowered once through family-specific typed lowerers. Shared scalar,
  vector, heap, segment, range, tile, and reduction emission primitives adapt that lowering to
  storage and execution modes; the design neither duplicates semantics across heap/native and
  scalar/vector modes nor concentrates all families in one generator.
- CPU analysis selects one complete valid route, access regime, representation, specialization,
  strategy, and materialization plan. Finalization after shared slot assignment deterministically
  realizes that selection. Without a trusted root it emits, verifies, and defines in memory; with
  a root it may verify and define a compatible stored hit or emit a miss. Runtime invokes only the
  cold-resolved handle.
- Task 0004 represents each selected portable buffer form through the existing
  `CpuKernelSpecialization.Argument` carrier/access vocabulary rather than a duplicate storage
  enum. One direct typed candidate source supplies deterministic preference order; CPU 0004 adds
  no registry, service locator, universal priority, tuning evidence, benchmark, or broad cost
  model.
- `CpuBorrowedBuffer` remains CPU 0001's non-owning `HostTensorStorage` representation/lifetime
  boundary and begins production portable cold-binding use in task 0004. Direct
  `HostTensorStorage`-to-argument binding is rejected because it would bypass Runtime's
  `BufferRepresentation` lifecycle boundary. The borrowed wrapper is not an artifact cache.
- Because generated method signatures vary, task 0004 keeps signature-specific invocation
  construction family-owned through a typed cold binder. The immutable portable executable
  strongly retains both `CpuGeneratedKernel` and its exact direct handle; a bound invocation
  retains direct typed carrier/workspace/worker fields and performs no hot argument
  classification.
- Historical task 0004 received an explicit trusted root and worker group. Task 0005A makes the
  root optional and removes the unused parallel object; neither becomes a public Config API.
- Deterministic generated bytes, structural identity, verification, and safe in-memory definition
  are mandatory. Filesystem persistence under an explicit trusted root is optional cold-path
  policy. Task 0005D recorded `KEEP_DISABLED`, so the default remains persistence-free. A hit can
  avoid emission but cannot preserve JIT machine code or profile.
- The generated-artifact key includes a generator schema/version and exactly every fact that
  changes bytecode or compatibility. Thread count or chunk size is keyed only when emitted code
  changes; otherwise it remains prepared parallel-execution configuration. Key construction and
  equality are deterministic and must be validated directly.
- The reset stores and compares the complete canonical IR/specialization structure plus generator,
  CPU/JDK/Class-File, generated-class, entry-name, and entry-descriptor compatibility facts. A path
  digest or checksum alone cannot establish a hit. Concrete compatible extents and every instance,
  graph, slot, address, run, handle, class-loader, and store identity are excluded by default.
- Compatible age never invalidates an artifact. The store has no time-to-live, hit-rate policy,
  automatic disk eviction, quota, background service, or valid-entry maintenance. Corrupt or
  incompatible bytes are never defined and may be replaced only through verified generation and
  forced temporary-file plus atomic-move publication.
- Equal requests share process-local compatible realization. A prepared executable reachable from
  `PreparedExecution` strongly retains the class, lookup, and exact handle for active use. The
  optional store is not a correctness owner and the reset has no legacy-schema migration reader.
- Checksums and structural verification detect accidental corruption but do not authenticate
  executable bytes. The artifact root is an explicit trusted-local-cache security boundary whose
  write isolation and administration belong to the caller/composition owner.
- The generated-class artifact store is distinct from the persistent tuning cache: the former
  reuses exact verified executable class bytes after route selection, while the latter records
  compatible route/configuration-selection evidence. Neither performs Runtime hot-path work.
- Matrix-multiplication candidates may include supported JDK Vector API species and strategy,
  unroll, tile, parallelism, and OpenBLAS thread configurations derived and pruned from target
  capabilities, workload facts, and budget.
- Scalar/vector and single/parallel are typed strategies inside the portable route. OpenBLAS and
  vendor integrations are typed peer-route configurations, never booleans in
  `Map<String,Object>`. Operation family does not key one universal configuration.
- Concrete CPU analysis generates a complete typed candidate set for each operation occurrence or
  partition. It filters by platform and provider availability, operation and attributes, data
  type, `Shape`, layout, exact numerical and determinism compatibility, and resource validity
  before comparing call overhead, safe workload heuristics, or a compatible tuning-cache entry.
- Common CPU analysis also owns exact division eligibility, exponent classification, numerical
  eligibility, and any selected `POW` realization plan. Emitters and vendor adapters consume those
  decisions and never invent independent fast-math behavior. Complete CPU 0005F admits same-typed
  FLOAT32/FLOAT64 binary/scalar DIV and scalar power with a direct fallback for every exponent.
  Exact typed positive or negative zero may produce positive one, positive one may select
  identity, positive two may select one typed multiply, and negative one may select one typed
  division while remaining semantic scalar power. Other integral exponents retain direct power:
  multiply chains and exponentiation by squaring have unproved intermediate rounding, overflow,
  and underflow behavior. `POW(0.5)` is not silently replaced by `SQRT`. Tensor exponents require
  compiler-owned immutable uniform-constant facts and are never inferred from Tensor storage or
  factory history.
- The selected numerical mode and every realization-changing `POW` plan participate in
  specialization/cache compatibility and the cold lowering manifest. The hot path performs no
  policy lookup. These rules apply equally to forward and compiler-generated gradient operations;
  CPU has no gradient-specific numerical policy.
- Exact semantics, determinism, data type, `Shape`, layout, alignment, lifetime, and provider
  eligibility are hard filters before cost comparison. CPU preparation then chooses routes and
  physical representations jointly over the relevant CPU dataflow and partition uses. It compares
  complete valid plan cost, including kernel time, Java/native call overhead, allocation,
  copying/materialization, packing or reorder work, and resource requirements; it must not select
  a locally fastest kernel when the resulting transition plan is worse.
- Native-backed CPU representations remain directly usable by scalar Java and Vector API routes.
  Crossing into an FFM provider call alone causes no copy. A preceding Java kernel may write
  directly into the selected native output buffer so a downstream compatible native consumer
  needs no intermediate materialization.
- Specialized opaque or prepacked layouts, CPU-to-device transfer, incompatible layout or
  alignment, and explicit heap export may still require distinct representations or explicit
  materialization. The canonical internal representation is not a claim that all model data or
  all external inputs are native.
- Compile and Planning remain logical and backend-neutral. Concrete CPU preparation owns route
  plus representation requirements; shared Prepare assigns and reconciles slots and explicit
  materializations across the complete prepared uses; Runtime executes that prepared schedule and
  tracks representation validity and residency.
- CPU uses canonical native internal buffers plus per-value, use-aware handling of borrowed and
  specialized representations. It does not introduce an all-Java versus all-native model mode.
- The Model `HostTensorStorage` contract is unchanged: `MemorySegmentStorage` remains borrowed and
  accepts compatible heap-backed and native-backed segments without gaining Runtime ownership,
  alignment, allocation, or route semantics.
- There is no fixed global vendor priority. A small workload may select Vector API or scalar even
  when a compatible native provider is available.
- The portable Class-File/Vector route remains the supported semantic baseline and fallback even
  after native routes land. OpenBLAS is only a narrow BLAS-compatible fallback. Accelerate,
  oneMKL/oneDNN, and AOCL/ZenDNN are exact-capability peer families compared by whole-plan cost,
  not a priority ladder and not separate `BackendId` values.
- Intel oneMKL and oneDNN remain separate low-level provider/integration boundaries because they
  serve standalone math and DNN-partition concerns respectively. AOCL-BLAS/AOCL-LibM and ZenDNN
  likewise remain distinct. CPU, not any provider leaf, owns capability reporting, coordination,
  fallback, route choice, and tuning.
- On Apple CPU, Accelerate BLAS, vDSP, and vForce are CPU routes. MPSGraph and custom Metal
  kernels remain exclusively Metal-backend routes after Planning selects Metal ownership; CPU
  never treats Metal as an internal optimization route.
- ARM has no hard-coded provider default. Apple Silicon may admit the Accelerate candidates above;
  other ARM systems retain portable code generation unless an explicitly verified provider task
  adds an eligible peer.
- On AMD CPU, portable scalar and Vector API routes remain available alongside AOCL-BLAS rather
  than becoming a lower-priority emergency path. AOCL-LibM enters only eligible sufficiently
  large vector-math candidate sets; ZenDNN remains a separate later DNN-partition option.
- Intel oneMKL/oneDNN and AMD AOCL/ZenDNN native candidates prioritize BFLOAT16 where the exact
  ABI, ISA/hardware, operation, and measured workload benefit are established. FLOAT16 enters only
  after Model task 0026 and the same exact evidence; it is not inferred from a 16-bit carrier or
  vendor availability.
- Vendor fast- or relaxed-math entry points require a future explicit numerical policy. They are
  not candidates under current exact/default semantics.
- Benchmarks, safe heuristics, and tuning may compare only candidates already eligible under the
  ordinary operation contract and explicit caller permission. Measurement never grants relaxed
  permission.

## Risks

- Leaking route selection into planning or splitting CPU routes into false backends.
- Exposing private CPU knobs through string dispatch, reflection annotations, a central registry,
  or a generic configuration language.
- Treating installed native libraries as a priority list rather than generating and filtering the
  valid candidates for the exact occurrence or partition.
- Moving capability truth, fallback, lifetime coordination, or tuning into a low-level vendor
  provider.
- Treating route choice and storage choice as coupled global modes, or choosing kernels locally
  without accounting for required representation transitions across consumers.
- Accidentally treating the CPU-private random mapping as a portable Model bitstream, omitting a
  generator-version or probability/state semantic from schema-19 identity, or making results
  depend on worker chunking.
- Advertising BFLOAT16 dropout merely because a `short` carrier exists, or accepting output/input
  or output/output aliasing that can mutate explicit state, value, or mask before validation.
- Preserving bridge-only or generic per-element scan/aggregate dispatch because semantic tests
  pass, or weakening the per-case near-parity threshold to accommodate a known slow generated
  loop.
- Narrowing the universal long/general carrier and layout contract while optimizing the proved
  dense heap-array int-index form, or allowing benchmark evidence to mutate Runtime behavior or
  ordinary preparation choices.
- Treating one passing movement or bridge-family measurement as evidence for another mapping,
  weakening a per-case gate after measurement, or comparing against a direct loop with different
  semantics or algorithmic work.
- Combining bridge-only indexing, scatter, fold, ordering, or dropout into one implementation task
  despite their distinct validation, workspace, multi-output, ordering, accumulation, and replay
  contracts, making a regression or failed parity case difficult to isolate.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
