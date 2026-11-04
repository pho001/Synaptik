# Compose existing [CPU](../../../glossary.md#term-cpu) [operation](../../../glossary.md#operation) emission

## Purpose and status

CPU0070 replaces the active CPU0066 [scalar](../../../glossary.md#term-scalar) → [vector](../../../glossary.md#term-vector) → worker sequence. Its original
non-authoritative replacement plan was independently approved and MAIN accepted on 2026-10-29,
not implementation or capability authorization. MAIN accepted this [planning](../../../glossary.md#term-planning)-only amendment after
targeted independent diff review on 2026-10-29; no implementation authorization follows.
[CPU0066](tasks/0066-shared-scan-body-emission-plan.md#result-and-independent-review)
remains a completed historical planning result; its [proposal](shared-body-emission-proposal.md)
and CPU0067–0069 direction are superseded. No source is changed or deleted by CPU0070.
CPU0071 and [CPU0072](tasks/0072-common-pointwise-sum-composition.md#result) are now Complete
after independent review and MAIN's recorded gates. Their source/measurement frontiers are closed;
the resulting CPU0073 exact design/contract packet and implementation start were user-approved
2026-10-30. Boole's authority APPROVE (0 P1/P2) and MAIN acceptance/verified [Ready](../../../glossary.md#term-ready) receipt made
CPU0073 the sole serial authorized source frontier at that time. Its implementation, independent
final reviews, MAIN named join and documentation closure are now Complete on 2026-10-30;
the [Result](tasks/0073-row-map-multi-reduction-fusion.md#result) owns exact [validation](../../../glossary.md#term-validation)/transfer limits.
Source/measurement frontiers None; performance unmeasured, no automatic successor or [commit](../../../glossary.md#term-commit)/push.

The [model](../../../glossary.md#term-model) is one existing family implementation with different generation-time [bindings](../../../glossary.md#term-binding).
Ordinary entries bind invocation operands and ranges; [Scan](../../../glossary.md#term-cumulative-scan) binds time slices, old/new [carry](../../../glossary.md#term-carry),
activity and saved histories around the same [operation](../../../glossary.md#operation) regions. A region means emitted code,
not a [runtime](../../../glossary.md#term-runtime) [callback](../../../glossary.md#term-callback), instruction object or second body [graph](../../../glossary.md#graph). Single-instruction-multiple-data
([SIMD](../../../glossary.md#simd)) math is reusable; parallel execution is a separate orchestration question.

The [target](../../../glossary.md#term-target) is general composition of existing ordinary emitters under Scan traversal and bindings,
not a second Scan math implementation. Body-family support is still [occurrence](../../../glossary.md#term-occurrence)-qualified:
ordinary forward support does not qualify Scan forward, forward does not qualify actual [Compiler](../../../glossary.md#term-compiler)
backward, and either scalar path does not qualify SIMD or parallel orchestration. CPU0072 closes
only the currently admitted [pointwise](../../../glossary.md#term-pointwise)/SUM domain. This direction mandates neither all-operation
support nor a speculative family-by-family implementation campaign or new task cascade.

## Contracts and retained capability

[Root Core invariants](../../../../ARCHITECTURE.md#core-invariants) and its
[scope index](../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../glossary.md#term-scope))</span> route authority to
[CPU backend routes](../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [backend glossary](../../../glossary.md#backend))</span>,
[Accepted ordinary symbolic-batch CPU contract](../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>,
[General differentiable scan](../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../glossary.md#general-differentiable-scan))</span>,
[Qualification, performance and migration gates](../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../glossary.md#term-qualification))</span>,
[Qualification and mechanism gates](../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Runtime preflight and implementation gates](../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../glossary.md#term-preflight))</span>,
[Actual first-order closure and arithmetic](../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Activity, raw identity and retained histories](../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings))</span>,
[Prepare lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../glossary.md#prepare); [lifecycle glossary](../../../glossary.md#lifecycle))</span> and
[Run lifecycle](../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../glossary.md#term-run))</span>.
These headings, not this plan, govern numerics, resources and the single [Runtime](../../../glossary.md#term-runtime) [owner](../../../glossary.md#term-owner).

Retain current static all-valid [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding), symbolic all-valid F32, and static/symbolic length-mode
F32/F64 forward and independently qualified actual [Compiler](../../../glossary.md#term-compiler) reverse. F32/F64 [mean](../../../glossary.md#term-mean) [FLOAT32](../../../glossary.md#term-float32)/[FLOAT64](../../../glossary.md#term-float64).
Keep dense actual-read/[scalar](../../../glossary.md#term-scalar)-read exceptions, formal/write guards and every unsupported case.
In particular, shared code does not admit all-valid symbolic F64, ordinary symbolic F64 [MATMUL](../../../glossary.md#matrix-multiplication--matmul),
low types, new body families/[layouts](../../../glossary.md#layout), native/[OpenBLAS](../../../glossary.md#term-openblas)/[Metal](../../../glossary.md#term-metal) [Scan](../../../glossary.md#term-cumulative-scan), higher-order [AD](../../../glossary.md#autograd) or [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration.
Ordinary `CpuMatmulIr.batchGeometry` remains F32/direct-scalar/sequential/no-[epilogue](../../../glossary.md#term-epilogue) only.
Compiler0015/0016 and Prepare0015 remain accepted boundaries, not new work in this plan.

## Grouped current family seams

This is a complete grouped inventory of the standard generated/direct-delegating family [dispatch](../../../glossary.md#term-dispatch),
not an all-[operation](../../../glossary.md#operation) [Scan](../../../glossary.md#term-cumulative-scan) support claim or semantics audit. Names below share the source prefix
`modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.
The [generator dispatch](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java)
(`generateClassBytes`, `generatePartialReduction`),
[lowering dispatch](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java),
[portable IR inventory](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java)
and [route encoding](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java)
provide the coverage index. A typed family [IR](../../../glossary.md#term-intermediate-representation-ir)'s encoded `CpuKernelIr` delegates to its existing
emitter; it is not a generic [runtime](../../../glossary.md#term-runtime) interpreter. `CpuPreparedExecutable` owns ordinary execution
and pre-write checks, not these math emitters. Resource descriptions below are family-specific,
not declarations that every form has the same workspace or parallel eligibility.

| Group and existing [owners](../../../glossary.md#term-owner) ([lowering](../../../glossary.md#lowering) / IR / emission) | [Binding](../../../glossary.md#term-binding), setup, resources and execution seam | Scan disposition |
|---|---|---|
| [Pointwise](../../../glossary.md#term-pointwise), comparisons/[masks](../../../glossary.md#term-boolean-mask), [scalar](../../../glossary.md#term-scalar) attributes and CAST: `CpuPartitionLowering` / `CpuKernelIr`, `CpuPointwiseOpcode` / generator scalar body, `emitVectorBody`, `CpuScalarEmitter`, `CpuCastEmitter`, `CpuVectorInstructionEmitter`, `CpuVectorMath` | Explicit input/output ordinals and access plans; [carrier](../../../glossary.md#term-storage-carrier)/address locals; [layout](../../../glossary.md#layout)/species/mask-bridge setup before loops; ordinary [element](../../../glossary.md#term-element) ranges. Cross-type CAST remains separately restricted. | CPU0072 shares current admitted composition; not all pointwise opcodes. |
| Affine [views](../../../glossary.md#term-view-meanings)/copies and non-affine movement: `CpuAffineLayoutLowering`, `CpuNonAffineMovementLowering` / `CpuAffineCopyIr`, `CpuDataMovementIr` / `CpuAffineCopyEmitter`, `CpuDataMovementEmitter` | Cold-composed address pairs or packed bases/[strides](../../../glossary.md#term-stride) and [occurrence](../../../glossary.md#term-occurrence) maps; raw copy/cursor loops; selected [materialization](../../../glossary.md#materialization) explicit. [PAD](../../../glossary.md#padding)/TILE/CONCAT/STACK/UNFOLD/SLICE_UPDATE keep their distinct mappings. | Existing Scan view bindings only; no general movement-family [admission](../../../glossary.md#term-admission). |
| [Gather](../../../glossary.md#gather)/one-hot: `CpuIndexingLowering` / `CpuIndexingIr` / `CpuIndexingEmitter` | Typed data/index/result roles, mapped output ranges; complete index [validation](../../../glossary.md#term-validation) precedes writer; no intrinsic scratch. | Inventory only. |
| Functional scatter and overlap fold: `CpuScatterLowering`, `CpuFoldLowering` / matching IR / matching emitters | Output-owned complete update/contributor traversal; index/uniqueness [preflight](../../../glossary.md#term-preflight) for scatter; optional exact-product scratch for floating scatter MUL; fold has no intrinsic scratch. | Inventory only; no hidden effect/body admission. |
| Ordering/top-K: `CpuOrderingLowering` / `CpuOrderingIr` / `CpuOrderingEmitter` | Whole [axis](../../../glossary.md#term-axis)-[slice](../../../glossary.md#slice) ranges, exact values/indices, two assigned [INT64](../../../glossary.md#data-type--datatype) merge regions per range; stable comparison/setup. | Inventory only. |
| Explicit-state initializers/[dropout](../../../glossary.md#dropout--inverted-dropout): `CpuRandomLowering` / `CpuRandomIr` / `CpuRandomEmitter` | Ordered state/value/mask roles and packed bases; state prologue, counters and range semantics, including empty range, remain owned here. | Inventory only; [RNG](../../../glossary.md#graph-rng-state--graphrngstate) remains excluded from pure Scan source. |
| Cumulative SUM/PROD: `CpuScanLowering` / `CpuScanIr` / `CpuScanEmitter` | Two boundaries, complete independent slices; sequential axis traversal, forward/reverse and inclusive/exclusive modes; no intrinsic scratch. | Not [general differentiable Scan](../../../glossary.md#general-differentiable-scan); no nested-scan grant. |
| Ordinary aggregate SUM/[MEAN](../../../glossary.md#term-mean)/PROD/[extrema](../../../glossary.md#saturation-pole-and-extremum)/[BOOL](../../../glossary.md#data-type--datatype) and [SUM-to-Shape](../../../glossary.md#sum-to-shape): `CpuAggregateLowering` / `CpuAggregateIr` / `CpuAggregateEmitter`, `CpuExactSumEmitter`, `CpuExactProductEmitter` | Output-cell ranges, normalized axes/[target](../../../glossary.md#term-target) mapping and actual counts; exact floating states use declared per-range slices, raw/no-axis forms do not become arithmetic. | CPU0072 shares [normalization](../../../glossary.md#term-normalization)/[ownership](../../../glossary.md#term-ownership-meanings) of existing SUM variants, not PROD/extrema/BOOL admission. |
| Arg-extrema, masked and advanced reductions: `CpuArgExtremaLowering`, `CpuMaskedReductionLowering`, `CpuAdvancedReductionLowering` / matching IR / `CpuArgExtremaEmitter`, `CpuMaskedReductionEmitter`, `CpuLogSumExpEmitter`, `CpuStatisticalReductionEmitter`, `CpuNormEmitter` | Complete cells/slices; index outputs, mask-before-data, multi-pass setup; exact-state scratch only for the selected masked/statistical/[L1](../../../glossary.md#term-l1) forms, not arg-extrema/[log-sum-exp](../../../glossary.md#log-sum-exp)/[L2](../../../glossary.md#term-l2). | Inventory only. |
| Partial modular reductions: `CpuPartialReductionLowering` / `CpuPartialReductionIr` / `CpuPartialReductionEmitter` | Separate generated `partialBody` and `orderedCombine`; primitive state array, fixed partial count and ordered combine. `CpuPreparedExecutable` binds their orchestration. | Not a reusable sequential-Scan SUM or parallel-body solution. |
| [Softmax](../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../glossary.md#softmax--log-softmax) and Layer/[RMS normalization](../../../glossary.md#rms-normalization): `CpuSoftmaxLowering`, `CpuTrailingNormalizationLowering` / `CpuSoftmaxIr`, `CpuTrailingNormalizationIr` / `CpuSoftmaxEmitter`, `CpuLayerNormEmitter`, `CpuRmsNormEmitter` | Whole slices/leading rows, ordered multi-pass math, optional affine roles and [finite](../../../glossary.md#term-finite)-input validation; Layer exact-mean scratch where required, RMS/softmax no intrinsic scratch. | Inventory only. |
| [Batch normalization](../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) [inference](../../../glossary.md#term-inference)/training: matching `CpuBatchNorm*Lowering` / matching IR / matching emitters | [Channel](../../../glossary.md#term-channel)/non-channel range selection; inference channel invariants; training complete channels, [statistic](../../../glossary.md#term-statistic) outputs and exact-state slice. | Inventory only; statistic transitions are not implicit Scan state. |
| [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and recognized epilogues: `CpuMatmulLowering` / `CpuMatmulIr` / `CpuMatmulEmitter` | Typed ordered operands/[bias](../../../glossary.md#term-bias)/output, geometry/range adapter; direct scalar/N-[vector](../../../glossary.md#term-vector) and 2×2 scalar/vector tiles, full K; species/layout setup mixed into current entries. | CPU0071 common bound [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) seam; ordinary variants retained, Scan admission unchanged. |
| [Conv2d](../../../glossary.md#convolution--conv2d)/3d and Pool2d/3d: matching lowerers / matching IR / matching emitters | Packed channel/window/group geometry and output-cell ranges; Conv scalar/output-width [SIMD](../../../glossary.md#simd) with borders/tails; Pool full-window scalar; emitter-local setup differs from shared carrier setup. | Inventory only. `CpuConv1dCompositionLowering`/`CpuPool1dCompositionLowering` delegate to existing 2d geometry, not new math families. |
| Attention and [losses](../../../glossary.md#loss): `CpuAttentionLowering`, `CpuLossLowering` / matching IR / `CpuAttentionEmitter`, `CpuLossEmitter` | Attention complete rows with assigned score scratch; losses role [alias](../../../glossary.md#aliasing-and-arena) maps, reduction/[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) ranges and target validation. Loss owns class-level emission; eligible unreduced [MSE](../../../glossary.md#mean-squared-error--mse) has existing SIMD, reduced/categorical forms remain scalar. | Inventory only; no recognition, scratch or [AD](../../../glossary.md#autograd) expansion. |

The fixed recurrent Java [kernels](../../../glossary.md#kernel)/executable and [OpenBLAS](../../../glossary.md#term-openblas) are separate execution seams, not
automatically inline regions or migration targets; reference oracles remain validation only. [General Scan](../../../glossary.md#general-differentiable-scan)'s
`CpuGeneralScanLowering`/IR/emitter retain composition bindings; the two Scan-only math subemitters
were retired by CPU0071/0072 after current forward/actual-reverse [qualification](../../../glossary.md#term-qualification).
The [matrix](../../../glossary.md#term-matrix-meanings) does not authorize an all-family rewrite; only the two bounded tasks below are detailed.

## Three different [bindings](../../../glossary.md#term-binding) and two emission lifetimes

Keep these spaces distinct; replacing every integer called a slot with one manager loses meaning.

| Space | Existing [owner](../../../glossary.md#term-owner) | Example and required adaptation |
|---|---|---|
| [JVM](../../../glossary.md#term-java-virtual-machine-jvm) local-variable slots | `CodeBuilder.allocateLocal(TypeKind)` | [Carrier](../../../glossary.md#term-storage-carrier) reference, address, [accumulator](../../../glossary.md#accumulator-type) and species locals; long/double occupy two JVM slots. Reuse this allocator; no generic slot manager. |
| Packed geometry indices | Family `Geometry.pack`, [Scan](../../../glossary.md#term-cumulative-scan) `Facts.pack` and invocation-owned `long[]` | [MATMUL](../../../glossary.md#matrix-multiplication--matmul) indices 4/5/6 denote M/K/N, not JVM locals or [Runtime](../../../glossary.md#term-runtime) [buffers](../../../glossary.md#term-buffer). Decode [invariant](../../../glossary.md#term-invariant) facts once; bind time-varying bases directly without per-step repacking. |
| Runtime buffer/workspace selections | [CPU](../../../glossary.md#term-cpu) analysis → shared [Prepare](../../../glossary.md#prepare) assignment → cold binder under one [`RunState`](../../../glossary.md#run-state--runstate) | Outer buffers and declared workspace; body-local value ordinals never select outer slots by numeric coincidence. Cuts address existing declared regions or demanded ports. |

[`CpuCarrierEmitter`](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java)
already accepts `parameterSlot` and `addressLocal` in `load`/`vectorLoad`/stores, and carrier/byte
address locals in `loadBytes`. Reuse these explicit generation-time mappings. Its fixed six-[layout](../../../glossary.md#layout)
block is reserved once at entry before later local allocation. Do not reserve it per [operation](../../../glossary.md#operation).

Split family entry adaptation from bound operation emission in existing owners. Invocation setup
decodes [immutable](../../../glossary.md#term-immutable) geometry, creates required typed layout/species facts and prepares invariant
[strides](../../../glossary.md#term-stride) outside time; the region consumes those facts plus selected operands, output ranges,
addresses and generation-time activity proofs. `CpuMatmulEmitter.emitDirectVector` currently creates
a fresh carrier emitter, calls `prepareVectorSpecies` and reads its fixed entry geometry. Calling
that method physically inside a generated time loop would repeat setup, even with collision-free
locals. A generation-time `prepare` call emits runtime instructions at its current placement.
Share setup per compatible [dtype](../../../glossary.md#data-type--datatype)/access policy; a cached [F32](../../../glossary.md#floating-point-precision-and-rounding) species cannot serve [F64](../../../glossary.md#floating-point-precision-and-rounding) or byte [masks](../../../glossary.md#term-boolean-mask).
Preserve initialization dominance through zero-trip, reverse, tail, fallback and final-copy edges.
Do not evaluate hoisted body operands/math at T=0 or when length-mode has no active time.

Proposed binding change (conceptual generation flow, not a new [API](../../../glossary.md#term-application-programming-interface-api)):

```text
before ordinary: entry ABI -> geometry/layout/species setup + family math
before Scan:     Scan bindings -> private Scan contraction/reduction loops
after ordinary: entry adapter -> shared invocation setup -> common bound family region
after Scan:     Scan setup    -> time/activity/state bindings -> same family region
```

Only setup/[lowering](../../../glossary.md#lowering) facts are generation objects. Executed code contains direct typed math and
loads/stores, no access object, [graph](../../../glossary.md#graph) walk, [callback](../../../glossary.md#term-callback), per-step allocation or interpreter.
Common math must not depend on `CpuGeneralScanEmitter.Shaped` or parse Scan stage diagnostics.
Scan translates existing Stage/[View](../../../glossary.md#term-view-meanings) facts into family inputs; it does not become the common owner.

## First foundation and stable immediate successor

Serial [DAG](../../../glossary.md#term-directed-acyclic-graph-dag): accepted CPU0064/0065 + retained Compiler0015/0016/Prepare0015 → CPU0070 accepted
[planning](../../../glossary.md#term-planning) → [CPU0071](tasks/0071-common-matmul-binding-foundation.md) →
[CPU0072](tasks/0072-common-pointwise-sum-composition.md#result): all Complete.
CPU0072's independent Class C source/[Javadoc](../../../glossary.md#term-javadoc)/docs/[provenance](../../../glossary.md#provenance) review and MAIN's combined [module](../../../glossary.md#term-module)/
Javadoc/named gates closed on 2026-10-29. Source frontier: None; no measurement budget is open.
The tasks migrated [ownership](../../../glossary.md#term-ownership-meanings)/[bindings](../../../glossary.md#term-binding) of existing algorithms, not [scalar](../../../glossary.md#term-scalar) → [vector](../../../glossary.md#term-vector) → worker stages.
No new named production type, package, enum, record or public [signature](../../../glossary.md#term-signature) was added.

CPU0071 is one complete [MATMUL](../../../glossary.md#matrix-multiplication--matmul) foundation: common [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [validation](../../../glossary.md#term-validation) and setup/body separation
in existing MATMUL/[carrier](../../../glossary.md#term-storage-carrier)/loop [owners](../../../glossary.md#term-owner); ordinary direct/tiled scalar/[SIMD](../../../glossary.md#simd) paths keep their math,
including CPU0065 [TANH](../../../glossary.md#term-tanh), while current scalar [Scan](../../../glossary.md#term-cumulative-scan) uses the common [contraction](../../../glossary.md#contraction-dimension) [owner](../../../glossary.md#term-owner). Its
indexed/cursor variants and contributor gating moved to common math, not a renamed Scan owner.
`CpuGeneralScanMatmulEmitter` was retired after forward/actual-reverse caller [qualification](../../../glossary.md#term-qualification).
Do not rewrite unrelated ordinary high-[rank](../../../glossary.md#term-rank)/vector forms just to force one universal loop.

CPU0072 completed shared [pointwise](../../../glossary.md#term-pointwise)/SUM composition for the currently admitted algebra through the
accepted binding/setup seam. Existing scalar/vector instructions retain their math; ordinary and
flat/Shaped Scan share load/instruction/store composition and facts-only [normalization](../../../glossary.md#term-normalization).
Sequential SUM now belongs to the existing common aggregate emitter, and
`CpuGeneralScanReductionEmitter` is retired after closure.
It does not add Scan SIMD capability or convert exact SUM to sequential SUM.
It closed the pointwise/SUM duplication left by CPU0071; no farther implementation brief is [Ready](../../../glossary.md#term-ready).

Prioritize this shared operand/address/geometry setup seam in existing family owners; Scan owns
traversal and temporal/state bindings, while common math serves operations inside and outside it.
Keep ordinary vector/worker algorithms and the current per-submission Job unchanged. Optional
Job reuse is not a prerequisite or blocker for CPU0071/0072 or later standard worker composition.
The current foundation remains scalar inline with no new per-step allocation/[callback](../../../glossary.md#term-callback) or route
[widening](../../../glossary.md#term-widening); future Scan SIMD/parallel use requires the separate gates below.

CPU0072's [source-grounded helper packet](tasks/0072-common-pointwise-sum-composition.md#files-current-package-map-and-proposed-helper-seams)
was implemented and independently approved, with MAIN gates accepted on 2026-10-29. If a
new named type proves necessary, stop with its precise source need, package/name/purpose/signature
and obtain **user** approval under Scan
[Qualification and mechanism gates](../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates).
The four CPU0066 proposed types and `PhaseState` are not requirements or preapproved alternatives.
No second captured body, [opcode](../../../glossary.md#term-opcode)-to-math [registry](../../../glossary.md#term-registry), generic access [runtime](../../../glossary.md#term-runtime) or [resource owner](../../../glossary.md#ownership-borrowing-and-lease) is added.

## CPU0072 extraction and evidence policy

CPU0072 implemented generation-only extraction in existing [owners](../../../glossary.md#term-owner) without a new type, [Runtime](../../../glossary.md#term-runtime)
[handoff](../../../glossary.md#term-handoff) or authority change. Its [Result](tasks/0072-common-pointwise-sum-composition.md#result)
records the frozen source [manifest](../../../glossary.md#term-manifest), [validation](../../../glossary.md#term-validation) and [provenance](../../../glossary.md#provenance); this policy grants no later work.
The frozen implementation retains MAIN-approved package-local [signatures](../../../glossary.md#term-signature) indexed by the brief.
`CpuPartitionLowering` already owns shared [opcode](../../../glossary.md#term-opcode)/immediate/[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-access [normalization](../../../glossary.md#term-normalization);
`CpuScalarEmitter` already owns typed [scalar](../../../glossary.md#term-scalar) math. Do not re-extract a second opcode switch.
Its implemented `validatePointwise` shares only current common [arity](../../../glossary.md#term-arity)/attrs/[dtype](../../../glossary.md#data-type--datatype)/broadcast checks;
ordinary broader instructions and each caller's provider/binder [admission](../../../glossary.md#term-admission) stay independent.
`CpuAggregateLowering.normalizedAxes/reductionForm/keepDimensions` replace duplicated ordinary
SUM/[MEAN](../../../glossary.md#term-mean) and [Scan](../../../glossary.md#term-cumulative-scan) SUM facts. Exact [Dimensions](../../../glossary.md#dimension), not sampled extents, identify named B. Preserve
`BatchGeometry.bind`'s actual [singleton](../../../glossary.md#term-singleton) filtering and resource guards; share no eligibility waiver.

`CpuClassFileKernelGenerator.emitBoundPointwise` owns the ordered composition, using existing
scalar/[vector](../../../glossary.md#term-vector) instruction emitters. Ordinary entries preload boundaries and store after the
instruction list; Scan binds demand-driven operands and exact producing-site stores per instruction.
Those are distinct existing load/store schedules, not a reason to move saved sites or preload
inactive operands. Availability and hoisted-origin [aliases](../../../glossary.md#aliasing-and-arena) stay generation facts. The common
orchestrator emits entry loads, then each bound load/math/store, then exit stores; adapters supply
no-op generation actions at unused phases. Nothing is called back at runtime.
The original flat `CpuGeneralScanEmitter.emit` also uses common composition: entry formal loads
and pre-step histories, per-instruction produced histories, then bank stores at exit. Its flat
path/bytes are retained, without Shaped conversion or moving either caller's saved sites.

`CpuAggregateEmitter.emitRepresentedSum` owns represented reset/fold/result control, not a renamed
Shaped-based reducer. The existing Scan adapter supplies selected/retained [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) mappings,
actual singleton branch, [extent](../../../glossary.md#term-extent) handles, activity-wrapped cell/contributor traversal and physical
sinks. These [bindings](../../../glossary.md#term-binding) contain no SUM arithmetic. Common fold order is output cell → typed +0
reset → legacy [invariant](../../../glossary.md#term-invariant) preload → selected contributors → producing [pointwise](../../../glossary.md#term-pointwise) sites → load
[accumulator](../../../glossary.md#accumulator-type)/contributor → typed fadd/dadd → store accumulator → producing result/saved sinks →
same-cell [epilogue](../../../glossary.md#term-epilogue). Loop/activity adapters bind coordinates before invoking each generation action;
row gates dominate both [producer](../../../glossary.md#term-producer) arithmetic and contributor loads, including ROW(1) and SHARED.
Length paths have no invariant contributor preload. [Fused](../../../glossary.md#specialization-and-fusion) producer availability is fresh per
contributor [scope](../../../glossary.md#term-scope), seeded only by proved invariant preload; the epilogue initially has only the
completed reduction result, not stale contributor locals.

Algorithm choice is explicit in the selected entry, not an accuracy enum or diagnostic-string
interpretation: ordinary `emit` retains `CpuExactSumEmitter`, guarded exact-state specializations
and rational MEAN; Scan
selects `emitRepresentedSum` with dtype, accumulator and generation-time raw/direct-singleton
proof flags. Never construct an under-resourced `CpuAggregateIr.Kind.SUM` to select sequential math.
Raw no-[axis](../../../glossary.md#term-axis) movement remains its existing [carrier](../../../glossary.md#term-storage-carrier) copy. Raw sumToShape singleton paths bypass add;
only the existing combined direct-singleton proof omits reset/contributor loops. Other raw paths
keep even their unused reset/local reservations for byte retention. B>1 with one active row is
not domain one. Retain result-wrapper [fan-out](../../../glossary.md#term-fan-out), restored epilogue coordinates and the distinct
per-time shared [carry](../../../glossary.md#term-carry) ADD. Exact state/empty resources stay as described below; Scan gains no
exact-state workspace, extra scratch, [buffer](../../../glossary.md#term-buffer) or compulsory copy.

Extract without changing emitted instruction/[constant](../../../glossary.md#term-constant)-pool/local-allocation order, including
currently unused reservations, setup placement, method [descriptors](../../../glossary.md#term-descriptor), generated names, resources
or binding. CPU0071 already demonstrates this constraint at the common [MATMUL](../../../glossary.md#matrix-multiplication--matmul) seam. Do not take
its static-MATMUL cutover as authorization to change pointwise/SUM or v9. Retain literal current
[canonical](../../../glossary.md#term-canonical-form) keys and both [identity](../../../glossary.md#term-identity-meanings) streams/[schema](../../../glossary.md#term-schema)/family 71, shaped-v4, MATMUL-v5, static length-v1/
retained-MATMUL-v2, symbolic-v3, combined-v9 and ordinary/fixed/pointwise controls. An unexpected
changed byte is a stop, not a name-normalized comparison or opportunistic namespace increment.
If retention necessarily prevents the required extraction, report the exact branch/[shape](../../../glossary.md#shape) conflict
and seek a narrow user-approved owning-contract cutover before implementation resumes. No such
conflict is currently established; no new production type or cutover is proposed.

After [Ready](../../../glossary.md#term-ready)/GO but before production edits, freeze a separate CPU0072 accepted-current baseline
from MAIN's [immutable](../../../glossary.md#term-immutable) before-source and actual old [artifacts](../../../glossary.md#term-artifact), preserving historical CPU0071 data.
Include its 226 controls plus 21 ordinary controls, capturing the 68 CPU0071 cutover rows at their
accepted NEW values. Use representative existing pointwise/SUM producer/reduction/tail, copy/
singleton, ordinary exact SUM/MEAN, scalar/vector pointwise and actual-[AD](../../../glossary.md#autograd) cases, not a new Cartesian
[qualification](../../../glossary.md#term-qualification) campaign or vector SUM route. Compare literal ClassFiles, streams, [ABI](../../../glossary.md#term-application-binary-interface-abi), numeric packs,
resource recipes and binding facts. [Capture](../../../glossary.md#term-capture-meanings) species so nonmatching vector hosts report unqualified
vector rows without suppressing scalar checks. A [finite](../../../glossary.md#term-finite) corpus requires independent review of all
changed generation/allocation paths; a source [hash](../../../glossary.md#term-hash) or current-code self-baseline is insufficient.

MAIN accepted conditional evidence transfer specifically for literal executable/identity equality
and unchanged ABI/resource/binding paths on 2026-10-29, using the compact table below;
actual equality must pass after extraction. No new numerical/performance qualification of existing
operations, per-[fixture](../../../glossary.md#term-fixture) certification ledger or additional [planning](../../../glossary.md#term-planning) task is required. Retain all
original limitations: historical parity gaps and incomplete performance dispositions do not block
pure extraction, and the extraction does not close them. No new timing protocol/budget is needed
for unchanged executed work. If work necessarily changes, stop at the compatibility/scope gate,
not a silent byte-normalization or a new benchmark. Any later authorized changed-work timing must
fix exact fixtures/algorithms/[seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/arms and entry boundaries, cold/admission/[kernel](../../../glossary.md#kernel)/allocation
separation, [warmup](../../../glossary.md#term-warmup)/stability observations, [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/order/[fork](../../../glossary.md#term-fork-meanings) caps and watchdog/no-retry limit.

### CPU0072 evidence transfer table

Read each row as preservation of its existing evidence, not a new support or parity claim.
CPU0071's [accepted Result](tasks/0071-common-matmul-binding-foundation.md#result) supplies the
current full-[module](../../../glossary.md#term-module)/named-join and frozen-control evidence; CPU0072 compares accepted-current
[artifacts](../../../glossary.md#term-artifact) rather than rerunning or improving historical performance campaigns.

| Existing group | Concrete current source/tests | Accepted evidence transferred and limits |
|---|---|---|
| Ordinary [pointwise](../../../glossary.md#term-pointwise) [scalar](../../../glossary.md#term-scalar)/[vector](../../../glossary.md#term-vector) | [Generated pointwise tests](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java), [paired structural oracle](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastStructuralOracleTest.java) | CPU0071 accepted module/control preservation; existing typed semantics and selected-entry hygiene. [Oracle](../../../glossary.md#term-test-oracle) is deliberately partial; vector is not claimed identical to scalar Java or optimally fast. |
| Exact ordinary SUM/[MEAN](../../../glossary.md#term-mean) | [Aggregate generation](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java), [dynamic-batch closure](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryDynamicBatchTest.java)<span data-glossary-navigation="true"> ([batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span> | CPU0071 accepted module and 21 ordinary controls include typed [F64](../../../glossary.md#floating-point-precision-and-rounding) SUM/MEAN/[AD](../../../glossary.md#autograd); retain exact-state/count/resource rules and existing performance dispositions, not new reduction timing or vector support. |
| [Scan](../../../glossary.md#term-cumulative-scan) sequential SUM and [fused](../../../glossary.md#specialization-and-fusion) [producer](../../../glossary.md#term-producer)/tail | [Symbolic/combined generated tests](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuSymbolicScanGeneratedKernelTest.java): `fusedSumToShapeKeepsProducerTailAndSingletonRepresentedIdentity`, raw-[singleton](../../../glossary.md#term-singleton)/activity/v9 proofs | Accepted module/controls; [CPU0064 Result](tasks/0064-scan-clean-java-generator-port.md#result) supplies bounded v9 reverse benefit at T4/F3/H2/B7,B32, with differing entry [ABI](../../../glossary.md#term-application-binary-interface-abi) and no other-[shape](../../../glossary.md#shape)/[dtype](../../../glossary.md#data-type--datatype)/[Engine](../../../glossary.md#term-engine) parity claim. No exact ordinary-SUM substitution. |
| Actual [Compiler](../../../glossary.md#term-compiler) AD | [Static binding tests](../../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMatmulBindingTest.java) and symbolic/combined tests above, including saved [activation](../../../glossary.md#term-activation) injection | CPU0071 module/named joins and existing actual sibling/clean-Java/raw saved-site checks. Its sole eight-row diagnostic is closed: unstable arms, no [C2](../../../glossary.md#term-c2) causal [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), length-reverse ~1.50 [F32](../../../glossary.md#floating-point-precision-and-rounding)/~1.42 F64 remains unresolved. |

CPU0072 met this conditional transfer gate for its literal 278-ClassFile control corpus and
unchanged ABI/resources/[bindings](../../../glossary.md#term-binding), with independent review of every changed generation/allocation
path. Original [captures](../../../glossary.md#term-capture-meanings) remain [immutable](../../../glossary.md#term-immutable); eight nondeterministic binding diagnostics have separate
verified before-source value witnesses, not waived [metadata](../../../glossary.md#term-metadata) or candidate-derived expectations.
The [Result](tasks/0072-common-pointwise-sum-composition.md#result) owns evidence and the original
failed module command plus targeted inventory correction. Species-mismatched vectors remain
unqualified; [finite](../../../glossary.md#term-finite) equality and unchanged source paths add no universal parity certification.

## Non-[fused](../../../glossary.md#specialization-and-fusion) [MATMUL](../../../glossary.md#matrix-multiplication--matmul) to [operation](../../../glossary.md#operation) boundary

Goal: preserve an unfused [dependency](../../../glossary.md#term-dependency) without a prepared [callback](../../../glossary.md#term-callback) or compulsory copy. Assume an
already admitted all-valid static [F32](../../../glossary.md#floating-point-precision-and-rounding) body, T=2/B=2, Xstep `[2,2]`, W `[2,1]`, [bias](../../../glossary.md#term-bias) `[1]`.
For the first step X=`[[1,2],[3,4]]`, W=`[[1],[2]]`, bias=`[1]`, source is
`q=Xstep@W; next=q.add(bias).tanh()`. Demand q's [residual](../../../glossary.md#scan-residual-port-and-history-demand) so its represented store is observable.
Analysis already has a cut or demanded-port [binding](../../../glossary.md#term-binding) for q; finalization must not invent one.

Conceptual generated sequence (not runnable source or a supported [API](../../../glossary.md#term-application-programming-interface-api)):

```text
outside time: bind invariant carriers/geometry/layouts and declared q region
at each t: bind Xstep and the current q history slice
          emit common MATMUL: finish every required q cell (full K)
          emit common ADD/TANH reading that same q region
          save demanded producing sites, then transition carry
```

The intermediate is q=`[[5],[11]]`; next is `[[tanh(6)],[tanh(12)]]` at represented F32 sites.
The next operation reads the completed intermediate directly. If q is private instead, its existing
declared workspace can be reused only after its inclusive last use and all old-[carry](../../../glossary.md#term-carry) reads;
no mandatory copy or T-fold allocation follows. A demanded history cannot be reused as ephemeral
scratch. Independent ordinary [unrolling](../../../glossary.md#term-unrolling) must retain the same saved sites and algorithm boundaries.

For length L=`[2,0]`, skip row 1's operand loads/arithmetic, keep its raw carry and store positive-zero
inactive history/emission. Actual reverse can contract ROW(1) into SHARED outputs: increasing active
K contributors and any-active SHARED guards must dominate loads and dependent math. Ordinary
MATMUL's entry lacks these predicates; merely renumbering [carrier](../../../glossary.md#term-storage-carrier) slots cannot implement this.
The common binding/proof seam must express them without a second [contraction](../../../glossary.md#contraction-dimension) algorithm.
Serial emission establishes full dependency completion today. A future parallel route would need
proved completion before a consumer or cut reuse; this example supplies no [synchronization](../../../glossary.md#term-synchronization) mechanism.

## Numerics, saved sites and v9 preservation

Keep ordinary exact-state SUM/[MEAN](../../../glossary.md#term-mean) and [Scan](../../../glossary.md#term-cumulative-scan) represented-sequential SUM as distinct existing
algorithms under common aggregate [ownership](../../../glossary.md#term-ownership-meanings). `CpuAggregateIr.Kind.SUM` remains exact; never bypass
its scratch guards or encode sequential arithmetic as an under-resourced exact sum. Common
[normalization](../../../glossary.md#term-normalization) uses explicit algorithm inputs, not an accuracy selector or diagnostic string.
Preserve raw [singleton](../../../glossary.md#term-singleton)/no-[axis](../../../glossary.md#term-axis) movement, empty [positive zero](../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../glossary.md#term-nan)/Inf/[signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) rules, [F64](../../../glossary.md#floating-point-precision-and-rounding)
[subnormals](../../../glossary.md#term-subnormal) and ordinary rational-MEAN [rounding](../../../glossary.md#term-rounding). Retain [F32](../../../glossary.md#floating-point-precision-and-rounding) five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs)/48-byte and bounded symbolic
F64 34-limb/280-byte state, static F64 domain-sized state, and zero exact-state bytes for Scan SUM.
[Cancellation](../../../glossary.md#term-cancellation)-sensitive `[1e20,1,-1e20]` distinguishes exact from sequential results; an ordinary
[unrolling](../../../glossary.md#term-unrolling) with exact reductions is not an exact [oracle](../../../glossary.md#term-test-oracle) for Scan sequential reductions.

Preserve exact bits produced at saved [TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid)/state sites and read them in actual reverse;
no replay, cross-saved [fusion](../../../glossary.md#term-fusion) or B×[parameter](../../../glossary.md#term-parameter) [gradient](../../../glossary.md#term-gradient) banks. Keep one ordered shared-[carry](../../../glossary.md#term-carry) ADD per
time, positional ties, both [seed](../../../glossary.md#term-seed) classes/traversals, previous-carry-before-step histories and original
time indexing. Activity guards apply to ROW(0), ROW(1) and SHARED, including hoisted/shared work.

Move and retain v9's direct stable entry carriers/temporal bases, explicit-B affine facts,
coefficient-proved zero scratch bases, cell-scoped address sharing and [pointwise](../../../glossary.md#term-pointwise) prefixes, indexed
[feature](../../../glossary.md#term-software-feature)/active-row contractions and bank-free scheduling. Existing
[`CpuGeneralScanStageSchedule.order`](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanStageSchedule.java)
checks producing dependencies, physical final-port hazards and inclusive all-B cut lifetimes.
Keep those temporal proofs; extract a common proof only where both callers actually need it,
without adding a second [optimizer](../../../glossary.md#term-optimizer). Do not enlarge scratch or add copies to simplify adapters.
Preserve fallback when a proof fails and exact resource declaration before assigned slots.

## [SIMD](../../../glossary.md#simd) [admission](../../../glossary.md#term-admission) and parallel composition gates

CPU0074 now supplies the bounded [selected-region mechanism](tasks/0074-row-fusion-simd-parallel.md)
and implemented, MAIN-accepted
[authority exception](../../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception).
Authority/docs APPROVE and MAIN's [Ready](../../../glossary.md#term-ready) receipt preceded implementation; CPU0074 is now Complete.
The [brief Result](tasks/0074-row-fusion-simd-parallel.md#result) records final [validation](../../../glossary.md#term-validation) and
the sole comparison, with steady-state profitability unqualified and [JIT](../../../glossary.md#term-just-in-time-compilation-jit)-pressure follow-up separate.
Only selected exec-v1 regions gain [feature](../../../glossary.md#term-software-feature)/output SIMD and row-partial workers; no all-[operation](../../../glossary.md#operation)
[Scan](../../../glossary.md#term-cumulative-scan) vectorization or feature-K splitting follows. Existing [vector](../../../glossary.md#term-vector) math is reused, with preferred
species/masked feature tails and exact selected-realization saved sites. [Scalar](../../../glossary.md#term-scalar) fallback is explicit.

Parallel composition was not an automatic property of inline math. MAIN's worker
audit supplies the bounded source facts:
[`CpuPreparedExecutable.bindCompatible`](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java) binds `RangeCall[]`
and per-range geometry; `Invocation.executeBound` calls `workerGroup.execute(calls)`.
[`CpuWorkerGroup.execute`](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroup.java)
clones the calls, allocates job/failure/[latch](../../../glossary.md#control-flow-graph) state, enqueues participants and synchronously joins;
workers reject nested submissions.
[`CpuGeneralScanPreparedExecutable`](../../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java)
retains one inline budget permit on unselected paths; selected parallel rows reserve actual
participant demand once around entry and lend it synchronously. Standard prepared/worker
calls are [runtime](../../../glossary.md#term-runtime) execution boundaries, not generation-time inline instruction [aliases](../../../glossary.md#aliasing-and-arena) or free
callbacks. CPU0074 implements the sole selected-region [synchronous](../../../glossary.md#term-synchronous) exception; all other repeating
[dispatch](../../../glossary.md#term-dispatch) is still prohibited. Ordinary workers and per-submission Job allocation stay unchanged.

CPU0074's coordinated patch tolerates existing Job/queue/failure/latch [metadata](../../../glossary.md#term-metadata), not arbitrary
body allocations. It retains ordinary symbolic-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) guards and the no-blanket-[widening](../../../glossary.md#term-widening) boundary.
The accepted authority diff and verified Ready brief preceded source work; the
[brief Result](tasks/0074-row-fusion-simd-parallel.md#result), not this plan, records evidence and pending gates.

The existing design/budget-[ownership](../../../glossary.md#term-ownership-meanings) gate must preserve one [RunState](../../../glossary.md#run-state--runstate), resource/[carry](../../../glossary.md#term-carry)/cut lifetimes,
full-K/output ownership, after-[producer](../../../glossary.md#term-producer) [preflight](../../../glossary.md#term-preflight)-before-writes, accessible storage, errors,
interruption/close, budget accounting and [quiescence](../../../glossary.md#term-quiescence). Scan's retained whole-entry inline permit
cannot be assumed safe around a worker submission that acquires participant permits: nested
acquisition can [deadlock](../../../glossary.md#term-deadlock). CPU0074 selects one sufficient whole-entry reservation, synchronously
lent without nested acquisition or concurrent caller math. Preserve
the no-nested-worker-submission rule. For an unfused producer → consumer, ordinary synchronous
worker join must complete before the consumer, carry transition or cut reuse; no mandatory copy
or new barrier follows merely from Scan. This is implemented and MAIN-accepted CPU0074,
not a retrospective CPU0071/0072 capability claim.

Scheduler metadata is distinct from avoidable body costs: duplicate arithmetic emission, per-step
[graph](../../../glossary.md#graph) access/re-[lowering](../../../glossary.md#lowering), geometry repacking, [tensor](../../../glossary.md#tensor)-[buffer](../../../glossary.md#term-buffer) allocation and T-fold graph/code
expansion remain excluded. No [singleton](../../../glossary.md#term-singleton), new job pool, mandatory cohort/phase/barrier, second
executor or Runtime [handoff](../../../glossary.md#term-handoff) is selected. Job reuse may be considered only as an optional,
separately approved evidence-driven follow-up, never a [dependency](../../../glossary.md#term-dependency) for normal worker composition.

### Deferred Job reuse evidence

Completed off-tree experiments changed no repository source. A used current fresh Job+[latch](../../../glossary.md#control-flow-graph);
B used fresh Job+monitor; C reused Job with the same monitor protocol. One timing [JVM](../../../glossary.md#term-java-virtual-machine-jvm), three
rounds, 2/4 workers and minimal/GEMM16/GEMM96 payloads showed no win over A: observed C medians
were 0.085–5.50% slower. These descriptive observations establish no statistical significance
and are not an end-to-end [Scan](../../../glossary.md#term-cumulative-scan) benchmark.

The separate one-JVM memory [run](../../../glossary.md#term-run) used three rounds × 100000 dispatches per arm/configuration
(300000 each; 1.8M total). Same-protocol B→C saved 135.855680/167.756693 B per [dispatch](../../../glossary.md#term-dispatch)
(42.54/41.78%; 38.868622/47.995575 MiB total for 2/4 workers). A→C's 57.49/54.42% also changes
completion protocol and is not reuse-only. C's shallow Job+wrapper [metadata](../../../glossary.md#term-metadata) was 160/192 B,
excluding [borrowed](../../../glossary.md#ownership-borrowing-and-lease) references' [targets](../../../glossary.md#term-target), payloads, [native memory](../../../glossary.md#aliasing-and-arena) and group footprint; queue/[lease](../../../glossary.md#term-lease)
allocations remain. Less [allocation churn](../../../glossary.md#term-allocation-churn) proves neither lower peak [RSS](../../../glossary.md#term-resident-set-size-rss)/[model](../../../glossary.md#term-model)/[tensor](../../../glossary.md#tensor) [RAM](../../../glossary.md#term-random-access-memory-ram) nor
Scan/[lifecycle](../../../glossary.md#lifecycle) [qualification](../../../glossary.md#term-qualification). Keep current scheduling; reuse is deferred, optional, not a blocker.
[Provenance](../../../glossary.md#provenance) only (not required dependencies): `/private/tmp/synaptik-job-ab.4JshRA/results.txt`
and `/private/tmp/synaptik-job-memory.0manXX/results.txt`; the durable decision and limits are above.

## [Validation](../../../glossary.md#term-validation), retirements and [handoff](../../../glossary.md#term-handoff)

The subsequent [row-map/multiple-terminal proposal](row-map-multi-reduction-fusion-plan.md) and
[CPU0073 brief](tasks/0073-row-map-multi-reduction-fusion.md) record user approval on 2026-10-30 of
exact names/[signatures](../../../glossary.md#term-signature)/[identities](../../../glossary.md#term-identity-meanings)/contract packet and implementation start. The
[backend authority packet](../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend))</span>
has Boole's independent APPROVE (0 P1/P2) and MAIN acceptance/[Ready](../../../glossary.md#term-ready) receipt on 2026-10-30.
One ordinary lowered unit or a region inside the existing [Scan](../../../glossary.md#term-cumulative-scan) unit shares current math, not a
per-step [Runtime](../../../glossary.md#term-runtime) unit/Job. CPU0073 is Complete after clean implementation, independent final
code/tests/[Javadoc](../../../glossary.md#term-javadoc)/evidence-transfer APPROVE, MAIN's named join and documentation closure.
Its Result preserves the failed full-[module](../../../glossary.md#term-module) [run](../../../glossary.md#term-run) separately from passing targeted closure and
accepted unchanged-outcome transfer. No performance claim, measurement grant, [commit](../../../glossary.md#term-commit)/push or
automatic successor follows. CPU0070/0071/0072 remain Complete; CPU0049 stays queued.

The task briefs own falsifiable semantic/resource/[identity](../../../glossary.md#term-identity-meanings) acceptance and focused commands.
Each needs clean Class C implementation and independent code/docs review, then its named
affected-module/[conformance](../../../glossary.md#term-conformance)/[Engine](../../../glossary.md#term-engine) [checkpoint](../../../glossary.md#term-checkpoint) once. Specialized clean Java and Class-File checks
compare matching algorithms, setup dominance and hot-loop/[dataflow](../../../glossary.md#term-dataflow) without hidden overhead.
CPU0071's implemented [compatibility exception](../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)
owns the narrow static [MATMUL](../../../glossary.md#matrix-multiplication--matmul) cutover. Unaffected bytes stay frozen as
a validation [target](../../../glossary.md#term-target), not assumed proof. No actual B/L/addresses/slots enter [canonical](../../../glossary.md#term-canonical-form) identity.
Unexpected changed namespaces or authority conflicts stop work.

CPU0070 granted no measurement. CPU0071 completed its sole USER-approved conditional [finite](../../../glossary.md#term-finite)
protocol after source review and named gates: eight rows, one [JVM](../../../glossary.md#term-java-virtual-machine-jvm), hard 60-second watchdog,
all before/after semantic checks PASS. The [Result](tasks/0071-common-matmul-binding-foundation.md#result)
records unsteady timing and unresolved generated/direct differences; no speed/parity claim.
That budget is consumed/closed: no repeat/retry/additional [fork](../../../glossary.md#term-fork-meanings). MAIN accepted CPU0072's conditional
byte-identical [evidence transfer above](#cpu0072-extraction-and-evidence-policy) on 2026-10-29,
with cold/[admission](../../../glossary.md#term-admission)/entry/[kernel](../../../glossary.md#kernel) costs separated. No inherited
CPU0067/0068 one-JVM budget, old timing selector, retry or five-fork campaign is carried forward.
Next analysis is [post-integration actual-Scan evaluation](#post-integration-actual-scan-evaluation)<span data-glossary-navigation="true"> ([evaluation glossary](../../../glossary.md#term-evaluation))</span>,
before optional Job reuse, not automatic further measurement.

Retirement is family-local: CPU0071 removed the Scan MATMUL subemitter and CPU0072 removed the
Scan reduction subemitter after [qualification](../../../glossary.md#term-qualification), sharing current [pointwise](../../../glossary.md#term-pointwise)/[normalization](../../../glossary.md#term-normalization) composition.
Keep Scan traversal/temporal [bindings](../../../glossary.md#term-binding)/roles/history/[lifetime](../../../glossary.md#term-lifetime) proofs, common
standard math and setup, ordinary workers, existing prepared [owners](../../../glossary.md#term-owner), all unsupported-case guards,
v9 optimizations and historical evidence. This plan records retirement; it grants no new execution.
Other [matrix](../../../glossary.md#term-matrix-meanings) families stay untouched until a concrete capability request reaches a stable frontier.

### Post-integration actual [Scan](../../../glossary.md#term-cumulative-scan) [evaluation](../../../glossary.md#term-evaluation)

The bounded independent-Java comparison now identifies repeated whole-stage traversals and delta
scratch traffic as a general [fusion](../../../glossary.md#term-fusion) opportunity. Its small-case raw matches are not general proof
or timing evidence; [CPU0073's proposal](row-map-multi-reduction-fusion-plan.md#motivation-and-evidence-limits)
records [provenance](../../../glossary.md#provenance)/limits; CPU0073 now supplies durable repository test oracles. Row fusion and optional
across-time [carry](../../../glossary.md#term-carry) promotion are distinct; the approved CPU0073 design selects row fusion only.
MAIN accepted the implemented bounded row fusion, independent reviews and named join on
2026-10-30. The source frontier is closed; performance is unmeasured and no measurement is granted.

CPU0071/0072/0073 are integrated. Historical split Scan backward trailed optimized matched Java;
whether that gap remains after CPU0073 is unmeasured. Any separately authorized evaluation starts with complete generated
and direct code, including the full reverse entry and its callers, not only shared math helpers or
another benchmark table. Compare loop nests/trips, address calculation/reloads, activity tests,
saved-value loads/stores, intermediate staging/copies, carry transitions and entry/[binding](../../../glossary.md#term-binding) costs.
Inventory avoidable work with exact source/ClassFile locations and algorithm/resource/saved-site
constraints; distinguish necessary semantic work from removable overhead before proposing repairs.
Use actual [Compiler](../../../glossary.md#term-compiler) reverse with matching [dtype](../../../glossary.md#data-type--datatype), traversal, [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), lengths, geometry, saved sites,
[carrier](../../../glossary.md#term-storage-carrier)/access policy and entry work. Ordinary exact reductions are not an [oracle](../../../glossary.md#term-test-oracle) for Scan's
represented-sequential folds. Keep forward, backward and any future parallel [qualification](../../../glossary.md#term-qualification) distinct.

CPU0071's diagnostic length-reverse new/direct medians were about 1.50 [F32](../../../glossary.md#floating-point-precision-and-rounding) / 1.42 [F64](../../../glossary.md#floating-point-precision-and-rounding). Some arms
warmed unstably; no [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names) [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) prove a [C2](../../../glossary.md#term-c2) cause. Those observations identify an unresolved
question, not a universal optimal-Java parity baseline, new [regression](../../../glossary.md#term-regression) or causal diagnosis.
First use retained source/ClassFiles and evidence; no inherited timing allowance or automatic [JVM](../../../glossary.md#term-java-virtual-machine-jvm),
retry, task, implementation or campaign follows. If timing is needed to discriminate a concrete
code hypothesis, obtain a fresh user-approved [finite](../../../glossary.md#term-finite) protocol. Any later speed/coordinator+worker
allocation/coordination-share evaluation must separate cold/[admission](../../../glossary.md#term-admission)/entry/[kernel](../../../glossary.md#kernel) work and state
measurement/stability limits. Optional Job reuse remains later and is not the presumed explanation.

Documentation profile: [Planning](../../../glossary.md#term-planning), with General and Example format for the conceptual boundary.
Implementation updated affected Javadocs and the [CPU](../../../glossary.md#term-cpu) [backend](../../../glossary.md#backend) explanation; final independent
documentation review found no new glossary meaning or authority/[ADR](../../../glossary.md#term-architecture-decision-record-adr) change.
CPU0070 changes no Java/[Javadoc](../../../glossary.md#term-javadoc), glossary meaning, contract or build.
Original CPU0070 independent review was APPROVE with no unresolved findings; MAIN accepted that
planning delivery on 2026-10-29. Separately, MAIN accepted this Job/composition planning amendment
after targeted independent diff review on 2026-10-29, with no unresolved findings.
CPU0071 is Complete on 2026-10-29: implementation, corrected independent source/Javadoc review,
MAIN-accepted [module](../../../glossary.md#term-module)/named gates, sole diagnostic and final shared-document pass are recorded in
its Result. CPU0072 is Complete after Banach's final independent APPROVE/P2 closure and MAIN's
combined module/Javadoc/named acceptance. CPU0073 is also Complete after independent final review,
accepted targeted closure/transfer, MAIN's named join and independent documentation closure.
Source/measurement frontiers None; no automatic successor or performance claim.
Deferred Job reuse and separate Scan [SIMD](../../../glossary.md#simd)/parallel gates are unchanged. CPU0072 ran no measurement;
all prior budgets remain closed, and complete backward-code comparison precedes optional Job reuse.
