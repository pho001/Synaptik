# CPU0074 proposed row-[fusion](../../../glossary.md#term-fusion) execution boundary

## Purpose and approval boundary

This [planning](../../../glossary.md#term-planning) packet specifies one extension of CPU0073, not a new architecture authority.
[ARCHITECTURE.md](../../../../ARCHITECTURE.md) and its incorporated contracts still govern.
MAIN accepted the authority patch and independent APPROVE (0 P1/P2) before the verified [Ready](../../../glossary.md#term-ready)
receipt on 2026-10-30. The [CPU0074 brief](tasks/0074-row-fusion-simd-parallel.md#result) now records
implemented and MAIN-accepted functional capability, Complete; its source/measurement frontier is closed.
MAIN addendum accepted 2026-10-30 after focused independent APPROVE of the two exact generation/
cold-[binding](../../../glossary.md#term-binding) seams below. No new type, [runtime](../../../glossary.md#term-runtime) protocol or capability-completion claim follows.
MAIN/focused-reviewer correction accepted 2026-10-30: [Scan](../../../glossary.md#term-cumulative-scan) minimum-row policy transport
replaced its unimplemented seven-[argument](../../../glossary.md#term-argument) overload.
MAIN/Boyle byte-seam addendum accepted 2026-10-30: the two generation-only checked-segment
methods below are implemented. Final [CPU](../../../glossary.md#term-cpu)/[Javadoc](../../../glossary.md#term-javadoc) and the sole comparison completed on frozen
source; functional acceptance does not qualify steady-state profitability or authorize rollout.
This planning document is not authority. No new top-level production type,
public numerical mode, executor, global Job, job pool or general [pointwise](../../../glossary.md#term-pointwise) refactor is proposed.

## Source-bound mechanism

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
The pre-CPU0074 source facts below motivated only the accepted additions; they are not a claim
that the old [scalar](../../../glossary.md#term-scalar)-only guards still describe selected exec-v1 implementation.

| Current [owner](../../../glossary.md#term-owner) / [symbol](../../../glossary.md#term-symbol) | Source fact and proposed change |
|---|---|
| `internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan` | Already owns execution strategy, selectedRangeCount and minimumElementsPerWorker. Retain worker policy here; add none to [canonical](../../../glossary.md#term-canonical-form) row [IR](../../../glossary.md#term-intermediate-representation-ir). |
| `internal.cache.CpuKernelSpecialization` | Scalar-only row/[Scan](../../../glossary.md#term-cumulative-scan) guards and fixed entry type need a selective optional emission [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) and [ABI](../../../glossary.md#term-application-binary-interface-abi) branch. Preserve old constructor delegates and scalar bytes. |
| `internal.codegen.emit.CpuRowMapReductionEmitter.emitBound` | Generation-only full-row orchestration; add a bounded range variant, retaining scalar entry behavior and local [validation](../../../glossary.md#term-validation). |
| `CpuVectorInstructionEmitter.emit` / `CpuVectorMath` | Existing typed [vector](../../../glossary.md#term-vector) [pointwise](../../../glossary.md#term-pointwise) math, including [TANH](../../../glossary.md#term-tanh); [SIGMOID](../../../glossary.md#term-sigmoid) currently has VectorForm.NONE. Reuse only admitted opcodes, with scalar fallback for unsupported regions. |
| `CpuMatmulEmitter.emitDirectVector` / private vector helpers near 788 | Existing output-[feature](../../../glossary.md#term-software-feature) vectorization keeps full feature K; expose the exact common vector math seams below rather than copy arithmetic. |
| `CpuAggregateEmitter.emitRepresented*` | Current shared Scan SUM seams are scalar, separate from exact ordinary SUM. Add only their represented vector counterparts below. |
| `CpuCarrierEmitter.vectorLoad/vectorStore` | Existing [contiguous](../../../glossary.md#term-contiguous)/broadcast loads and contiguous stores, not arbitrary segment gathers. Add masked feature-tail overloads; stage only proved small transposes. |
| `CpuGeneralScanEmitter.emitRegionalStage` | One selected regional marker and emitted T loop; retain temporal [bindings](../../../glossary.md#term-binding)/activity and generated caller-side merge/[carry](../../../glossary.md#term-carry). No [runtime](../../../glossary.md#term-runtime) stage interpreter. |
| `CpuGeneratedKernel.lookup` / `hiddenLookup()` / `lossEntryPointFor` | Retains the defining [lookup](../../../glossary.md#term-lookup), exposes it through the existing public hiddenLookup(), and explicitly resolves [loss](../../../glossary.md#loss) helpers cold. Extend named accessors, not a [registry](../../../glossary.md#term-registry) or per-step lookup. |
| `CpuPreparedExecutable.bindCompatible/Invocation` | Binds RangeCall[] and geometry once; workers synchronously join. Extend ordinary regional partial/merge binding, not ordinary exact reductions. |
| `CpuGeneralScanPreparedExecutable.Invocation.executeBound` | Currently holds budget.acquire(1) around the whole generated entry. Replace only selected parallel [admission](../../../glossary.md#term-admission) with one adequate [borrowed](../../../glossary.md#ownership-borrowing-and-lease) reservation. |
| `CpuWorkerGroup.execute` / `CpuConcurrencyBudget.Lease` | RangeCall is package-private; [dispatch](../../../glossary.md#term-dispatch) clones calls, creates Job/failure/[latch](../../../glossary.md#control-flow-graph)/queue [metadata](../../../glossary.md#term-metadata) and joins. Add same-budget reservation validation and a no-acquire dispatch overload. |

Final cold metadata also uses existing `CpuFusionDecision`, `CpuFusionProfitabilitySelector`
and plan validation: `ROW_REGION_STATE` labels ordinary selected scratch only with matching
row IR/emission and no Scan/recurrent owner. Internal row consumers stay in their indivisible
unit; ordinary pointwise accounting/scoring is unchanged. This is not a new type, seam or authority.

## Proposed exact internal declarations

These are declaration excerpts, not runnable Java or supported application APIs. Unchanged names
and constructors remain. Additional named types/common [signatures](../../../glossary.md#term-signature) require MAIN approval.
No `Realization` is added to `CpuRowMapReductionIr`; its structuralKey remains unchanged.

```java
// Nested in existing internal.cache.CpuKernelSpecialization: emitted layout facts, not policy.
public record RowFusionEmission(int maximumPartialCount,
    List<Integer> invariantTransposeValues, List<Integer> temporalTransposeValues,
    int stepOperandCount, int stepFrameLongCount) {}
// Append Optional<RowFusionEmission> rowFusionEmission to the specialization record.
// Keep every old constructor, including its former canonical signature, delegating empty.

// Existing internal.codegen.emit.CpuRowMapReductionEmitter; generation only.
static void emitRangeBound(CodeBuilder code, CpuRowMapReductionIr ir,
    int rowIndex, int rowBegin, int rowEnd,
    int[][] valueLocals, int[][] terminalAccumulatorLocals,
    Runnable init, Consumer<Runnable> activeRow, IntConsumer emitRowSite,
    IntConsumer emitTerminalStep, Runnable finish);

// Existing CpuMatmulEmitter; all locals/loaders are matching FloatVector or DoubleVector.
static void emitVectorInit(CodeBuilder code, DataType type, int accumulator);
static void emitVectorContributorStep(CodeBuilder code, DataType type, int accumulator,
    Runnable left, Runnable right);
static void emitVectorFinish(CodeBuilder code, DataType type, int accumulator,
    IntConsumer sink);
// Existing CpuAggregateEmitter; represented Scan SUM only, never CpuExactSumEmitter.
static void emitRepresentedVectorInit(CodeBuilder code, DataType type, int accumulator,
    boolean directSingleton);
static void emitRepresentedVectorContributorStep(CodeBuilder code, DataType type,
    int accumulator, boolean rawSingleton, Runnable load);
static void emitRepresentedVectorFinish(CodeBuilder code, DataType type, int accumulator,
    IntConsumer sink);

// Existing CpuCarrierEmitter: overload current full forms; maskLocal holds matching VectorMask.
void vectorLoad(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
    boolean broadcast, boolean intAddress, int maskLocal);
void vectorStore(DataType type, CarrierAccess access, int parameterSlot, int addressLocal,
    int valueLocal, boolean intAddress, int maskLocal);
// Existing CpuCarrierEmitter; package-private generation-only checked-segment byte access.
void vectorLoadBytes(DataType type, int carrierSlot, int byteAddressSlot, int maskLocal);
void vectorStoreBytes(DataType type, int carrierSlot, int byteAddressSlot,
    int valueLocal, int maskLocal);

// Existing CpuGeneratedKernel: resolve named generated methods once during cold binding.
public MethodHandle ordinaryRowRangeEntryPoint(); // exact specialization.entryType()
public MethodHandle ordinaryRowMergeEntryPoint(); // exact specialization.entryType()
public MethodHandle scanRowRangeEntryPoint();
// scanRowRange helper type: (MemorySegment[], long[], long begin, long end,
//                           long partialByteOffset)void
// Existing package-private CpuGeneralScanEmitter; generates that helper through its private binder.
void emitRowRange(CodeBuilder code, CpuKernelSpecialization specialization, CpuGeneralScanIr ir);
// Existing ordinary symbolic CpuPreparedExecutable constructor: append unit policy/borrowed group.
public CpuPreparedExecutable(PreparedMemoryPlan memoryPlan, List<BufferSelection> selections,
    CpuGeneratedKernel artifact, CpuPortableRoutePlan route,
    Optional<WorkspaceSelection> workspace, CpuConcurrencyBudget budget,
    int selectedRangeCount, long minimumElementsPerWorker, CpuWorkerGroup workerGroup);
// Preserve the old six-argument constructor, delegating (1, 1, null).

// Nested in existing internal.executable.CpuGeneralScanPreparedExecutable.
public static final class StepDispatch {
    private StepDispatch(MemorySegment[] operands, long[] frame,
        CpuWorkerGroup.RangeCall[] calls, CpuWorkerGroup workerGroup);
    public MemorySegment[] operands();
    public long[] frame();
    public void execute();
}
// Existing CpuGeneralScanPreparedExecutable: old six-argument constructor delegates (null,1).
public CpuGeneralScanPreparedExecutable(PreparedMemoryPlan plan, List<BufferSelection> buffers,
    Optional<WorkspaceSelection> workspace, Facts facts, CpuGeneratedKernel artifact,
    CpuConcurrencyBudget budget, CpuWorkerGroup workerGroup, long minimumElementsPerWorker);
// No retention of the unimplemented seven-argument overload is required.
// Existing executable-package seams; no public RangeCall or generic callback is added.
void CpuWorkerGroup.execute(RangeCall[] calls, CpuConcurrencyBudget.Lease reservation);
boolean CpuConcurrencyBudget.Lease.covers(CpuConcurrencyBudget budget, int demand);
```

`RowFusionEmission` snapshots nonnull ordinal lists; lists are disjoint, duplicate-free and refer
to proved source values. Counts are checked nonnegative (maximumPartialCount positive), fixed
before declarations and consistent with existing unit/species facts. It records code/resource
[shape](../../../glossary.md#shape) only: no route selection, worker [ownership](../../../glossary.md#term-ownership-meanings) or [mutable](../../../glossary.md#term-mutable) data. No [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) for CPU0073 [scalar](../../../glossary.md#term-scalar).
The generator derives [feature](../../../glossary.md#term-software-feature) [masks](../../../glossary.md#term-boolean-mask) and block [layout](../../../glossary.md#layout) from authenticated region accesses/species.

emitRowRange is generation-only and reuses the private temporal/row binder, not a [runtime](../../../glossary.md#term-runtime) [callback](../../../glossary.md#term-callback).
The ordinary symbolic overload transports existing unit-owned policy and [borrowed](../../../glossary.md#ownership-borrowing-and-lease) workers to
cold numeric B [binding](../../../glossary.md#term-binding); no per-B [artifact](../../../glossary.md#term-artifact) or worker policy in [canonical](../../../glossary.md#term-canonical-form) row [IR](../../../glossary.md#term-intermediate-representation-ir) is added.
The [Scan](../../../glossary.md#term-cumulative-scan) overload validates positive minimumElementsPerWorker (otherwise IllegalArgumentException).
Finalizer transports unit.minimumElementsPerWorker() cold; keep policy in the unit, not IR/Facts/
[specialization](../../../glossary.md#term-specialization). Its old six-[argument](../../../glossary.md#term-argument) constructor delegates (null,1), preserving inline behavior.

vectorLoadBytes/vectorStoreBytes require checked MEMORY_SEGMENT [FLOAT32](../../../glossary.md#term-float32)/[FLOAT64](../../../glossary.md#term-float64), [target](../../../glossary.md#term-target) preferred
species, LONG byte-offset local and matching mask/value species. Emit direct native-byte-order
masked fromMemorySegment/intoMemorySegment; disabled [lanes](../../../glossary.md#term-lane) access no storage. Scalar broadcasts
use existing loadBytes plus broadcast. Keep old [element](../../../glossary.md#term-element) methods and array paths literally unchanged;
migrate only selected exec-v1 Scan byte addresses and ordinary private partial scratch. Removing
the emitted /width*width roundtrip is the rationale, not proof of [JIT](../../../glossary.md#term-just-in-time-compilation-jit) behavior or a slowdown/speed
claim. No new type/layer, runtime entry [ABI](../../../glossary.md#term-application-binary-interface-abi), resource or global [schema](../../../glossary.md#term-schema) bump follows. New selected
exec-v1 is unreleased: final inspection/performance must use a fresh private artifact root so old
candidate [cache](../../../glossary.md#term-cache) entries cannot hide the changed generated shape.

The range emitter retains caller-allocated tables and generation-only callbacks. Scalar locals
are typed primitive cells; [vector](../../../glossary.md#term-vector) bodies use REFERENCE locals by fixed feature chunk. The caller
selects and validates the table convention before emission; no runtime [dtype](../../../glossary.md#data-type--datatype)/strategy [dispatch](../../../glossary.md#term-dispatch).
Rows advance from rowBegin inclusive to rowEnd exclusive. Initialization/finish occur once;
zero rows still publish valid required [identities](../../../glossary.md#term-identity-meanings) through the complete-unit wrapper.

Vector init uses the selected target dtype's preferred species; specialization validates matching
preferred bits and [accumulator](../../../glossary.md#accumulator-type)/load/mask species, never arbitrary species. It creates typed +0
unless directSingleton omits reset. Contributor loaders leave
matching vectors; [MATMUL](../../../glossary.md#matrix-multiplication--matmul) emits separate per-lane MUL then ADD, never [FMA](../../../glossary.md#term-fused-multiply-add-fma). Represented SUM emits
typed ADD or raw [singleton](../../../glossary.md#term-singleton) assignment, with no synthetic +0 for a raw copy. Finish passes the
completed reference local to its generation-only sink without extra arithmetic. Source scalar
math and exact ordinary SUM/[MEAN](../../../glossary.md#term-mean) remain unchanged. Feature-tail masks exclude storage access
and [publication](../../../glossary.md#term-publication) outside the real cells; unused vector lanes never enter a horizontal reduction.

## [SIMD](../../../glossary.md#simd) [dimension](../../../glossary.md#dimension), temporal staging and [oracle](../../../glossary.md#term-test-oracle)

Select [feature](../../../glossary.md#term-software-feature)/output [lanes](../../../glossary.md#term-lane) within a single active row. Gate the row before any row/[seed](../../../glossary.md#term-seed)/history
load and dependent math. Use masked [contiguous](../../../glossary.md#term-contiguous) feature loads/stores, [scalar](../../../glossary.md#term-scalar) broadcast operands
and existing [vector](../../../glossary.md#term-vector) opcodes. Feature contractions traverse the original K in order; terminal
vectors traverse R in order within each worker. No vector reduction across different features.

Primary actual [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) reverse has F=3, H=2: compute its two delta cells once from exact saved
primals; vectorize dX(F3), dH(H2), dW(three H2 vectors), dU(two H2 vectors), db(H2).
Extract needed delta feature lanes once to scalar locals for ordered feature-K broadcasts;
do not assemble arbitrary strided vectors with per-lane withLane chains.
W `[3,2]` and U `[2,2]` need [canonical](../../../glossary.md#term-canonical-form) transposed right operands for dX/dH. Emit bounded raw
[transpose](../../../glossary.md#term-transpose) copies into predeclared scratch: 6+4 cells = 40 [F32](../../../glossary.md#floating-point-precision-and-rounding) / 80 [F64](../../../glossary.md#floating-point-precision-and-rounding) bytes before [alignment](../../../glossary.md#term-alignment).
If their [bindings](../../../glossary.md#term-binding) are [invariant](../../../glossary.md#term-invariant) across T, populate once per [run](../../../glossary.md#term-run) under an any-active gate.
SHARED history or changing cuts instead populate at the exact active original time, after
their [producer](../../../glossary.md#term-producer) and before [dispatch](../../../glossary.md#term-dispatch); reuse the same ten-cell area only after all readers join.
If stability/access/[lifetime](../../../glossary.md#term-lifetime) cannot be proved, the vector region falls back, not stale reuse.
No staging runs for T=0/B=0/all-inactive; no padded/history reads are justified by zero fill.

This is generic bounded row-region binding, not a second RNN emitter. Fixed-feature [provenance](../../../glossary.md#provenance),
unit-[stride](../../../glossary.md#term-stride)/broadcast mapping or the explicit small transpose proof must cover every access.
Unproved ROW(1), dynamic feature strides and unsupported vector opcodes retain scalar fallback.
An independent optimal Java oracle for each realization must include the same necessary transpose,
masked feature algorithm, full feature-K order and exact saved sites. Compare complete code/[dataflow](../../../glossary.md#term-dataflow);
if a better current-need clean-Java [shape](../../../glossary.md#shape) avoids staging, justify the retained cost or revise
the proposal before coding. No claim that masked H2/F3 is faster on every species or small B.
The current optimal reference retains twelve typed [carry](../../../glossary.md#term-carry) cells across T while preserving one
distinct ADD per active time. Generator across-time carry promotion remains a non-goal; keep
the reference intact and disclose the dataflow difference in performance interpretation.

## [Ownership](../../../glossary.md#term-ownership-meanings), budget and [synchronous](../../../glossary.md#term-synchronous) [dispatch](../../../glossary.md#term-dispatch)

[Mental model](../../../glossary.md#term-mental-model): cold plan → cold [run](../../../glossary.md#term-run) [binding](../../../glossary.md#term-binding) → emitted active-time setup → workers/join →
generated terminal merge/[carry](../../../glossary.md#term-carry) → next time. Only the generated outer loop owns time progression.
StepDispatch bridges that one selected region to prebound typed calls; it is not a body [callback](../../../glossary.md#term-callback)
[registry](../../../glossary.md#term-registry). Its public accessors return its own [mutable](../../../glossary.md#term-mutable) run arrays without copying so the generated
entry can bind them once. Generated code updates only current operand references/time/base facts;
it does not repack geometry, rediscover [views](../../../glossary.md#term-view-meanings) or resolve methods per step.

Existing unit fields fix maximum ranges W and minimum row work m. At cold run binding use
N=min(W,ceil(R/m)) for R>0, otherwise zero; [partition](../../../glossary.md#partition) [0,R) by quotient/remainder in ascending
range index. This numeric rule is fixed before execution, [constant](../../../glossary.md#term-constant) across time, not a value-based
profitability choice. If N<2, execute inline under one permit without a worker submission.
Stage/partial [layout](../../../glossary.md#layout) reserves W bounded slots; no B×[parameter](../../../glossary.md#term-parameter) bank. Each partial has typed
terminal cells plus an eight-byte validity flag, initialized each dispatch. Invalid empty/inactive
partials are skipped, not merged as artificial +0 contributors. Geometry-[singleton](../../../glossary.md#term-singleton) raw reductions
keep their copy path; one active row in a larger B is not a geometry singleton.

For selected parallel [Scan](../../../glossary.md#term-cumulative-scan) acquire max(1,N) from the exact finalizer budget before entry and
install that reservation privately in this invocation's StepDispatch for this synchronous call.
Caller setup, merge and carry execute under it. Dispatch verifies same-budget [identity](../../../glossary.md#term-identity-meanings), live
status and sufficient demand, then calls the overload which neither acquires nor [releases](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) permits.
The caller performs no concurrent math while workers run. Started workers always quiesce before
return/throw; only then may caller merge, advance carry/time, reuse scratch or [release](../../../glossary.md#term-release) the [lease](../../../glossary.md#term-lease).
Remove the private reservation binding in finally after [quiescence](../../../glossary.md#term-quiescence). Reject wrong/released/short
leases before enqueue; never share a reservation across RunStates or concurrent dispatches.
Capacity one uses the inline path. Parallel Scan requires the coordinated group/budget pair;
legacy uncoordinated compatibility behavior of existing ordinary calls remains unchanged.

Keep current deterministic failure ordering, [cancellation](../../../glossary.md#term-cancellation) of unclaimed ranges, interruption/status
restoration, close/join and no-worker-nested-submission rules. Validate worker accessibility of
all selected segments/scratch before writes, rechecking [producer](../../../glossary.md#term-producer)-sensitive [preflight](../../../glossary.md#term-preflight) as today.
One [RunState](../../../glossary.md#run-state--runstate) owns the StepDispatch, arrays, calls, partials and staged values; prepared [owners](../../../glossary.md#term-owner)
retain only [immutable](../../../glossary.md#term-immutable) recipes/handles and [borrowed](../../../glossary.md#ownership-borrowing-and-lease) group/budget references.
Per-run costs are one dispatch object, two bounded arrays, prebound calls and declared scratch;
none scales with T. Existing calls.clone/Job/failure/[latch](../../../glossary.md#control-flow-graph)/queue [metadata](../../../glossary.md#term-metadata) per dispatch is the
specific proposed exception, not body/[element](../../../glossary.md#term-element) allocation permission. Job reuse is optional.

## Numerical permission and selective [identity](../../../glossary.md#term-identity-meanings)

Foundational [Aggregate floor](../../../glossary.md#term-aggregate-floor) already permits rounded trees. Amend only the narrower ordered
CPU0073/[Scan](../../../glossary.md#term-cumulative-scan) rule for selected terminal R/B contractions and represented Scan SUM. Preserve same
compute/[accumulator](../../../glossary.md#accumulator-type) [dtype](../../../glossary.md#data-type--datatype), separate products/additions, sequential time and one distinct shared
[carry](../../../glossary.md#term-carry) ADD per active time. No implicit [FMA](../../../glossary.md#term-fused-multiply-add-fma), [TF32](../../../glossary.md#term-tf32), lower [precision](../../../glossary.md#term-precision), [feature](../../../glossary.md#term-software-feature)-K regrouping or ordinary
exact SUM/[MEAN](../../../glossary.md#term-mean) relaxation. [Masks](../../../glossary.md#term-boolean-mask), indices, casts, raw movement/singletons/empties and saved values
remain exact under the selected realization. No cross-realization/thread-count/[device](../../../glossary.md#term-device) bitwise claim.
Workers use fixed row partitions and merge valid partials in ascending range index on the caller;
no shared [atomic](../../../glossary.md#resource-transaction-and-rollback) floating updates. Test arithmetic with per-[operation](../../../glossary.md#operation)/dtype [finite](../../../glossary.md#term-finite) tolerances and
separate [cancellation](../../../glossary.md#term-cancellation)/special/raw assertions, not a public numerical policy.

Preserve envelope 71, ordinary projection 72, Scan projection 71, [canonical](../../../glossary.md#term-canonical-form) row structuralKey,
old [scalar](../../../glossary.md#term-scalar) constructors/ABIs/keys/ClassFiles/resources and noneligible controls. Only new selected
realizations append `|ordinary-row-map-multi-reduction-exec-v1=` or
`|scan-row-map-multi-reduction-exec-v1=` after their existing [payload](../../../glossary.md#artifact-envelope-and-atomic-publication), identically in both streams.
Bind strategy/species, ordered partial/merge algorithm, masks, [transpose](../../../glossary.md#term-transpose) ordinals/lifetimes,
staging/partial/frame [layout](../../../glossary.md#layout), maximum partials, helper names/types, range convention, exact resource
recipes and all baked bounds. Key fixed geometry/bytes when baked; exclude invocation B/L,
addresses/assigned slots/resolved [run](../../../glossary.md#term-run) bytes. Existing unit policy remains preparation-owned;
key any policy fact only where it actually [shapes](../../../glossary.md#shape) emitted code or compatibility. Actual B binds
the fixed numeric range rule once; it does not select a new [artifact](../../../glossary.md#term-artifact). Reject stale selected-mode
helpers and validate method count/types, size/local limits and explicit accessors when loading.

## Staged authority diff map

MAIN accepted the edits below, independent APPROVE and verified [Ready](../../../glossary.md#term-ready) receipt before source work.
The mechanism is implemented and MAIN accepted. This historical cutover map adds no authority.

| Authority location | Accepted bounded change |
|---|---|
| Root Core invariants / [scope](../../../glossary.md#term-scope) index | Link the selected CPU0074 exception; retain single [RunState](../../../glossary.md#run-state--runstate), staged [Prepare](../../../glossary.md#prepare) and profile-free rules. Distinguish implementation, [validation](../../../glossary.md#term-validation) and MAIN completion acceptance. |
| [Backend](../../../glossary.md#backend) [CPU](../../../glossary.md#term-cpu) backend routes | Qualify no-per-step-[dispatch](../../../glossary.md#term-dispatch)/allocation and [scalar](../../../glossary.md#term-scalar)-only language solely for the nested [synchronous](../../../glossary.md#term-synchronous) selected-region seam and existing scheduler [metadata](../../../glossary.md#term-metadata). |
| Backend CPU0073 Physical region and [admission](../../../glossary.md#term-admission) / Cold proof, resources and range | Retain scalar v1; add the bounded [feature](../../../glossary.md#term-software-feature)-[vector](../../../glossary.md#term-vector)/row-partial [matrix](../../../glossary.md#term-matrix-meanings), private workspace and internal worker ranges while retaining one complete ordinary unit and exact ordinary SUM separation. |
| Backend CPU0073 Exact types, [signatures](../../../glossary.md#term-signature) and generation [lifetime](../../../glossary.md#term-lifetime) / Selective compatibility and retention | Add the exact nested [payload](../../../glossary.md#artifact-envelope-and-atomic-publication)/dispatch, common math/[mask](../../../glossary.md#term-boolean-mask) seams, accessors, reservation overload and selected-mode [identity](../../../glossary.md#term-identity-meanings) exceptions above; no top-level type. |
| Backend Accepted ordinary symbolic-[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) CPU contract | Exception only for admitted selected [F32](../../../glossary.md#floating-point-precision-and-rounding) row regions; retain [F64](../../../glossary.md#floating-point-precision-and-rounding)/control/exact-SUM and all unsupported-domain guards. |
| [Scan](../../../glossary.md#term-cumulative-scan) Actual first-order closure and arithmetic / Activity, raw identity and retained histories / First-order length-mode reverse | Permit selected terminal contributor grouping, not feature-K/time regrouping; replace increasing-b requirement only there. Retain activity dominance, selected saved values and distinct [carry](../../../glossary.md#term-carry) ADD. |
| Scan [Qualification](../../../glossary.md#term-qualification), performance and migration gates / Qualification and mechanism gates / [Runtime](../../../glossary.md#term-runtime) [preflight](../../../glossary.md#term-preflight) and implementation gates | Narrow typed synchronous dispatch/scheduler exception and independently qualify [fused](../../../glossary.md#specialization-and-fusion) forward/actual reverse. Reconcile relevant scalar/vector exclusion summaries without admitting no-region Scan. |
| Runtime Prepare [lifecycle](../../../glossary.md#lifecycle) / [Run](../../../glossary.md#term-run) lifecycle | Predeclare all tile/partial resources; allow one-run prebound CPU context and synchronous reservation lending, not a new Runtime protocol/execution [owner](../../../glossary.md#term-owner) or [tensor](../../../glossary.md#tensor) allocation per step. |
| Foundational Floating arithmetic primitives / [Aggregate floor](../../../glossary.md#term-aggregate-floor) | Read/retain unchanged; no global relaxed-numerics rewrite. |

The staged patch also updates focused `docs/architecture/general-differentiable-scan.md`,
`runtime-prepare-backend-boundary.md`, the CPU backend guide, glossary budget entry and ADR0053.
Master/roadmap record CPU0074 Complete, functional capability accepted and the frontier closed.
Final Javadocs preceded MAIN's passing post-fix CPU/[Javadoc](../../../glossary.md#term-javadoc) gate; no source changed afterward.

## [Validation](../../../glossary.md#term-validation) budget and retained evidence

MAIN froze 590 current CPU0073 production classes at
`/private/tmp/synaptik-cpu0074-run.brjyRN/baseline-classes`, [manifest](../../../glossary.md#term-manifest)
`/private/tmp/synaptik-cpu0074-run.brjyRN/baseline-classes.sha256` before CPU0074 source edits.
The 2565-path source freeze is
`/private/tmp/synaptik-cpu0073-run.DOIRA8/final-doc-evidence/pass.20261030-83686-1rt5r13/after-inventory.json`.
Use these for literal controls; create new test oracles independently, never from candidate output.
Subsequent user-approved CPU0073 experiments are closed evidence, not a rewritten CPU0073 Result:
split→[fused](../../../glossary.md#specialization-and-fusion) paired gains about 3.24–3.99; independent [runtime](../../../glossary.md#term-runtime)-B/T Java gap about 11–19%, with
[F32](../../../glossary.md#floating-point-precision-and-rounding)/B7 harness-[JIT](../../../glossary.md#term-just-in-time-compilation-jit) limitations. Reports remain at `/private/tmp/synaptik-cpu0073-ab.gjqEAL/REPORT.txt`
and `/private/tmp/synaptik-cpu0073-clean-java.dKKoC5/REPORT.txt`; actual decompiles/comparison are in
`/private/tmp/synaptik-cpu0073-code-compare.bsepCz`. No old experiment is rerun implicitly.

The exhausted MAIN-approved CPU0074 budget was one measured [JVM](../../../glossary.md#term-java-virtual-machine-jvm) with a 60-second total
watchdog, no retry/additional [fork](../../../glossary.md#term-fork-meanings); setup uses a separately bounded untimed correctness check.
Measure ordinary F3/H2 fused actual [derivative](../../../glossary.md#term-derivative) and combined actual F3/H2 reverse, F32/[F64](../../../glossary.md#floating-point-precision-and-rounding),
T=4 for [Scan](../../../glossary.md#term-cumulative-scan), B=32 and B=256, with explicit two-range configuration. Each of eight rows compares
four generated realizations with their four matched optimal Java arms: one 50ms warm window and
three alternating-order 50ms measured windows per arm, retaining every [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters). This consumes
12.8 seconds of requested windows, leaving bounded setup/watchdog margin; if [censored](../../../glossary.md#term-censored), report
incomplete evidence without an automatic retry. Check outputs outside windows; separate cold
generation/[binding](../../../glossary.md#term-binding), [kernel](../../../glossary.md#kernel) entry and worker [dispatch](../../../glossary.md#term-dispatch) costs, record JIT/[host](../../../glossary.md#term-host) limitations and all
staging/scheduler allocation costs. Diagnose staging overhead; no all-[shapes](../../../glossary.md#shape)-faster/parity gate.
[Finite](../../../glossary.md#term-finite)/special/exact acceptance and budget/close/concurrent-[run](../../../glossary.md#term-run) tests remain ordinary executable
tests, not timed campaigns. The brief records owning commands and the single MAIN join.

## [Planning](../../../glossary.md#term-planning) delivery and remaining gate

Independent mechanism review APPROVE, 0 P1 / 0 P2: the exact [vector](../../../glossary.md#term-vector) init/contributor/finish seams
close the sole P2. Final independent authority/docs APPROVE (0 P1/P2), MAIN authority acceptance
and the verified CPU0074 [Ready](../../../glossary.md#term-ready) receipt are recorded on 2026-10-30, including the Job-reuse wording
correction. The clean implementation is now frozen; CPU0074 implements no Job reuse
and does not depend on it.
The brief Result centralizes final [CPU](../../../glossary.md#term-cpu)/[Javadoc](../../../glossary.md#term-javadoc), named integration, retention and performance
receipts. MAIN accepted functional capability as Complete. The sole comparison is [nonstationary](../../../glossary.md#term-nonstationary);
steady-state profitability is unqualified and concrete [JIT](../../../glossary.md#term-just-in-time-compilation-jit)-pressure diagnosis is recommended
separately, not a new task/frontier/campaign. No [warm-up](../../../glossary.md#benchmark-measurement-vocabulary)-only dismissal, speed/parity/[ranking](../../../glossary.md#term-ranking)
claim, production-default change, rollout or reopened timing budget follows.
