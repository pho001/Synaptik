# CPU0074 proposed row-fusion execution boundary

## Purpose and approval boundary

This planning packet specifies one extension of CPU0073, not a new architecture authority.
[ARCHITECTURE.md](../../../../ARCHITECTURE.md) and its incorporated contracts still govern.
MAIN accepted the authority patch and independent APPROVE (0 P1/P2) before the verified Ready
receipt on 2026-10-30. The [CPU0074 brief](tasks/0074-row-fusion-simd-parallel.md#result) now records
implemented and MAIN-accepted functional capability, Complete; its source/measurement frontier is closed.
MAIN addendum accepted 2026-10-30 after focused independent APPROVE of the two exact generation/
cold-binding seams below. No new type, runtime protocol or capability-completion claim follows.
MAIN/focused-reviewer correction accepted 2026-10-30: Scan minimum-row policy transport
replaced its unimplemented seven-argument overload.
MAIN/Boyle byte-seam addendum accepted 2026-10-30: the two generation-only checked-segment
methods below are implemented. Final CPU/Javadoc and the sole comparison completed on frozen
source; functional acceptance does not qualify steady-state profitability or authorize rollout.
This planning document is not authority. No new top-level production type,
public numerical mode, executor, global Job, job pool or general pointwise refactor is proposed.

## Source-bound mechanism

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
The pre-CPU0074 source facts below motivated only the accepted additions; they are not a claim
that the old scalar-only guards still describe selected exec-v1 implementation.

| Current owner / symbol | Source fact and proposed change |
|---|---|
| `internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan` | Already owns execution strategy, selectedRangeCount and minimumElementsPerWorker. Retain worker policy here; add none to canonical row IR. |
| `internal.cache.CpuKernelSpecialization` | Scalar-only row/Scan guards and fixed entry type need a selective optional emission payload and ABI branch. Preserve old constructor delegates and scalar bytes. |
| `internal.codegen.emit.CpuRowMapReductionEmitter.emitBound` | Generation-only full-row orchestration; add a bounded range variant, retaining scalar entry behavior and local validation. |
| `CpuVectorInstructionEmitter.emit` / `CpuVectorMath` | Existing typed vector pointwise math, including TANH; SIGMOID currently has VectorForm.NONE. Reuse only admitted opcodes, with scalar fallback for unsupported regions. |
| `CpuMatmulEmitter.emitDirectVector` / private vector helpers near 788 | Existing output-feature vectorization keeps full feature K; expose the exact common vector math seams below rather than copy arithmetic. |
| `CpuAggregateEmitter.emitRepresented*` | Current shared Scan SUM seams are scalar, separate from exact ordinary SUM. Add only their represented vector counterparts below. |
| `CpuCarrierEmitter.vectorLoad/vectorStore` | Existing contiguous/broadcast loads and contiguous stores, not arbitrary segment gathers. Add masked feature-tail overloads; stage only proved small transposes. |
| `CpuGeneralScanEmitter.emitRegionalStage` | One selected regional marker and emitted T loop; retain temporal bindings/activity and generated caller-side merge/carry. No runtime stage interpreter. |
| `CpuGeneratedKernel.lookup` / `hiddenLookup()` / `lossEntryPointFor` | Retains the defining lookup, exposes it through the existing public hiddenLookup(), and explicitly resolves loss helpers cold. Extend named accessors, not a registry or per-step lookup. |
| `CpuPreparedExecutable.bindCompatible/Invocation` | Binds RangeCall[] and geometry once; workers synchronously join. Extend ordinary regional partial/merge binding, not ordinary exact reductions. |
| `CpuGeneralScanPreparedExecutable.Invocation.executeBound` | Currently holds budget.acquire(1) around the whole generated entry. Replace only selected parallel admission with one adequate borrowed reservation. |
| `CpuWorkerGroup.execute` / `CpuConcurrencyBudget.Lease` | RangeCall is package-private; dispatch clones calls, creates Job/failure/latch/queue metadata and joins. Add same-budget reservation validation and a no-acquire dispatch overload. |

Final cold metadata also uses existing `CpuFusionDecision`, `CpuFusionProfitabilitySelector`
and plan validation: `ROW_REGION_STATE` labels ordinary selected scratch only with matching
row IR/emission and no Scan/recurrent owner. Internal row consumers stay in their indivisible
unit; ordinary pointwise accounting/scoring is unchanged. This is not a new type, seam or authority.

## Proposed exact internal declarations

These are declaration excerpts, not runnable Java or supported application APIs. Unchanged names
and constructors remain. Additional named types/common signatures require MAIN approval.
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
shape only: no route selection, worker ownership or mutable data. No payload for CPU0073 scalar.
The generator derives feature masks and block layout from authenticated region accesses/species.

emitRowRange is generation-only and reuses the private temporal/row binder, not a runtime callback.
The ordinary symbolic overload transports existing unit-owned policy and borrowed workers to
cold numeric B binding; no per-B artifact or worker policy in canonical row IR is added.
The Scan overload validates positive minimumElementsPerWorker (otherwise IllegalArgumentException).
Finalizer transports unit.minimumElementsPerWorker() cold; keep policy in the unit, not IR/Facts/
specialization. Its old six-argument constructor delegates (null,1), preserving inline behavior.

vectorLoadBytes/vectorStoreBytes require checked MEMORY_SEGMENT FLOAT32/FLOAT64, target preferred
species, LONG byte-offset local and matching mask/value species. Emit direct native-byte-order
masked fromMemorySegment/intoMemorySegment; disabled lanes access no storage. Scalar broadcasts
use existing loadBytes plus broadcast. Keep old element methods and array paths literally unchanged;
migrate only selected exec-v1 Scan byte addresses and ordinary private partial scratch. Removing
the emitted /width*width roundtrip is the rationale, not proof of JIT behavior or a slowdown/speed
claim. No new type/layer, runtime entry ABI, resource or global schema bump follows. New selected
exec-v1 is unreleased: final inspection/performance must use a fresh private artifact root so old
candidate cache entries cannot hide the changed generated shape.

The range emitter retains caller-allocated tables and generation-only callbacks. Scalar locals
are typed primitive cells; vector bodies use REFERENCE locals by fixed feature chunk. The caller
selects and validates the table convention before emission; no runtime dtype/strategy dispatch.
Rows advance from rowBegin inclusive to rowEnd exclusive. Initialization/finish occur once;
zero rows still publish valid required identities through the complete-unit wrapper.

Vector init uses the selected target dtype's preferred species; specialization validates matching
preferred bits and accumulator/load/mask species, never arbitrary species. It creates typed +0
unless directSingleton omits reset. Contributor loaders leave
matching vectors; MATMUL emits separate per-lane MUL then ADD, never FMA. Represented SUM emits
typed ADD or raw singleton assignment, with no synthetic +0 for a raw copy. Finish passes the
completed reference local to its generation-only sink without extra arithmetic. Source scalar
math and exact ordinary SUM/MEAN remain unchanged. Feature-tail masks exclude storage access
and publication outside the real cells; unused vector lanes never enter a horizontal reduction.

## SIMD dimension, temporal staging and oracle

Select feature/output lanes within a single active row. Gate the row before any row/seed/history
load and dependent math. Use masked contiguous feature loads/stores, scalar broadcast operands
and existing vector opcodes. Feature contractions traverse the original K in order; terminal
vectors traverse R in order within each worker. No vector reduction across different features.

Primary actual RNN reverse has F=3, H=2: compute its two delta cells once from exact saved
primals; vectorize dX(F3), dH(H2), dW(three H2 vectors), dU(two H2 vectors), db(H2).
Extract needed delta feature lanes once to scalar locals for ordered feature-K broadcasts;
do not assemble arbitrary strided vectors with per-lane withLane chains.
W `[3,2]` and U `[2,2]` need canonical transposed right operands for dX/dH. Emit bounded raw
transpose copies into predeclared scratch: 6+4 cells = 40 F32 / 80 F64 bytes before alignment.
If their bindings are invariant across T, populate once per run under an any-active gate.
SHARED history or changing cuts instead populate at the exact active original time, after
their producer and before dispatch; reuse the same ten-cell area only after all readers join.
If stability/access/lifetime cannot be proved, the vector region falls back, not stale reuse.
No staging runs for T=0/B=0/all-inactive; no padded/history reads are justified by zero fill.

This is generic bounded row-region binding, not a second RNN emitter. Fixed-feature provenance,
unit-stride/broadcast mapping or the explicit small transpose proof must cover every access.
Unproved ROW(1), dynamic feature strides and unsupported vector opcodes retain scalar fallback.
An independent optimal Java oracle for each realization must include the same necessary transpose,
masked feature algorithm, full feature-K order and exact saved sites. Compare complete code/dataflow;
if a better current-need clean-Java shape avoids staging, justify the retained cost or revise
the proposal before coding. No claim that masked H2/F3 is faster on every species or small B.
The current optimal reference retains twelve typed carry cells across T while preserving one
distinct ADD per active time. Generator across-time carry promotion remains a non-goal; keep
the reference intact and disclose the dataflow difference in performance interpretation.

## Ownership, budget and synchronous dispatch

Mental model: cold plan → cold run binding → emitted active-time setup → workers/join →
generated terminal merge/carry → next time. Only the generated outer loop owns time progression.
StepDispatch bridges that one selected region to prebound typed calls; it is not a body callback
registry. Its public accessors return its own mutable run arrays without copying so the generated
entry can bind them once. Generated code updates only current operand references/time/base facts;
it does not repack geometry, rediscover views or resolve methods per step.

Existing unit fields fix maximum ranges W and minimum row work m. At cold run binding use
N=min(W,ceil(R/m)) for R>0, otherwise zero; partition [0,R) by quotient/remainder in ascending
range index. This numeric rule is fixed before execution, constant across time, not a value-based
profitability choice. If N<2, execute inline under one permit without a worker submission.
Stage/partial layout reserves W bounded slots; no B×parameter bank. Each partial has typed
terminal cells plus an eight-byte validity flag, initialized each dispatch. Invalid empty/inactive
partials are skipped, not merged as artificial +0 contributors. Geometry-singleton raw reductions
keep their copy path; one active row in a larger B is not a geometry singleton.

For selected parallel Scan acquire max(1,N) from the exact finalizer budget before entry and
install that reservation privately in this invocation's StepDispatch for this synchronous call.
Caller setup, merge and carry execute under it. Dispatch verifies same-budget identity, live
status and sufficient demand, then calls the overload which neither acquires nor releases permits.
The caller performs no concurrent math while workers run. Started workers always quiesce before
return/throw; only then may caller merge, advance carry/time, reuse scratch or release the lease.
Remove the private reservation binding in finally after quiescence. Reject wrong/released/short
leases before enqueue; never share a reservation across RunStates or concurrent dispatches.
Capacity one uses the inline path. Parallel Scan requires the coordinated group/budget pair;
legacy uncoordinated compatibility behavior of existing ordinary calls remains unchanged.

Keep current deterministic failure ordering, cancellation of unclaimed ranges, interruption/status
restoration, close/join and no-worker-nested-submission rules. Validate worker accessibility of
all selected segments/scratch before writes, rechecking producer-sensitive preflight as today.
One RunState owns the StepDispatch, arrays, calls, partials and staged values; prepared owners
retain only immutable recipes/handles and borrowed group/budget references.
Per-run costs are one dispatch object, two bounded arrays, prebound calls and declared scratch;
none scales with T. Existing calls.clone/Job/failure/latch/queue metadata per dispatch is the
specific proposed exception, not body/element allocation permission. Job reuse is optional.

## Numerical permission and selective identity

Foundational Aggregate floor already permits rounded trees. Amend only the narrower ordered
CPU0073/Scan rule for selected terminal R/B contractions and represented Scan SUM. Preserve same
compute/accumulator dtype, separate products/additions, sequential time and one distinct shared
carry ADD per active time. No implicit FMA, TF32, lower precision, feature-K regrouping or ordinary
exact SUM/MEAN relaxation. Masks, indices, casts, raw movement/singletons/empties and saved values
remain exact under the selected realization. No cross-realization/thread-count/device bitwise claim.
Workers use fixed row partitions and merge valid partials in ascending range index on the caller;
no shared atomic floating updates. Test arithmetic with per-operation/dtype finite tolerances and
separate cancellation/special/raw assertions, not a public numerical policy.

Preserve envelope 71, ordinary projection 72, Scan projection 71, canonical row structuralKey,
old scalar constructors/ABIs/keys/ClassFiles/resources and noneligible controls. Only new selected
realizations append `|ordinary-row-map-multi-reduction-exec-v1=` or
`|scan-row-map-multi-reduction-exec-v1=` after their existing payload, identically in both streams.
Bind strategy/species, ordered partial/merge algorithm, masks, transpose ordinals/lifetimes,
staging/partial/frame layout, maximum partials, helper names/types, range convention, exact resource
recipes and all baked bounds. Key fixed geometry/bytes when baked; exclude invocation B/L,
addresses/assigned slots/resolved run bytes. Existing unit policy remains preparation-owned;
key any policy fact only where it actually shapes emitted code or compatibility. Actual B binds
the fixed numeric range rule once; it does not select a new artifact. Reject stale selected-mode
helpers and validate method count/types, size/local limits and explicit accessors when loading.

## Staged authority diff map

MAIN accepted the edits below, independent APPROVE and verified Ready receipt before source work.
The mechanism is implemented and MAIN accepted. This historical cutover map adds no authority.

| Authority location | Accepted bounded change |
|---|---|
| Root Core invariants / scope index | Link the selected CPU0074 exception; retain single RunState, staged Prepare and profile-free rules. Distinguish implementation, validation and MAIN completion acceptance. |
| Backend CPU backend routes | Qualify no-per-step-dispatch/allocation and scalar-only language solely for the nested synchronous selected-region seam and existing scheduler metadata. |
| Backend CPU0073 Physical region and admission / Cold proof, resources and range | Retain scalar v1; add the bounded feature-vector/row-partial matrix, private workspace and internal worker ranges while retaining one complete ordinary unit and exact ordinary SUM separation. |
| Backend CPU0073 Exact types, signatures and generation lifetime / Selective compatibility and retention | Add the exact nested payload/dispatch, common math/mask seams, accessors, reservation overload and selected-mode identity exceptions above; no top-level type. |
| Backend Accepted ordinary symbolic-batch CPU contract | Exception only for admitted selected F32 row regions; retain F64/control/exact-SUM and all unsupported-domain guards. |
| Scan Actual first-order closure and arithmetic / Activity, raw identity and retained histories / First-order length-mode reverse | Permit selected terminal contributor grouping, not feature-K/time regrouping; replace increasing-b requirement only there. Retain activity dominance, selected saved values and distinct carry ADD. |
| Scan Qualification, performance and migration gates / Qualification and mechanism gates / Runtime preflight and implementation gates | Narrow typed synchronous dispatch/scheduler exception and independently qualify fused forward/actual reverse. Reconcile relevant scalar/vector exclusion summaries without admitting no-region Scan. |
| Runtime Prepare lifecycle / Run lifecycle | Predeclare all tile/partial resources; allow one-run prebound CPU context and synchronous reservation lending, not a new Runtime protocol/execution owner or tensor allocation per step. |
| Foundational Floating arithmetic primitives / Aggregate floor | Read/retain unchanged; no global relaxed-numerics rewrite. |

The staged patch also updates focused `docs/architecture/general-differentiable-scan.md`,
`runtime-prepare-backend-boundary.md`, the CPU backend guide, glossary budget entry and ADR0053.
Master/roadmap record CPU0074 Complete, functional capability accepted and the frontier closed.
Final Javadocs preceded MAIN's passing post-fix CPU/Javadoc gate; no source changed afterward.

## Validation budget and retained evidence

MAIN froze 590 current CPU0073 production classes at
`/private/tmp/synaptik-cpu0074-run.brjyRN/baseline-classes`, manifest
`/private/tmp/synaptik-cpu0074-run.brjyRN/baseline-classes.sha256` before CPU0074 source edits.
The 2565-path source freeze is
`/private/tmp/synaptik-cpu0073-run.DOIRA8/final-doc-evidence/pass.20261030-83686-1rt5r13/after-inventory.json`.
Use these for literal controls; create new test oracles independently, never from candidate output.
Subsequent user-approved CPU0073 experiments are closed evidence, not a rewritten CPU0073 Result:
split→fused paired gains about 3.24–3.99; independent runtime-B/T Java gap about 11–19%, with
F32/B7 harness-JIT limitations. Reports remain at `/private/tmp/synaptik-cpu0073-ab.gjqEAL/REPORT.txt`
and `/private/tmp/synaptik-cpu0073-clean-java.dKKoC5/REPORT.txt`; actual decompiles/comparison are in
`/private/tmp/synaptik-cpu0073-code-compare.bsepCz`. No old experiment is rerun implicitly.

The exhausted MAIN-approved CPU0074 budget was one measured JVM with a 60-second total
watchdog, no retry/additional fork; setup uses a separately bounded untimed correctness check.
Measure ordinary F3/H2 fused actual derivative and combined actual F3/H2 reverse, F32/F64,
T=4 for Scan, B=32 and B=256, with explicit two-range configuration. Each of eight rows compares
four generated realizations with their four matched optimal Java arms: one 50ms warm window and
three alternating-order 50ms measured windows per arm, retaining every sample. This consumes
12.8 seconds of requested windows, leaving bounded setup/watchdog margin; if censored, report
incomplete evidence without an automatic retry. Check outputs outside windows; separate cold
generation/binding, kernel entry and worker dispatch costs, record JIT/host limitations and all
staging/scheduler allocation costs. Diagnose staging overhead; no all-shapes-faster/parity gate.
Finite/special/exact acceptance and budget/close/concurrent-run tests remain ordinary executable
tests, not timed campaigns. The brief records owning commands and the single MAIN join.

## Planning delivery and remaining gate

Independent mechanism review APPROVE, 0 P1 / 0 P2: the exact vector init/contributor/finish seams
close the sole P2. Final independent authority/docs APPROVE (0 P1/P2), MAIN authority acceptance
and the verified CPU0074 Ready receipt are recorded on 2026-10-30, including the Job-reuse wording
correction. The clean implementation is now frozen; CPU0074 implements no Job reuse
and does not depend on it.
The brief Result centralizes final CPU/Javadoc, named integration, retention and performance
receipts. MAIN accepted functional capability as Complete. The sole comparison is nonstationary;
steady-state profitability is unqualified and concrete JIT-pressure diagnosis is recommended
separately, not a new task/frontier/campaign. No warm-up-only dismissal, speed/parity/ranking
claim, production-default change, rollout or reopened timing budget follows.
