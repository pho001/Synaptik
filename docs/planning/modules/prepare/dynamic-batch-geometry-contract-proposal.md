# Dynamic batch geometry and resource contract proposal

## Purpose and status

Complete — reviewed DESIGN delivery, 2026-10-24; user architecture approval is now recorded.
Independent Class C design/docs APPROVE, zero open P1/P2; MAIN accepts delivery under its conditional
admission. User “ano” explicitly accepts authority cutover, subsequent ordinary CPU forward/actual
first-order backward and bounded final A/B. [Prepare0012A](tasks/0012a-dynamic-batch-authority-cutover.md)
is Complete: authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2, docs/glossary review complete. No Ready Java brief until MAIN separately admits planner output.
[Prepare0012](tasks/0012-dynamic-batch-geometry-and-resource-contract.md) records launch/review evidence;
[current authority](../../../../ARCHITECTURE.md) carries the enacted/accepted future protocol;
current Java dynamic batch remains unsupported. The proposal body remains historical design. Base:
HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted dirty tree.

Recommend one symbolic compilation/preparation: Compiler binds semantic metadata, Prepare fixes
identities, CPU prepares numeric recipes, Runtime creates isolated resources, Engine publishes actual
descriptors. Runtime already allocates freshly per run; the gap is sizing, not freshness. The
[source inventory](../../dynamic-batch-execution.md#current-truth) identifies current fixed-only seams.

First scope: ordinary homogeneous FLOAT32 CPU forward **and actual first-order backward**, one named
DynamicDimension B, fixed feature/parameter dimensions, rank-at-most-two boundaries and needed AD
views. General Scan follows with fixed T/F/H. No B_max, per-B fullprepare cache, prepared mutation,
new session/executor, second DAG/proof runtime or graph interpreter. Native/mixed, wider dtype,
variable T, higher-order, persistence and tuning remain separate gates.

Size exception: this atomic cross-owner packet exceeds the 25 KB target because it keeps the exact
four boundary signatures, nine-type/replacement map, empty/resource example and draft authority/ADR
changes together for one decision. Splitting them would separate the mode/sizing/publication contract
from its migration and approval conditions. The task brief remains below 200 lines/15 KB.

## Exact proposed signatures and semantics

These are future declarations, not runnable examples. Package prefix: `io.github.pho001.synaptik.`.
For every signature below: references/results are non-null; lists snapshot membership and reject null
elements in encounter order. Null throws `NullPointerException`, invalid fields/metadata
`IllegalArgumentException`, checked overflow `ArithmeticException`, invalid dense index
`IndexOutOfBoundsException`. Immutable recipes are thread-safe/storage-free; resolved arrays belong
to one run. Additional lifecycle failures retain their existing exception/precedence contracts.

### Compiler metadata

```java
// compiler.BatchBindingPlan: public final, private construction
public static BatchBindingPlan prepare(CompileArtifacts artifacts);
public BatchBindingPlan.Binding bind(List<TensorDescriptor> actualInputs);
public static final class Binding { // private constructor
    public long batchSize();
    public int valueCount();
    public TensorDescriptor descriptor(int valueIndex);
}
```

`prepare` runs once from final graph facts and same-package `CompileDiagnostics.constraints()`.
Preindex descriptor templates, input/value/publication positions and exact typed obligations; reject
unsupported symbols/expressions. Preserve all eight CompileArtifacts components/order and logical
memory descriptors. No diagnostic parsing, public predicate language or Tensor/storage sidecar.
Existing typed predicates/GraphPredicateProof gain bound evaluation inside Compiler only.

`bind` consumes actual fully static descriptors in final compiled input order. Derive B>=0 from the
first declared named-axis occurrence, not storage bytes; require every occurrence, including unused
inputs, to agree. Match rank/fixed axes/dtype/requiresGrad exactly; parameters/seeds stay fixed.
Only expressions determined by B/constants qualify; output-only symbols/anonymous unknowns reject
at prepare. Unsatisfied predicates/negative dimensions reject before borrowing. No per-run graph
traversal, operation inference, recapture, optimization or backend selection.

Dynamic caller templates require resolved zero-offset dense layouts with actual canonical strides;
`isView=true` dense reads may qualify. Absent layout is not silently canonical. Static templates
keep exact existing admission, including positional unresolved-layout behavior. Proved AD transpose/
expand views are separate numeric mappings. Materialized outputs become canonical non-view;
qualified view publications retain actual geometry. `descriptor(index)` resolves **only requested**
metadata on demand; Engine calls it for publications. Do not eagerly allocate descriptors/Shapes for
intermediates. `valueCount` is graph list size, not maximum ValueId. Numeric cold cost is
O(inputs + requirements + obligations + publications), never a DAG rebuild/per-element evaluation.

### Runtime declarations and creation

```java
// runtime.memory.PreparedMemoryPlan: new mode component; old constructor remains
public record PreparedMemoryPlan(List<BufferEntry> buffers, List<WorkspaceEntry> workspaces,
        boolean requiresBatchSize) {
    public PreparedMemoryPlan(List<BufferEntry> buffers, List<WorkspaceEntry> workspaces);
    public interface ByteSize {
        OptionalLong fixedByteSize();
        long byteSize(long batchSize);
    }
    public record FixedByteSize(long bytes) implements ByteSize { /* methods above */ }
    public record BufferEntry(BufferSlot slot, ByteSize size, long byteAlignment) {
        public BufferEntry(BufferSlot slot, long byteSize, long byteAlignment);
        public long byteSize();
    }
    public record WorkspaceEntry(WorkspaceSlot slot, ByteSize size, long byteAlignment) {
        public WorkspaceEntry(WorkspaceSlot slot, long byteSize, long byteAlignment);
        public long byteSize();
    }
}
// prepare.analysis.PreparationResourceRequirement: existing nested variants
record Buffer(ValueId valueId, PreparedMemoryPlan.ByteSize size, long byteAlignment)
        implements PreparationResourceRequirement {
    public Buffer(ValueId valueId, long byteSize, long byteAlignment);
    public long byteSize();
}
record Workspace(long requirementId, PreparedMemoryPlan.ByteSize size, long byteAlignment)
        implements PreparationResourceRequirement {
    public Workspace(long requirementId, long byteSize, long byteAlignment);
    public long byteSize();
}
// runtime.run.RunMemoryGeometry: public final, private construction
public static RunMemoryGeometry resolve(PreparedMemoryPlan plan, OptionalLong batchSize);
public PreparedMemoryPlan memoryPlan();
public OptionalLong batchSize();
public long bufferByteSize(int bufferIndex);
public long workspaceByteSize(int workspaceIndex);
// Existing RunState adds constructor/accessor; keep original constructor
public RunState(PreparedMemoryPlan plan, List<List<BufferRepresentationBinding>> buffers,
        List<WorkspaceRepresentation> workspaces, RunMemoryGeometry geometry);
public RunMemoryGeometry geometry();
// runtime.run.PreparedExecutionRunner adds overload; keep original static overload
public RunResult run(PreparedExecution execution, List<BufferRepresentation> callerInputs,
        RunMemoryGeometry geometry);
// Existing package-private RunStateCreation adds overload
static RunState create(PreparedRepresentationPlan plan, List<BufferRepresentation> inputs,
        RunMemoryGeometry geometry);
// Existing PreparedRepresentationPlan creator SAMs are REPLACED
BufferRepresentation create(long byteSize, long byteAlignment); // BufferCreator
WorkspaceRepresentation create(long byteSize, long byteAlignment); // WorkspaceCreator
```

Sizes are bytes, nonnegative long; fixed alignment is bytes, power of two from 1 through `1L<<62`.
Workspace IDs stay nonnegative/local. `size()` retains the recipe. Old long constructors wrap
FixedByteSize; old `byteSize()` returns its exact count (including zero), but throws
`IllegalStateException` for a run-bound declaration. No fake zero/-1/maximum count. Record-component/
reflection/binary changes and creator SAM migration are explicit; static constructor/accessor source
compatibility is retained, not a record ABI promise.

`fixedByteSize()` is a **structural declaration marker**: present for FixedByteSize, absent for
run-bound recipes even if their results happen to be constant. `byteSize(B)` validates B>=0 and
returns exact checked bytes. Recipes contain numeric facts only: no graphs, Model expressions,
value reads, caches/providers, route selection, physical allocation or requirement discovery.
Prepare compares structural declarations/alignment/coverage without evaluating recipes. Runtime
evaluates them; CPU owns their physical meaning. This explicitly changes today's fixed-only contract.

**Execution mode is independent of byte independence.** Prepare sets `requiresBatchSize=true`
from the qualified symbolic projection even if every size is constant, e.g. `X[B,0].sum()`.
Old two-list plan construction means false and rejects run-bound declarations. `resolve` requires
present B iff this mode is true, then resolves all buffers/workspaces in plan order into private
arrays. Absent-B runner/creation/state overloads reject a true-mode plan before invoking creators
or accepting resource ownership, even for all-constant bytes. State-aware compatibility defaults
also use this explicit mode, never `fixedByteSize` as a mode test.

Geometry authenticates the exact plan reference; foreign geometry rejects. Engine resolves before
borrowing; Runtime runner still acquires its unique prepared lease before inspecting supplied
inputs/geometry. Static overloads retain static admission/order. Creators validate numeric/type
constraints and allocate exactly the requested usable span (allocator padding is not capacity),
returning fresh non-null objects even for zero bytes. InitializedBuffer fills the actual count.
Keep complete caller validation before creation, duplicate-result rejection, origins/validity and
reverse rollback. Allocation errors propagate; null results fail; backend owns rollback before
return, setup/run afterward. No unsafe zero-argument dynamic adapter.

### CPU numeric recipes and binding

```java
// backend.cpu.internal.memory.CpuRepresentationRecipes
public record BatchByteSize(long constantBytes, long bytesPerBatch, int elementByteWidth)
        implements PreparedMemoryPlan.ByteSize { /* methods above */ }
// backend.cpu.internal.ir.CpuAccessPlan
public record BatchBinding(CpuAccessPlan plan, List<Long> extentConstants,
        List<Long> extentBatchFactors, long baseElementOffset,
        List<Long> strideConstants, List<Long> strideBatchFactors) {
    public CpuAccessPlan.Binding bind(long batchSize);
}
// backend.cpu.internal.lowering.CpuMatmulLowering
public record BatchGeometry(DataType dataType, CpuAccessPlan.BatchBinding left,
        CpuAccessPlan.BatchBinding right, CpuAccessPlan.BatchBinding result) {
    public CpuMatmulLowering.Geometry bind(long batchSize);
}
// backend.cpu.internal.lowering.CpuAggregateLowering
public record BatchGeometry(CpuAggregateIr.Kind kind, DataType dataType,
        CpuAggregateIr.Form form, List<Integer> axes, boolean keepDimensions,
        CpuAccessPlan.BatchBinding input, CpuAccessPlan.BatchBinding output)
        implements PreparedMemoryPlan.ByteSize {
    public CpuAggregateLowering.Geometry bind(long batchSize);
    public OptionalLong fixedByteSize(); // empty: run-bound declaration
    public long byteSize(long batchSize); // bind(B).workspaceBytes(1)
}
// Existing runtime.execution.PreparedExecutable adds protected state-aware hooks
protected boolean acceptsBufferRepresentation(RunState state, int selectionIndex,
        BufferRepresentation representation);
protected boolean acceptsWorkspaceRepresentation(RunState state, int selectionIndex,
        WorkspaceRepresentation representation);
```

CPU records are public only in existing internal packages. BatchBinding validates equal-rank lists,
nonnegative coefficients/base and computes extent/stride `constant+factor*B` in **elements**, with
checked counts/spans. One B-bearing axis per represented shape; proved canonical/transpose/expand
mappings only. Preserve exact Model empty strides: `[F,0]` has `[0,1]`, not max(1,B). Regimes/axis
roles stay prepared; no fake representative B. BatchByteSize uses checked `constantBytes +
bytesPerBatch*B`, nonnegative coefficients, positive supported width, divisibility and CPU limits.

Rank-two MATMUL resolves existing Geometry/pack: forward M/K/N=`B/F/H`, dX=`B/H/F`, dW=`F/B/H`
with actual transposed strides. Analysis fixes generated scalar, single-range routes independently
of B; no admission-time profitability selection. Carrier pattern/code stays fixed; unsupported
carriers reject, not select another artifact. Pack actual geometry once into existing Invocation.

Aggregate BatchGeometry binds actual domains/output counts and conditional workspace, not maximum
banks. For first-slice FLOAT32/int-limited N, `signedBits=278+ceil(log2(N))` (cardinality term zero
at N<=1), at most 309 bits: **five limbs for all admitted N, including 0/1**. Retain optimal constant-
five-limb CpuExactSumEmitter. Single-range scratch is 48 bytes (`8+8*5`) if outputCount>0, otherwise
zero: non-affine actual sizing. Do not require dynamic-limb emission for this slice; F64/long domains
need later qualification. SUM-to-Shape handles B=1 copy/singleton at entry in the same code/route/
slot; zero scratch never discovers/removes a requirement. Preserve actual count division/numerics.

Do not run Model lowerers or element-enumerating injectivity checks per invocation: canonical
write injectivity is proved at prepare, binding checks numeric bounds. New hooks default to old
static hooks only for false-mode plans; CPU overrides true-mode actual bytes from state.geometry(),
retaining type/access/alignment/read-only/overlap checks. bindCompatible/Invocation retains direct
typed carriers and fresh packed/access arrays, no shared currentB/ThreadLocal or hot recipe dispatch.

### Engine metadata and publication

```java
// Existing package-scoped EngineBackendComposition adds exact-prefix borrowing
BufferRepresentation borrow(HostTensorStorage storage, long referencedByteSize);
// Existing package-private RunResult constructor adds occurrence descriptors
RunResult(AdvancedRunResult owner, List<CompiledGraph.PublicationSpec> specifications,
        List<TensorDescriptor> actualDescriptors);
// Existing private Publication constructor adds actual descriptor
Publication(RunResult result, int index, CompiledGraph.PublicationSpec specification,
        TensorDescriptor actualDescriptor);
public TensorDescriptor descriptor(); // existing RunResult.Publication accessor
```

Public Engine signatures stay. `run` keeps exact-ID/arbitrary-order semantics; fresh B values use
runValues without prototype mutation. Dynamic order: existing Engine/owner/closed-prepared gates;
snapshot/count/null; semantic and numeric/size preflight; snapshot all storage associations once
and validate all; borrow; Runtime lease/create/cold bind; execute; complete result. New borrowing
wraps exactly the referenced prefix, not excess capacity, without owning caller storage; old static
borrow stays. Invalid prefix rejects; failure cleans wrappers only.

Before borrow/allocation check fixed axes/type/flags, strides/counts, byte products, referenced
span/offset sums and aggregate allocation bounds including alignment padding/representation
multiplicity. First dynamic CPU extents/counts/strides/offsets/referenced-element spans <=
Integer.MAX_VALUE; bytes/aggregate bounds <=Long.MAX_VALUE. These are representability limits,
not public B_max. Zero count cannot hide overflowing fixed trailing geometry or unsupported unused
declarations. Require matching, present/live/thread-accessible storage and sufficient capacity.
Physical address/alignment and writable overlap checks follow allocation, before execution.

Engine preindexes publication-to-value-list positions and resolves only requested descriptors.
Exact occurrence count is required. Gradient metadata comes from generated gradient **values**,
not target flags; retain tensorId/role/derivativeOrder/targetIndex and distinct authenticated aliases.
Materialization uses actual shape/layout/bytes and returns detached HostTensorValue. Failure closes
inward result then reverse borrows with distinct suppression, publishing nothing. No Model metadata
or Compiler dependency enters Runtime.

## Complete type and replacement map

These nine additions include nested types; no type moves/removals or new module/dependency.

| Added type under the package prefix | Visibility; why existing type cannot serve; consumers |
|---|---|
| `compiler.BatchBindingPlan` | Public final; diagnostics are not admission recipes. Engine retains prepared semantic metadata. |
| `compiler.BatchBindingPlan.Binding` | Public static final/private constructor; CompileArtifacts cannot hold per-run metadata. Engine requests descriptors. |
| `runtime.memory.PreparedMemoryPlan.ByteSize` | Public nested interface; long cannot describe run-bound sizing. Prepare transports, Runtime evaluates, CPU implements. |
| `runtime.memory.PreparedMemoryPlan.FixedByteSize` | Public nested record; preserves static long declarations without fake counts/backend dependency. |
| `runtime.run.RunMemoryGeometry` | Public final/private constructor; RunState lacks authenticated pre-allocation sizes. Engine/runner/state/CPU consume; no new lease. |
| `backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize` | Public nested record; existing factories capture constants. Affine byte/width/limit recipe. |
| `backend.cpu.internal.ir.CpuAccessPlan.BatchBinding` | Public nested record; existing Binding is resolved. Lowerers prepare, CPU binder resolves. |
| `backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry` | Public nested record; resolved Geometry cannot describe B-varying M/K. Finalizer/executable consume. |
| `backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry` | Public nested record; resolved Geometry cannot predeclare conditional scratch. Supplies actual domains/workspace. |

| Existing owner/symbols affected | Proposed change / retained path |
|---|---|
| Model DynamicDimension/DimensionExpressions/TensorDescriptor; Planning LogicalMemoryRequirement | No record replacement: symbolic meaning/unresolved layouts remain. |
| Compiler CompileDiagnostics.constraints, GraphPredicateProof/typed predicates, inference/AD preflight/final layout closure | Bounded semantic binding and symbolic closure; retain eight CompileArtifacts fields and static/diagnostic APIs. |
| Planning OperationCapabilityQuery/BackendOwnerPlanning; CpuCapabilityProvider static/layout helpers | Keep transport/ownership; add precise **symbolic** answers for every final forward/actual-AD op. Relaxing Prepare alone cannot turn static [64,...] into symbolic support; native/mixed/unqualified ops reject. |
| PrepareContext, PreparationResourceRequirement.Buffer/Workspace, GraphPreparation/assignment/contribution validation | Qualified symbolic branch; set explicit plan mode; compare complete size declarations without evaluation. Preserve static guards, scoped IDs/demand, slot/position order and transactions. |
| PreparedMemoryPlan and nested entries; PreparedRepresentationPlan creator SAMs/CreatedBuffer/InitializedBuffer; RunState/RunStateCreation/PreparedExecutionRunner/PreparedExecutable.bind/hooks | Signatures/mode above; preserve origins/leases/validity/cleanup. Mechanically migrate static/native creator lambdas, not delete their routes. |
| CpuRepresentationRecipes factories; CpuPartitionLowering.LoweredPartition, CpuPartitionPreparationPlan/preparer/finalizer; CpuPreparedExecutable.accepts*/bindCompatible/Invocation | Retain typed recipes and actual-byte dynamic branch. Keep static executable/carrier paths; Invocation is existing, not a tenth addition. |
| CpuAccessPlan.Binding; CpuMatmulLowering.Geometry.pack; CpuAggregateLowering.Geometry/Layout; CpuMatmulIr/CpuAggregateIr/CpuKernelSpecialization, both identity streams and existing loop/MATMUL/aggregate/exact-sum emitters | Reuse numeric formats; `ordinary-batch-v1` discriminates changed baked meaning, excluding actual B/bytes/slots/addresses. Preserve static keys/bytes and fixed-five-limb code. |
| Engine PreparedExecution.valueElementSpans/adapters, AdvancedEngine.runOrdinaryOpen/validateStorage/publication validation, EngineBackendComposition.borrow, RunResult/Publication/materialization | Add dynamic metadata/prefix/actual publication; retain static fast spans, exact-ID path and public owners. |
| Later TrainingSession.bindValues/assembleInputs/validateInputStorage/publication checks; NN ModelExecutionState.validateValues/byteSize/acquisition | Separately qualify dynamic external matching; preserve fixed parameters/seeds, transaction/exclusion/native Arena and saved Models. No new type. |
| Later Scan/ScanBody, ScanBodyInference/ScanGradientRules, PrepareContext.validateScan, CpuGeneralScanIr/Lowering/PreparedExecutable/emitters | Symbolic-row branch, fixed T, same resource owners; preserve static all-valid/fixed/Prepare0011 and independent length gate. No further type. |

Only dynamic branches supersede fixed-only guards/captured sizes. Delete an adapter/guard later
only if all callers have equivalent typed replacements and static tests. No deletion needed here;
useful static/native/NN paths are not obsolete. Inward record/SAM changes require consumer migration.

## Concrete reuse and empty policy

Conceptual acceptance, not current support: compile FLOAT32 X `[B,4]`, W `[4,3]` (both grad-eligible),
publish Y=X@W and L=mean(Y), request actual dX/dW from L with scalar-one seed. Prepare once; all-one
values, no updates, B=64→40→1→64. Y=4, L=4, dY=1/(3B), dX=1/B, every dW=1/3 under existing
rounding. ReductionGradientRules.mean/count uses `oneLike(input).sum(...)`, never a baked 192.

| B | Y shape/bytes | dX shape/bytes | dW shape/bytes | MEAN domain/scratch bytes |
|---:|---|---|---|---|
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 192 / 48 |
| 40 | `[40,3]` / 480 | `[40,4]` / 640 | `[4,3]` / 48 | 120 / 48 |
| 1 | `[1,3]` / 12 | `[1,4]` / 16 | `[4,3]` / 48 | 3 / 48 |
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 192 / 48 |

L=4 bytes, borrowed X=16B/W=48 bytes; materialized dY=12B bytes. Separately materialized AD count
SUM produces `[1,1]`=4 bytes with its own 48-byte scratch. Demand/fusion determines helper buffers
versus locals: record actual final requirements, seeds/transposes/contiguous resources, not an
invented universal total. dW contracts K=B, no B×parameter bank. Identities/positions/routes/code/
schedule stay fixed; all numeric arrays/writable resources are fresh/run-owned.

Recommend B=0 when graph predicates allow zero: Y/dX/dY empty, scalar L=NaN from mandatory empty
MEAN division, nonempty `[4,3]` dW typed-zero empty contraction. MEAN/count scratch still exists;
never skip the whole run or read nonexistent dY. Preserve constituent/static empty, SUM identity,
signed-zero/special-value and rejection behavior. Other operations may reject via their predicates.
Later Training keeps nonempty-parameter/finite-objective-gradient checks: empty-MEAN training
rejects transactionally; qualified empty SUM may supply zero gradients. No automatic step skip or
sample weighting. Failure below representability limits still rolls back and permits retry.

## Scan timing and ownership

Later fixed-T Scan: TBF stride B*F, banks B*H, row history T*B*retained-row-elements, shared history
T*shared-elements, no invented B axis. Dead histories have zero bytes/stores; parameter accumulators
stay parameter-shaped. Training accumulation is SUM; lengths do not change outer MEAN/masking.

Admission reads no lengths. After length producers execute and before this Scan's first output/
padding/history/scratch write, validate accessible aligned INT64 `[B]` span and **all B** values in
`[0,T]`. Invalid last value rejects before earlier rows write; T=0 validates all are zero. B=0 has
empty value domain/no body work, but shared outputs retain their semantic identities. T=0 all-valid
carry preserves initial raw values, dInitialCarry preserves dFinalCarry, invariant sums are typed
zero. Empty domains bypass no validity check; Scan-local failure does not roll back upstream writes.

Preserve caller-owned borrowed storage through result close, result/prepared leases, close precedence,
nonwaiting Runtime close arbitration, reverse attempt-all cleanup/distinct suppression, persistent
rollback and result-after-prepared-close materialization. Failure publishes nothing/closes no caller
storage. Different-B runs/results share no mutable arrays/banks/outputs and each has one RunState.
NN keeps exclusion/Arena lifetime; Training keeps nonempty overlap rejection and parameter/optimizer
transactions. No device-overlap/throughput promise.

## Exact proposed authority and ADR packet

All following text is **PROPOSAL**, not live authority. Insert only after independent final review,
MAIN and user acceptance; root remains sole index, each scoped owner owns only its concern.

1. Root [Core lifecycle](../../../../ARCHITECTURE.md#core-lifecycle), replace universal fixed-only
   fresh-value wording: “Static runValues templates require exact descriptors. Explicitly qualified
   symbolic-batch templates bind B from actual static descriptors through Compiler-owned metadata
   admission. Fixed feature/type/parameter semantics, code, routes and resource identities remain.
   Runtime is Model/Compiler-free and receives numeric recipes/run geometry only.” Add a scope-index
   routing row to the owners below; dependency rules unchanged.
2. [Runtime/Prepare/Engine](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
   modules/runtime: “Each entry declares constant bytes or an immutable numeric recipe. PreparedMemoryPlan
   explicitly declares whether B is required, independently of byte-size independence. Runtime resolves
   same-plan geometry before creators receive exact bytes/fixed alignment; no Model/graph interpretation.”
   In modules/prepare and Prepare lifecycle replace fixed-only exact-byte clauses with “exact fixed bytes
   or complete explicit run-size recipes”; add “Prepare validates mode/declarations/coverage without
   evaluating backend geometry; identities/routes are fixed before finalization.” Run lifecycle adds
   “Actual-byte checks/arrays are run-owned; length VALUE checks remain producer-ordered and scan-local.”
   Public reusable prepared execution replaces dynamic rejection with the admission/layout/publication
   rules above; retain static and close/lease rules unchanged.
3. [Compiler compile artifacts](../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
   add “Eight components remain. Compiler derives BatchBindingPlan from final metadata/exact private
   obligations. Engine invokes bounded descriptor binding, not diagnostic predicates or graph passes;
   nothing is exposed to Runtime.” Automatic differentiation adds “Qualify actual generated dynamic
   closure with existing count/contraction algebra; forward support implies no backward support.”
4. [CPU routes](../../../architecture/contracts/backend-execution.md#cpu-backend-routes), add
   “Qualified single-B generated routes fix code/carriers/routes/declarations during preparation.
   CPU numeric recipes bind actual access/contraction/workspace once before loops. Dynamic identities
   key changed baked meaning, not actual B. Preserve static keys/bytes; no native/mixed/vector/parallel grant.”
5. [General Scan](../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan), add
   “Symbolic B qualifies separately with fixed T/features/state dimensions. Static all-valid/fixed
   behavior remains. Ordinary dynamic CPU enables no Scan path. Geometry admission never replaces
   all-B length VALUE checks after producers/before first writes, including T=0/B=0.” Keep length semantics.
6. [Owned Models](../../../architecture/contracts/extensions-training.md#owned-tensor-models) and
   modules/training, add “Dynamic external signatures qualify separately over public Engine with one
   parameter/optimizer state and existing exclusion/transactions; accumulation is SUM. Symbolic graph/
   defineTensor acquisition needs Compiler cold reconstruction/Checkpoint copy, save/load additionally
   wire qualification. Builder acquisition qualifies independently. None blocks ordinary Engine+CPU;
   static saved Models/readers stay supported.” Generic graph source/values retains static admission.

Embedded exact ADR draft, future `docs/design/decisions/0050-dynamic-batch-geometry.md`
(reserve number at accepted cutover):

> # ADR 0050 Dynamic batch geometry through numeric recipes
>
> Status: Proposed — 2026-10-24.
>
> Context and drivers: symbolic Model dimensions meet fixed-only capability/sizing/publication.
> Require one immutable compile/prepare, actual allocation, semantic ownership, hot paths and static lifetime compatibility.
>
> Options: fixed-only cannot meet reuse; maximum banks/per-B fullprepare caches impose different
> lifecycle/allocation. Select prepared numeric recipes with per-run binding.
>
> Decision: recommend Prepare0012's signatures/nine-type map. Keep eight CompileArtifacts components.
> Compiler binds metadata, Prepare fixes identities/mode without numeric evaluation, CPU owns recipes,
> Runtime creates one run and Engine publishes actual descriptors. Permit B=0 under operation predicates.
>
> Rationale: conditional scratch is not universally affine; execution mode is not byte independence.
> No extra graph/owner is needed. Keep proved five-limb F32 code; no speculative dynamic-limb generality.
>
> Consequences: migrate inward record/creator consumers explicitly; no binary compatibility claim.
> Keep static/native paths. Qualify actual AD closure, sizes/publication and failure/concurrency.
> Ordinary CPU precedes Scan and separate NN/cold/wire/native/tuning gates. Matched-work uses invocation
> B on both sides, not an identical-JIT/no-penalty promise.
>
> Related: Prepare0012 brief/proposal and root/scoped headings linked above.

At cutover synchronize existing [Runtime Prepare backend boundary](../../../architecture/runtime-prepare-backend-boundary.md)
and [module boundaries](../../../architecture/module-boundaries.md). Finalize affected Javadoc for
every signature/record/hook: all inputs/results, units, null/overflow/precedence, ownership/threading,
static migration. Target glossary dimension-binding/prepared-memory/run-state terms; no edits now.

## Acceptance and next gate

Independent final design review and MAIN delivery acceptance are complete. User architecture
approval is explicit; MAIN admitted [Prepare0012A](tasks/0012a-dynamic-batch-authority-cutover.md)
for authority only after its 2026-10-24 17:41:11 UTC dependency/base re-verification. Authority,
[ADR0050](../../../design/decisions/0050-dynamic-batch-geometry.md), explanations and glossary are
enacted and MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open P1/P2, docs/glossary
review complete. User implementation approval exists, but no Ready Java brief until MAIN separately
admits planner output; no automatic successor or current dynamic execution claim. Design body preserved.

First vertical must compile/query genuinely symbolic B and execute actual AD closure: MATMUL,
SUM/MEAN/count, arithmetic/splats, expand/divide, rank edits, transpose/contiguous/sumToShape and K=B
adjoints, not forward-only provider support. Prove 64→40→1→64 before generic Scan; NN/Training
one-state reuse follows separately. NN cold acquisition, wire and independent builder gates are not
Engine prerequisites; static length skipping is independent/not cancelled or held on distant NN work.

Later focused/module tests/Javadoc extend PrepareContextTest, CompileDiagnosticsTest,
RunStateCreationTest, EngineTypedLifecycleTest, TrainingSessionValuesTest and CPU access/MATMUL/
aggregate tests. Include B=0/1, `X[B,0]` all-constant-byte mode rejection before creators, nonempty
empty contractions, conditional 0/48 scratch/five-limb proof, shape/type/flags/layout mismatch,
absent/dead/short/inaccessible storage, overflow before borrow, produced invalid-last length/T=0,
dead histories, allocation/bind/publication failures, suppression/retry/close races/different-B
concurrency. Retain static/native/view/empty/length/persistence guards. MAIN owns architecture/
conformance/integration join; enforce no Runtime Model/Compiler edge or Prepare backend interpretation,
unchanged extension edges and nine-type inventory.

Generated/direct evidence gives both sides fixed F/H/T and **invocation B**. Compare algorithm,
hot-loop/dataflow and constant-five-limb state; inspect Class-Files for hidden per-element/step
dispatch/allocation/boxing/reflection/recipe work. Short warmed matched-work timing separates cold
generation/classload/JIT, admission and kernel costs. Diagnose/justify overhead, promise no identical
JIT or zero penalty. This design runs source/contract, links, size, whitespace and protection checks only.
