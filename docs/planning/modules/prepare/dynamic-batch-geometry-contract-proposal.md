# Dynamic [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) geometry and resource contract proposal

## Purpose and status

Complete — reviewed DESIGN delivery, 2026-10-24; user architecture approval is now recorded.
Independent Class C design/docs APPROVE, zero open P1/P2; MAIN accepts delivery under its conditional
[admission](../../../glossary.md#term-admission). User “ano” explicitly accepts authority cutover, subsequent ordinary [CPU](../../../glossary.md#term-cpu) forward/actual
first-order backward and bounded final A/B. [Prepare0012A](tasks/0012a-dynamic-batch-authority-cutover.md)
is Complete: authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2, docs/glossary review complete. No [Ready](../../../glossary.md#term-ready) Java brief until MAIN separately admits planner output.
[Prepare0012](tasks/0012-dynamic-batch-geometry-and-resource-contract.md) records launch/review evidence;
[current authority](../../../../ARCHITECTURE.md) carries the enacted/accepted future protocol;
current Java dynamic [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) remains unsupported. The proposal body remains historical design. Base:
HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted dirty tree.

Recommend one symbolic [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names)/preparation: [Compiler](../../../glossary.md#term-compiler) binds semantic [metadata](../../../glossary.md#term-metadata), [Prepare](../../../glossary.md#prepare) fixes
[identities](../../../glossary.md#term-identity-meanings), CPU prepares numeric recipes, [Runtime](../../../glossary.md#term-runtime) creates isolated resources, [Engine](../../../glossary.md#term-engine) publishes actual
[descriptors](../../../glossary.md#term-descriptor). Runtime already allocates freshly per [run](../../../glossary.md#term-run); the gap is sizing, not freshness. The
[source inventory](../../dynamic-batch-execution.md#current-truth) identifies current fixed-only seams.

First [scope](../../../glossary.md#term-scope): ordinary [homogeneous](../../../glossary.md#term-homogeneous) [FLOAT32](../../../glossary.md#term-float32) CPU forward **and actual first-order backward**, one named
DynamicDimension B, fixed [feature](../../../glossary.md#term-software-feature)/[parameter](../../../glossary.md#term-parameter) [dimensions](../../../glossary.md#dimension), [rank](../../../glossary.md#term-rank)-at-most-two boundaries and needed [AD](../../../glossary.md#autograd)
[views](../../../glossary.md#term-view-meanings). [General Scan](../../../glossary.md#general-differentiable-scan) follows with fixed T/F/H. No B_max, per-B fullprepare [cache](../../../glossary.md#term-cache), prepared mutation,
new session/executor, second [DAG](../../../glossary.md#term-directed-acyclic-graph-dag)/proof runtime or [graph](../../../glossary.md#graph) interpreter. Native/mixed, wider [dtype](../../../glossary.md#data-type--datatype),
variable T, higher-order, [persistence](../../../glossary.md#term-persistence) and tuning remain separate gates.

Size exception: this [atomic](../../../glossary.md#resource-transaction-and-rollback) cross-[owner](../../../glossary.md#term-owner) packet exceeds the 25 KB [target](../../../glossary.md#term-target) because it keeps the exact
four boundary [signatures](../../../glossary.md#term-signature), nine-type/replacement map, empty/resource example and draft authority/[ADR](../../../glossary.md#term-architecture-decision-record-adr)
changes together for one decision. Splitting them would separate the mode/sizing/[publication](../../../glossary.md#term-publication) contract
from its migration and approval conditions. The task brief remains below 200 lines/15 KB.

## Exact proposed [signatures](../../../glossary.md#term-signature) and semantics

These are future declarations, not runnable examples. Package prefix: `io.github.pho001.synaptik.`.
For every signature below: references/results are non-null; lists [snapshot membership](../../../glossary.md#occurrence-identity-and-canonical-form) and reject null
[elements](../../../glossary.md#term-element) in encounter order. Null throws `NullPointerException`, invalid fields/[metadata](../../../glossary.md#term-metadata)
`IllegalArgumentException`, checked [overflow](../../../glossary.md#term-overflow) `ArithmeticException`, invalid dense index
`IndexOutOfBoundsException`. [Immutable](../../../glossary.md#term-immutable) recipes are [thread-safe](../../../glossary.md#term-thread-safe)/storage-free; resolved arrays belong
to one [run](../../../glossary.md#term-run). Additional [lifecycle](../../../glossary.md#lifecycle) failures retain their existing exception/precedence contracts.

### [Compiler](../../../glossary.md#term-compiler) [metadata](../../../glossary.md#term-metadata)

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

[`prepare`](../../../glossary.md#prepare) runs once from final [graph](../../../glossary.md#graph) facts and same-package `CompileDiagnostics.constraints()`.
Preindex [descriptor](../../../glossary.md#term-descriptor) templates, input/value/[publication](../../../glossary.md#term-publication) positions and exact typed obligations; reject
unsupported symbols/expressions. Preserve all eight CompileArtifacts components/order and logical
memory descriptors. No diagnostic parsing, public predicate language or [Tensor](../../../glossary.md#tensor)/storage sidecar.
Existing typed predicates/GraphPredicateProof gain bound [evaluation](../../../glossary.md#term-evaluation) inside Compiler only.

`bind` consumes actual fully static descriptors in final compiled input order. Derive B>=0 from the
first declared named-[axis](../../../glossary.md#term-axis) [occurrence](../../../glossary.md#term-occurrence), not storage bytes; require every occurrence, including unused
inputs, to agree. Match [rank](../../../glossary.md#term-rank)/fixed axes/[dtype](../../../glossary.md#data-type--datatype)/requiresGrad exactly; [parameters](../../../glossary.md#term-parameter)/[seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) stay fixed.
Only expressions determined by B/[constants](../../../glossary.md#term-constant) qualify; output-only symbols/anonymous unknowns reject
at prepare. Unsatisfied predicates/negative [dimensions](../../../glossary.md#dimension) reject before [borrowing](../../../glossary.md#term-borrowing). No per-[run](../../../glossary.md#term-run) graph
traversal, [operation](../../../glossary.md#operation) [inference](../../../glossary.md#term-inference), recapture, optimization or [backend](../../../glossary.md#backend) selection.

Dynamic caller templates require resolved zero-offset dense [layouts](../../../glossary.md#layout) with actual [canonical](../../../glossary.md#term-canonical-form) [strides](../../../glossary.md#term-stride);
`isView=true` dense reads may qualify. Absent layout is not silently canonical. Static templates
keep exact existing [admission](../../../glossary.md#term-admission), including positional unresolved-layout behavior. Proved [AD](../../../glossary.md#autograd) [transpose](../../../glossary.md#term-transpose)/
expand [views](../../../glossary.md#term-view-meanings) are separate numeric mappings. Materialized outputs become canonical non-view;
qualified view publications retain actual geometry. `descriptor(index)` resolves **only requested**
metadata on demand; [Engine](../../../glossary.md#term-engine) calls it for publications. Do not eagerly allocate descriptors/[Shapes](../../../glossary.md#shape) for
intermediates. `valueCount` is graph list size, not maximum [ValueId](../../../glossary.md#logical-valueid-versus-physical-memory-slot). Numeric cold cost is
O(inputs + requirements + obligations + publications), never a [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) rebuild/per-[element](../../../glossary.md#term-element) evaluation.

### [Runtime](../../../glossary.md#term-runtime) declarations and creation

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

Sizes are bytes, nonnegative long; fixed [alignment](../../../glossary.md#term-alignment) is bytes, power of two from 1 through `1L<<62`.
Workspace IDs stay nonnegative/local. `size()` retains the recipe. Old long constructors wrap
FixedByteSize; old `byteSize()` returns its exact count (including zero), but throws
`IllegalStateException` for a [run](../../../glossary.md#term-run)-bound declaration. No fake zero/-1/maximum count. Record-component/
reflection/[binary](../../../glossary.md#term-binary-descriptor-and-metadata) changes and creator SAM migration are explicit; static constructor/accessor source
compatibility is retained, not a record [ABI](../../../glossary.md#term-application-binary-interface-abi) promise.

`fixedByteSize()` is a **structural declaration marker**: present for FixedByteSize, absent for
run-bound recipes even if their results happen to be [constant](../../../glossary.md#term-constant). `byteSize(B)` validates B>=0 and
returns exact checked bytes. Recipes contain numeric facts only: no [graphs](../../../glossary.md#graph), [Model](../../../glossary.md#term-model) expressions,
value reads, caches/providers, route selection, physical allocation or requirement discovery.
[Prepare](../../../glossary.md#prepare) compares structural declarations/alignment/coverage without evaluating recipes. Runtime
evaluates them; [CPU](../../../glossary.md#term-cpu) owns their physical meaning. This explicitly changes today's fixed-only contract.

**Execution mode is independent of byte independence.** Prepare sets `requiresBatchSize=true`
from the qualified symbolic projection even if every size is constant, e.g. `X[B,0].sum()`.
Old two-list plan construction means false and rejects run-bound declarations. `resolve` requires
present B iff this mode is true, then resolves all [buffers](../../../glossary.md#term-buffer)/workspaces in plan order into private
arrays. Absent-B runner/creation/state overloads reject a true-mode plan before invoking creators
or accepting resource [ownership](../../../glossary.md#term-ownership), even for all-constant bytes. State-aware compatibility defaults
also use this explicit mode, never `fixedByteSize` as a mode test.

Geometry authenticates the exact plan reference; foreign geometry rejects. [Engine](../../../glossary.md#term-engine) resolves before
[borrowing](../../../glossary.md#term-borrowing); Runtime runner still acquires its unique prepared [lease](../../../glossary.md#term-lease) before inspecting supplied
inputs/geometry. Static overloads retain static [admission](../../../glossary.md#term-admission)/order. Creators validate numeric/type
constraints and allocate exactly the requested usable span (allocator [padding](../../../glossary.md#padding) is not capacity),
returning fresh non-null objects even for zero bytes. InitializedBuffer fills the actual count.
Keep complete caller [validation](../../../glossary.md#term-validation) before creation, duplicate-result rejection, origins/validity and
reverse [rollback](../../../glossary.md#term-rollback). Allocation errors propagate; null results fail; [backend](../../../glossary.md#backend) owns rollback before
return, setup/run afterward. No unsafe zero-[argument](../../../glossary.md#term-argument) dynamic adapter.

### [CPU](../../../glossary.md#term-cpu) numeric recipes and [binding](../../../glossary.md#term-binding)

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

CPU records are public only in existing internal packages. BatchBinding validates equal-[rank](../../../glossary.md#term-rank) lists,
nonnegative coefficients/base and computes [extent](../../../glossary.md#term-extent)/[stride](../../../glossary.md#term-stride) `constant+factor*B` in **[elements](../../../glossary.md#term-element)**, with
checked counts/spans. One B-bearing [axis](../../../glossary.md#term-axis) per represented [shape](../../../glossary.md#shape); proved [canonical](../../../glossary.md#term-canonical-form)/[transpose](../../../glossary.md#term-transpose)/expand
mappings only. Preserve exact [Model](../../../glossary.md#term-model) empty strides: `[F,0]` has `[0,1]`, not max(1,B). Regimes/axis
roles stay prepared; no fake representative B. BatchByteSize uses checked `constantBytes +
bytesPerBatch*B`, nonnegative coefficients, positive supported width, divisibility and CPU limits.

[Rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../glossary.md#matrix-multiplication--matmul) resolves existing Geometry/pack: forward M/K/N=`B/F/H`, dX=`B/H/F`, dW=`F/B/H`
with actual transposed strides. Analysis fixes generated [scalar](../../../glossary.md#term-scalar), single-range routes independently
of B; no [admission](../../../glossary.md#term-admission)-time profitability selection. [Carrier](../../../glossary.md#term-storage-carrier) pattern/code stays fixed; unsupported
carriers reject, not select another [artifact](../../../glossary.md#term-artifact). Pack actual geometry once into existing Invocation.

Aggregate BatchGeometry binds actual domains/output counts and conditional workspace, not maximum
banks. For first-[slice](../../../glossary.md#slice) [FLOAT32](../../../glossary.md#term-float32)/int-limited N, `signedBits=278+ceil(log2(N))` ([cardinality](../../../glossary.md#term-cardinality) term zero
at N<=1), at most 309 bits: **five limbs for all admitted N, including 0/1**. Retain optimal [constant](../../../glossary.md#term-constant)-
five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs) CpuExactSumEmitter. Single-range scratch is 48 bytes (`8+8*5`) if outputCount>0, otherwise
zero: non-affine actual sizing. Do not require dynamic-limb emission for this slice; [F64](../../../glossary.md#floating-point-precision-and-rounding)/long domains
need later [qualification](../../../glossary.md#term-qualification). [SUM-to-Shape](../../../glossary.md#sum-to-shape) handles B=1 copy/[singleton](../../../glossary.md#term-singleton) at entry in the same code/route/
slot; zero scratch never discovers/removes a requirement. Preserve actual count division/numerics.

Do not [run](../../../glossary.md#term-run) Model lowerers or element-enumerating [injectivity](../../../glossary.md#storage-mapping-contiguity-and-injectivity) checks per invocation: canonical
write injectivity is proved at [prepare](../../../glossary.md#prepare), binding checks numeric bounds. New hooks default to old
static hooks only for false-mode plans; CPU overrides true-mode actual bytes from state.geometry(),
retaining type/access/[alignment](../../../glossary.md#term-alignment)/read-only/overlap checks. bindCompatible/Invocation retains direct
typed carriers and fresh packed/access arrays, no shared currentB/ThreadLocal or hot recipe [dispatch](../../../glossary.md#term-dispatch).

### [Engine](../../../glossary.md#term-engine) [metadata](../../../glossary.md#term-metadata) and [publication](../../../glossary.md#term-publication)

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

Public Engine [signatures](../../../glossary.md#term-signature) stay. [`run`](../../../glossary.md#term-run) keeps exact-ID/arbitrary-order semantics; fresh B values use
runValues without [prototype](../../../glossary.md#term-prototype) mutation. Dynamic order: existing Engine/[owner](../../../glossary.md#term-owner)/closed-prepared gates;
[snapshot](../../../glossary.md#term-snapshot)/count/null; semantic and numeric/size [preflight](../../../glossary.md#term-preflight); snapshot all storage associations once
and validate all; borrow; [Runtime](../../../glossary.md#term-runtime) [lease](../../../glossary.md#term-lease)/create/cold bind; execute; complete result. New [borrowing](../../../glossary.md#term-borrowing)
wraps exactly the referenced prefix, not excess capacity, without owning caller storage; old static
borrow stays. Invalid prefix rejects; failure cleans wrappers only.

Before borrow/allocation check fixed [axes](../../../glossary.md#term-axis)/type/flags, [strides](../../../glossary.md#term-stride)/counts, byte products, [referenced
span](../../../glossary.md#referenced-element-span)/offset sums and aggregate allocation bounds including [alignment](../../../glossary.md#term-alignment) [padding](../../../glossary.md#padding)/representation
multiplicity. First dynamic [CPU](../../../glossary.md#term-cpu) extents/counts/strides/offsets/referenced-[element](../../../glossary.md#term-element) spans <=
Integer.MAX_VALUE; bytes/aggregate bounds <=Long.MAX_VALUE. These are representability limits,
not public B_max. Zero count cannot hide overflowing fixed trailing geometry or unsupported unused
declarations. Require matching, present/live/thread-accessible storage and sufficient capacity.
Physical address/alignment and writable overlap checks follow allocation, before execution.

Engine preindexes publication-to-value-list positions and resolves only requested [descriptors](../../../glossary.md#term-descriptor).
Exact [occurrence](../../../glossary.md#term-occurrence) count is required. [Gradient](../../../glossary.md#term-gradient) metadata comes from generated gradient **values**,
not [target](../../../glossary.md#term-target) flags; retain tensorId/role/derivativeOrder/targetIndex and distinct authenticated [aliases](../../../glossary.md#aliasing-and-arena).
[Materialization](../../../glossary.md#materialization) uses actual [shape](../../../glossary.md#shape)/[layout](../../../glossary.md#layout)/bytes and returns [detached](../../../glossary.md#detached-model-results) [HostTensorValue](../../../glossary.md#host-tensor-value--hosttensorvalue). Failure closes
inward result then reverse borrows with distinct suppression, publishing nothing. No [Model](../../../glossary.md#term-model) metadata
or [Compiler](../../../glossary.md#term-compiler) [dependency](../../../glossary.md#term-dependency) enters Runtime.

## Complete type and replacement map

These nine additions include nested types; no type moves/removals or new [module](../../../glossary.md#term-module)/[dependency](../../../glossary.md#term-dependency).

| Added type under the package prefix | Visibility; why existing type cannot serve; consumers |
|---|---|
| `compiler.BatchBindingPlan` | Public final; diagnostics are not [admission](../../../glossary.md#term-admission) recipes. [Engine](../../../glossary.md#term-engine) retains prepared semantic [metadata](../../../glossary.md#term-metadata). |
| `compiler.BatchBindingPlan.Binding` | Public static final/private constructor; CompileArtifacts cannot hold per-[run](../../../glossary.md#term-run) metadata. Engine requests [descriptors](../../../glossary.md#term-descriptor). |
| `runtime.memory.PreparedMemoryPlan.ByteSize` | Public nested interface; long cannot describe run-bound sizing. [Prepare](../../../glossary.md#prepare) transports, [Runtime](../../../glossary.md#term-runtime) evaluates, [CPU](../../../glossary.md#term-cpu) implements. |
| `runtime.memory.PreparedMemoryPlan.FixedByteSize` | Public nested record; preserves static long declarations without fake counts/[backend](../../../glossary.md#backend) dependency. |
| `runtime.run.RunMemoryGeometry` | Public final/private constructor; [RunState](../../../glossary.md#run-state--runstate) lacks authenticated pre-allocation sizes. Engine/runner/state/CPU consume; no new [lease](../../../glossary.md#term-lease). |
| `backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize` | Public nested record; existing factories [capture](../../../glossary.md#term-capture-meanings) [constants](../../../glossary.md#term-constant). Affine byte/width/limit recipe. |
| `backend.cpu.internal.ir.CpuAccessPlan.BatchBinding` | Public nested record; existing [Binding](../../../glossary.md#term-binding) is resolved. Lowerers prepare, CPU binder resolves. |
| `backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry` | Public nested record; resolved Geometry cannot describe B-varying M/K. Finalizer/executable consume. |
| `backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry` | Public nested record; resolved Geometry cannot predeclare conditional scratch. Supplies actual domains/workspace. |

| Existing [owner](../../../glossary.md#term-owner)/symbols affected | Proposed change / retained path |
|---|---|
| [Model](../../../glossary.md#term-model) DynamicDimension/DimensionExpressions/TensorDescriptor; [Planning](../../../glossary.md#term-planning) LogicalMemoryRequirement | No record replacement: symbolic meaning/unresolved [layouts](../../../glossary.md#layout) remain. |
| [Compiler](../../../glossary.md#term-compiler) CompileDiagnostics.constraints, GraphPredicateProof/typed predicates, [inference](../../../glossary.md#term-inference)/[AD](../../../glossary.md#autograd) [preflight](../../../glossary.md#term-preflight)/final layout closure | Bounded semantic binding and symbolic closure; retain eight CompileArtifacts fields and static/diagnostic APIs. |
| Planning [OperationCapabilityQuery](../../../glossary.md#operation-capability-query--operationcapabilityquery)/BackendOwnerPlanning; CpuCapabilityProvider static/layout helpers | Keep transport/[ownership](../../../glossary.md#term-ownership-meanings); add precise **symbolic** answers for every final forward/actual-AD op. Relaxing Prepare alone cannot turn static [64,...] into symbolic support; native/mixed/unqualified ops reject. |
| [PrepareContext](../../../glossary.md#prepare-context-preparecontext), PreparationResourceRequirement.[Buffer](../../../glossary.md#term-buffer)/Workspace, [GraphPreparation](../../../glossary.md#graph-preparation--graphpreparation)/assignment/contribution [validation](../../../glossary.md#term-validation) | Qualified symbolic branch; set explicit plan mode; compare complete size declarations without [evaluation](../../../glossary.md#term-evaluation). Preserve static guards, scoped IDs/demand, slot/position order and transactions. |
| [PreparedMemoryPlan](../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and nested entries; [PreparedRepresentationPlan](../../../glossary.md#prepared-representation-plan--preparedrepresentationplan) creator SAMs/CreatedBuffer/InitializedBuffer; RunState/RunStateCreation/[PreparedExecutionRunner](../../../glossary.md#prepared-execution-runner--preparedexecutionrunner)/[PreparedExecutable](../../../glossary.md#prepared-executable--preparedexecutable).bind/hooks | [Signatures](../../../glossary.md#term-signature)/mode above; preserve origins/leases/validity/cleanup. Mechanically migrate static/native creator lambdas, not delete their routes. |
| CpuRepresentationRecipes factories; CpuPartitionLowering.LoweredPartition, CpuPartitionPreparationPlan/preparer/finalizer; CpuPreparedExecutable.accepts*/bindCompatible/Invocation | Retain typed recipes and actual-byte dynamic branch. Keep static executable/[carrier](../../../glossary.md#term-storage-carrier) paths; Invocation is existing, not a tenth addition. |
| CpuAccessPlan.Binding; CpuMatmulLowering.Geometry.pack; CpuAggregateLowering.Geometry/Layout; CpuMatmulIr/CpuAggregateIr/CpuKernelSpecialization, both [identity](../../../glossary.md#term-identity-meanings) streams and existing loop/[MATMUL](../../../glossary.md#matrix-multiplication--matmul)/aggregate/exact-sum emitters | Reuse numeric formats; `ordinary-batch-v1` discriminates changed baked meaning, excluding actual B/bytes/slots/addresses. Preserve static keys/bytes and fixed-five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs) code. |
| Engine [PreparedExecution](../../../glossary.md#prepared-execution--preparedexecution).valueElementSpans/adapters, AdvancedEngine.runOrdinaryOpen/validateStorage/[publication](../../../glossary.md#term-publication) validation, EngineBackendComposition.borrow, [RunResult](../../../glossary.md#run-result--runresult)/Publication/[materialization](../../../glossary.md#materialization) | Add dynamic metadata/prefix/actual publication; retain static fast spans, exact-ID path and public owners. |
| Later [TrainingSession](../../../glossary.md#training-session--trainingsession).bindValues/assembleInputs/validateInputStorage/publication checks; [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) ModelExecutionState.validateValues/byteSize/acquisition | Separately qualify dynamic external matching; preserve fixed [parameters](../../../glossary.md#term-parameter)/[seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), [transaction](../../../glossary.md#term-transaction)/exclusion/native [Arena](../../../glossary.md#term-arena) and saved Models. No new type. |
| Later [Scan](../../../glossary.md#term-cumulative-scan)/ScanBody, ScanBodyInference/ScanGradientRules, PrepareContext.validateScan, CpuGeneralScanIr/[Lowering](../../../glossary.md#lowering)/PreparedExecutable/emitters | Symbolic-row branch, fixed T, same resource owners; preserve static all-valid/fixed/Prepare0011 and independent length gate. No further type. |

Only dynamic branches supersede fixed-only guards/captured sizes. Delete an adapter/[guard](../../../glossary.md#term-guard) later
only if all callers have equivalent typed replacements and static tests. No deletion needed here;
useful static/native/NN paths are not obsolete. Inward record/SAM changes require consumer migration.

## Concrete reuse and empty policy

Conceptual acceptance, not current support: [compile](../../../glossary.md#compile) [FLOAT32](../../../glossary.md#term-float32) X `[B,4]`, W `[4,3]` (both grad-eligible),
publish Y=X@W and L=[mean](../../../glossary.md#term-mean)(Y), request actual dX/dW from L with [scalar](../../../glossary.md#term-scalar)-one [seed](../../../glossary.md#term-seed). [Prepare](../../../glossary.md#prepare) once; all-one
values, no updates, B=64→40→1→64. Y=4, L=4, dY=1/(3B), dX=1/B, every dW=1/3 under existing
[rounding](../../../glossary.md#term-rounding). ReductionGradientRules.mean/count uses `oneLike(input).sum(...)`, never a baked 192.

| B | Y [shape](../../../glossary.md#shape)/bytes | dX shape/bytes | dW shape/bytes | MEAN domain/scratch bytes |
|---:|---|---|---|---|
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 192 / 48 |
| 40 | `[40,3]` / 480 | `[40,4]` / 640 | `[4,3]` / 48 | 120 / 48 |
| 1 | `[1,3]` / 12 | `[1,4]` / 16 | `[4,3]` / 48 | 3 / 48 |
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 192 / 48 |

L=4 bytes, [borrowed](../../../glossary.md#ownership-borrowing-and-lease) X=16B/W=48 bytes; materialized dY=12B bytes. Separately materialized [AD](../../../glossary.md#autograd) count
SUM produces `[1,1]`=4 bytes with its own 48-byte scratch. Demand/[fusion](../../../glossary.md#term-fusion) determines helper [buffers](../../../glossary.md#term-buffer)
versus locals: record actual final requirements, [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/transposes/[contiguous](../../../glossary.md#term-contiguous) resources, not an
invented universal total. dW contracts K=B, no B×[parameter](../../../glossary.md#term-parameter) bank. [Identities](../../../glossary.md#term-identity-meanings)/positions/routes/code/
schedule stay fixed; all numeric arrays/writable resources are fresh/[run](../../../glossary.md#term-run)-owned.

Recommend B=0 when [graph](../../../glossary.md#graph) predicates allow zero: Y/dX/dY empty, scalar L=[NaN](../../../glossary.md#term-nan) from mandatory empty
MEAN division, nonempty `[4,3]` dW typed-zero empty [contraction](../../../glossary.md#contraction-dimension). MEAN/count scratch still exists;
never skip the whole run or read nonexistent dY. Preserve constituent/static empty, SUM [identity](../../../glossary.md#term-identity-meanings),
[signed-zero](../../../glossary.md#nan-infinity-and-signed-zero)/special-value and rejection behavior. Other operations may reject via their predicates.
Later Training keeps nonempty-parameter/[finite](../../../glossary.md#term-finite)-[objective](../../../glossary.md#term-objective)-[gradient](../../../glossary.md#term-gradient) checks: empty-MEAN training
rejects transactionally; qualified empty SUM may supply zero gradients. No automatic step skip or
[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) weighting. Failure below representability limits still rolls back and permits retry.

## [Scan](../../../glossary.md#term-cumulative-scan) timing and [ownership](../../../glossary.md#term-ownership-meanings)

Later fixed-T Scan: [TBF](../../../glossary.md#term-tbf) [stride](../../../glossary.md#term-stride) B*F, banks B*H, row history T*B*retained-row-[elements](../../../glossary.md#term-element), shared history
T*shared-elements, no invented B [axis](../../../glossary.md#term-axis). Dead histories have zero bytes/stores; [parameter](../../../glossary.md#term-parameter) accumulators
stay parameter-shaped. Training accumulation is SUM; lengths do not change outer [MEAN](../../../glossary.md#term-mean)/masking.

[Admission](../../../glossary.md#term-admission) reads no lengths. After length [producers](../../../glossary.md#term-producer) execute and before this Scan's first output/
[padding](../../../glossary.md#padding)/history/scratch write, validate accessible aligned [INT64](../../../glossary.md#data-type--datatype) `[B]` span and **all B** values in
`[0,T]`. Invalid last value rejects before earlier rows write; T=0 validates all are zero. B=0 has
empty value domain/no body work, but shared outputs retain their semantic [identities](../../../glossary.md#term-identity-meanings). T=0 all-valid
[carry](../../../glossary.md#term-carry) preserves initial raw values, dInitialCarry preserves dFinalCarry, [invariant](../../../glossary.md#term-invariant) sums are typed
zero. Empty domains bypass no validity check; Scan-local failure does not roll back upstream writes.

Preserve caller-owned [borrowed](../../../glossary.md#ownership-borrowing-and-lease) storage through result close, result/prepared [leases](../../../glossary.md#term-lease), close precedence,
nonwaiting [Runtime](../../../glossary.md#term-runtime) close arbitration, reverse attempt-all cleanup/distinct suppression, persistent
[rollback](../../../glossary.md#term-rollback) and result-after-prepared-close [materialization](../../../glossary.md#materialization). Failure publishes nothing/closes no caller
storage. Different-B runs/results share no [mutable](../../../glossary.md#term-mutable) arrays/banks/outputs and each has one [RunState](../../../glossary.md#run-state--runstate).
[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) keeps exclusion/[Arena](../../../glossary.md#term-arena) [lifetime](../../../glossary.md#term-lifetime); Training keeps nonempty overlap rejection and parameter/[optimizer](../../../glossary.md#term-optimizer)
transactions. No [device](../../../glossary.md#term-device)-overlap/[throughput](../../../glossary.md#term-throughput) promise.

## Exact proposed authority and [ADR](../../../glossary.md#term-architecture-decision-record-adr) packet

All following text is **PROPOSAL**, not live authority. Insert only after independent final review,
MAIN and user acceptance; root remains sole index, each scoped [owner](../../../glossary.md#term-owner) owns only its concern.

1. Root [Core lifecycle](../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle))</span>, replace universal fixed-only
   fresh-value wording: “Static runValues templates require exact descriptors. Explicitly qualified
   symbolic-batch templates bind B from actual static descriptors through Compiler-owned metadata
   admission. Fixed feature/type/parameter semantics, code, routes and resource identities remain.
   Runtime is Model/Compiler-free and receives numeric recipes/run geometry only.” Add a scope-index
   routing row to the owners below; dependency rules unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([descriptors glossary](../../../glossary.md#term-descriptor); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Compiler glossary](../../../glossary.md#term-compiler); [metadata glossary](../../../glossary.md#term-metadata); [admission glossary](../../../glossary.md#term-admission); [feature glossary](../../../glossary.md#term-software-feature); [parameter glossary](../../../glossary.md#term-parameter); [Runtime glossary](../../../glossary.md#term-runtime); [Model glossary](../../../glossary.md#term-model); [run glossary](../../../glossary.md#term-run); [scope glossary](../../../glossary.md#term-scope); [dependency glossary](../../../glossary.md#term-dependency))</span>
2. [Runtime/Prepare/Engine](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime)<span data-glossary-navigation="true"> ([Prepare glossary](../../../glossary.md#prepare); [Engine glossary](../../../glossary.md#term-engine))</span>,
   modules/runtime: “Each entry declares constant bytes or an immutable numeric recipe. PreparedMemoryPlan
   explicitly declares whether B is required, independently of byte-size independence. Runtime resolves
   same-plan geometry before creators receive exact bytes/fixed alignment; no Model/graph interpretation.”
   In modules/prepare and Prepare lifecycle replace fixed-only exact-byte clauses with “exact fixed bytes
   or complete explicit run-size recipes”; add “Prepare validates mode/declarations/coverage without
   evaluating backend geometry; identities/routes are fixed before finalization.” Run lifecycle adds
   “Actual-byte checks/arrays are run-owned; length VALUE checks remain producer-ordered and scan-local.”
   Public reusable prepared execution replaces dynamic rejection with the admission/layout/publication
   rules above; retain static and close/lease rules unchanged.<span data-glossary-navigation="true"> ([modules glossary](../../../glossary.md#term-module); [constant glossary](../../../glossary.md#term-constant); [immutable glossary](../../../glossary.md#term-immutable); [PreparedMemoryPlan glossary](../../../glossary.md#prepared-memory-plan--preparedmemoryplan); [alignment glossary](../../../glossary.md#term-alignment); [graph glossary](../../../glossary.md#graph); [backend glossary](../../../glossary.md#backend); [producer glossary](../../../glossary.md#term-producer); [scan glossary](../../../glossary.md#term-cumulative-scan); [prepared execution glossary](../../../glossary.md#prepared-execution--preparedexecution); [layout glossary](../../../glossary.md#layout); [publication glossary](../../../glossary.md#term-publication); [lease glossary](../../../glossary.md#term-lease))</span>
3. [Compiler compile artifacts](../../../architecture/contracts/compiler-autograd.md#compile-artifacts)<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../glossary.md#compile-artifacts))</span>,
   add “Eight components remain. Compiler derives BatchBindingPlan from final metadata/exact private
   obligations. Engine invokes bounded descriptor binding, not diagnostic predicates or graph passes;
   nothing is exposed to Runtime.” Automatic differentiation adds “Qualify actual generated dynamic
   closure with existing count/contraction algebra; forward support implies no backward support.”<span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding); [Automatic differentiation glossary](../../../glossary.md#autograd); [contraction glossary](../../../glossary.md#contraction-dimension))</span>
4. [CPU routes](../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu))</span>, add
   “Qualified single-B generated routes fix code/carriers/routes/declarations during preparation.
   CPU numeric recipes bind actual access/contraction/workspace once before loops. Dynamic identities
   key changed baked meaning, not actual B. Preserve static keys/bytes; no native/mixed/vector/parallel grant.”<span data-glossary-navigation="true"> ([vector glossary](../../../glossary.md#term-vector))</span>
5. [General Scan](../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General Scan glossary](../../../glossary.md#general-differentiable-scan))</span>, add
   “Symbolic B qualifies separately with fixed T/features/state dimensions. Static all-valid/fixed
   behavior remains. Ordinary dynamic CPU enables no Scan path. Geometry admission never replaces
   all-B length VALUE checks after producers/before first writes, including T=0/B=0.” Keep length semantics.<span data-glossary-navigation="true"> ([dimensions glossary](../../../glossary.md#dimension))</span>
6. [Owned Models](../../../architecture/contracts/extensions-training.md#owned-tensor-models) and
   modules/training, add “Dynamic external signatures qualify separately over public Engine with one
   parameter/optimizer state and existing exclusion/transactions; accumulation is SUM. Symbolic graph/
   defineTensor acquisition needs Compiler cold reconstruction/Checkpoint copy, save/load additionally
   wire qualification. Builder acquisition qualifies independently. None blocks ordinary Engine+CPU;
   static saved Models/readers stay supported.” Generic graph source/values retains static admission.<span data-glossary-navigation="true"> ([signatures glossary](../../../glossary.md#term-signature); [optimizer state glossary](../../../glossary.md#term-optimizer-state); [cold reconstruction glossary](../../../glossary.md#term-cold-reconstruction); [Checkpoint glossary](../../../glossary.md#term-checkpoint); [qualification glossary](../../../glossary.md#term-qualification))</span>

Embedded exact ADR draft, future `docs/design/decisions/0050-dynamic-batch-geometry.md`
(reserve number at accepted cutover):

> # [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0050 Dynamic [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) geometry through numeric recipes
>
> Status: Proposed — 2026-10-24.
>
> Context and drivers: symbolic Model dimensions meet fixed-only capability/sizing/publication.
> Require one immutable compile/prepare, actual allocation, semantic ownership, hot paths and static lifetime compatibility.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [publication glossary](../../../glossary.md#term-publication); [immutable glossary](../../../glossary.md#term-immutable); [compile glossary](../../../glossary.md#compile); [prepare glossary](../../../glossary.md#prepare); [lifetime glossary](../../../glossary.md#term-lifetime))</span>
>
> Options: fixed-only cannot meet reuse; maximum banks/per-B fullprepare caches impose different
> lifecycle/allocation. Select prepared numeric recipes with per-run binding.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle))</span>
>
> Decision: recommend Prepare0012's signatures/nine-type map. Keep eight CompileArtifacts components.
> Compiler binds metadata, Prepare fixes identities/mode without numeric evaluation, CPU owns recipes,
> Runtime creates one run and Engine publishes actual descriptors. Permit B=0 under operation predicates.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [metadata glossary](../../../glossary.md#term-metadata); [evaluation glossary](../../../glossary.md#term-evaluation); [CPU glossary](../../../glossary.md#term-cpu); [Runtime glossary](../../../glossary.md#term-runtime); [Engine glossary](../../../glossary.md#term-engine); [descriptors glossary](../../../glossary.md#term-descriptor); [operation glossary](../../../glossary.md#operation))</span>
>
> Rationale: conditional scratch is not universally affine; execution mode is not byte independence.
> No extra graph/owner is needed. Keep proved five-limb F32 code; no speculative dynamic-limb generality.<span data-glossary-navigation="true"> ([graph glossary](../../../glossary.md#graph); [owner glossary](../../../glossary.md#term-owner); [limb glossary](../../../glossary.md#exact-superaccumulator-and-limbs); [F32 glossary](../../../glossary.md#floating-point-precision-and-rounding))</span>
>
> Consequences: migrate inward record/creator consumers explicitly; no binary compatibility claim.
> Keep static/native paths. Qualify actual AD closure, sizes/publication and failure/concurrency.
> Ordinary CPU precedes Scan and separate NN/cold/wire/native/tuning gates. Matched-work uses invocation
> B on both sides, not an identical-JIT/no-penalty promise.<span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [AD glossary](../../../glossary.md#autograd); [concurrency glossary](../../../glossary.md#term-concurrency); [Scan glossary](../../../glossary.md#term-cumulative-scan); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [JIT glossary](../../../glossary.md#term-just-in-time-compilation-jit))</span>
>
> Related: Prepare0012 brief/proposal and root/scoped headings linked above.

At cutover synchronize existing [Runtime Prepare backend boundary](../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend))</span>
and [module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module))</span>. Finalize affected [Javadoc](../../../glossary.md#term-javadoc) for
every [signature](../../../glossary.md#term-signature)/record/hook: all inputs/results, units, null/[overflow](../../../glossary.md#term-overflow)/precedence, [ownership](../../../glossary.md#term-ownership-meanings)/threading,
static migration. [Target](../../../glossary.md#term-target) glossary [dimension](../../../glossary.md#dimension)-[binding](../../../glossary.md#term-binding)/prepared-memory/[run](../../../glossary.md#term-run)-state terms; no edits now.

## Acceptance and next gate

Independent final design review and MAIN delivery acceptance are complete. User architecture
approval is explicit; MAIN admitted [Prepare0012A](tasks/0012a-dynamic-batch-authority-cutover.md)
for authority only after its 2026-10-24 17:41:11 UTC [dependency](../../../glossary.md#term-dependency)/base re-verification. Authority,
[ADR0050](../../../design/decisions/0050-dynamic-batch-geometry.md), explanations and glossary are
enacted and MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open P1/P2, docs/glossary
review complete. User implementation approval exists, but no [Ready](../../../glossary.md#term-ready) Java brief until MAIN separately
admits planner output; no automatic successor or current dynamic execution claim. Design body preserved.

First vertical must [compile](../../../glossary.md#compile)/query genuinely symbolic B and execute actual [AD](../../../glossary.md#autograd) closure: [MATMUL](../../../glossary.md#matrix-multiplication--matmul),
SUM/[MEAN](../../../glossary.md#term-mean)/count, arithmetic/splats, expand/divide, [rank](../../../glossary.md#term-rank) edits, [transpose](../../../glossary.md#term-transpose)/[contiguous](../../../glossary.md#term-contiguous)/sumToShape and K=B
adjoints, not forward-only provider support. Prove 64→40→1→64 before generic [Scan](../../../glossary.md#term-cumulative-scan); [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training
one-state reuse follows separately. NN cold acquisition, wire and independent builder gates are not
[Engine](../../../glossary.md#term-engine) prerequisites; static length skipping is independent/not cancelled or held on distant NN work.

Later focused/[module](../../../glossary.md#term-module) tests/[Javadoc](../../../glossary.md#term-javadoc) extend PrepareContextTest, CompileDiagnosticsTest,
RunStateCreationTest, EngineTypedLifecycleTest, TrainingSessionValuesTest and [CPU](../../../glossary.md#term-cpu) access/MATMUL/
aggregate tests. Include B=0/1, `X[B,0]` all-[constant](../../../glossary.md#term-constant)-byte mode rejection before creators, nonempty
empty contractions, conditional 0/48 scratch/five-[limb](../../../glossary.md#exact-superaccumulator-and-limbs) proof, [shape](../../../glossary.md#shape)/type/flags/[layout](../../../glossary.md#layout) mismatch,
absent/dead/short/inaccessible storage, [overflow](../../../glossary.md#term-overflow) before borrow, produced invalid-last length/T=0,
dead histories, allocation/bind/[publication](../../../glossary.md#term-publication) failures, suppression/retry/close races/different-B
[concurrency](../../../glossary.md#term-concurrency). Retain static/native/[view](../../../glossary.md#term-view-meanings)/empty/length/[persistence](../../../glossary.md#term-persistence) guards. MAIN owns architecture/
[conformance](../../../glossary.md#term-conformance)/integration join; enforce no [Runtime](../../../glossary.md#term-runtime) [Model](../../../glossary.md#term-model)/[Compiler](../../../glossary.md#term-compiler) edge or [Prepare](../../../glossary.md#prepare) [backend](../../../glossary.md#backend) interpretation,
unchanged extension edges and nine-type inventory.

Generated/direct evidence gives both sides fixed F/H/T and **invocation B**. Compare algorithm,
hot-loop/[dataflow](../../../glossary.md#term-dataflow) and constant-five-limb state; inspect Class-Files for hidden per-[element](../../../glossary.md#term-element)/step
[dispatch](../../../glossary.md#term-dispatch)/allocation/boxing/reflection/recipe work. Short warmed matched-work timing separates cold
generation/classload/[JIT](../../../glossary.md#term-just-in-time-compilation-jit), [admission](../../../glossary.md#term-admission) and [kernel](../../../glossary.md#kernel) costs. Diagnose/justify overhead, promise no identical
JIT or zero penalty. This design runs source/contract, links, size, whitespace and protection checks only.
