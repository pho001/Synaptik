# [Runtime](../glossary.md#term-runtime), [Prepare](../glossary.md#prepare), and [Backend](../glossary.md#backend) Boundary

This document explains the boundary defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

Start with [the execution atlas's distinctions](../explanation/packages/execution.md#the-distinctions-to-keep-while-reading)
if slots, representations or [leases](../glossary.md#term-lease) are unfamiliar. A **slot** is a planned position, a
**representation** is the physical storage bound there for a [run](../glossary.md#term-run), and a **workspace** is temporary
backend scratch rather than a transferable [graph value](../glossary.md#graph-value). A **projection** passes only the facts
needed by the receiving boundary. **[Opaque](../glossary.md#term-opaque)** means shared code can retain and transport a
backend's plan without interpreting its private fields.

## Start with one value crossing the boundary

Suppose Compiler assigns a producer to CPU and a consumer to Metal. Engine passes the immutable
recipe to shared Prepare, which calls each backend analyzer with only its partition's
descriptions. A single logical value can receive one slot with **two representation positions**:
one CPU buffer and one Metal buffer. Backend finalizers receive those exact assigned positions;
they do not allocate one invocation's writable output.

At run setup Runtime calls the prepared creation recipes, then binds a transfer to the actual
CPU and Metal representations. The transfer copies the bytes before the Metal consumer uses
them; merely reserving both positions did not make both copies valid. The logical value remains
the same value throughout. See [explicit composition and cold routing](#current-explicit-engine-composition-and-cold-routing)
for the qualified transfer domain, and [Metal ingress versus borrowing](../backend-guide/metal-backend.md#follow-the-storage-boundary-not-just-the-method-name)
for why an internal Runtime borrow and a host-to-device copy are different boundaries.

Two lifetimes protect different resources. Runtime's prepared-execution lease permits a
synchronous invocation to use reusable code/resources and ends before that call returns.
The returned result's lease keeps run resources and input borrows alive until result close.
Neither lease is a copy of the output; [explicit materialization](../explanation/ownership.md#explicit-engine-use)
creates the independent value. The detailed inventory below describes these mechanisms rather
than adding another runner or ownership contract.


Runtime currently implements the [immutable](../glossary.md#term-immutable) [`BufferSlot`](../glossary.md#buffer-slot--bufferslot) and [`WorkspaceSlot`](../glossary.md#workspace-slot--workspaceslot) [identities](../glossary.md#term-identity-meanings),
[`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan) final slot geometry, [nominal](../glossary.md#term-nominal) [buffer](../glossary.md#term-buffer)/workspace representation roles,
[borrowed](../glossary.md#ownership-borrowing-and-lease)/run-owned buffer [bindings](../glossary.md#term-binding), immutable prepared representation origins and backend-owned
creator callbacks, package-private cold creation with [rollback](../glossary.md#term-rollback), structural per-run [residency](../glossary.md#residency) and
explicit buffer-copy validity, the array-backed one-run [`RunState`](../glossary.md#run-state--runstate) [lifecycle](../glossary.md#lifecycle), immutable
[`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) and [`PreparedBufferTransfer`](../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) recipes, per-run [`BoundInvocation`](../glossary.md#bound-invocation--boundinvocation) and
[`BoundBufferTransfer`](../glossary.md#bound-buffer-transfer--boundbuffertransfer) objects, and the immutable creation-plus-execution-plus-transfer
[`PreparedSchedule`](../glossary.md#prepared-schedule--preparedschedule) contract described below, the dense final [publication](../glossary.md#term-publication) suffix, direct per-run
publication binding, the whole-`RunState` [`RunResult`](../glossary.md#run-result--runresult) lease, the immutable-recipe lifecycle [owner](../glossary.md#term-owner)
[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), explicit executable buffer-access declarations, and the stateless
prepared-execution runner. Prepare
currently implements the analysis-side projection, opaque analysis and tuning marker roles, the
exact-[partition](../glossary.md#partition) opaque tuning [handoff](../glossary.md#term-handoff), exact resource
declarations, analysis result, preparer collaboration, deterministic complete-set slot assignment,
typed backend finalization input/collaboration, the minimal prepared-partition association, exact
[compile](../glossary.md#compile) projection, one immutable partition-local [directed acyclic graph](../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../glossary.md#intermediate-representation-dag-and-ssa)) per backend
analysis context, explicit schedule assembly, complete schedule [validation](../glossary.md#term-validation), and construction of
the reusable prepared-execution root.
The [CPU](../glossary.md#term-cpu) backend supplies broad physical allocation, preparation, and access for
`Engine.standard()` and explicit composition. [Metal](../glossary.md#term-metal) supplies bounded [MPSGraph](../glossary.md#term-mpsgraph) and custom-[kernel](../glossary.md#kernel)
[occurrence](../glossary.md#term-occurrence)-qualified routes plus explicit native configuration, [host](../glossary.md#term-host) ingress, [materialization](../glossary.md#materialization), and
close. `Engine.builder()` currently registers opened CPU and/or Metal integrations and composes
their non-empty partition sets through one shared preparation. It preflights every owner and
required directed transfer before backend analysis, then [captures](../glossary.md#term-capture-meanings) direct per-input, executable,
transfer, and publication references. Current CPU/Metal transfer covers all seven [model](../glossary.md#term-model) data types
over exact [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping storage [layouts](../glossary.md#layout) with a
checked physical span; [CUDA](../glossary.md#term-cuda) and conversion-based transfer composition remain later work. The
lifecycle flow therefore combines current foundations with selected later stages, and each focused
section states its implementation status.
[ADR 0011](../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> defines the
resource-[ownership](../glossary.md#term-ownership) and cold-binding architecture.
[ADR 0013](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md) defines
the persistent prepared-[resource ownership](../glossary.md#term-resource-ownership) and [close lifecycle](../glossary.md#ownership-borrowing-and-lease).
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) selects explicit
[Engine](../glossary.md#term-engine) registration and construction [ownership](../glossary.md#term-ownership-meanings).
[ADR 0016](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md) selects owner-indexed
representations, physical contribution composition, and explicit bidirectional transfer.

## Boundary in one flow

```text
CompileArtifacts
  -> Engine validates every partition owner and required directed transfer
  -> shared Prepare validation and per-partition analysis
  -> concrete backend lowering, route choice, and exact resource declaration
  -> deterministic logical buffer/workspace assignment
     + one buffer representation position per participating owner
  -> concrete backend finalization against exact slots and representation positions
  -> backend physical-creation contributions
  -> Engine composition of partition execution, explicit transfer, and publication occurrences
  -> validated PreparedSchedule + PreparedMemoryPlan
  -> PreparedExecution
  -> Runtime execution with one heterogeneous RunState
```

Read the arrows from [compile](../glossary.md#compile)-time facts toward reusable [runtime](../glossary.md#term-runtime) state. [Prepare](../glossary.md#prepare) turns an [immutable](../glossary.md#term-immutable)
compile recipe into runtime-[ready](../glossary.md#term-ready) state, but executable construction is deliberately split around
shared slot assignment. Runtime executes the result without rediscovering, [lowering](../glossary.md#lowering), or selecting
[backend](../glossary.md#backend) work.

## Accepted symbolic-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [handoff](../glossary.md#term-handoff)

The [resource contract](contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract),
[Compiler binding](contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([binding glossary](../glossary.md#term-binding))</span> and
[CPU recipes](contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract) now
are implemented for the bounded ordinary [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32), [rank](../glossary.md#term-rank)-at-most-two one-B [CPU](../glossary.md#term-cpu) closure.
Prepare0014 additionally implements the [bounded symbolic Scan execution](#symbolic-scan-execution-and-future-geometry-reuse)
[slice](../glossary.md#slice) through these same [owners](../glossary.md#term-owner); its task tracks final [qualification](../glossary.md#term-qualification).
Prepare0015 implements combined one-B/fixed-T/[features](../glossary.md#term-software-feature) [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) length-[Scan](../glossary.md#term-cumulative-scan) through this same
handoff. Owning [validation](../glossary.md#term-validation) and independent code/structural review passed; root validation and the sole bounded
probe completed (evidence and limits:
[Prepare0015 Result](../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. No new [API](../glossary.md#term-application-programming-interface-api)/type, [Runtime](../glossary.md#term-runtime) numeric protocol, storage [owner](../glossary.md#term-owner) or [lease](../glossary.md#term-lease)
follows.
The current-foundation sections below also describe the preserved fixed paths.
[ADR 0050](../design/decisions/0050-dynamic-batch-geometry.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> records why exact per-[run](../glossary.md#term-run) sizing
replaces mandatory maximum banks or per-batch full preparation without introducing another owner.

```text
compile/prepare once: Compiler metadata plan -> Prepare fixed identities -> CPU numeric recipes
each invocation:     Engine binds static descriptors -> checked numeric sizes -> borrow
                       -> Runtime lease/create/cold bind -> execute -> actual publications
```

Read the upper line as [immutable](../glossary.md#term-immutable) semantic/physical recipes, the lower as one invocation's state.
[Compiler](../glossary.md#term-compiler) alone interprets symbolic [dimensions](../glossary.md#dimension) and private typed constraints; [Prepare](../glossary.md#prepare) validates
complete declarations/coverage without executing CPU geometry. Runtime receives same-plan numeric
geometry, not [Model](../glossary.md#term-model)/compiler objects. CPU packs actual access/[contraction](../glossary.md#contraction-dimension) domains once and emits
no per-[element](../glossary.md#term-element) recipe [evaluation](../glossary.md#term-evaluation). [Engine](../glossary.md#term-engine) resolves only [publication](../glossary.md#term-publication) [descriptors](../glossary.md#term-descriptor), not intermediate
[Shapes](../glossary.md#shape) or a new [graph](../glossary.md#graph). The eight CompileArtifacts components and [dependency](../glossary.md#term-dependency) edges do not change.

Execution mode is not a size test: `X[B,0].sum()` still requires B even when its declarations all
have [constant](../glossary.md#term-constant) bytes. [PreparedMemoryPlan](../glossary.md#prepared-memory-plan--preparedmemoryplan)'s explicit mode rejects absent B before creators. Entries
[carry](../glossary.md#term-carry) exact fixed bytes or immutable run-size recipes; creators receive actual bytes/[alignment](../glossary.md#term-alignment),
not a zero-[argument](../glossary.md#term-argument) closure over a guessed B. Slots, representation positions, routes and generated
code remain fixed; all [mutable](../glossary.md#term-mutable) arrays and writable resources belong to the run. This preserves
different-B concurrent results without shared currentB or a new lease protocol.

Implemented low-level acceptance scenario: [compile](../glossary.md#compile) grad-eligible FLOAT32 X `[B,4]`,
W `[4,3]`, Y=X@W and L=[mean](../glossary.md#term-mean)(Y), requesting actual dX/dW with [scalar](../glossary.md#term-scalar)-one [seed](../glossary.md#term-seed). Prepare once;
all-one data and no updates give the following actual outputs under existing [rounding](../glossary.md#term-rounding):

| B | Y shape / bytes | dX shape / bytes | dW shape / bytes | Y / L / each dW |
|---:|---|---|---|---|
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 4 / 4 / 1/3 |
| 40 | `[40,3]` / 480 | `[40,4]` / 640 | `[4,3]` / 48 | 4 / 4 / 1/3 |
| 1 | `[1,3]` / 12 | `[1,4]` / 16 | `[4,3]` / 48 | 4 / 4 / 1/3 |
| 64 | `[64,3]` / 768 | `[64,4]` / 1024 | `[4,3]` / 48 | 4 / 4 / 1/3 |

The table distinguishes batch-sized outputs from fixed [parameter](../glossary.md#term-parameter) [gradients](../glossary.md#term-gradient): each dX entry is
1/B, and dW contracts K=B rather than allocating B×parameter storage. L is four bytes; each
materialized [AD](../glossary.md#autograd) helper has its own declared resource, so this is not a total-memory budget.
MEAN/count preserves existing algebra, while admitted FLOAT32 exact SUM retains five limbs and
conditional 0/48-byte scratch. B=0, if predicates allow, produces empty Y/dX, [NaN](../glossary.md#term-nan) empty MEAN and
nonempty typed-zero dW; it cannot skip the whole run. Changed feature width or inconsistent B
rejects before [borrowing](../glossary.md#term-borrowing), as do unsupported [layout](../glossary.md#layout), [overflow](../glossary.md#term-overflow) and invalid storage.

A counter-scenario is resolving Model Shape expressions inside Runtime or resizing [buffers](../glossary.md#term-buffer) while
running a [kernel](../glossary.md#kernel): that crosses semantic [ownership](../glossary.md#term-ownership-meanings) and invalidates fixed requirement coverage.
Instead, validate numeric sizes and referenced prefixes before borrow/allocation; preserve lease
precedence, caller ownership, actual publication descriptors and reverse failure cleanup.

Ordinary forward plus actual first-order backward qualifies independently of Scan. [Symbolic Scan](../glossary.md#symbolic-scan)
keeps T/[features](../glossary.md#term-feature) fixed; future length-value checks remain after [producers](../glossary.md#term-producer)/before first writes, not
in shape [admission](../glossary.md#term-admission). [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training one-state reuse, cold graph acquisition, symbolic wire and builder
qualification are separate; static saved Models/readers remain. Focused tests cover sizes/empty/
failure/[concurrency](../glossary.md#term-concurrency)/static retention and generated-versus-optimal-Java structure plus bounded
matched-work A/B. The public three-list Engine compile workflow separately executes scalar L plus
actual dX/dW; publishing Y and L while selecting L alone uses the existing low-level [functional
gradient request](../glossary.md#functional-gradient-request), not a new public overload.

<a id="future-symbolic-scan-and-geometry-reuse"></a>

## [Symbolic Scan](../glossary.md#symbolic-scan) execution and future geometry reuse

Prepare0014 implements bounded [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32) [scalar](../glossary.md#term-scalar), single-range [CPU](../glossary.md#term-cpu) forward and actual
[Compiler](../glossary.md#term-compiler)-generated first-order reverse; its [task](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md)
tracks final [qualification](../glossary.md#term-qualification). This execution path is separate from the still-unimplemented geometry
[memo](../glossary.md#cache-and-memoization). The following recurrence illustrates current geometry; the miss/hit scenario below is
explicitly future [metadata](../glossary.md#term-metadata) reuse, not a current [API](../glossary.md#term-application-programming-interface-api) or execution claim. The
[Scan contract](contracts/recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([Scan glossary](../glossary.md#term-cumulative-scan))</span> alone defines semantics;
the [Engine memo contract](contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine))</span>
owns the narrow metadata exception. [ADR0051](../design/decisions/0051-symbolic-scan-and-geometry-reuse.md)
compares the selected direction with static execution and per-B respecialization.

Goal: reuse one fixed recurrence and preparation across fresh same-B and changed-B inputs.
Use fixed T=3, F=4, H=3 and one named B. Inputs are
X `[3,B,4]`, h0 `[B,3]`, Wx `[4,3]`, Wh `[3,3]`, [bias](../glossary.md#term-bias) `[3]`. Conceptual body:

```text
hNext = tanh(xStep @ Wx + h @ Wh + bias)
nextCarry = hNext; emission = hNext
finalCarry[B,3]; Y[3,B,3]
```

At each original time index, [MATMUL](../glossary.md#matrix-multiplication--matmul) and bias produce the next state, whose exact [activation](../glossary.md#term-activation)
output is saved when the actual reverse needs it. With zero X/h0/Wx/Wh and bias=atanh(0.25),
each step is approximately 0.25 after FLOAT32 [rounding](../glossary.md#term-rounding). This illustrative value is not a
qualification [oracle](../glossary.md#term-test-oracle): numerical tests must use signed nonuniform data and [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) versus ordinary
[Tensor](../glossary.md#tensor) [unrolling](../glossary.md#term-unrolling) and specialized clean Java, including both traversals and joint output seeds.
dX/dh0 follow actual row [shapes](../glossary.md#shape), but dWx `[4,3]`, dWh `[3,3]` and dbias `[3]` stay fixed;
the [weight](../glossary.md#term-weight) adjoints contract K=B, while [mean](../glossary.md#term-mean)(finalCarry) counts actual B*3.

Current calls resolve fresh numeric geometry, with no memo [lookup](../glossary.md#term-lookup)/[publication](../glossary.md#term-publication):

```text
prepare once -> fixed scoped checks + CPU numeric recipes + slots/code
each call    -> Engine/owner/inward admission -> full fresh Compiler bind
             -> checked scoped/numeric/aggregate geometry -> ALL current storage admission
             -> borrow exact prefixes -> Runtime's sole lease -> fresh resources/cold binding
             -> execute -> actual publications and fresh result
```

Read downward: prepared [identities](../glossary.md#term-identity-meanings) are reused, not [mutable](../glossary.md#term-mutable) arrays, caller [descriptors](../glossary.md#term-descriptor) or resources.
For B64→40→1→64, X bytes are 3072→1920→48→3072 and final-[carry](../glossary.md#term-carry) bytes 768→480→12→768;
dWx remains `[4,3]` with 48 bytes. These are data sizes, not the complete [run](../glossary.md#term-run) budget. Scoped
unused declarations still participate in pre-borrow [admission](../glossary.md#term-admission), including overflowing geometry
behind zero extents. [The public example](../user-guide/running-models.md#symbolic-scan-with-changing-batch-size)
uses the implemented [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/runValues [lifecycle](../glossary.md#lifecycle).

### Future [memo](../glossary.md#cache-and-memoization) only

For a conceptual serial memo witness, start with one preparation and an empty [metadata](../glossary.md#term-metadata) entry, then supply
fresh values in compiled input order. Each row below still performs complete fresh [Compiler](../glossary.md#term-compiler)
bind/private obligations, input/storage/physical checks and new resource creation. [Payload](../glossary.md#artifact-envelope-and-atomic-publication) bytes
are exact [FLOAT32](../glossary.md#term-float32) data sizes, not the total declared memory budget.

| Call | B | Memo result | X bytes | finalCarry bytes | Y bytes |
|---:|---:|---|---:|---:|---:|
| 1 | 64 | miss; retain 64 facts | 3072 | 768 | 2304 |
| 2 | 64 | hit on pure [Engine](../glossary.md#term-engine) facts only | 3072 | 768 | 2304 |
| 3 | 40 | miss; replace with 40 facts | 1920 | 480 | 1440 |
| 4 | 40 | hit on pure Engine facts only | 1920 | 480 | 1440 |
| 5 | 64 | miss; 64 was replaced | 3072 | 768 | 2304 |

Read the memo column as metadata reuse, not reuse of a [Binding](../glossary.md#term-binding), supplied [descriptor](../glossary.md#term-descriptor) or invocation.
Input-backed publications always retain the current admitted descriptor, including `isView`;
only template-derived non-input metadata can hit. Code/routes/slots remain identical across all
five calls. Each gets fresh [RunState](../glossary.md#run-state--runstate), writable storage/workspaces (even at zero bytes), [mutable](../glossary.md#term-mutable)
working arrays and result. [Immutable](../glossary.md#term-immutable) exact-plan numeric facts/private backing may be shared;
the existing geometry test's sharing across states proves no Engine memo implementation.

The flow distinguishes semantic [admission](../glossary.md#term-admission) from numeric facts and physical resources:

```text
prepare once -> fixed scoped checks/layout recipes + CPU size/access recipes + slots/code
each call    -> Engine/owner/inward admission -> full fresh Compiler bind
             -> local memo snapshot or checked numeric candidate -> current storage admission
             -> atomic metadata publication if still open -> borrow exact prefixes
             -> Runtime's sole lease -> fresh resources/cold binding -> execute -> fresh result
```

An admission failure publishes nothing and leaves a good entry intact. Allocation or execution
failure may leave valid metadata, never resources/failures; retry revalidates and allocates fresh.
Concurrent calls keep local immutable snapshots, with candidate retention bounded by active
admissions. Close clears/prevents late [publication](../glossary.md#term-publication). After inward admission it suppresses memo
publication rather than imposing a new outward-close error; [Runtime](../glossary.md#term-runtime)'s existing [lease](../glossary.md#term-lease) still
arbitrates the later close/[run](../glossary.md#term-run) [race](../glossary.md#term-race). No execution lock or second lease is introduced, and the
exact coordination mechanism remains a later approved implementation choice.

Useful variations: B=0 empties row work but not shared outputs or [parameter](../glossary.md#term-parameter) [gradients](../glossary.md#term-gradient); empty [MEAN](../glossary.md#term-mean)
is [NaN](../glossary.md#term-nan), and empty contractions/SUM produce typed zeros. T=0 is a separate fixed-T preparation with
raw initial [carry](../glossary.md#term-carry) and no body/hoisted work. A changed F/H, inconsistent later input B, invalid
same-B [layout](../glossary.md#layout)/storage or overflowing latent local domain rejects before borrow. Neither emptiness
nor a hit can hide it. A false shortcut would retain a previous caller's dense-[view](../glossary.md#term-view-meanings) descriptor or
shared mutable packed array because B matched; it would break publication truth or run isolation.

No new type/public [signature](../glossary.md#term-signature) or Runtime [handoff](../glossary.md#term-handoff) is selected. Existing static [F32](../glossary.md#floating-point-precision-and-rounding)/[F64](../glossary.md#floating-point-precision-and-rounding) [Scan](../glossary.md#term-cumulative-scan)/read
exceptions, constituent arithmetic and compatibility [identities](../glossary.md#term-identity-meanings) remain unchanged. Length-value
[preflight](../glossary.md#term-preflight) stays after [producers](../glossary.md#term-producer)/before Scan writes. CPU0042 implements fixed-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) length
execution separately; symbolic-B plus lengths remains gated.
Prepare0014's execution tests cover both actual reverse [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings), exact demanded/dead/shared resources,
[aliases](../glossary.md#aliasing-and-arena) and failure/close/[concurrency](../glossary.md#term-concurrency). Future memo tests must additionally distinguish reused
Engine geometry work from still-executed binder work. Generated Class-File inspection and bounded
matched-work [CPU](../glossary.md#term-cpu) evidence compare specialized clean-Java algorithm/hot-loop/[dataflow](../glossary.md#term-dataflow); the initial
Engine timing setup failed and its approved environment-only follow-up is pending. Neither the
CPU [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters) nor this future memo example promises parity, speed or automatic
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/cold/wire/wider-[backend capability](../glossary.md#backend-capability).

## Current [Runtime](../glossary.md#term-runtime) memory geometry

The current `modules/runtime` production surface contains two nominally distinct non-negative
plan-local [identities](../glossary.md#term-identity-meanings):

- `io.github.pho001.synaptik.runtime.memory.BufferSlot(long value)` for a reusable buffer
  position; and
- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlot(long value)` for a reusable workspace
  position.

Neither record stores a plan reference. Equal numeric values are valid across the two domains,
and another plan may reuse either number without establishing cross-plan [identity](../glossary.md#term-identity-meanings).

The current [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan) carries final ordered geometry through [immutable](../glossary.md#term-immutable) [buffer](../glossary.md#term-buffer)-entry and
workspace-entry snapshots. Each entry retains one exact slot reference plus exact non-negative
byte-size declaration and positive power-of-two [byte alignment](../glossary.md#term-byte-alignment). `requiresBatchSize` is explicit;
[`RunMemoryGeometry`](../glossary.md#run-memory-geometry--runmemorygeometry) resolves same-plan per-[run](../glossary.md#term-run) numeric sizes without [Model](../glossary.md#term-model)/[Compiler](../glossary.md#term-compiler) dependencies.
Slots are unique within their respective
lists, the two domains remain separate, and empty plans are valid.

These Runtime contracts deliberately retain no [compile](../glossary.md#compile)-time [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot), analysis-local workspace
requirement ID, [Prepare](../glossary.md#prepare) requirement, or source-to-slot association. They do not derive assignment,
allocate or own a physical buffer or workspace, define [aliasing](../glossary.md#term-aliasing) or [lifetime](../glossary.md#term-lifetime), bind a run, provide
access, transfer data, or execute work. This distinction lets later prepared units refer to final
positions without moving [graph](../glossary.md#graph) objects or physical storage into the Runtime hot-path contract.

## Current analysis foundation

The current `modules/prepare` production surface is the
`io.github.pho001.synaptik.prepare.analysis` package. Its exact ten top-level declarations are:

- `BackendAnalysisInputs`, the marker role for one concrete backend's immutable target,
  capability, configuration, and compatible cached-decision inputs;<span data-glossary-navigation="true"> ([concrete backend glossary](../glossary.md#concrete-backend); [target glossary](../glossary.md#term-target))</span>
- `BackendPreparationPlan`, the marker role for that backend's immutable selected lowering,
  route, and private configuration;<span data-glossary-navigation="true"> ([lowering glossary](../glossary.md#lowering))</span>
- [`PartitionDag`](../glossary.md#partition-dag-partitiondag), the [immutable](../glossary.md#term-immutable) [topology](../glossary.md#term-topology)-only projection for exactly one planned [partition](../glossary.md#partition);
- [`PrepareContext`](../glossary.md#prepare-context-preparecontext), the validated partition projection;
- `PreparationResourceRequirement`, the [sealed](../glossary.md#term-sealed) [buffer](../glossary.md#term-buffer)/workspace declaration family;
- [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis), the immutable selected-plan and requirement result;
- [`BackendPartitionPreparer`](../glossary.md#backend-partition-preparer--backendpartitionpreparer), the typed [backend](../glossary.md#backend) analysis collaboration;
- `BackendTuningCandidateBatch`, the method-free role for one backend-owned immutable complete
  candidate batch;<span data-glossary-navigation="true"> ([candidate batch glossary](../glossary.md#candidate-batch))</span>
- `BackendTuningDecision`, the method-free role for one backend-owned immutable selected-decision
  reference; and
- `BackendPartitionTuningHandoff`, the three-component generic association between one exact
  partition, one typed candidate batch, and an optional typed decision.

`PartitionDag` retains the exact planned-partition reference and exact `CompiledNode` references
in validated stable [topological order](../glossary.md#term-topological-order). It derives [producer](../glossary.md#term-producer)/output-port, consumer/input-port, and
edge occurrences from ordered [node](../glossary.md#node) ports. Repeated inputs therefore remain distinct, each output
of a multi-output node has its own producer [occurrence](../glossary.md#term-occurrence), and query order follows partition node
order then port order. An input occurrence is external when its [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot) has no producer inside
this partition. A local sink is a node with no output consumed inside this partition; [publication](../glossary.md#term-publication)
and cross-partition consumers do not change that topology-only classification.

The [DAG](../glossary.md#term-directed-acyclic-graph-dag) contains neither the complete [`CompiledGraphModel`](../glossary.md#compiled-graph--compiledgraphmodel) nor nodes from another partition. It
does not infer regions, cross-partition [ownership](../glossary.md#term-ownership-meanings), transfer, publication, [materialization](../glossary.md#materialization), memory,
[fusion](../glossary.md#term-fusion), route, scheduling, or performance policy. Duplicate node IDs or produced values,
partition membership/order disagreement, self-dependencies, and later-producer dependencies fail
construction before backend analysis.

`PrepareContext` stores this exact DAG as its sole node/topology component. Its derived
`partition()` and `nodes()` delegate to the DAG. Its six components are the DAG, projected values,
logical-memory requirements, [constants](../glossary.md#term-constant), backend inputs and the [Scan](../glossary.md#term-cumulative-scan) activity map. Structural
ScanActivityPlan [metadata](../glossary.md#term-metadata) and full source [shapes](../glossary.md#shape) accompany the scoped body; these are not an
executable [Runtime](../glossary.md#term-runtime) [graph](../glossary.md#graph). Runtime receives prepared recipes, not another executor. The
[runtime contract](contracts/runtime-prepare-engine.md) owns the exact context/constructor surface.

Projected values are unique by `ValueId`, every node input and output resolves to one projected
value, and every projected value has one [descriptor](../glossary.md#term-descriptor)-matching logical-memory requirement. This is
intentionally asymmetric: the current contract does not separately require every projected value
to occur in a node input or output.

Projected Shapes follow qualified ordinary and Scan static/named-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) projection, not arbitrary
symbolic geometry. An immutable logical-splat constant may be projected
only for a projected graph input, and its exact [`ScalarValue`](../glossary.md#typed-scalar-value--scalarvalue) type must match that input's
descriptor. Neither the context nor any marker role exposes `CompileArtifacts` or another
[Compiler](../glossary.md#term-compiler)-owned aggregate.

The sealed resource family has two immutable records. `Buffer` associates one projected
`ValueId` with an exact non-negative byte size and positive power-of-two [byte alignment](../glossary.md#term-byte-alignment).
`Workspace` uses a non-negative analysis-local requirement ID with the same size and [alignment](../glossary.md#term-alignment)
rules. Buffer IDs and workspace IDs are unique within their respective domains in one
`BackendPartitionAnalysis`; neither [identity](../glossary.md#term-identity-meanings) is a Runtime slot, address, allocation, handle, or
per-[run](../glossary.md#term-run) [binding](../glossary.md#term-binding).

`BackendPartitionPreparer.analyze` must be deterministic from the complete context and return the
exact context partition reference. Analysis performs no measurement, tuning search, [cache](../glossary.md#term-cache)
mutation, allocation, executable construction, or slot assignment. The [CPU](../glossary.md#term-cpu) backend currently
implements this collaboration internally for its supported partitions; that does not make CPU
types part of the shared [Prepare](../glossary.md#prepare) surface.

### Current [opaque](../glossary.md#term-opaque) tuning transport

`BackendPartitionTuningHandoff<C, D>` is cold, [immutable](../glossary.md#term-immutable) transport state separate from
[`PrepareContext`](../glossary.md#prepare-context-preparecontext) and [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis). `C` must implement
`BackendTuningCandidateBatch`, and `D` must implement `BackendTuningDecision`. The record retains
the exact non-null `PlannedPartition` and candidate-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) references plus a non-null `Optional`
that is either empty or contains the exact decision reference. It copies no [backend](../glossary.md#backend) state and
acquires no [ownership](../glossary.md#term-ownership-meanings).

The two roles are deliberately method-free. Shared [Prepare](../glossary.md#prepare) can preserve the caller's concrete
generic types, associate the values with one [partition](../glossary.md#partition), and transport them, but it cannot
enumerate candidates, interpret compatibility, choose a winner, or apply a decision. [CPU](../glossary.md#term-cpu) [OpenBLAS](../glossary.md#term-openblas)
and [Metal](../glossary.md#term-metal) route batches and decisions adopt these [nominal](../glossary.md#term-nominal) roles without moving their components,
[schema](../glossary.md#term-schema), candidate generation, matching, [heuristic](../glossary.md#term-heuristic) fallback, or execution behavior out of the
backend. Each backend validates a supplied decision against a freshly generated local batch.

The record itself contains no measurement, [cache](../glossary.md#term-cache) schema, serialized form, executable, or [Runtime](../glossary.md#term-runtime)
state. Current public `Engine.prepareTuned(...)` first obtains the sole eligible single-[owner](../glossary.md#term-owner) local
[handoff](../glossary.md#term-handoff), maps it to [occurrence](../glossary.md#term-occurrence) 0 in partition 0 with [weight](../glossary.md#term-weight) 1, and passes it opaquely to Phase 1.
The exact authenticated local decision then goes unchanged to the same backend's separate
complete-plan [producer](../glossary.md#term-producer). [Engine](../glossary.md#term-engine) transports that second decision-empty handoff to Phase 2 and later
authenticates the decision-present result. Metal exposes this transport only for an exact
[singleton](../glossary.md#term-singleton) NEG two-route local batch and a one-candidate complete batch fixed to the local winner.
[Model](../glossary.md#term-model)-wide extraction, multiple occurrences, broader [graph](../glossary.md#graph)/partition search, multiple partitions,
mixed backends, and executable [persistence](../glossary.md#term-persistence) remain downstream work.

### Current representative-execution [lifecycle](../glossary.md#lifecycle)

[Engine](../glossary.md#term-engine) can now retain one lifecycle [admission](../glossary.md#term-admission) around a package-private [synchronous](../glossary.md#term-synchronous) representative
session. Admission precedes [owner](../glossary.md#term-owner) and input inspection, then exact compiled-input [binding](../glossary.md#term-binding)
snapshots every caller-owned [host storage](../glossary.md#host-storage) before [borrowing](../glossary.md#term-borrowing) non-owning wrappers. Each supplied
complete recipe runs through the stateless [Runtime](../glossary.md#term-runtime) runner in a fresh [`RunState`](../glossary.md#run-state--runstate) and completes all
publications. Existing Phase-1 trial actions remain completion-only: they validate the result
count and close the result without inspecting [publication](../glossary.md#term-publication) payloads.

The same session also owns the package-private correctness action used by current complete-plan
composition. Before the first correctness [run](../glossary.md#term-run), Engine validates every ordered compiled
publication [descriptor](../glossary.md#term-descriptor) and the aggregate [canonical](../glossary.md#term-canonical-form)-byte limit. The first fresh run copies every
[publication occurrence](../glossary.md#publication-occurrence) independently to [detached](../glossary.md#detached-model-results) canonical bytes while the result [lease](../glossary.md#term-lease) is open,
including repeated [aliases](../glossary.md#aliasing-and-arena) and empty values, then closes the result before returning an [opaque](../glossary.md#term-opaque)
same-session reference. Each later candidate uses another fresh `RunState`, completes all copies
and result cleanup, and only then compares exact represented bytes. Order, byte length, [signed
zero](../glossary.md#term-signed-zero), and [NaN payload](../glossary.md#nan-infinity-and-signed-zero) bits are significant. The comparison exposes only `MATCH` or `MISMATCH` and
no bytes, [Tensor](../glossary.md#tensor) [metadata](../glossary.md#term-metadata), Runtime representation, or [backend](../glossary.md#backend) value.

The session closes wrappers once in reverse order before fresh selected or ordinary fallback
preparation. Execution, correctness-copy, result-count, copied-length, or cleanup failure
invalidates the session and preserves the existing primary, suppression, poisoning, admission,
and once-only cleanup rules. A successfully completed correctness mismatch does not poison the
session, although a later Phase-2 owner must stop rather than time or select that candidate set.
Required tuning propagates its recoverable failure without ordinary preparation; allowed fallback
performs ordinary safe-[heuristic](../glossary.md#term-heuristic) preparation once only after successful cleanup. Engine closure
waits for the admission, and a close [race](../glossary.md#term-race) rejects the final selected or fallback recipe. The public
composition invokes tuning outside Runtime. [Cache](../glossary.md#term-cache) hits execute no trials; misses use configured
warmups and timed [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters), each through a freshly prepared trial and fresh `RunState`. Selected
and fallback production recipes are prepared afresh. No tuning state, cache access, [ranking](../glossary.md#term-ranking), or
route selection enters Runtime.

Current Engine adapts the correctness primitive and the selected backend's opaque complete-plan
[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) to the generic tools-only consumer. All candidate correctness actions and cleanup finish
before any [warmup](../glossary.md#term-warmup) or timed action. Every action freshly prepares the requested recipe, and one
additional fresh preparation creates the authenticated production winner after representative
cleanup. Current [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) complete-plan compatibility is `SESSION`, so the exact [model](../glossary.md#term-model)-plan
path is transported but receives no filesystem input/output. Metal local compatibility is also
`SESSION`; it cannot hit or publish a persistent workload entry. This composition changes no
candidate interpretation, backend contract, [Prepare](../glossary.md#prepare) [ownership](../glossary.md#term-ownership-meanings), or Runtime behavior; a future
`PERSISTENT` [producer](../glossary.md#term-producer) may use the same translation without making current decisions persistent.

## The staged [prepare](../glossary.md#prepare) [handoff](../glossary.md#term-handoff)

The first handoff is now represented by [`PrepareContext`](../glossary.md#prepare-context-preparecontext). While shared Prepare holds the complete
[compile](../glossary.md#compile) [graph](../glossary.md#graph), it constructs each [`PartitionDag`](../glossary.md#partition-dag-partitiondag) exactly once, constructs all contexts before the
first [backend](../glossary.md#backend) call, and passes only the [partition](../glossary.md#partition)-local projection plus stable semantic and
[Planning](../glossary.md#term-planning) facts and fully resolved prepare-time inputs. The context is not `CompileArtifacts` and
exposes no [Compiler](../glossary.md#term-compiler)-owned aggregate. Concrete backends may consume the projected [Model](../glossary.md#term-model) and
Planning contracts during analysis; [Runtime](../glossary.md#term-runtime) never receives those graph objects.

Through the current [`BackendPartitionPreparer`](../glossary.md#backend-partition-preparer--backendpartitionpreparer) collaboration, the owning backend deterministically
analyzes and lowers the partition from those explicit inputs. It selects one supported route and
configuration and returns a [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis). The result has two parts:

- an opaque backend plan retaining the selected lowering, fusion, route, and private
  configuration; and<span data-glossary-navigation="true"> ([opaque glossary](../glossary.md#term-opaque); [lowering glossary](../glossary.md#lowering); [fusion glossary](../glossary.md#term-fusion))</span>
- exact backend-neutral declarations for every buffer and workspace resource that shared
  preparation must assign.

The next [lifecycle](../glossary.md#lifecycle) step is current inside a package-private complete-set handoff. Shared Prepare
assigns stable Runtime-owned [`BufferSlot`](../glossary.md#buffer-slot--bufferslot) and [`WorkspaceSlot`](../glossary.md#workspace-slot--workspaceslot) [identities](../glossary.md#term-identity-meanings), retains exact
requirement-to-slot associations, and constructs one [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan). The current rule is
conservative: one [buffer slot](../glossary.md#buffer-slot--bufferslot) per distinct declared [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot), with maximum declared size and
[alignment](../glossary.md#term-alignment) only for repeated fixed-byte declarations. [Run](../glossary.md#term-run)-bound declarations require structural
recipe equality and identical alignment; Prepare never evaluates them to invent a numeric maximum.
There is one distinct [workspace slot](../glossary.md#workspace-slot--workspaceslot) per declaration, with exact recipe/alignment agreement.
No [aliasing](../glossary.md#term-aliasing), [lifetime](../glossary.md#term-lifetime), interference, or reuse model is inferred.

“Assignment” gives requirements stable names and positions; it does not allocate bytes.
For example, one logical value crossing [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) [ownership](../glossary.md#term-ownership-meanings) can have one buffer slot with
distinct [owner](../glossary.md#term-owner)-specific representations. A transfer copies between those representations; it
does not create another semantic [graph value](../glossary.md#graph-value) or silently make every copy current.

Backend finalization is also current as a shared contract. After assignment, the owning backend's
`BackendPartitionFinalizer` receives one typed `BackendPartitionFinalization` containing its exact
analysis, the exact shared plan, and assignments in declaration order. It returns one
`BackendPartitionFinalizationResult` containing a [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) plus an [immutable](../glossary.md#term-immutable) [snapshot](../glossary.md#term-snapshot)
of persistent resources in physical acquisition order; shared Prepare extracts only the
executable for the minimal current [`PreparedPartition`](../glossary.md#prepared-partition--preparedpartition) association. A finalizer owns and rolls
back everything it acquires until its complete result returns successfully. Finalization must not
change route choice, add an undeclared shared need, or allocate per-run physical representations.

The [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) handoff, entry, and result remain package-private implementation details. Public
`GraphPreparation.project(...)` and `GraphPreparation.prepare(...)` share one projection
implementation. The former exposes an exact partition-local `PrepareContext` so [Engine](../glossary.md#term-engine) can ask a
backend for tuning candidates; the latter [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) the complete positionally supplied backend
collaborations, producerless [constant](../glossary.md#term-constant) contributions, assignment, finalization, schedule assembly,
[validation](../glossary.md#term-validation), [rollback](../glossary.md#term-rollback), and final [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution). Concrete backends never receive the
`CompileArtifacts` aggregate. [Graph preparation](../glossary.md#graph-preparation--graphpreparation) does not select or discover a backend set;
current Engine composition supplies the collaborators explicitly.

After finalization, [`PreparedScheduleContext`](../glossary.md#prepared-schedule-context--preparedschedulecontext) carries only stable Model, Planning, Prepare, and
Runtime facts: planned partitions, graph values, Compiler-ordered bindable-source IDs, constants,
ordered [publication](../glossary.md#term-publication) IDs, the exact memory plan, prepared partitions, and [buffer](../glossary.md#term-buffer) assignments.
Prepare places bindable buffers first in that exact [occurrence](../glossary.md#term-occurrence) order, independently of partition
analysis/declaration order, and validates these associations before calling a concrete schedule
assembler. For a producerless
published constant, Prepare identifies and orders the exact logical role and asks the assembler
to contribute physical geometry from the stable graph value, logical-memory requirement, and
[scalar](../glossary.md#term-scalar). Prepare still validates complete coverage and owns assignment and transactionality.

Unknown/expression [Shapes](../glossary.md#shape) or symbolic geometry outside the bounded single named-B projection,
including its [rank](../glossary.md#term-rank)-three [Scan](../glossary.md#term-cumulative-scan) closure, fail `PrepareContext` construction before backend analysis. The implemented
[ordinary symbolic-batch handoff](#accepted-symbolic-batch-handoff) preserves selected routes,
declared identities and slots; Scan still requires its independent whole-body [qualification](../glossary.md#term-qualification),
not a per-run graph.

## Current explicit [Engine](../glossary.md#term-engine) composition and cold routing

The current public composition is an Engine-owned `AutoCloseable` builder. Concrete
`takeOwnership(...)` overloads accept already-opened [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) integrations, transfer
[ownership](../glossary.md#term-ownership-meanings) at method entry, and adapt them into a private registration-ordered [registry](../glossary.md#term-registry) keyed by
[`BackendId`](../glossary.md#backend-identity--backendid). Each registration [captures](../glossary.md#term-capture-meanings) one stable capability provider and one [immutable](../glossary.md#term-immutable)
point-in-time availability [snapshot](../glossary.md#term-snapshot). An equal duplicate ID is rejected and the newly transferred
integration is rolled back; no existing entry is replaced.

That map has only cold responsibilities. [Compile](../glossary.md#compile) receives the frozen provider/snapshot lists in
registration order so [Planning](../glossary.md#term-planning) selects [owner](../glossary.md#term-owner) [identities](../glossary.md#term-identity-meanings). [Prepare](../glossary.md#prepare) validates the complete non-empty
owner set and every required directed transfer before the first [backend](../glossary.md#backend) analysis call, resolves
each planned [partition](../glossary.md#partition) to its registered preparation, and performs one shared [`GraphPreparation`](../glossary.md#graph-preparation--graphpreparation)
[transaction](../glossary.md#term-transaction). Shared Prepare assigns one logical slot plus deterministic owner-indexed physical
representations, finalizes partitions, and collects backend [buffer](../glossary.md#term-buffer)/workspace contributions.
Engine then assembles partition execution in compile order, one transfer per logical value and
distinct destination owner immediately before that owner's first consumer, and the dense
[publication](../glossary.md#term-publication) suffix.

A missing owner, no planned partition, or an unsupported transfer fails before backend analysis.
The current CPU/Metal path supports exact native-[host](../glossary.md#term-host)-staged transfer in both directions for all
seven carriers over [rank](../glossary.md#term-rank)-0..16 fully static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [layouts](../glossary.md#layout) with
checked physical byte spans. It performs no conversion, retry, fallback, [heap](../glossary.md#term-heap) staging, or owner
substitution.

Once `GraphPreparation` returns, the registry map has no execution role. The outward prepared
handle captures direct non-owning adapter references in caller-input and [publication occurrence](../glossary.md#publication-occurrence)
order. Engine [run](../glossary.md#term-run) uses the input references for backend-owned caller-storage ingress, and each
outward publication uses its own captured adapter for authenticated host [materialization](../glossary.md#materialization); both
occur outside [Runtime](../glossary.md#term-runtime) and before integration closure.
The [prepared schedule](../glossary.md#prepared-schedule--preparedschedule) carries direct executable, representation-creation, transfer, and
publication recipes. Runtime cold [binding](../glossary.md#term-binding) turns them into typed direct-reference actions; neither
binding nor traversal consults Engine, the adapter, a backend ID, capability provider,
availability snapshot, integration, `ServiceLoader`, reflection, or a global registry.
`prepareTuned(...)` may use a selected CPU adapter or an exact eligible [singleton](../glossary.md#term-singleton)-NEG Metal
adapter. Metal [MPSGraph](../glossary.md#term-mpsgraph)-only, multiple-partition, and mixed-owner plans fail local eligibility
before trials. A safe-[heuristic](../glossary.md#term-heuristic) fallback never changes the selected adapter.

<a id="planned-fixed-recurrent-scan-handoff"></a>

## Fixed recurrent [scan](../glossary.md#term-cumulative-scan) [handoff](../glossary.md#term-handoff)

[Model](../glossary.md#term-model) now represents the fixed recurrent scan and its valid-length [Tensor](../glossary.md#tensor) as ordinary expression
[metadata](../glossary.md#term-metadata). The [CPU](../glossary.md#term-cpu) [scalar](../glossary.md#term-scalar) executable path uses an ordinary caller input whose
values alter bounded [backend](../glossary.md#backend) control flow while its [Shape](../glossary.md#shape) and the complete resource geometry stay
static. The valid-length Tensor is fully static [rank-one](../glossary.md#axis-rank-scalar-vector-and-matrix) `INT64[batch]`; only its per-[run](../glossary.md#term-run) values
vary. Model does not read those values. They do not make `time`, [`batch`](../glossary.md#term-batch-machine-learning-and-hyperparameters), [feature](../glossary.md#term-feature) [dimensions](../glossary.md#dimension),
[parameter](../glossary.md#term-parameter) Shapes, output Shapes, route selection, or slot requirements dynamic.

```text
Planning: one ordinary operation capability query and owner
  -> Prepare: one static partition projection
  -> backend analysis: one recurrent-loop plan plus exact resources
  -> shared assignment
  -> backend finalization: one reusable PreparedExecutable
  -> Engine: typed logical input/publication binding
  -> Runtime: invoke one cold-bound action
```

[Planning](../glossary.md#term-planning) does not interpret variant, direction, valid lengths, active rows, loop parameters, or a
[backend route](../glossary.md#backend-route). For this fixed family, shared [Prepare](../glossary.md#prepare) adds no body, region, loop, or [callback](../glossary.md#term-callback)
contract. Backend analysis may retain the fixed transition and loop geometry in its [opaque](../glossary.md#term-opaque) plan,
but it must declare every shared resource before assignment and must not create one executable
body per time step.
Finalization constructs one reusable executable for the compatible static [descriptors](../glossary.md#term-descriptor).

[Engine](../glossary.md#term-engine), not [Runtime](../glossary.md#term-runtime), owns typed mapping of logical caller inputs to the ordered Runtime
representations and typed output [publication](../glossary.md#term-publication). Runtime sees only its existing prepared recipes,
slots, representations, cold-bound invocations, schedules, and isolated [`RunState`](../glossary.md#run-state--runstate). Runtime does
not inspect the length Tensor, select loop bounds, compact active rows, [dispatch](../glossary.md#term-dispatch) an [operation](../glossary.md#operation), or
interpret recurrence.

The [concrete backend](../glossary.md#concrete-backend) validates the entire length [vector](../glossary.md#term-vector) and representation-specific access
preconditions before mutating any output representation. Each value must be in `[0, time]`.
Execution then writes original-time-aligned hidden outputs, exact [positive zero](../glossary.md#nan-infinity-and-signed-zero) for every padded
[coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix), and explicit final hidden and [LSTM](../glossary.md#term-long-short-term-memory-lstm) cell states. It traverses only valid prefixes in
`FORWARD` or `REVERSE` order and performs no recurrent arithmetic for invalid coordinates.
Branch-based skipping satisfies the first contract; physical active-row compaction, packed
batches, and stronger memory-work claims remain deferred.

CPU now admits [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32), [FLOAT64](../glossary.md#term-float64), [BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16) [RNN](../glossary.md#term-recurrent-neural-network-rnn), reset-after [GRU](../glossary.md#term-gated-recurrent-unit-gru) and LSTM
occurrences in either direction, with or without [bias](../glossary.md#term-bias). Resolved nonnegative-[stride](../glossary.md#term-stride)/offset inputs
include repeated and overlapping reads; outputs remain [canonical](../glossary.md#term-canonical-form) dense. Logical counts and
referenced spans are checked separately, and an empty [layout](../glossary.md#layout)'s unused origin is never accessed.
Analysis retains direct private facts rather than generated [IR](../glossary.md#term-intermediate-representation-ir). Finalization
creates one Java scan child per [occurrence](../glossary.md#term-occurrence) without consulting the generated-[artifact](../glossary.md#term-artifact) store.
Two packed projection rows require `2 * gates * hiddenSize * workingBytes` run-owned scratch bytes,
with width/[alignment](../glossary.md#term-alignment) 8 for FLOAT64 and 4 otherwise, independent
of time and lengths; empty time or batch omits scratch. Dense outputs still require `O(T*B*H)` storage.
Cold [binding](../glossary.md#term-binding) reads no lengths. Bound execution checks every length after any length [producer](../glossary.md#term-producer),
before [padding](../glossary.md#padding), initial-state copies or gate writes; an invalid last row leaves every scan output
unchanged. Strided lengths use their logical addresses, not poisoned gaps. FLOAT64 works in double;
low types use FLOAT32 work and exact low stored-state barriers, including LSTM cell-to-hidden
consumption. Read/read [aliases](../glossary.md#aliasing-and-arena) remain legal; writes/scratch reject overlap with whole selected
representations. Existing Engine cleanup propagates the unchecked failure and returns no result.

No shared executable code or contract changes were needed. If a later route proves
that a selected backend cannot preserve static route and resource decisions across compatible
valid-length values, it must expose the exact gap for a new architecture decision instead of
adding speculative control-flow state to Prepare or Runtime. See
[ADR 0012](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.

<a id="cpu0074-staged-synchronous-row-region-exception"></a>

## CPU0074 implemented [synchronous](../glossary.md#term-synchronous) row-region exception

Implemented and MAIN accepted: the [CPU mechanism](contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception)
and [run-owned reservation contract](contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)
permit only selected [fused](../glossary.md#specialization-and-fusion) regions to [dispatch](../glossary.md#term-dispatch) synchronous worker ranges from the emitted [Scan](../glossary.md#term-cumulative-scan)
loop. CPU0074 is Complete after final bounded [validation](../glossary.md#term-validation). Its
[Result](../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result) distinguishes
functional acceptance from unqualified steady-state profitability; defaults and rollout are unchanged.
Shared [Prepare](../glossary.md#prepare) and [Runtime](../glossary.md#term-runtime) APIs, one [RunState](../glossary.md#run-state--runstate) and synchronous [publication](../glossary.md#term-publication) do not change.

[Lifecycle](../glossary.md#lifecycle) example with two actual participants and a capacity-two budget: analysis declares
the bounded [feature](../glossary.md#term-software-feature)-[transpose](../glossary.md#term-transpose) scratch and two terminal partial/validity slots before assignment.
Cold [run](../glossary.md#term-run) [binding](../glossary.md#term-binding) creates the context/arrays/calls once. The caller acquires two permits, performs
after-[producer](../glossary.md#term-producer) [preflight](../glossary.md#term-preflight), and enters generated code. At the selected active-time region it lends
the same reservation to the two workers and waits without concurrent math. Workers acquire no
nested permits. Join completes before caller merge/[carry](../glossary.md#term-carry) or scratch reuse; the caller [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings)
only after all started work quiesces, also on failure/interruption/close. With one participant,
the inline path uses one permit and submits nothing. A wrong, released or insufficient reservation
rejects before enqueue. Holding one inline permit while separately acquiring worker permits is
not this protocol and can [deadlock](../glossary.md#term-deadlock).

Only existing scheduler Job/queue/failure/[latch](../glossary.md#control-flow-graph) [metadata](../glossary.md#term-metadata) may allocate per dispatch. Frames, typed
bindings and calls remain run-owned and prebound; no per-step [graph](../glossary.md#graph) [lookup](../glossary.md#term-lookup), geometry repacking,
[tensor](../glossary.md#tensor) allocation or second executor is introduced. True [invariant](../glossary.md#term-invariant) transposes stage once under
activity gating; SHARED history/changing cuts stage at the right active time or fall back.
The [Scan owner](contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception)<span data-glossary-navigation="true"> ([owner glossary](../glossary.md#term-owner))</span> retains
activity and separate per-time carry ADD semantics. Existing [scalar](../glossary.md#term-scalar) [handoff](../glossary.md#term-handoff) statements below
remain unchanged outside this explicit selected exec-v1 exception.

The implemented, independently reviewed
[CPU0075 scalar emission/cache cutover](contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse)
does not change this handoff. [CPU](../glossary.md#term-cpu) analysis must account for within-row cached values/addresses
and last-use slot reuse before resources/slots; finalization only emits the proved form. For
example, two reads of one saved cell may share a typed local within an active row, but never
across a possibly [aliasing](../glossary.md#term-aliasing) write or a time/bank transition. This avoids redundant generated
accesses without a runtime [cache](../glossary.md#term-cache), new run object, entry [parameter](../glossary.md#term-parameter) or changed worker reservation.
Only affected scalar compatibility [identities](../glossary.md#term-identity-meanings) advance; [SIMD](../glossary.md#simd)/nonregional controls stay literal.
The [decision record](../design/decisions/0054-row-fusion-scalar-load-address-reuse.md) explains why
the former literal scalar-retention promise is qualified instead of preserving two emitters.
MAIN's CPU/[Javadoc](../glossary.md#term-javadoc) gate passed; the [CPU0075 Result](../planning/backends/cpu/tasks/0075-row-fusion-scalar-load-address-reuse.md#result)
separates structural savings from [JIT](../glossary.md#term-just-in-time-compilation-jit)-qualified timing. Neither a [C1](../glossary.md#term-c1) repair nor a material Scan
speedup was established, and the remaining Java gap grants no wider staging or carry [lifetime](../glossary.md#term-lifetime).

<a id="cpu0076-prospective-scalar-invariant-staging"></a>

### CPU0076 implemented [scalar](../glossary.md#term-scalar) [invariant](../glossary.md#term-invariant) staging

The user-approved [CPU0076 contract](contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)
permits the implemented bounded load-placement step. [CPU](../glossary.md#term-cpu)
cold analysis proves a cell's storage/address/activity [lifetime](../glossary.md#term-lifetime) before cuts/resources/slots;
generation may then keep the represented value in method locals without adding a [Runtime](../glossary.md#term-runtime) [cache](../glossary.md#term-cache).
Shared [Prepare](../glossary.md#prepare) still assigns the same resources and Runtime executes the same prepared unit.

For example, an authenticated read-only [matrix](../glossary.md#term-matrix-meanings) cell with no row/time dependence may load once
under the invocation's nonempty/any-active [guard](../glossary.md#term-guard). A SHARED history of that matrix instead loads
its current original-time [slice](../glossary.md#slice) after its [producer](../glossary.md#term-producer) at active(t). Loading the first slice once
and reusing it through reverse traversal would read stale data. A worker range with no active
consumer cannot preload body cells just because another range is active. If the existing helper
[bindings](../glossary.md#term-binding) cannot express the required placement, its checked path remains supported. A [constant](../glossary.md#term-constant)-row
[view](../glossary.md#term-view-meanings) of a ROW history still needs that source row's activity; it is not a SHARED source. The
implemented scopes omit ROW/[carry](../glossary.md#term-carry)/bank/scratch origins and already-hoisted scalar actuals.

Only complete consumed fixed-[feature](../glossary.md#term-software-feature) views and complete regional bodies qualify; [rank](../glossary.md#term-rank) greater
than two or unrelated body producers keep checked reads. The shared cold query is reused with
identical authenticated [scope](../glossary.md#term-scope)/base facts before cuts, in preparation, both cache streams and
emission. Caller lifetime includes every demanded history base, even forwarded sources and
prepublication; both phases include duplicate wrapper [carrier](../glossary.md#term-storage-carrier)/base allowances. This prevents a
regional-only estimate from hiding locals that coexist with invocation staging.

Caller/helper live-slot proof remains separate and pre-slot; no new frame, bank, helper [ABI](../glossary.md#term-application-binary-interface-abi),
reservation or [resource owner](../glossary.md#ownership-borrowing-and-lease) follows. Carry accumulators remain memory-backed with separate
per-time ADDs. [ADR0055](../design/decisions/0055-row-invariant-load-staging.md) explains the
tradeoff. The [task result](../planning/backends/cpu/tasks/0076-row-invariant-load-staging.md#result)
records generic tests, actual ClassFile proof, literal retention and the sole comparison:
B256 observed 17.7547% lower time than accepted CPU0075, with logged primary/Entry/window [C2](../glossary.md#term-c2)
stable through its [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) envelope; B32 is outer-loop-[JIT](../glossary.md#term-just-in-time-compilation-jit)-confounded. This is not universal
stationarity, assembly, parity or [C1](../glossary.md#term-c1)-repair evidence. Final CPU/[Javadoc](../glossary.md#term-javadoc) and named [Engine](../glossary.md#term-engine) gates
passed; the independent documentation pass reuses that source-matched evidence without reruns.

## [General differentiable scan](../glossary.md#general-differentiable-scan) [handoff](../glossary.md#term-handoff)

This handoff is implemented for the bounded static [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) [CPU](../glossary.md#term-cpu) [scan](../glossary.md#term-cumulative-scan), including
[rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../glossary.md#matrix-multiplication--matmul) and independently qualified actual [Compiler](../glossary.md#term-compiler) first-order reverse bodies. CPU0033
derives [F64](../glossary.md#floating-point-precision-and-rounding) spans, [strides](../glossary.md#term-stride), histories, banks and cuts from eight-byte [elements](../glossary.md#term-element), with exact checked
resources before slots and typed capacity/[alignment](../glossary.md#term-alignment) checks at [binding](../glossary.md#term-binding). The historical Prepare0014
layer extended this handoff to bounded all-valid one-B FLOAT32 with unresolved template [layouts](../glossary.md#layout)
and exact [run](../glossary.md#term-run)-size declarations. Current ordinary CPU [general Scan](../glossary.md#general-differentiable-scan) qualifies named-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [F32](../glossary.md#floating-point-precision-and-rounding)/F64
forward and actual selected VJPs under single-B, fixed-T/[features](../glossary.md#term-feature) guards; this does not grant
ordinary symbolic F64 MATMUL, arbitrary symbolic geometry, symbolic [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training or [persistence](../glossary.md#term-persistence). The
[scan owner](contracts/recurrent-scan.md#general-differentiable-scan) specifies the complete semantic
protocol; the [Prepare](contracts/runtime-prepare-engine.md#modulesprepare) and
[CPU](contracts/backend-execution.md#cpu-backend-routes) contracts own only their boundaries.

```text
Compiler: validated scoped body + role/result bindings + ordinary residual edges/demand
  -> Planning: Tensor-free whole-body/binder query, one complete-body owner
  -> PrepareContext/PartitionDag: preserve attachment, separate local/outer descriptors
  -> CPU analysis: prepare once, exact checked resource declarations
  -> shared slot/representation assignment
  -> CPU finalization: emit outer time loop + admitted typed step emitters
  -> Runtime: cold-bind direct references, execute with one isolated RunState
```

Read downward: no generated executable is finalized before its resources have slots. Supported
children do not prove that a loop binder can execute their [topology](../glossary.md#term-topology). A body with a supported
[pointwise](../glossary.md#term-pointwise) child but an unsupported role/[carrier](../glossary.md#term-storage-carrier) binding must reject as a whole, not [dispatch](../glossary.md#term-dispatch) that
child at [runtime](../glossary.md#term-runtime) or split the body across [owners](../glossary.md#term-owner). Local numeric ValueIds can coincide with outer
IDs without identifying the same values; projection validates them in separate scopes.

Prepare0011 also projects length-aware forward and actual Compiler-generated reverse scans.
It separates the final [INT64](../glossary.md#data-type--datatype) lengths actual from floating body formals, preserves explicit
ROW(0)/ROW(1)/SHARED bindings and outer layouts, and validates static boundaries and complete-[graph](../glossary.md#graph)
demand before any analyzer. It neither repeats [Model](../glossary.md#term-model)/Compiler row-algebra proof nor reads length
values. The existing handoff transports [backend](../glossary.md#backend)-declared mixed-boundary resources before slots;
test-only analyzers verify those associations and [rollback](../glossary.md#term-rollback) without invoking physical creators
or executables. CPU0042 separately implements real fixed-geometry FLOAT32/FLOAT64 length [lowering](../glossary.md#lowering),
mixed-width declarations and generated forward/actual-reverse execution through this same handoff.

Prepare0015 extends length projection to static or one named B, without changing that sequence
or re-proving Model/Compiler algebra in [Prepare](../glossary.md#prepare). Before analyzers, validate
all scoped declarations/bindings/latent ports and complete demand. Every run, including repeated B,
retains fresh Compiler bind/private obligations, current input-backed [metadata](../glossary.md#term-metadata)/storage and checked
numeric sizing before borrow; after [producers](../glossary.md#term-producer), run-aware typed checks and all B lengths precede
the first Scan-local write. `[B,0]` still carries explicit B and batch-required mode. Actual
F32/F64 widths and separate eight-byte INT64 control feed existing declarations/creators and one
[RunState](../glossary.md#run-state--runstate); the [Engine](../glossary.md#term-engine) [memo](../glossary.md#cache-and-memoization) remains excluded.

[Canonical](../glossary.md#term-canonical-form) [residual](../glossary.md#scan-residual-port-and-history-demand) wrappers are not memory reservations. A reverse consumer makes selected
histories live through ordinary edges; [inference](../glossary.md#term-inference) with no residual [publication](../glossary.md#term-publication)/consumer has zero
history bytes/stores. Internal body uses require only step temporaries. Analysis declares checked
output, staging, [carry](../glossary.md#term-carry)-bank, temporary and live-history bytes before slots, including demand for
reverse residuals. No [opaque](../glossary.md#term-opaque) saved-workspace exchange or nested execution [owner](../glossary.md#term-owner) replaces this work.

CPU generation composes typed body emitters inside an emitted time loop. Existing carrier/MATMUL
entry assumptions need explicit step-local carrier/local/base/range bindings; CPU0032's generation-only
[matrix](../glossary.md#term-matrix-meanings) composer supplies them without an ordinary MATMUL entry call. A generation-time Consumer
is not a runtime [callback](../glossary.md#term-callback). Hoist [invariant](../glossary.md#term-invariant) setup, prefer direct time offsets, and justify
every [materialization](../glossary.md#materialization)/carry bank/temporary. Live saved histories and needed MATMUL storage remain;
the decision does not promise universally copy-free or faster execution. [Mutable](../glossary.md#term-mutable) carry, changing
addresses/geometry arrays and storage belong to the run, never shared prepared state. Body [shape](../glossary.md#shape)
is compiled once, not T times, with no runtime interpreter/per-op [lookup](../glossary.md#term-lookup)/per-step allocation.

Generated [identity](../glossary.md#term-identity-meanings) must key every baked body/role/demand fact while CpuKernelIr excludes extents,
slots and run bindings; stale meaning/format [artifacts](../glossary.md#term-artifact) reject. The current CPU route retains
[schema](../glossary.md#term-schema)/family 71 and old pointwise/shaped keys and bytes; a separate MATMUL-only versioned
[specialization](../glossary.md#term-specialization) [payload](../glossary.md#artifact-envelope-and-atomic-publication) includes every baked affine/resource fact in both identity streams.
The additive F64 variants use existing [dtype](../glossary.md#data-type--datatype) and baked eight-byte resource facts in those same
streams, preserving accepted F32 keys and generated bytes.
The fixed Java route above remains intact. CPU0042 adds the separately qualified fixed-geometry
length branch: INT64 control outside floating formals, after-[producer](../glossary.md#term-producer) [preflight](../glossary.md#term-preflight), and activity-dominated
generated work. All-valid static/symbolic keys and generated bytes remain unchanged. Cold/v5,
owned NN workflows remain separate gates. Prepare0015 adds a separate combined symbolic-length
branch, including typed outer/control closure and explicit B even behind zero features. Owning tests
and independent source/Class-File review passed; root [validation](../glossary.md#term-validation) and the sole bounded probe completed;
[Prepare0015 Result](../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) records
evidence and limits; final MAIN acceptance remains pending. See
[the focused scan explanation](general-differentiable-scan.md) and
[ADR 0045](../design/decisions/0045-general-differentiable-scan.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.

## What [prepare](../glossary.md#prepare) creates

The complete architecture prepare [lifecycle](../glossary.md#lifecycle) creates:

- [`PreparedPartition`](../glossary.md#prepared-partition--preparedpartition), which associates a planned partition and backend identity with its
  prepared work;<span data-glossary-navigation="true"> ([partition glossary](../glossary.md#partition); [backend identity glossary](../glossary.md#backend-identity--backendid))</span>
- a possible `PreparedUnit`, if the actual finalization or schedule consumer later establishes a
  distinct invariant beyond the executable and its selections;<span data-glossary-navigation="true"> ([invariant glossary](../glossary.md#term-invariant))</span>
- [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable), the current reusable cold-[binding](../glossary.md#term-binding) recipe for one prepared region;
- [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan), whose current Runtime contract defines final buffer/workspace slot
  geometry without allocating physical storage;<span data-glossary-navigation="true"> ([workspace slot glossary](../glossary.md#workspace-slot--workspaceslot))</span>
- [`PreparedSchedule`](../glossary.md#prepared-schedule--preparedschedule), which orders executable, transfer/[materialization](../glossary.md#materialization), and [publication](../glossary.md#term-publication) work; and
- [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution), the immutable reusable runtime-ready result, including any immutable
  persistent prepared resources that are not ordinary per-run workspace.<span data-glossary-navigation="true"> ([ready glossary](../glossary.md#term-ready))</span>

Today, `PreparedMemoryPlan`, [`PreparedRepresentationPlan`](../glossary.md#prepared-representation-plan--preparedrepresentationplan), `PreparedExecutable`,
`PreparedSchedule`, `PreparedExecution`, and `PreparedPartition` exist among the prepared/[runtime](../glossary.md#term-runtime)
contracts in this list; [`BoundInvocation`](../glossary.md#bound-invocation--boundinvocation) is
the current per-[run result](../glossary.md#run-result--runresult) of binding an executable. The current `PreparedExecution` retains one
exact plan, one schedule that reports that same plan reference, and a private [identity](../glossary.md#term-identity-meanings)-unique
[snapshot](../glossary.md#term-snapshot) of persistent prepared resources. It has explicit non-waiting close and [opaque](../glossary.md#term-opaque)
[synchronous](../glossary.md#term-synchronous) [run](../glossary.md#term-run) [leases](../glossary.md#term-lease), uses [owner](../glossary.md#term-owner) identity rather than [structural equality](../glossary.md#term-structural-equality), and exposes no
resource [lookup](../glossary.md#term-lookup). The current schedule retains one exact plan and an [immutable](../glossary.md#term-immutable) ordered
snapshot. It permits one optional first-only representation-creation prefix plus executable and
[buffer](../glossary.md#term-buffer)-transfer occurrences, followed by an optional dense publication-only suffix. Empty,
executable-only, transfer-only, and zero-publication schedules and repeated pre-publication
occurrences remain valid. Distinct publication positions may name the same exact representation.
It has no `PreparedUnit` and invokes no [callback](../glossary.md#term-callback), binding, publication, or execution. Current
public Prepare orchestration constructs and validates this recipe through one explicitly supplied
assembler. The analysis-side and finalization-side Prepare contracts described above are also
current. `modules/prepare` owns [`PrepareContext`](../glossary.md#prepare-context-preparecontext),
[`BackendPartitionPreparer`](../glossary.md#backend-partition-preparer--backendpartitionpreparer), [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis), shared resource declarations, current
assignment and source associations, `PreparedPartition`, [`GraphPreparation`](../glossary.md#graph-preparation--graphpreparation), and complete
schedule [validation](../glossary.md#term-validation). [Engine](../glossary.md#term-engine)-level composition supplies explicitly registered [backend](../glossary.md#backend) implementations and
their input facts. Prepare
does not interpret the backend's opaque route plan.

Shared prepare code does not implement concrete [CPU](../glossary.md#term-cpu), [Metal](../glossary.md#term-metal), or [CUDA](../glossary.md#term-cuda) [lowering](../glossary.md#lowering) and does not own backend-specific executable or storage implementations.

## What [runtime](../glossary.md#term-runtime) does

`modules/runtime` owns the contracts and state needed after preparation. It:

- executes [`PreparedSchedule`](../glossary.md#prepared-schedule--preparedschedule);
- creates exactly one [`RunState`](../glossary.md#run-state--runstate) for each active complete logical run and binds caller inputs as
  borrowed resources;<span data-glossary-navigation="true"> ([borrowed glossary](../glossary.md#ownership-borrowing-and-lease))</span>
- manages runtime slots, resources, and workspaces through prepared contracts;
- performs scheduled [residency](../glossary.md#residency), transfer, and [materialization](../glossary.md#materialization) work;
- invokes current cold-bound `BoundInvocation.execute()` objects from later scheduled work;
- updates residency after execution; and
- publishes outputs and [gradients](../glossary.md#term-gradient) according to the prepared plan and [run](../glossary.md#term-run) policy.

Runtime does not optimize the [graph](../glossary.md#graph), construct [autograd](../glossary.md#autograd), choose [backend ownership](../glossary.md#backend-ownership), discover [backends](../glossary.md#backend), lower partitions, or select [kernels](../glossary.md#kernel). Its [hot path](../glossary.md#term-hot-path) does not operate on [`Operation`](../glossary.md#operation) or `CompiledNode`.

[Runtime profiling](../glossary.md#runtime-profiling) is passive observation of this execution. Runtime may translate observed facts
into typed [trace](../glossary.md#trace) payloads, but neither profiling nor tracing selects or mutates execution
settings. Runtime performs no [model](../glossary.md#term-model)-[autotuning](../glossary.md#autotuning--model-autotuning) search, [cache](../glossary.md#term-cache) [lookup](../glossary.md#term-lookup) or mutation, or hot-path graph
inspection.

## Per-[run](../glossary.md#term-run) resources and cold [binding](../glossary.md#term-binding)

```text
immutable PreparedExecution
  -> one isolated RunState for the complete heterogeneous run
     -> BufferSlot -> one or more explicitly prepared representations
     -> WorkspaceSlot -> one backend-local scratch representation
  -> cold checked binding -> backend-owned typed direct-reference invocation
  -> hot-path execution -> publication -> cleanup of resources still owned by the run
```

Concurrent invocations reuse the prepared recipe but never a [`RunState`](../glossary.md#run-state--runstate) or run-owned [mutable](../glossary.md#term-mutable)
resource. [Runtime](../glossary.md#term-runtime) owns logical slot state, [ownership](../glossary.md#term-ownership-meanings) transitions, validity/[residency](../glossary.md#residency), cleanup
orchestration, and failure isolation. Concrete [backends](../glossary.md#backend) own physical representation classes and
perform allocation, [release](../glossary.md#term-release), transfer, and access.

The current Runtime foundation implements the prepared origin description, package-private cold
creation and [rollback](../glossary.md#term-rollback), structural residency, explicit per-copy [buffer validity](../glossary.md#buffer-validity), [carrier](../glossary.md#term-storage-carrier), cleanup,
and cold-binding portions of this flow.
`BufferRepresentation` and `WorkspaceRepresentation` are distinct [nominal](../glossary.md#term-nominal) closeable roles with
no physical access [API](../glossary.md#term-application-programming-interface-api). `BufferRepresentationBinding` marks one exact [buffer](../glossary.md#term-buffer) representation as
[borrowed](../glossary.md#ownership-borrowing-and-lease) or run-owned. `RunState` retains one exact [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan), copies supplied list
structure into private arrays in plan encounter order, rejects repeated representation [identity](../glossary.md#term-identity-meanings),
and exposes direct indexed access. Every supplied workspace is run-owned. Successful construction
transfers cleanup responsibility; failed construction transfers nothing and closes nothing.

Each bound representation is structurally resident until closure. One independent [boolean](../glossary.md#data-type--datatype) per
buffer copy records logical validity: borrowed caller inputs start valid, ordinary created
run-owned buffers start invalid, backend-created initialized buffers start valid, zero or multiple
copies may be valid, and workspaces remain scratch outside logical validity. The initialized
origin carries only a backend creator and no [graph](../glossary.md#graph) or [scalar](../glossary.md#term-scalar) fact. Query and mutation are
explicit [constant](../glossary.md#term-constant)-time array operations without
copying, backend work, ownership changes, or implicit coherence.

The package-private cold creation [operation](../glossary.md#operation) validates the complete dense caller-input list before
callbacks, creates buffers in buffer/representation order and then workspaces, and constructs one
state. A partial failure closes successfully created results once in reverse creation order,
preserves the original unchecked failure, suppresses cleanup failures, and never closes borrowed
inputs.

Current cleanup marks the state closed first, skips borrowed buffers, and attempts every owned
representation once in deterministic reverse order. It preserves the first unchecked exception
or error and suppresses later failures. The state is not [thread-safe](../glossary.md#term-thread-safe), but separate states may
share the [immutable](../glossary.md#term-immutable) plan while keeping run-owned representations and validity arrays isolated.
The current transfer foundation adds an immutable recipe for two distinct already-created
representation positions of one buffer. Cold binding validates exact plan/state association,
position bounds, and concrete source/destination compatibility once, then produces a backend-
owned bound action retaining direct concrete references. Its final action makes a valid
destination a no-op; otherwise it requires a valid source, invokes backend work once, and marks
only the destination valid after success. Backend failure leaves Runtime validity unchanged.
[Materialization](../glossary.md#materialization) is this same explicit transfer when it produces an equivalent already-created
destination representation. It adds no second operation kind, allocation, route search,
invalidation, or hidden coherence.

This implemented foundation still has no concrete allocation or storage-access implementation or
public output-value access.

The current [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) retains one exact plan reference plus private immutable snapshots
of ordered dense buffer/representation and workspace selections, with aligned read-only,
write-only, or read-write declarations. Empty and repeated selections are valid. `bind` requires
an open `RunState` associated by exact plan [reference identity](../glossary.md#term-reference-identity), resolves
buffers before workspaces in selection order, and invokes concrete-backend compatibility hooks
once per selected representation. A normal incompatibility returns `false` so Runtime can issue a
stable indexed failure.

After all compatibility checks pass, the backend constructs a current [`BoundInvocation`](../glossary.md#bound-invocation--boundinvocation) retaining
the exact [run state](../glossary.md#run-state--runstate) and direct concrete typed buffer/workspace fields. Binding may allocate fresh
nominal arrays and the invocation object, but it acquires no auxiliary closeable binding resource
and changes no ownership. `BoundInvocation.execute()` performs one state-open [guard](../glossary.md#term-guard), then calls
the backend implementation. It rejects execution after the state closes and owns or closes
nothing. One immutable executable may bind concurrently to distinct states; a [bound invocation](../glossary.md#bound-invocation--boundinvocation) is
not thread-safe and must not [race](../glossary.md#term-race) execution with closure.

Caller inputs are borrowed, internal buffers and workspaces are run-owned, and current published
results [lease](../glossary.md#term-lease) the complete `RunState` to [`RunResult`](../glossary.md#run-result--runresult); immutable persistent prepared resources
remain [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution)-owned. A workspace is backend-local scratch rather than a transferable
logical value; [host](../glossary.md#term-host) staging and [device](../glossary.md#term-device) scratch use separate workspace requirements.

The current immutable [`PreparedPublication`](../glossary.md#prepared-publication--preparedpublication) identifies one already-created buffer representation
with dense Runtime buffer, representation, and result positions against one exact plan. These are
prepared/run [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix), not [compiler](../glossary.md#term-compiler) graph or [Tensor](../glossary.md#tensor) identities. Its cold binding resolves the
selected physical representation once into a per-run [`BoundPublication`](../glossary.md#bound-publication--boundpublication). [Publication](../glossary.md#term-publication) requires
the exact selected copy to be valid at that moment, changes only a local one-shot flag, and
performs no [lookup](../glossary.md#term-lookup), transfer, fallback, conversion, allocation, backend [callback](../glossary.md#term-callback), validity change,
or ownership mutation. Publication occurrences form the schedule's dense final suffix. Different
result positions may intentionally [alias](../glossary.md#aliasing-and-arena) one exact representation.

After all bound occurrences publish successfully, `RunResult` privately snapshots their direct
representation references and leases cleanup of the complete open state. Empty results are valid.
Partial publication or failed result construction transfers no cleanup responsibility. Closing
the result delegates to the existing state cleanup, so borrowed inputs remain caller-owned and
run-owned resources retain deterministic reverse cleanup. The result exposes count and [lifecycle](../glossary.md#lifecycle)
only; output value, representation, Tensor, and state access remain deliberately absent.

Java representation compatibility is now checked explicitly once at the current cold binding
boundary. The backend creates typed bound objects with direct references, so the [hot path](../glossary.md#term-hot-path) needs no map lookup,
reflection, string [dispatch](../glossary.md#term-dispatch), graph inspection, backend discovery, [kernel](../glossary.md#kernel) selection, or repeated
unsafe cast. The shared contracts use no raw `Object`, unchecked generic API, public backend type
switch, [registry](../glossary.md#term-registry), or [service locator](../glossary.md#term-service-locator).

[`PreparedExecutionRunner`](../glossary.md#prepared-execution-runner--preparedexecutionrunner) creates one isolated state, binds all remaining schedule occurrences
before the first action, and traverses direct bound references in encounter order. It validates
executable reads before invalidating every copy of each output buffer and validates only exact
writes after success. Failure leaves those output copies invalid and closes the state while
preserving the original unchecked failure. The stateless runner supports concurrent isolated
calls, but each call is [synchronous](../glossary.md#term-synchronous) and uses one orchestrating thread.

The initial [model](../glossary.md#term-model) adds no automatic [pooling](../glossary.md#term-pooling), reuse, [aliasing](../glossary.md#term-aliasing), hidden coherence/write-back,
distributed sharding, or multi-device scheduling. Transfer/materialization recipes and their
success-only validity transition, [prepared publication](../glossary.md#prepared-publication--preparedpublication), result lease, executable-output
invalidation, and schedule traversal are current. Ordinary public [Engine](../glossary.md#term-engine) output access is current
through explicit bounded materialization into [detached](../glossary.md#detached-model-results) host values and exposes neither Runtime
representations nor backend storage.

The selected persistent-resource extension keeps the recipe data immutable while making
`PreparedExecution` an explicit lifecycle [owner](../glossary.md#term-owner). Runtime exposes only a nominal close contract;
backend implementations retain native types and perform physical release. The execution snapshots
unique resource identities once, and schedule repetition never duplicates ownership. A runner
must acquire a lease before creating run state. Close rejects later leases immediately and either
[releases](../glossary.md#commit-release-distribution-log-and-seed-meanings) resources in the closing thread or defers reverse attempt-all cleanup to the final active
lease. It never waits for an active run.

That Runtime owner and lease protocol is current. The [Prepare](../glossary.md#prepare) finalizer-return [transaction](../glossary.md#term-transaction) is also
current: `BackendPartitionFinalizationResult` snapshots one executable and its acquisition-ordered
resources, shared Prepare rejects repeated exact identities, and [graph preparation](../glossary.md#graph-preparation--graphpreparation) transfers the
ordered unique resources only through successful `PreparedExecution` construction. Current [CPU](../glossary.md#term-cpu)
finalization returns an empty resource list. One ordinary outward prepared handle owns its exact inward preparation; advanced prepared
handles retain their existing [opaque](../glossary.md#term-opaque) ownership. Closing a prepared handle rejects later runs and closes its exact
inward Runtime execution once without closing an already returned result. A run that already
acquired the prepared lease may finish; close does not wait, and its final lease release performs
deferred prepared-resource cleanup. Engine closes temporary, trial, and rollback preparations
instead of publishing another owner, while Runtime remains the sole prepared-resource and run-lease
authority. Engine shutdown waits for admitted operations, closes retained results in reverse
run-publication order, then ordinary and advanced prepared handles in reverse
preparation-publication order, and finally the backend integrations.
Completed [Metal](../glossary.md#term-metal) 0002 and 0003 consume this ownership chain for persistent [MPSGraph](../glossary.md#term-mpsgraph) executable and
custom-kernel pipeline resources. Metal 0004 adds typed route-candidate generation and [cache](../glossary.md#term-cache)
compatibility without changing this lifecycle.

Preparation is transactional across this [handoff](../glossary.md#term-handoff). A finalizer cleans resources acquired before a
failed return. After a successful return, shared Prepare tracks the unique resources in acquisition
order and closes them in reverse if another finalizer, schedule assembly, schedule [validation](../glossary.md#term-validation), or
final aggregate construction fails. Ownership transfers only when the complete
`PreparedExecution` is constructed successfully.

A **[resource transaction](../glossary.md#resource-transaction-and-rollback)** is an all-or-clean-up ownership handoff, not a database transaction
and not rollback of arithmetic already executed. Read the successful path downward in this
diagram; side arrows show who cleans acquired resources if preparation stops at that stage.
“Rollback” releases those resources rather than publishing a partly prepared result.

```mermaid
flowchart TD
    accTitle: Persistent prepared-resource ownership transaction
    accDescr: A backend finalizer owns newly acquired resources until it returns successfully, shared Prepare then owns rollback, and successful PreparedExecution construction transfers sole ownership to Runtime; failure at each earlier stage cleans the resources owned there.
    BACKEND["Backend finalizer acquires resources"] --> RETURN["Complete finalizer result returns"]
    BACKEND -->|failure before return| LOCAL["Backend releases its acquisitions"]
    RETURN --> PREPARE["Shared Prepare tracks unique resources"]
    PREPARE -->|later preparation failure| ROLLBACK["Prepare closes acquisitions in reverse order"]
    PREPARE -->|complete construction succeeds| OWNER["PreparedExecution owns resources once"]
    OWNER --> CLOSE["Close and final active lease determine release"]
```

Duplicate rejection does not undo the resource transfer from a returned result. For example,
if one result lists resource identities `[A, A, B]`, Prepare records both unique owners before
rejecting the second `A`, then attempts to close `B` followed by `A` exactly once each. The same
cleanup order applies when an earlier result listed `A` and a later result lists `[A, B]`.
Prepare does not validate or use the rejected executable and calls no later finalizer. A close
failure does not stop cleanup: the first duplicate diagnostic stays primary, with distinct close
failures suppressed in cleanup order and self-suppression skipped. This prevents a trailing owner
such as `B` from leaking just because its result is invalid.

## What a [concrete backend](../glossary.md#concrete-backend) does

Each concrete backend owns preparation and execution details for the partitions assigned to it. This includes:

- [backend](../glossary.md#backend)-owned analysis and [lowering](../glossary.md#lowering) of planned [graph](../glossary.md#graph) regions;
- backend-specific [fusion](../glossary.md#term-fusion) and [specialization](../glossary.md#term-specialization);
- concrete [kernel](../glossary.md#kernel) or executable route selection;
- exact declaration of shared [buffer](../glossary.md#term-buffer) and workspace requirements;
- construction of [immutable](../glossary.md#term-immutable) [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) implementations during the current finalization
  contract;
- physical buffer and workspace representation implementations plus their
  allocation/release/transfer/access mechanics;<span data-glossary-navigation="true"> ([release glossary](../glossary.md#term-release))</span>
- current backend-owned typed [`BoundInvocation`](../glossary.md#bound-invocation--boundinvocation) subclasses with direct representation references;
- backend-specific [materialization](../glossary.md#materialization) and [native bridge](../glossary.md#term-native-bridge) integration; and
- backend [trace](../glossary.md#trace) contributions.

[Planning](../glossary.md#term-planning) chooses an [owner](../glossary.md#term-owner) such as [CPU](../glossary.md#term-cpu), [Metal](../glossary.md#term-metal), or [CUDA](../glossary.md#term-cuda). The owning backend then chooses a [scalar](../glossary.md#term-scalar)
route, [Vector API](../glossary.md#term-vector-api) route, generated [JVM](../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../glossary.md#term-bytecode) CPU [computation kernel](../glossary.md#kernel), [OpenBLAS](../glossary.md#term-openblas) route, [MPSGraph](../glossary.md#term-mpsgraph)
executable, custom Metal kernel, CUDA kernel, or another backend-internal route during [prepare](../glossary.md#prepare).

For a [generated CPU computation kernel](../glossary.md#generated-cpu-computation-kernel), CPU analysis owns lowering, specialization, route choice,
and exact resource declarations. CPU finalization may generate and define the selected kernel or
reuse a compatible CPU-owned generated [artifact](../glossary.md#term-artifact) after shared slot assignment. [Runtime](../glossary.md#term-runtime) receives
only the [prepared executable](../glossary.md#prepared-executable--preparedexecutable) and does not generate, [cache](../glossary.md#term-cache), or select the kernel.

Route selection occurs during analysis, before shared slot assignment. Executable construction
occurs during finalization, after slot assignment. Neither step consumes `CompileArtifacts` or
[Compiler](../glossary.md#term-compiler) internals, and neither is repeated by Runtime.

Each backend also owns typed, version-controlled, tested candidate generators beside the routes
they configure. A generator uses [target](../glossary.md#term-target) capabilities, [canonical](../glossary.md#term-canonical-form) workload facts, and the tuning
budget to return complete valid configurations. The [operation](../glossary.md#operation) family selects the generator; it is
not a [cache key](../glossary.md#term-cache-key) for one universal family-wide setting. Hardware and supported [JDK](../glossary.md#term-java-development-kit-jdk) Vector API
species constrain CPU [vector](../glossary.md#term-vector) candidates, so no candidate can promise an arbitrary physical [lane](../glossary.md#term-lane)
count.

The current narrow Prepare transport can associate a complete backend [candidate batch](../glossary.md#candidate-batch) and an
optional selected decision with one exact [partition](../glossary.md#partition). Shared code does not interpret route,
vector, thread, tile, kernel, or other private fields. Future tuning orchestration may use this
transport while the backend remains responsible for compatibility [validation](../glossary.md#term-validation). During ordinary
preparation, the backend may reuse a compatible entry from an explicit workload cache or apply
safe heuristics. [Model autotuning](../glossary.md#autotuning--model-autotuning) remains optional for correctness.

The [model](../glossary.md#term-model)-specific tuning result is an explicit prepared plan or artifact, not hidden global
state. Its persistent plan record and the reusable workload cache are loaded and updated outside
runtime; incompatible or corrupt entries fall back safely. Physical cache formats and prepared-
executable [serialization](../glossary.md#term-serialization) remain deferred.

## Why there is no shared `backend.lowering` [module](../glossary.md#term-module)

[Lowering](../glossary.md#lowering) is inseparable from a [backend](../glossary.md#backend)'s implementation [model](../glossary.md#term-model), [fusion](../glossary.md#term-fusion) opportunities, [specialization](../glossary.md#term-specialization), storage, and executable construction. A shared lowering module would either encode [concrete backend](../glossary.md#concrete-backend) knowledge in a supposedly common layer or reduce lowering to abstractions that do not own the real decision.

Backend-neutral [graph canonicalization](../glossary.md#graph-canonicalization) and [decomposition](../glossary.md#term-decomposition) belong in [compiler](../glossary.md#term-compiler) passes. Backend-specific lowering belongs in each concrete backend's [prepare](../glossary.md#prepare) implementation.

## Why there is no [runtime](../glossary.md#term-runtime) [service locator](../glossary.md#term-service-locator)

Runtime must execute an already-[prepared schedule](../glossary.md#prepared-schedule--preparedschedule). Looking up a [backend](../glossary.md#backend) or [kernel](../glossary.md#kernel) service during execution would move selection and executable construction into the [hot path](../glossary.md#term-hot-path) and hide dependencies.

The [engine](../glossary.md#term-engine) instead registers backends explicitly and supplies their capability and preparation services before runtime execution. [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) provides the backend-independent runtime call boundary.

## Why reflective [backend](../glossary.md#backend) discovery is not the core mechanism

[Classpath](../glossary.md#term-classpath) scanning, annotation scanning, reflection, or `ServiceLoader` as the default mechanism would make backend availability implicit and mix composition with execution concerns. [Engine](../glossary.md#term-engine) is the [composition root](../glossary.md#term-composition-root), so backends are registered explicitly when the engine is built.

An optional convenience layer for [plugin](../glossary.md#term-plugin) discovery would require the architecture update specified by the contract. It must not become a [runtime](../glossary.md#term-runtime) hot-path mechanism.

See [Lifecycle](lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../glossary.md#lifecycle))</span> for the complete stage flow, [Module Boundaries](module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../glossary.md#term-module))</span> for [ownership](../glossary.md#term-ownership-meanings), and [Dependency Rules](dependency-rules.md) for prohibited [dependency](../glossary.md#term-dependency) edges.
See [Performance Evidence and Tuning](performance-evidence-and-tuning.md) for the optimization
workflow boundaries that feed [prepare](../glossary.md#prepare) without entering runtime.

## Numerical semantics at the boundary

[`PrepareContext`](../glossary.md#prepare-context-preparecontext) carries complete [occurrence](../glossary.md#term-occurrence) and [ownership](../glossary.md#term-ownership-meanings) facts, not a numerical selector.
[Model](../glossary.md#term-model) owns one family/[dtype](../glossary.md#data-type--datatype) contract. [Planning](../glossary.md#term-planning)'s provider query rejects unsupported occurrences;
[backend](../glossary.md#backend) analysis qualifies a concrete route against exact stored/discrete/special boundaries and
test-only [finite](../glossary.md#term-finite) arithmetic criteria before committing its resource declarations. [Prepare](../glossary.md#prepare) does
not infer capability from a passing [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters). The frozen v1 [`ACCELERATOR`](../glossary.md#term-accelerator) [capability ledger](../glossary.md#capability-ledger) records
cutover answers; the profile-free v2 ledger snapshots actual current provider answers. A reviewed
post-cutover addition can change v2 without changing v1. At the Task-0074 [checkpoint](../glossary.md#term-checkpoint)
(2026-10-09), the bounded no-[gradient](../glossary.md#term-gradient) [FLOAT32](../glossary.md#term-float32) [`EXP`](../glossary.md#term-expx) occurrence is true while its independently
queried [BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16) occurrences and their v2 answers were false. Planning could
assign only the true occurrence; Prepare then fixes its direct [MPSGraph](../glossary.md#term-mpsgraph) route before declaring
shared resources. For example, an explicit low-to-FLOAT32 cast followed by FLOAT32 `EXP` may use an
internal MPSGraph boundary step, but the enclosing low-containing [partition](../glossary.md#partition) still uses
`CUSTOM_PROGRAM`; direct low-valued `EXP` was unsupported at that checkpoint. Task 0075 now
admits each [homogeneous](../glossary.md#term-homogeneous) BFLOAT16 and FLOAT16 `EXP` occurrence in the bounded [canonical](../glossary.md#term-canonical-form), no-gradient
domain through its own qualified custom step and independently true current v2 provider answer.
Each step retains FLOAT32 working [evaluation](../glossary.md#term-evaluation) and one final [narrowing](../glossary.md#term-narrowing) to its original low type
under Model semantics, and the whole low-containing partition remains on `CUSTOM_PROGRAM`.
The explicit-cast FLOAT32 route and the low-valued custom routes are distinct. This is an occurrence
and dtype distinction fixed during Prepare, not a [runtime](../glossary.md#term-runtime) decision.
Task 0076 separately admits bounded canonical no-gradient FLOAT32 [`SIGMOID`](../glossary.md#term-sigmoid) through a fixed
sign-guarded MPSGraph composition. Task 0077 separately admits BFLOAT16 and FLOAT16 `SIGMOID`
through typed custom steps. An explicit low-to-FLOAT32 cast permits the FLOAT32 step alongside
a low custom step inside one low-containing `CUSTOM_PROGRAM` partition without changing its
whole-partition route. See the
[backend capability contract](contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../glossary.md#backend-capability))</span>
and [ADR 0027](../design/decisions/0027-post-cutover-capability-evolution.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.

[Metal](../glossary.md#term-metal) retains eligible FLOAT32 MPSGraph routes and the fixed custom program for low-containing
partitions; [CPU](../glossary.md#term-cpu) retains generated and [OpenBLAS](../glossary.md#term-openblas) routes.
The current Metal image is [schema](../glossary.md#term-schema) 20 with a 124-byte header and tuning [identity](../glossary.md#term-identity-meanings) 36; CPU generator
schema is 70; native [ABI](../glossary.md#term-application-binary-interface-abi) 7 retains thirteen exports. CPU/Metal transfer accepts canonical or
resolved positive-[stride](../glossary.md#term-stride) non-overlapping all-[carrier](../glossary.md#term-storage-carrier) storage [layouts](../glossary.md#layout) without [widening](../glossary.md#term-widening) [operation](../glossary.md#operation)
capability. Runtime requires no profile, [tolerance](../glossary.md#term-tolerance), [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions), or policy [lookup](../glossary.md#term-lookup).
