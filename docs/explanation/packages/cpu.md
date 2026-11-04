# CPU execution and native providers

> This is an explanation of the implementation, not a capability or lifecycle contract.
> [ARCHITECTURE.md](../../../ARCHITECTURE.md) and the incorporated
> [CPU backend routes](../../architecture/contracts/backend-execution.md#cpu-backend-routes),
> [low-precision cutover](../../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover),
> and [Prepare/Run lifecycles](../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
> remain authoritative.

This atlas explains how a CPU-owned graph partition becomes reusable executable code, and why
its code can be shared while its mutable memory cannot. Start with the
[framework explanation](../framework.md) for the whole lifecycle or the
[tutorials](../../tutorials/index.md) for application code. The
[package index](index.md) connects this page to Model, Compiler, execution and training.

The central distinction is **backend versus route**. Planning assigns work to the `cpu` backend.
CPU preparation then chooses a *route*: generated scalar or Vector API code, a retained direct
Java recurrence, or a qualified native implementation. OpenBLAS and oneDNN are not additional
backends. Installing a native library does not make an unsupported operation supported.

## Read the pipeline before the packages

The arrows below show handoffs, not calls made once per tensor element:

```text
operation + exact descriptors/attributes/gradient roles
  -> CPU capability answer used by Planning
  -> CPU-owned planned partition
  -> CPU analysis: form units, lower, prove layouts, choose routes
     | immutable typed IR + specialization + exact resource declarations
     v
shared Prepare: assign buffer/workspace slots and representation positions
  -> CPU finalization: verify assignments, realize code/native resources
  -> immutable prepared executable + physical creation recipes
  -> Runtime creates one RunState, borrows inputs, creates run-owned memory
  -> cold bind: typed direct references, geometry, validators, native handles
  -> synchronous execution of already-bound work
  -> completed result owning run resources until close
```

A **unit** is CPU's selected computation region inside a partition. **Lowering** translates graph
meaning into CPU-specific instructions and access recipes. The **intermediate representation
(IR)** records those instructions, not live buffers. A **specialization** records facts that
shape generated code. A **slot** identifies assigned storage in a prepared memory plan; it is
not the storage itself. A **workspace** is run-owned temporary storage or native invocation state.

Analysis must discover storage before shared assignment. Finalization can realize an already
selected recipe, not invent another workspace. Runtime binds and executes the result; it does not
inspect the graph, select a route, generate code or recover from a native failure by changing routes.
The important sources are [CpuPartitionPreparer][preparer], [CpuPartitionFinalizer][finalizer],
and [CpuPreparedPartitionExecutable][partition-executable].

### Following the learning program

This is a conceptual trace of [LearningJourney](../../examples/README.md), not an additional
runnable example or a report of a run performed for this page. Its prerequisites are the tutorial's
configured public Model/Engine/Training workflow and the audited private artifacts described there.
The CPU discussion assumes that workflow is using CPU execution.

| Teaching input or step | Arithmetic and CPU significance |
| --- | --- |
| FLOAT32 input `[2,1]` with values `[1,3]`; supplied gradient-eligible weight `[1,1]` initially `0`; no bias | `prediction = input @ transpose(weight)`. Model copies the supplied parameter; CPU does not own the optimizer's parameter state. |
| First prediction | `[0,0]`; each output contraction has `K=1`. Generated MATMUL preserves exactly one typed product, without an extra identity ADD or FMA. |
| Target `[2,6]`; squared-error SUM | `(0-2)^2 + (0-6)^2 = 40`. CPU lowers the actual graph produced by the objective and Compiler, not a hard-coded training algorithm. |
| Compiler-produced first-order weight gradient | `2*((0-2)*1 + (0-6)*3) = -40`. CPU executes the derivative operations through the same qualified ordinary machinery. |
| `Sgd(0.05)` step | Training updates `weight = 0 - 0.05*(-40) = 2`; CPU is not the SGD owner. |
| Repeated prediction; fresh input `[2,4]` with the same signature | `[2,6]`, then `[4,8]`. Prepared code must bind current storage, not cache the initial weight or an earlier run's values. |

The forward `K=1` product is excluded from both native MATMUL routes. A changed example with more
features would change `K`, but that alone would still not guarantee native selection: exact dtype,
layout, implementation qualification, resource costs and cold selection also matter. Similarly,
changing values to shape `[1,2]` is not reuse of the tutorial's static `[2,1]` signature; application
admission rejects a mismatched descriptor before CPU numerical work. The tutorial and
[reference entry point](../../reference/index.md) explain the public workflow and relevant errors.

## io.github.pho001.synaptik.backend.cpu

**Purpose and entry points.** [CpuBackendIntegration][integration] is the supported
service-provider integration boundary used by Engine and module composition, not an ordinary
application execution facade. [CpuCapabilityProvider][capability] supplies the stable `cpu`
identity and an occurrence-specific answer: operation, original input/output descriptors,
attributes and gradient roles all participate. A FLOAT32 answer is not a blanket answer for a
low-precision or symbolic occurrence.

**Mechanism and relations.** `open()` creates CPU composition with bounded optional OpenBLAS
setup; portable execution remains available if that optional setup does not qualify. Explicit
`openOneDnn(path, api)` requires the exact external library and fixes an API preference before
analysis. The preference restricts native candidates, not which route wins. The integration
exposes partition preparation, schedule contribution and host-storage borrowing to shared
execution owners. Capability itself performs no numerical execution or resource allocation.

**State, ownership and errors.** The adapter owns its composition and native-library lifetime;
returned preparation/tuning collaborators borrow that lifetime. New preparation/run admission
requires an open integration. Already issued native resources have their own retained lifetimes
under Runtime's resource protocol. Required oneDNN load, ABI or symbol failures throw; they are
not converted into successful portable execution. The capability predicate and complete-partition
lowering are separate checks: admitting an operation does not promise arbitrary fusion, physical
aliasing or every native realization.

**Diagnostics, not steering.** [CpuTraceObserver][observer] receives opt-in cold preparation events
with [CpuPreparationDiagnostics][diagnostics] payloads: `SELECTED`, `ANALYSIS_FAILED`, `FINALIZED`
and `FINALIZATION_FAILED`, correlated by preparation identity. Selected route/unit summaries
and bounded rejection reasons distinguish unqualified code, native unavailability and a losing
heuristic. Candidate rejection positions need not be positions in the selected plan. Payloads
retain no tensor values, buffers, native handles or throwable objects. Observer callbacks run
synchronously on preparing threads and may be concurrent; the borrowed sink must be thread-safe.
Observer failures disable later diagnostics without changing business/native outcomes. These
hooks do not instrument every run or prove that a selected executable was invoked.
[CpuLocalWorkloadTuning][local-tuning] and [CpuCompletePlanTuning][plan-tuning] expose cold
candidate/decision collaborations; they do not themselves measure workloads or run benchmarks.

**Learning connection.** This package lets the normal Engine composition admit the tutorial's
FLOAT32 operations and prepare them. It neither builds the linear layer nor differentiates the
objective nor applies SGD. Diagnostics can explain a cold route choice without changing the
program's arithmetic.

## io.github.pho001.synaptik.backend.cpu.internal

This [namespace boundary][internal-package] marks unsupported implementation contracts, including
technically public Java types needed for collaboration across CPU subpackages. It is not a second
application API or a module to import for a shorter tutorial.

The children divide the CPU task into graph lowering, immutable recipes, code realization,
physical representations and invocation. They consume shared Model/Planning/Prepare facts and
return shared Runtime contracts; they keep CPU policy out of those shared owners. The root
package contains no independent resource owner or executable service. Its key invariant is the
cold/hot separation: graph inspection and artifact realization occur before the generated loop,
not through lookups inside that loop.

**Learning connection.** These implementation packages participate beneath the public learning
workflow. LearningJourney should never construct their plans, slots or invocation objects itself;
doing so would bypass the Engine's lifecycle and ownership checks.

## io.github.pho001.synaptik.backend.cpu.internal.lowering

**Purpose and entry points.** [CpuPartitionDagDecomposer][decomposer] turns a complete partition
into a deterministic unit sequence; [CpuPartitionLowering][lowering] translates each admitted unit
into typed CPU facts. Focused owners such as [CpuMatmulLowering][matmul-lowering] handle a family's
shape/layout rules rather than asking a route to reinterpret Model operations.

**How it works.** Lowering follows partition-local producer/consumer occurrences. External
boundaries are ordered by first use; eligible private intermediate results become virtual typed
values. Published or cross-unit values retain stores. Pointwise fusion therefore removes an
unneeded physical intermediate without removing the arithmetic site that produced it. Layout
composition derives actual bases and strides; a private transpose feeding MATMUL can become
strided operand reads instead of a transpose copy. Non-affine movement, indexing, scatter,
folding, reductions, convolution, normalization, loss and Scan use family-specific lowerers.
[CpuOutputInjectivity][injectivity] proves that distinct logical output positions do not write the
same cell. [CpuRepresentationPlanner][representation-planner] describes eligible copy alternatives;
it is not an allocator.

**Inputs, outputs and invariants.** Input is the shared Prepare projection of semantic, DAG and
logical-memory facts. Output is immutable IR plus concrete cold geometry/access bindings and
resource requirements. Lowering owns no arena, worker or assigned Runtime slot. It rejects
unsupported topology, shape/type disagreement, unproved layouts and invalid write geometry;
checked span/byte arithmetic can overflow. Fusion respects publication, saved-value, use and
numerical boundaries rather than merging merely because two operations are adjacent.
Symbolic geometry uses proved numeric binding recipes, not a sample batch substituted into a
static graph. General Scan reuses ordinary unit lowering for its local body, while keeping
activity, carries and saved history explicit; that is not permission for every native or vector
standalone unit to execute inside Scan.

**Learning connection.** The linear projection and squared-error/derivative graph enter here as
ordinary operations. The transpose may be absorbed where proved. The objective's required SUM
and gradient contributions cannot be deleted just because the initial parameter is zero: lowering
does not inspect the example's data to specialize its mathematics.

## io.github.pho001.synaptik.backend.cpu.internal.ir

**Purpose and entry points.** [CpuPortableKernelIr][portable-ir] is the common typed family boundary;
[CpuKernelIr][kernel-ir] describes pointwise values, ordered instructions, loop access forms and
stores. Family records such as [CpuMatmulIr][matmul-ir] retain contraction algorithm/numerical
facts. [CpuAccessPlan][access-plan] separates an access *form* from its concrete binding.

**Mechanism.** Topology-local ordinals replace graph identities. Values distinguish external input,
virtual local and stored output. The dtype belongs to each value, so a byte carrier does not turn
FP8 into BOOL. A CAST retains ordered source/target types and its explicit conversion boundary.
Normalized access regimes describe how a generated loop advances through dense, broadcast or
more general affine storage; bindings retain actual extents, offsets, strides and referenced spans.
This lets compatible structures share code without sharing invocation addresses.

The pointwise canonical identity excludes graph IDs, assigned slots, live segments and run
bindings. Code-shaping family geometry that is baked into a specialization must still enter its
appropriate compatibility identity; “canonical” does not mean every generated class is
shape-independent. Record validation rejects malformed ordinals, arity, types and family facts.
IR owns immutable description data, never tensor memory, worker state or a native engine.

**Learning connection.** `K=1` MATMUL facts and the actual objective/AD operations are represented
here before code exists. The input and weight values `[1,3]` and `0` are not the canonical graph
instruction identity. Replacing the parameter's current storage contents must not rewrite IR.

## io.github.pho001.synaptik.backend.cpu.internal.prepare

**Purpose and entry points.** [CpuPartitionPreparer][preparer] analyzes;
[CpuPartitionPreparationPlan][preparation-plan] retains the selected result;
[CpuPartitionFinalizer][finalizer] realizes it after shared assignment.
[CpuPreparedScheduleAssembler][schedule-assembler] contributes physical recipes/schedule facts to
shared Prepare rather than creating an alternative scheduler.

**Analysis, then assignment, then realization.** Analysis decomposes and lowers the complete
partition, evaluates legal fusion/representation alternatives, chooses scalar/vector computation
and single-range/bounded-parallel orchestration, and compares eligible native peers. Ordinary
preparation retains direct representation topology; explicit candidate workflows may select
already-described copy forms. Bounded complete-plan enumeration is not a whole-partition size
limit: longer partitions have a deterministic direct multi-unit path. Analysis declares exact
buffer spans and workspace sizes/alignment (or qualified run-size recipes) before shared Prepare
assigns storage positions. It performs no generated-artifact lookup or persistence access.

Finalization checks every assigned buffer/workspace and required borrowed execution owner before
artifact realization. It cannot change selected topology, route, carrier pattern, strategy or
resource geometry. Portable units obtain generated artifacts; fixed direct recurrent units do not
need an artifact-store lookup. Native units authenticate their retained provider session and
configuration. oneDNN recompilation must reproduce the analyzed scratch size and implementation.
The returned executable retains the exact shared memory plan. Persistent native resources transfer
to Runtime in acquisition order; failure closes newly acquired resources in reverse order,
preserving cleanup failures as suppressed exceptions, without retrying another route.

**Owned state and failure boundaries.** Plans contain immutable requirements, choices and
metadata, not run-owned buffers. Wrong assignments, closed borrowed owners, incompatible native
metadata, unsupported lowering or checked geometry overflow fail before a successful preparation
is published. [CpuTraceProducer][trace-producer] supplies the cold diagnostics described above.

**Learning connection.** The first use prepares the concrete forward/training graph; later calls
reuse its executable recipes. The updated weight is an invocation input, not an opportunity for
finalization to rebuild a class or silently acquire undeclared training storage.

## io.github.pho001.synaptik.backend.cpu.internal.codegen.emit

**Purpose and entry points.** [CpuClassFileKernelGenerator][generator] uses the Java 26 Class-File
API to emit JVM bytecode directly. It does not generate Java source and call `javac`.
[CpuGeneratedKernel][generated-kernel] strongly retains the verified bytes, hidden-class lookup and
exact static entry `MethodHandle`. Focused emitters include [CpuMatmulEmitter][matmul-emitter],
[CpuScalarEmitter][scalar-emitter], [CpuCastEmitter][cast-emitter] and
[CpuGeneralScanBodyEmitter][scan-body-emitter].

**What the class actually contains.** The generator consumes a selected specialization, not
Model operations. It writes typed loads, locals, arithmetic, loops and stores for the chosen
family and exact ordered primitive-array/`MemorySegment` carrier pattern. A pointwise entry takes
primitive half-open `start`/`end` bounds. Proved dense heap paths narrow bounds/bases once and
use integer addressing; segment/general paths keep long addressing. Eligible vector bodies emit
Vector API operations and scalar tails; parallel range orchestration is outside ordinary
unit classes. Focused families emit their own contraction, contributor, index, movement or
reduction loops rather than calling a generic graph interpreter. This is not a claim that every
body is helper-free: fixed scalar mathematics may share reference helpers, and composed Scan
bodies use statically emitted helper calls.

Generated classes have no fields or interfaces. Generation and persisted-byte reuse both check
JVM verification and the emission owner's complete method inventory, descriptors and structural
limits. Definition uses `Lookup.defineHiddenClass(..., NESTMATE)` and resolves the exact static
entry once. A hidden class may contain subsidiary methods for composed work; it is not necessarily
one giant method. These checks reject stale/malformed class shapes; they neither authenticate
hostile cache storage nor prove numerical execution. The Java Virtual Machine subsequently owns
just-in-time compilation; stored Class-File bytes are not stored machine code.

**Ordinary low precision.** BFLOAT16 and FLOAT16 use raw short carriers; FLOAT8_E4M3 and
FLOAT8_E5M2 use raw bytes. Numerical working sites and accumulators use FLOAT32, with one final
declared narrowing per logical low result, not a narrowing after every reduction contributor.
An explicitly stored/converted intermediate remains its own rounding boundary even when adjacent
work is fused. Exact/discrete movement preserves represented bits. Pointwise low code uses scalar
arithmetic, not a claim of low Vector API support. Cross-type CAST emits the Model-defined
conversion into the target-typed local rather than calling the cold Model conversion oracle.
These rules are visible in the generator and the focused
[low-fold regression source][low-fold-test]; they do not imply that every conceivable low occurrence,
symbolic layout or native implementation qualifies.

**Resources and errors.** Emitters own generation-time builders and immutable emitted artifacts,
not run buffers. Invalid IR/specialization, structural limits, verification or hidden-class/entry
resolution failures reject preparation. General Scan emits an actual time loop and fixed ordinary
body helpers; selected row-region execution has narrowly qualified prebound synchronous range
dispatch. It is not a general per-step callback/interpreter facility.

**Learning connection.** The generated forward contraction respects the singleton typed product;
objective and gradient units execute the actual lowered arithmetic. Reusing the class after SGD
shares code, not the old weight or old output. The FLOAT32 example exercises none of the low
narrowing paths; changing dtype requires fresh exact capability and full workflow qualification.

## io.github.pho001.synaptik.backend.cpu.internal.cache

**Purpose and entry points.** [CpuKernelSpecialization][specialization] and
[CpuLoweringFingerprint][fingerprint] describe generated compatibility;
[CpuGeneratedKernelArtifactStore][artifact-store] realizes or reuses an artifact.
[CpuSpecializationBudget][specialization-budget] bounds generated forms. This cache is not a
parameter cache, output cache or workload-tuning decision store.

**Mechanism.** Before even a weak-cache hit, the store authenticates current IR/specialization
agreement. A process-wide map weakly interns compatible live artifacts. With an explicit trusted
root it may read one bounded current-schema envelope containing compatibility metadata, deterministic
class bytes and a checksum. A persisted hit verifies the bytes and defines a fresh hidden class;
it preserves no earlier JIT machine code or profiling state. Missing, corrupt, stale, oversized or
inaccessible entries are safe misses, followed by verified in-memory generation. Optional
publication failure does not change correctness. A checksum detects corruption, not hostile code
substitution: the root must be trusted.

**Identity and ownership.** Identity includes the lowering fingerprint, relevant generator/family
compatibility, typed carrier pattern, chosen compute form, applicable vector species and every
baked code-shaping fact. Concrete addresses, physical resources and worker identity are not cache
keys. Do not generalize pointwise geometry sharing to family specializations that bake geometry.
Prepared recipes strongly retain artifacts; the store retains them weakly and drains collected
entries at a later cold realization. It starts no cleanup thread and does no invocation-time lookup.
The artifact has no explicit close lifecycle.

**Learning connection.** The same structural code may be reused for repeated predictions or a
compatible fresh preparation. The cache must not cache `weight=0`, the objective `40`, a borrowed
input arena, or the updated parameter `2`. The
[artifact-store test source][artifact-test] illustrates fresh, weak and persisted realization paths;
its presence is not a claim that these paths were run while writing this atlas.

## io.github.pho001.synaptik.backend.cpu.internal.route.portable

**Purpose and entry points.** [CpuPortableRoutePlan][portable-plan] pairs route-independent IR with
its selected Class-File specialization and any admitted numeric batch-binding recipe. It is a
realization description, not a scalar reference interpreter or provider selector.

**Mechanism and state.** Constructors cross-check canonical fingerprints, typed boundary/carrier
patterns, optional family payloads and baked geometry. Mutable carriers, bases, ranges and scratch
slices stay outside the route plan. Scalar/vector compute and range orchestration are different
choices: several scalar ranges can invoke the same scalar artifact. Optional generated copy plans
already describe the adjusted consumer form before this seam; the portable route does not decide
to copy during execution.

The package also holds retained direct Java fixed-recurrence implementations, including
[CpuRecurrentScanKernel][recurrent-kernel] and typed siblings. They are not the reference oracles
and not the general differentiable-Scan implementation. They consume checked geometry and
invocation segments, finish projections before replacing carried coordinates, and keep mutable
carry/scratch in the invocation. General Scan instead uses the emitted time-loop/body machinery
in lowering, code generation and executable packages. Both paths require exact independent
occurrence admission; a class's presence does not grant universal recurrence support.

**Learning connection.** The canonical program uses portable ordinary MATMUL and ordinary
objective/derivative operations, not any fixed recurrence. Its route plan describes reusable code;
Training supplies current parameters on each call. Corrupt or inconsistent plan/specialization
facts fail before a valid executable is constructed.

## io.github.pho001.synaptik.backend.cpu.internal.executable

**Purpose and entry points.** [CpuPreparedExecutable][unit-executable] is the immutable direct-unit
recipe; [CpuPreparedPartitionExecutable][partition-executable] composes fixed topological children.
[CpuGeneralScanPreparedExecutable][scan-executable] binds emitted general Scan work.
[CpuWorkerGroup][workers] and [CpuConcurrencyBudget][budget] coordinate CPU execution, not Runtime
ownership or asynchronous jobs.

**Bind once, execute direct work.** Cold binding checks exact memory-plan/representation positions,
logical dtype, carrier pattern, liveness, accessibility, mutability, alignment, size, geometry and
write-related overlap. It then captures typed arrays/segments, primitive address state, fixed
ranges and direct handles in an invocation. The composite binds all children before execution and
executes them in list order; its dependency indices are diagnostic facts, not a dynamic scheduler.
Required copy units complete before their consumers. No route is selected here.

Metadata validation and value validation have different timing. Index bounds, replacement
uniqueness, canonical BOOL bytes, index-loss targets or Scan lengths may depend on earlier
producers. Their validators are prebound without reading values and execute over the current
eligible domain after producers, before the consumer's numerical writes or worker submission.
A physical byte carrier alone never enables BOOL validation for FP8. Validation boundaries do not
mean an entire multi-unit run can be rolled back byte-for-byte; failure stops execution and Runtime
returns no successful result.

**Threads and owned state.** A unit borrows its worker group. The group owns named daemon platform
threads and must be closed by its owner; it is not a common pool or public executor. Ordinary
parallel submission acquires the selected participant demand from the shared budget, starts
already-bound disjoint ranges and joins all started work before returning. Failure cancels
unclaimed ranges and chooses worker failures deterministically by range index. Interruption joins
started work and restores interrupt status. A narrowly selected row-region path borrows a live
run-owned reservation rather than nesting budget acquisition. MATMUL ranges own complete cells,
rows or tiles, never slices of the contraction `K`. Scan's changing geometry, carry banks,
histories, staging and partials remain run-owned, not mutable fields shared by preparations.

**Learning connection.** A prepared prediction binds `[1,3]` and the current weight; a subsequent
prediction binds fresh values/updated weight. Its output belongs to that run. Keeping a result
open does not turn its buffers into mutable fields of the reusable executable, and closing a
prepared owner is not permission to close caller-owned parameter/input storage.

## io.github.pho001.synaptik.backend.cpu.internal.memory

**Purpose and entry points.** [CpuBufferRepresentation][buffer-representation] is the CPU view of
Runtime buffer storage; [CpuBufferArgument][buffer-argument] carries typed direct access facts.
[CpuRepresentationRecipes][representation-recipes] supplies deferred creation callbacks.
[CpuHostSnapshotExporter][snapshot-exporter] explicitly exports caller-owned host snapshots.

**Physical ownership.** [CpuBorrowedBuffer][borrowed-buffer] retains exact caller host storage
without copying it, extending its segment scope or taking its close duty. The caller must keep it
alive while borrowed. [CpuNativeBuffer][native-buffer] allocates one exact aligned segment in a
shared arena and closes that arena at most once. Shared arena accessibility supports the
orchestrator, qualified worker threads and FFM calls; it does not by itself make conflicting
writes safe. [CpuContiguousWorkspace][workspace] owns aligned run-local temporary bytes.
Representation recipes defer allocation and constant initialization until Runtime creates a run;
constructing a preparation does not allocate its tensor buffers.

**Invariants and failures.** Logical dtype and physical carrier are separate. FP8 byte storage
preserves raw representations independently of BOOL's canonical zero/one rule. Referenced span,
not just logical element count, determines required storage for strided layouts. Sizes,
alignments, whole-element widths, actual batch geometry and access permissions are checked;
overflow, dead/inaccessible segments, insufficient spans or incompatible carriers fail admission.
A materialization workspace may serve a declared contiguous copy, or family scratch may be sliced
into disjoint per-range regions; neither creates a new Runtime lifetime owner. Runtime orchestrates
creation/release and keeps borrowed storage distinct from owned buffers/workspaces.

**Learning connection.** Input and current parameter storage are borrowed; generated outputs and
required scratch are fresh run-owned resources. Materializing a result is an explicit snapshot
operation, not a way to retain an implicitly live native result after its owner closes. The fresh
`[2,4]` invocation must read those values rather than a snapshot of `[1,3]`.

## io.github.pho001.synaptik.backend.cpu.internal.reference

**Purpose and entry points.** [CpuScalarReferenceKernel][scalar-reference] and focused oracles such
as [CpuMatmulReferenceKernel][matmul-reference] provide clean Java semantic/conformance references
for already-lowered CPU facts. They are not a selectable runtime fallback route.

**Mechanism.** References traverse the same typed instructions/access or family geometry in
readable Java. The MATMUL oracle, for example, traverses increasing logical `K`, keeps `K=0`
positive zero, treats `K=1` as one typed product, and stores a completed result once. Low floating
MATMUL uses FLOAT32 working values and final conversion to its represented output. This gives
generated-code tests a precise algorithm to compare rather than merely comparing two unrelated
framework outputs. Some pure scalar mathematics helpers are intentionally shared by generated
bytecode; that does not turn reference graph evaluation into runtime execution. Generated CAST
emits conversion directly, while the cold reference can call the Model conversion oracle.

Reference inputs/outputs are caller-supplied carriers; references do not own Runtime slots,
provider sessions, prepared resources or a thread pool. Carrier/type and checked-address errors
reject invalid oracle inputs. Their scope is supported lowered forms, not an interpreter for
arbitrary Model operations.

**Learning connection.** The program's singleton projection and gradient contractions can be
reasoned about with these algorithms, but LearningJourney does not invoke them through Runtime.
The numerical walkthrough above is arithmetic explanation, not a claim that the reference or
production generated code was executed for this documentation change.

## io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas

**Purpose and entry points.** [CpuBackendComposition][composition] owns CPU lifetime composition;
[CpuOpenBlasDiscovery][blas-discovery] attempts bounded loading;
[CpuOpenBlasQualifier][blas-qualifier] issues exact-session compatibility evidence;
[CpuOpenBlasRouteSelector][blas-selector] compares a native candidate with the already-complete
portable lowering; [CpuOpenBlasPreparedExecutable][blas-executable] binds the selected route.
This policy belongs to CPU, not to the leaf provider described below.

**Loading is not qualification or selection.** Discovery may be disabled, use an exact name or
absolute path, or use a fixed platform table. Loading resolves symbols only. Qualification checks
bounded host/ABI/binary facts and small exact-output/special-value GEMM smoke cases against the
exact live coordinator. Absolute-path inspection can retain binary identity; name loading is
session-only evidence. This is not hostile-binary authentication or a universal numerical or
performance certificate. Optional failure leaves ordinary CPU available before preparation;
there is no native failure-triggered route switch during execution.

**The selected domain and copies.** The production route is a positive rank-two same-type FLOAT32
or FLOAT64 bare product with `K>1`, optionally preceded by proved private transpose prefixes. It
has no low-precision, batched/broadcast, `K=0`, `K=1` or arbitrary epilogue grant. Common lowering
supplies effective reads; the non-transposed dense row-major provider may require generated affine
copy-in for either operand and copy-out for a noncanonical result. Each is a separately declared
run-owned workspace before shared assignment. The cost comparison includes these transitions
and thread configuration, not only GEMM arithmetic. Ineligibility, overflow, a tie or insufficient
benefit retains portable execution during cold selection. No benchmark/speedup follows from the
dimensionless cost model.

**State and threads.** [CpuOpenBlasCoordinator][blas-coordinator] owns the transferred provider
resource and coordinates calls/configuration with the common CPU budget. It captures the original
positive native thread count, installs/verifies the selected count, excludes configuration writers
from admitted calls, and on close quiesces borrowers, restores/verifies the original count, then
closes the provider. Its lock does not serialize arbitrary outside native users of the same binary.
A wrong session/target, closed owner, failed count verification/restoration or invocation error
fails rather than retrying portable execution. Prepared plans retain immutable facts; actual
segments and copies are run-owned.

**Learning connection.** The forward `K=1` projection is deliberately ineligible, even if OpenBLAS
is installed. The tutorial's graph does not establish native selection for its derivative units
either: each actual prepared occurrence must independently meet route and cost conditions.

## io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn

**Purpose and entry points.** [CpuOneDnnProjection][dnn-projection] projects common CPU units into
typed native recipes; [CpuOneDnnSelector][dnn-selector] performs cold whole-plan comparison;
[CpuOneDnnCompiler][dnn-compiler] authenticates structural/API qualification and realizes the chosen
recipe through [CpuOneDnnPrimitives][dnn-primitives] or [CpuOneDnnGraph][dnn-graph]. Graph here is a
vendor execution API, not Synaptik's graph compiler or automatic differentiation.

**External library and exact admission.** [CpuOneDnnLibrary][dnn-library] loads an explicitly
required absolute path, checks exact oneDNN `3.14.0`, little-endian LP64 and sequential CPU runtime
`SEQ` (code `1`), then owns the FFM lookup/CPU engine. There is no bundled oneDNN binary or optional
automatic oneDNN discovery. Loading/authentication failure throws. API preference `AUTO`,
`PRIMITIVES` or `GRAPH` does not force native execution. The bare MATMUL baseline requires positive
static dense/permuted-dense or normalized broadcast-batch layouts, rank at most 12 and `K>1`.
Neither symbolic geometry nor Scan orchestration inherits native admission.

Some ordinary pointwise/reduction/movement units and literal substeps of a composite unit have
independently qualified projections. A composite pipeline retains the common unit's arithmetic
sites and saved/public stores; it is not permission to replace convolution, activation,
normalization or pooling wholesale with an opaque vendor operator. Exact implementation identity
matters after descriptor construction. For example, raw reorder, extrema and elementary operations
have implementation-specific checks; identity reductions and several Graph forms are rejected.
Source enums or a native descriptor being available are not a support matrix.

**Current low/native limits.** Typed low inputs retain their real dtype, origin, span and alignment.
Primitives use strict FLOAT32 working/accumulation attributes. Qualified low MATMUL implementations
are explicitly checked (including selected reference, AArch64 BF16/F16 and selected x86 FP8 forms),
not accepted by a broad optimized-name prefix. Graph 3.14 rejects FP8 arithmetic; low Graph binary
and reduction forms have additional restrictions. CPU native FLOAT64 is unavailable; OpenBLAS
remains the separate FLOAT32/FLOAT64 peer. Native FP8 requires an available supported x86 target;
no native FP8 execution is claimed on Arm. An enum tag, reference recipe or rejected-target test
is not positive execution evidence on another host.

Current low logical outputs use a declared FLOAT32 result workspace followed by the ordinary
generated final CAST via [CpuOneDnnOutputNarrowing][dnn-narrowing]. This preserves exact Model
rounding and the logical output/saved boundary instead of trusting an unproved vendor destination
conversion. It does not promote FP8 operands to FLOAT32 and call that native FP8 execution.
The [low-native test source][low-native-test] separates exact generated final narrowing from
optional real typed-input native tests gated by an explicit library and available implementation.
It is source evidence, not a report of native tests run for this page.

**Prepare and run resources.** Analysis queries handle-free descriptor/scratch facts; primitive
queries create no executable, whereas a Graph scratch query temporarily compiles and closes one.
Finalization recompiles exactly the chosen API/recipe, verifies scratch and implementation identity,
and transfers persistent kernels to Runtime with transactional cleanup on failure. Native caches
remain library-owned and are unrelated to the CPU Class-File artifact store.

[CpuOneDnnPreparedExecutable][dnn-executable] supplies workspace creators for fresh run-owned
streams, memory/tensor handles and ABI tables. Issued workspaces retain their kernel/library until
close independently of prepared-owner closure. Binding validates typed spans/origins and rejects
input/output/scratch aliasing. Matching logical-value/origin/span ingress copies may be shared
within one RunState; the first native consumer refreshes them every run, never across-run weight
snapshots. Graph inputs are variable. Fixed copy/execute/wait work runs synchronously under one
CPU-budget permit; no late selection, allocation of native control objects, numerical repair,
retry or portable fallback is installed in the hot invocation.

**Learning connection.** The forward singleton MATMUL is not native-eligible. Existing Compiler
AD, if a different occurrence qualifies natively, still produces the ordinary derivative graph;
oneDNN supplies neither vendor backward semantics nor a second training pipeline. Closing or
failing a required oneDNN integration is not silently repaired by the tutorial's portable path.

## io.github.pho001.synaptik.backend.provider.openblas

**Purpose and entry points.** This separate
[openblas-provider module](../../../modules/backend/openblas-provider) is a JDK-only leaf below CPU.
[OpenBlasLibrary][provider-library] explicitly loads the caller's name/path;
[FfmOpenBlasNativeAccess][provider-ffm] binds `cblas_sgemm`, `cblas_dgemm`,
`openblas_set_num_threads` and `openblas_get_num_threads` through the Foreign Function and Memory
(FFM) API. C integer arguments select the ordinary 32-bit-`blasint` interface.
[OpenBlasGemmInvocation][provider-gemm] validates and invokes normalized calls;
[OpenBlasThreadControl][provider-threads] exposes thread utilities.

**Mechanism and boundary.** A call computes one dense row-major non-transposed
`C[m,n] = alpha*(A[m,k] @ B[k,n]) + beta*C[m,n]` for FLOAT32 or FLOAT64. Segments start at byte
offset zero. The provider does not interpret Config, Tensor graphs, layouts, batching, packing,
costs, fallback or backend ownership. CPU depends on this provider, never the reverse. In
particular, a successful raw GEMM call does not establish production CPU route qualification.

**Resources, errors and invariants.** The library handle owns a shared-arena lookup lifetime, not
caller matrix memory. Calls retain, allocate, copy, reinterpret and close no caller segments.
Inputs may be read-only and overlap each other; the required output must be writable, correctly
aligned/sized and disjoint from both required input spans. Dimensions must be nonnegative and all
scope/thread accessibility checks hold. Validation finishes even for output-empty calls, which
then make no native call. A raw positive-output `K=0` call still invokes GEMM so `beta` applies;
this differs deliberately from production CPU native-route exclusion of `K<=1`.

Loading/binding failure is reported by `OpenBlasLoadException`; malformed arguments, closed/dead
storage or native invocation failures are not converted into a different algorithm. Callers must
keep library and segments alive and avoid racing close with calls. Native thread count is mutable
library/process state; the provider supplies no call serialization or coordination with unrelated
native consumers. CPU's coordinator adds its bounded policy above this leaf. The provider makes
no numerical, determinism or performance guarantee for arbitrary OpenBLAS binaries.

**Learning connection.** Application code does not use raw GEMM or set native thread counts to
implement the lesson. The forward singleton product remains portable. If a different prepared
CPU workload selects OpenBLAS, this package performs only its already-normalized GEMM and thread
operations; objective construction, AD, SGD and result ownership remain elsewhere.

## Native path accounting

“Native” describes several different relationships. Read this table from repository code to the
external implementation it uses; it is not an installation or publishing promise.

| Repository path / external component | Actual role and packaging distinction |
| --- | --- |
| [modules/backend/cpu](../../../modules/backend/cpu), especially `internal/route/nativeops/onednn` | Java FFM bindings and CPU-owned recipes/qualification/lifetime handling. oneDNN is an external exact shared library, not a checked-in C/C++ CPU bridge. |
| [modules/backend/openblas-provider](../../../modules/backend/openblas-provider) | Java FFM leaf binding to an external OpenBLAS shared library. The provider module is not the OpenBLAS binary. CPU's discovery/qualification code lives in `internal/route/nativeblas/openblas`, not here. |
| External OpenBLAS name or absolute path | Provides GEMM and thread-control symbols. Optional bounded CPU discovery is distinct from a caller explicitly opening a provider handle. |
| Explicit external oneDNN 3.14.0 absolute path | Provides primitive/Graph C APIs and CPU engine; exact ABI/runtime loading is required, not optional discovery. Library and issued resources have retained lifetimes. |
| [modules/backend/metal/macos-arm64](../../../modules/backend/metal/macos-arm64) | The colocated native Metal bridge/source/build/package tree belongs to the Metal backend, not to either CPU route. See the [Metal atlas](metal.md) for its native source accounting. |

The CPU and provider paths above contain Java integration, not a packaged Synaptik CPU JNI bridge.
Classpath FFM deployments require `--enable-native-access=ALL-UNNAMED`; this flag does not install
a library, authenticate an ABI, qualify an implementation or force a route. Portable generated
CPU code does not require installing OpenBLAS or oneDNN. For exact ownership rules consult the
[OpenBLAS provider contract](../../architecture/contracts/backend-execution.md#openblas-provider)
and [qualified oneDNN contract](../../architecture/contracts/backend-execution.md#qualified-onednn-cpu-realization).

## What to take away

The canonical program needs public Model/Engine/Training owners, not fourteen backend imports.
CPU capability admits exact occurrences; lowering preserves their meaning; shared Prepare assigns
resources; finalization creates immutable recipes and code; each run owns current values and
mutable resources. Native routes can replace only independently qualified physical realizations
chosen before execution. Neither a native library, a cache hit nor a generated class proves that
the learning program ran, that a host supports native FP8, or that execution will be faster.

[integration]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java
[capability]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java
[observer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuTraceObserver.java
[diagnostics]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuPreparationDiagnostics.java
[local-tuning]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java
[plan-tuning]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCompletePlanTuning.java
[internal-package]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/package-info.java
[preparer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java
[finalizer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java
[preparation-plan]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java
[schedule-assembler]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPreparedScheduleAssembler.java
[trace-producer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuTraceProducer.java
[decomposer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java
[lowering]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java
[matmul-lowering]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulLowering.java
[injectivity]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuOutputInjectivity.java
[representation-planner]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java
[portable-ir]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java
[kernel-ir]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java
[matmul-ir]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuMatmulIr.java
[access-plan]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java
[generator]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java
[generated-kernel]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernel.java
[matmul-emitter]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuMatmulEmitter.java
[scalar-emitter]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java
[cast-emitter]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastEmitter.java
[scan-body-emitter]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanBodyEmitter.java
[specialization]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java
[fingerprint]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuLoweringFingerprint.java
[specialization-budget]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuSpecializationBudget.java
[artifact-store]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java
[artifact-test]: ../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java
[low-fold-test]: ../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLowFoldFp32AccumulationTest.java
[portable-plan]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java
[recurrent-kernel]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuRecurrentScanKernel.java
[unit-executable]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java
[partition-executable]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java
[scan-executable]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java
[workers]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuWorkerGroup.java
[budget]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuConcurrencyBudget.java
[buffer-representation]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferRepresentation.java
[buffer-argument]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferArgument.java
[representation-recipes]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuRepresentationRecipes.java
[snapshot-exporter]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuHostSnapshotExporter.java
[borrowed-buffer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBorrowedBuffer.java
[native-buffer]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuNativeBuffer.java
[workspace]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuContiguousWorkspace.java
[scalar-reference]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java
[matmul-reference]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuMatmulReferenceKernel.java
[composition]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java
[blas-discovery]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscovery.java
[blas-qualifier]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasQualifier.java
[blas-selector]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java
[blas-executable]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasPreparedExecutable.java
[blas-coordinator]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasCoordinator.java
[dnn-projection]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnProjection.java
[dnn-selector]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnSelector.java
[dnn-compiler]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnCompiler.java
[dnn-primitives]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnPrimitives.java
[dnn-graph]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnGraph.java
[dnn-library]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnLibrary.java
[dnn-narrowing]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnOutputNarrowing.java
[dnn-executable]: ../../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnPreparedExecutable.java
[low-native-test]: ../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeops/onednn/CpuOneDnnLowPrecisionTest.java
[provider-library]: ../../../modules/backend/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java
[provider-ffm]: ../../../modules/backend/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/FfmOpenBlasNativeAccess.java
[provider-gemm]: ../../../modules/backend/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasGemmInvocation.java
[provider-threads]: ../../../modules/backend/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasThreadControl.java
