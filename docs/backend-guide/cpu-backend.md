# CPU backend

## Outcome and status

This guide defines the CPU integration boundary and helps contributors avoid treating CPU routes
as separate backends. The current CPU module accepts eleven bounded, fully static portable families.
A pointwise partition is one supported occurrence or one connected straight-line chain of at most
eight occurrences. A static affine partition is one connected one-input/one-output chain of at
most eight resolved-layout view occurrences. Either family lowers to one computation unit, one
route-independent canonical kernel intermediate representation (IR), one generated Java 26 class,
and one partition-level prepared executable. Internal single-use results remain graph and logical-
memory values but are virtual in the unit. Pointwise analysis declares its derived external
boundaries and sole final output; affine analysis declares exactly the original source and final
result. A static movement partition is exactly one resolved-layout PAD, TILE, CONCAT, STACK,
UNFOLD_AXIS, or UNFOLD2D occurrence. It declares each distinct input once in first-occurrence
order plus one distinct injective output while preserving every semantic composition occurrence.
UNFOLD_AXIS copies all six represented types; UNFOLD2D copies only FLOAT64, FLOAT32, and BFLOAT16
under the Model's canonical NCHW columns contract.
The fourth family is exactly one resolved-layout `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, or
`ONE_HOT` occurrence. It validates every logical INT32/INT64 index deterministically before any
output write and then uses scalar or parallel-scalar generated output ranges. The movement family
also accepts exactly one resolved-layout `SLICE_UPDATE` with ordered `[base, update]` inputs and
either signed finite-coordinate or target-relative placement. It writes a distinct injective
result with the base Shape and leaves both inputs unchanged.
The fifth family is exactly one resolved-layout functional scatter occurrence:
`SCATTER_ELEMENTS`, Gather-compatible fixed-add `SCATTER_ADD`, or `SCATTER_ND`. All three consume
ordered `[data, indices, updates]`, validate before writing, and create a fresh data-shaped result
without mutating inputs. Scalar and parallel-scalar execution own disjoint output coordinates.
The sixth family is exactly one resolved-layout overlap fold occurrence: general-axis
`FOLD_AXIS`, or canonical columns-to-NCHW `FOLD2D`. Every result coordinate begins at represented
positive zero and accumulates its logical input contributions in canonical input row-major order.
Both families accept FLOAT64, FLOAT32, and BFLOAT16; `FOLD_AXIS` additionally accepts INT32 and
INT64. `FOLD2D` excludes symmetric-padding and ceil-tail positions outside the unpadded result.
Schema-26 generated fold entries embed the selected family mapping, represented addition, and
typed array or native-order segment access. Dense heap arrays use integer loop/address state;
general layouts and segment or mixed carriers normally use long addressing. Schema 35 adds one
guarded bounded exception for the frozen mixed-carrier FLOAT32 FOLD2D shape; every unproved case
retains the typed general-long body.
The seventh family is exactly one resolved-layout stable ordering occurrence: `SORT`, `ARGSORT`,
or two-output `TOP_K`. It accepts all six represented types, orders complete logical-axis slices,
keeps every floating NaN after non-NaNs in both directions, reverses signed-zero order with the
requested direction, and preserves increasing original logical index for equal values and NaNs.
SORT and TOP_K copy selected represented bits; ARGSORT and TOP_K write zero-based INT64 axis
coordinates. Unsorted TOP_K deterministically orders the selected pairs by increasing original
index. Scalar or complete-slice parallel-scalar execution uses one exact run-owned workspace with
two primitive INT64 merge-index regions per selected range.
Schema-27 generated ordering entries embed the typed stable merge, comparison, optional
selected-pair index ordering, and represented value/index stores. Proved dense heap-array forms
use integer loop and address state; general layouts and segment or mixed carriers retain typed
long addressing. Scratch access remains direct through the assigned segment in both forms.
The eighth family is exactly one resolved-layout `INITIAL_STATE` or FLOAT64/FLOAT32 `DROPOUT`
occurrence. It uses explicit key/counter state and the versioned CPU-private
`SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` mapping. INITIAL_STATE writes key then counter without a draw.
DROPOUT consumes one draw per row-major logical value ordinal, writes the dropped value, a
canonical one-byte BOOL keep mask, and the next key/counter state, with identical scalar and
parallel-scalar results and zero workspace. Schema-28 generated entries embed the typed state
prologue, counter mapping, threshold, value, mask, and element loop. Dense heap arrays use direct
integer addressing; arbitrary layouts and segment or mixed carriers normally use typed long
addressing. Schema 35 adds one guarded bounded exception for the frozen rank-one FLOAT32 mixed-
carrier dropout shape; every unproved case retains the typed general-long body.
The ninth family is exactly one resolved-layout `CUM_SUM` or `CUM_PROD` occurrence. It accepts
FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 in inclusive/exclusive and forward/reverse modes.
Each logical scan slice keeps one sequential typed accumulator; scalar or parallel-scalar
execution may distribute only whole independent slices. The family declares one input and one
distinct output, with zero workspace or materialization. Schema 38 adds one completely
guarded fixed `[1024,1024]` axis-one exclusive reverse INT64 product body over segment carriers;
all other admitted scan geometries retain their existing typed bodies.
The tenth family is exactly one resolved-layout ordinary or binding-aware aggregate occurrence. `SUM`, `PROD`,
`MIN`, and `MAX` accept FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64; `MEAN` accepts the three
floating types; and Boolean `ALL` and `ANY` accept canonical BOOL. Exact full, single-axis, and
multi-axis forms share one output-cell geometry, including an empty multi-axis selection as a
one-value point domain. Scalar or parallel-scalar execution distributes only whole output cells.
Floating SUM, MEAN, and PROD use declared run-owned exact-state workspace; every other row uses
zero workspace. No row uses input materialization, partial reductions, or a combine step.
Binding-aware SUM-to-Shape adds fully bound right-aligned SUM over the same five numeric types;
it uses exact floating or modular integral arithmetic when axes reduce and raw represented copying
when none does.
The eleventh family is exactly one resolved-layout, one-axis `ARG_MIN` or `ARG_MAX` occurrence.
It accepts FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64 and writes a zero-based logical axis
coordinate as INT64. The normalized axis, keep/remove-Dimension form, FIRST/LAST tie policy, NaN
preference, signed-zero order, and signed integral order come from the Model contract. Scalar or
parallel-scalar execution owns complete output cells and uses no workspace or materialization.

Generated scalar and Java 26 Vector API entries accept primitive `start` and `end` bounds.
Compatible concrete extents bind on the cold path and share identical class bytes and one
process-local loaded compatibility identity. The pointwise semantic matrix uses exactly five
executable types—FLOAT64, FLOAT32, INT32, INT64, and BOOL—and one forty-eight-opcode CPU-private
vocabulary. Affine copies accept all six current Model data types and transfer represented bits
without conversion. Outside the cumulative-scan, ordinary-aggregate, and overlap-fold families
described below,
BFLOAT16 uses the existing raw `short[]` representation or native-order two-byte segment access
only; raw movement support is not general BFLOAT16 arithmetic or numerical support. The access
family covers scalar/rank/singleton/multi-axis
broadcasting, zero extents, offsets, positive and broadcast-zero strides, and ordered
heap/segment/mixed carrier patterns for the derived boundaries. CPU 0005H closes all nineteen
same-typed FLOAT32/FLOAT64 unary kinds, while preserving the three separate floating-classification
kinds and CPU 0005G's extrema, clamp, Tensor power, and canonical-BOOL logic. CPU 0005J preserves
that semantic inventory while adding exact preferred-species vector parity for selected floating
extrema/clamp/ReLU/sign/cast, signed-integral arithmetic/extrema/cast, canonical-BOOL logic/cast,
and narrowly virtual floating comparison/classification masks through logical masks into WHERE.
Cold analysis selects scalar, vector, parallel-scalar, or parallel-vector execution;
vector-ineligible admitted geometry or opcodes fall back to scalar compute. Analysis can also
compare direct access with at most one CPU-private contiguous FLOAT64 input copy from explicit
dimensionless cold cost evidence. A selected copy uses one declared run-owned workspace and
completes before consumer execution. Capability and lowering fail closed for every other
operation, type, shape, layout, parameter, alias, fan-out, publication, carrier, or route. In
particular, CAST is same-type only and BFLOAT16 remains representation-only for affine movement;
CPU does not invent cross-type conversion semantics. Native, tuning, excluded pointwise rows,
general partition-DAG fusion and later operation families remain Draft. Functional scatter,
overlap fold, stable ordering, explicit-state random, cumulative-scan, and ordinary-aggregate
execution are current
only within the exact
static portable boundaries described below; they do not imply native, vector, dynamic-layout, or
universal backend coverage.

The lower-level OpenBLAS provider separately implements explicit library loading, required-symbol
binding, a caller-owned lookup lifetime, low-level FLOAT32/FLOAT64 dense row-major general matrix
multiplication (GEMM), and direct positive thread-count query/control. Those leaf capabilities do
not make OpenBLAS a CPU route until a later CPU prepare task supplies truthful eligibility,
normalization, fallback, and executable integration.

## Prerequisites and terms

Contributors need JDK 26 and the ownership rules in the [architecture
contract](../../ARCHITECTURE.md#cpu-backend-routes). Real loading also requires a compatible
OpenBLAS binary already installed or otherwise available to the operating-system loader, plus
deployment JVM permission for restricted native access.

- A **route** is one backend-local implementation choice. The glossary entry for
  [backend route](../glossary.md#backend-route) explains why OpenBLAS is not a separate backend.
- The **Foreign Function and Memory (FFM) API** is the JDK native-interoperability API used to
  load the library and bind C symbols.
- An [OpenBLAS library handle](../glossary.md#openblas-library-handle--openblaslibrary) is the
  caller-owned Java lifetime for one complete lookup and binding set. It exposes mutable native
  thread state without owning a thread-selection policy.

## Ownership mental model and scope

The CPU backend owns capability reporting, physical CPU representations, whole-partition lowering,
specialization, fusion, route selection, executable units, and typed tracing.
The Class-File/Vector portable baseline, narrow OpenBLAS fallback, and exact-capability vendor peers
remain routes within one CPU owner. Fusion and specialization describe common lowering or a route
configuration; they are not additional backend identities or provider-owned graph pipelines. The
architecture does not require a particular bytecode-generation API.

```text
planning: owner = CPU
CPU prepare: common lowering -> eligible portable / OpenBLAS / vendor peer plans -> whole-plan choice
runtime: invoke prepared CPU executable
```

The low-level OpenBLAS provider owns library loading, symbol binding, GEMM calls, and thread
control only. It does not interpret graphs or choose fusion, broadcasting, representations,
materialization, or Runtime lifetime. Dependency direction is
`backends/cpu -> backends/openblas-provider`.

## Historical provisional foundation

The following foundation description records the implementation delivered by superseded CPU
tasks 0001 through 0005. CPU 0005A removed its flat `backend.cpu.execution` package, per-node
candidate pipeline, worker/vector placeholders, and mandatory durable-store behavior. It remains
useful as historical evidence only; the current implementation is described under
[Atomic partition-kernel reset](#atomic-partition-kernel-reset).

`CpuCapabilityProvider` is the only public CPU type. It returns the exact stable
`BackendId("cpu")` constant. Its `supports` method returns `true` only for a parameterless binary
`ADD` occurrence with exactly two inputs and one output, exact equal fully static shapes, and one
exact common `FLOAT64`, `FLOAT32`, `INT32`, or `INT64` type. Every layout must be unresolved or
resolved as canonical dense contiguous, zero-offset, non-view geometry. An unresolved layout is
accepted because CPU preparation selects a canonical materialized representation; a resolved
view, offset, or non-canonical stride is rejected. The identity itself still does not assert
availability or registration, and infrastructure or a related type never broadens capability.

The backend-private execution package supplies two physical ownership forms:

- `CpuBorrowedBuffer` retains one exact caller-owned `HostTensorStorage` and segment. Its close is
  a no-op and it never copies, slices, reinterprets, or closes caller memory.
- `CpuNativeBuffer` and `CpuNativeWorkspace` each own one exact aligned native segment in a
  distinct shared arena. They are run-owned, accessible across permitted CPU worker/Foreign
  Function and Memory (FFM) threads while open, and close their arena at most once.

Cold binding classifies each selected buffer independently and does not materialize bytes. If an
exact primitive heap carrier is observable and matches the logical data type, the bound argument
retains that array plus the carrier-relative byte offset and exact byte size. Otherwise the
`CpuBufferArgument.Segment` form retains the exact selected `MemorySegment` or slice, reports
`byteOffset() == 0`, and preserves its byte size and read-only state. `Segment` is an access form,
not a native-provenance claim: genuine native segments use it, and so do heap-backed segments
whose exact primitive carrier is unavailable. In JDK 26, a read-only heap segment has an empty
`heapBase()`, so CPU binding keeps that exact read-only segment without copying.

Every binding checks representation kind, liveness, current-thread access, exact byte size,
writability, logical data type, carrier compatibility when observable, and required alignment.
Mixed array and exact-segment selections are ordinary ordered inputs and outputs. The later
route-specific binder receives fresh cold arrays once and must retain the needed direct typed
fields in its bound invocation. The hot call performs no representation lookup, heap-base
discovery, type switch, route selection, materialization, or allocation.

The fixed worker group owns daemon platform workers and splits a non-empty half-open range into
bounded deterministic contiguous ranges. It provides synchronous completion, isolated concurrent
submissions, cooperative cancellation between ranges, first/suppressed failure propagation,
interrupt restoration, and idempotent shutdown. It does not select an operation, kernel, route,
thread count from configuration, or reduction combine policy.

## Historical generated-kernel foundation

The generated-kernel foundation turns one already-selected typed specialization and one
family-owned emitter into verified Java class bytes and a directly invocable hidden-class
artifact. It is deliberately narrower than a CPU route:

```text
typed specialization + matching family emitter
  -> deterministic Java 26 class bytes
  -> Class-File API verification
  -> fresh hidden nestmate class
  -> exact static MethodHandle entry point
```

The Java Class-File API is the current CPU-internal bytecode builder, and the Java Vector API is
the current CPU-internal vector mechanism. Both are implementation choices, not architecture
invariants. The Vector API remains an incubator module in Java 26 and is enabled only for CPU
compile, test, and Javadoc tasks.

`CpuKernelSpecialization` is an immutable, structurally comparable description of every fact that
may change emitted bytes. It contains the generator schema and Java class-file target, a
family-owned lowering fingerprint, ordered argument carrier and access facts, baked byte offsets,
element strides and extents, dynamic-extent count, execution mode, exact Vector species when
applicable, byte order, unroll/tile/tail structure, the current exact/default numerical mode, and
partial-combine ordering. A second SHA-256 content fingerprint covers that complete ordered
specialization. These fingerprints are deterministic compatibility identities, not security
authentication, Model-operation capability, or a persistent cache format.

The portable mode vocabulary has exactly four values:

| Mode | Emission form | Invocation range |
|---|---|---|
| Scalar single-thread | Scalar instructions | One complete element count |
| Scalar parallel | Scalar instructions | Caller-assigned half-open start/end plus range index |
| Vector API single-thread | Exact-species Vector API instructions | One complete element count |
| Vector API parallel | Exact-species Vector API instructions | Caller-assigned half-open start/end plus range index |

The mode owns only the structural choice between the scalar and Vector family callback. A family
emitter owns semantic instruction construction, while distinct carrier, scalar, Vector, range/
tile/tail, and partial/combine emitters provide low-level bytecode seams. The first production
family emitter implements only scalar pointwise `ADD`; all other semantic families and its Vector
callback remain unsupported. The range controls do not create parallel work: `CpuWorkerGroup` owns
worker dispatch, synchronization, cancellation, and failure propagation outside generated code.

Generated entry signatures use the selected primitive array carrier (`double[]`, `float[]`,
`short[]`, `int[]`, `long[]`, or `byte[]`), an exact `MemorySegment`, or an ordered mixture. A
primitive-array byte offset is either baked into the specialization or passed as a primitive
`long`; exact segments use a segment-relative zero base. Dynamic extents and the single-thread
count or parallel range controls are also primitive entry arguments. Generation performs no copy
and owns none of the supplied arrays or segments. Read-only, write-only, and read-write access
facts constrain emitted loads and stores; exact-segment liveness, thread access, size, and
writability remain cold-binding obligations.

`CpuGeneratedKernel` retains the specialization, verified bytes, full-privilege hidden lookup,
hidden class, exact static method handle, and method type. Retaining the artifact retains that
hidden-class lifetime, but the foundation promises no unloading time. Calling the generator's
convenience generation path directly still defines a fresh hidden class. The durable store
described below instead reuses compatible class bytes and weakly interns a loaded artifact while
it remains live.

Foundation tests still invoke bounded synthetic copy and structural probes across heap,
exact-segment, and mixed signatures in all four modes. Those tests prove the generic generator
and direct-carrier seams only. Separate CPU `ADD` tests prove the narrow production route described
below; they do not broaden the generic foundation to other operations or modes.

## Historical durable generated-artifact store

`CpuGeneratedKernelArtifactStore` is the package-private cold-loading boundary for deterministic
generated class bytes. A caller constructs it with one explicit trusted local root; construction
normalizes that path but does not touch the filesystem. A later CPU finalizer can ask the store
for one already-selected `CpuKernelSpecialization` and matching family emitter after shared
Prepare assigns slots. The current CPU-private finalizer performs that integration and makes the
prepared executable a strong owner. Public composition still has no owner for the explicit root
or worker-group lifetime.

```text
explicit root + complete specialization + matching emitter
  -> deterministic .cpuclass path
  -> validate a complete compatible envelope, or emit verified bytes
  -> force a temporary file and atomically replace the final entry
  -> re-read, revalidate, define a hidden class, and resolve the exact handle
  -> return a loaded CpuGeneratedKernel
```

The final relative path has the fixed form
`generated-kernels/v1/sha256/<two hex digits>/<62 hex digits>.cpuclass`. Its SHA-256 address covers
a domain separator and complete canonical compatibility metadata. The self-contained binary
envelope records a format version, bounded metadata and class lengths, a class-byte checksum, the
full metadata, and the class bytes. A path digest, specialization fingerprint, Java hash code, or
checksum alone is never accepted as compatibility proof. The store compares the complete
metadata, rejects trailing or truncated data, verifies the checksum, and delegates Java class-file
verification and exact class-shape validation to `CpuClassFileKernelGenerator` before definition.

A missing, incompatible, or corrupt final entry is a cache miss. The store emits deterministic
verified bytes, writes one uniquely named temporary regular file in the final directory, forces
its contents, and performs an atomic replacement. It never falls back to a non-atomic move. A
mandatory final re-read means only the bytes visible at the deterministic final path are defined.
Separate Java Virtual Machine (JVM) processes may emit redundantly, but atomic replacement and
complete validation ensure that readers do not accept a partial publication. Ordinary directory,
permission, read, write, force, or atomic-move failures fail cold loading instead of silently
bypassing the explicit store. Compatible age and access history never invalidate an entry; the
store performs no expiry, eviction, quota enforcement, directory sweep, or background cleanup.

Equal requests in one process share one in-flight attempt even when they use different store
instances normalized to the same root. Waiters observe the same loaded artifact or the same
unchecked failure, and an interrupted waiter restores its interrupt status without cancelling the
attempt. A failed attempt is removed and may be retried. Loaded artifacts are interned only through
weak references with reference-queue cleanup during later calls. The store has no strong completed
map or hidden-class unloading promise. A caller-held artifact remains invocable; task 0004 makes
the prepared executable its strong lifetime owner while that prepared recipe is usable.

The root is an executable-code trust boundary. SHA-256 checksums and structural verification
detect accidental corruption and incompatible data; they do not authenticate bytecode. The
caller must choose a local root whose permissions and ancestor paths prevent untrusted writers
from replacing its contents. An attacker who can modify both stored class bytes and their checksum
is outside this mechanism's security claim. The store is also distinct from a workload tuning
cache: it reuses exact executable bytes after route and specialization selection, while tuning
evidence would help select a route or configuration before finalization.

## Historical typed portable preparation

The current CPU-private preparation path connects one already CPU-owned partition to one reusable
Runtime recipe. Candidate sources return complete partition candidates, each with a non-empty
ordered kernel sequence:

```text
PrepareContext<CpuPortableAnalysisInputs>
  -> injected family-owned candidate source
  -> exact validation and first eligible candidate in source order
  -> CpuPortablePreparationPlan + exact shared resource declarations
  -> shared Prepare assigns slots and validates declaration geometry
  -> CpuPortablePartitionFinalizer
  -> compatible artifact load or verified generation
  -> CpuPortablePreparedExecutable
  -> Runtime cold binding through CpuBorrowedBuffer or CpuNativeBuffer
  -> family-owned typed BoundInvocation
```

`CpuPortableAnalysisInputs` snapshots the exact supported Vector species and retains one immutable
prepared parallel configuration. The configuration records positive worker count, positive
minimum range size, and deterministic-range intent; it does not own a worker group or discover a
target. A directly injected `CpuPortableCandidateSource` supplies complete candidates in
deterministic preference order. There is no registry, reflective discovery, generic parameter
map, tuning lookup, or universal route priority.

Each node-level `CpuPortableKernelCandidate` binds one specialization and matching family emitter
to ordered buffer/workspace declarations, identity-bound uses, and a signature-specific cold
binder. `CpuPortablePartitionCandidate` groups those node recipes and their exact shared
declarations for the complete partition. CPU
analysis rejects non-CPU ownership, an unprojected buffer value, a specialization data type that
disagrees with that projected value, unsupported Vector species, and malformed declarations or
uses. It selects the first valid eligible candidate. The declared byte size and alignment remain
backend-owned opaque facts at this point: task 0004 does not derive dense byte size from a tensor
descriptor or introduce a layout/materialization policy. The existing shared finalization handoff
checks that each assigned Runtime slot satisfies the exact declaration geometry.

The finalizer receives an explicit trusted artifact root and an already-owned open
`CpuWorkerGroup`. It resolves every shared declaration and every node use to assigned dense plan
positions before the first artifact-store request, then loads or generates artifacts in node
order. It neither changes a selected specialization nor adds a resource. The resulting immutable
executable strongly retains every `CpuGeneratedKernel` and exact direct `MethodHandle`; it borrows
rather than closes the worker group and owns no per-run representation.

Runtime remains the lifetime boundary. Caller storage enters the run only through the non-owning
`CpuBorrowedBuffer` implementation of `BufferRepresentation`; no executable or binder accepts
`HostTensorStorage` directly. Cold binding first applies Runtime and the task-0001 representation
checks, then validates the exact specialization carrier, baked or dynamic array offset form, and
parallel worker accessibility. A family binder copies the direct handle and required carrier,
segment, workspace, worker, and primitive range fields into a guard-free node call. One
partition-level `BoundInvocation` performs the sole run-state-open guard, then invokes those
direct child calls in node order. The hot path has no slot lookup, storage discovery, argument
classification, artifact access, reflection, handle adaptation, route selection, operation
dispatch, or allocation.

The foundation tests use fixed synthetic emitters and signature-specific invocations to exercise all
four portable modes, heap and exact-segment carriers, mixed signatures, repeated resources,
artifact miss and reuse, shared assignment, parallel worker access, and concurrent binding. Normal
return proves only that the staged lifecycle and direct-call boundary work for those synthetic
inputs. Production pointwise tests separately exercise the exact route below; neither test family
claims backend conformance, public Engine integration, or performance.

## Historical dense ADD route

The only production operation route is parameterless `BinaryArithmeticKind.ADD`. CPU analysis
visits a non-empty CPU-owned partition in stored node order and rejects the complete partition if
any node falls outside the exact capability matrix. It does not split the partition, skip a node,
or fall back after Planning has selected CPU ownership.

For each accepted node, `CpuPointwiseAddCandidateSource` derives one identity-free lowering and
one scalar single-thread specialization whose signature is:

```text
(readable left MemorySegment,
 readable right MemorySegment,
 writable output MemorySegment,
 element count) -> void
```

Shared buffer requirements are interned once per graph `ValueId` in first encounter order while
visiting each node's inputs and output. Exact byte size is the checked static element count times
the data-type width, alignment is that width, and a zero-element value requests zero bytes. A
value read by one node and written by another receives partition-level read-write access. Every
node recipe still keeps `[left, right, output]` argument order and uses the exact shared
declaration objects.

The generated loop follows logical flat order. `FLOAT64` and `FLOAT32` use ordinary Java Virtual
Machine IEEE binary addition; `INT32` and `INT64` use Java fixed-width two's-complement modular
addition. For example, a two-node INT32 partition with inputs `[1, 2]` and `[10, 20]` can compute:

```text
node 0: intermediate = [1, 2] + [10, 20] = [11, 22]
node 1: output       = intermediate + [1, 2] = [12, 24]
```

The intermediate uses one shared assigned buffer. Finalization resolves the complete assignment
mapping before artifact access, loads the two generated artifacts in node order, and cold binding
creates two direct guard-free child calls. Normal execution produces `[12, 24]` and proves the
ordered native-segment ADD route for this example; it does not prove heap, Vector, parallel,
fused, dynamic-shape, or non-canonical-layout support.

The historical route supported scalars and zero extents because their shapes were fully static. It deliberately
excludes broadcasting, mixed types, BFLOAT16, BOOL, FLOAT16, resolved views or non-canonical
strides, input/output aliasing, heap or mixed carriers, Vector API and parallel execution,
fusion, workspace, tuning, and vendor libraries. CPU 0005A replaced this route atomically; later
family expansion remains Draft.

## Atomic partition-kernel reset

Completed [CPU task 0005A](../planning/backends/cpu/tasks/0005a-atomic-partition-kernel-architecture-reset.md)
atomically replaced the historical per-node ADD sequence before operation-family expansion.

The current CPU-private flow is:

```text
complete CPU-owned partition
  -> computation-oriented execution units
  -> legal then profitable fusion
  -> route-independent canonical kernel IR and normalized access-plan form
  -> exact post-fusion buffer declarations
  -> portable Class-File/Vector baseline or eligible peer native route
  -> one partition-level PreparedExecutable
  -> direct cold-bound execution
```

The proving slice is one fully static FLOAT64 ADD -> exact GELU -> MUL chain. ADD and MUL use the
current Model right-aligned broadcast result, GELU preserves the ADD result Shape exactly, and all
boundary layouts are resolved. Its ADD and
GELU results remain graph values and kernel-IR values, but they are virtual within the one fused
unit and receive no buffer declaration or Runtime slot. This does not remove them from
`LogicalMemoryPlan`: logical graph-value planning and backend physical materialization are
different decisions. A scalar reference realization checks conformance and provides a fail-closed
fallback; it is not an `Operation` or IR interpreter inside Runtime.

The portable route is bytecode-first Java 26 Class-File generation plus the Vector API and remains
the always-available semantic fallback for every occurrence it supports. CPU 0005C completes all
four execution strategies for the proving slice. OpenBLAS is a narrow cross-platform native fallback for eligible
BLAS-compatible linear algebra, not the universal or preferred CPU route. Vendor/platform peers
include Accelerate BLAS/vDSP/vForce on Apple CPU, distinct oneMKL BLAS/VML and oneDNN families on
Intel, and distinct AOCL-BLAS/AOCL-LibM and optional ZenDNN families on AMD.

There is no fixed vendor priority and no route-specific `BackendId`. Planning selects the single
CPU owner; CPU Prepare filters exact capability, platform, semantics, numerical/determinism,
layout, representation, and resources, then compares whole-plan cost. Apple Silicon may admit
Accelerate. Other ARM targets use portable code generation unless a later task adds an explicitly
verified provider; ARM does not imply one hard-coded native library.

Portable generated loops use universal primitive `start`/`end` loops. Compatible
concrete extents and element count bind on the cold path and do not normally change class/cache
identity. Focused tests prove that one exact byte sequence and loaded compatibility identity serve
two compatible extents of the same fused topology. Additional planner-visible fixed-shape or
unrolled variants remain later, budgeted, evidence-selected specializations. The current hard
ceiling admits four complete analysis candidates and realizes one artifact; additional
fixed-shape and unrolled variant budgets are both zero. This budget counts distinct preparation
candidates and artifact identities. A generator may still emit a fixed-trip straight-line body
inside that one artifact after a complete cold geometry guard proves it, without creating another
route or Runtime choice.

The canonical IR records typed boundary/virtual values, ordered semantics, structural access-plan
form, loop model, and stores. It excludes selected route, thread count, vector species, artifact
root, graph/Runtime identity, generator version, and invocation bindings. The current single
right-aligned access system consumes Model `ShapeBroadcast` and `LayoutDescriptor` semantics. It
supports fully static scalar, expanded-rank, singleton/multi-axis/zero broadcast, offsets,
positive and zero strides, and heap, segment, or mixed carriers. Current Prepare has no exact
dynamic binding and rejects non-static projected shapes, so CPU fails closed for dynamic/symbolic
dimensions; a future task requires an explicit
exact-binding contract. The access family does not duplicate `WHERE`, elementwise, or fused
planners. Broadcast gradients remain `SUM_TO_SHAPE` and later reduction work.

Carrier access form is structural when it changes generated code. Each generated class contains
exactly one direct static entry whose ordered typed primitive-array/`MemorySegment` signature
matches the prepared unit. FLOAT64, FLOAT32, INT32, INT64, and BOOL use `double[]`, `float[]`,
`int[]`, `long[]`, and canonical `byte[]` respectively, or an exact native-order segment. The
ordered type/carrier pattern participates in specialization and class/cache compatibility;
exact carrier objects, byte offsets, extents, strides, slots, addresses, and run identity remain
cold bindings. Equal topology/access structure/carrier patterns may reuse class bytes and loaded
identity across compatible extents, while a different carrier pattern intentionally selects a
different specialization. The generator does not emit every possible carrier combination into
each class. CPU analysis receives the backend-owned prepared pattern, finalization realizes the
one matching artifact, and Runtime binding only validates matching concrete carriers; it neither
generates nor specializes code.

`CpuPartitionAnalysisInputs.DEFAULT` disables the lowering manifest, persistence,
materialization, vector preference, and parallel execution. Its empty explicit carrier list means
"select one exact `MemorySegment` form per boundary derived by lowering"; it is no longer a
four-boundary topology contract. An explicit composition-created input may instead supply a
non-null ordered heap/segment pattern. CPU analysis snapshots it and validates its count, type,
and order against the derived declarations. No physical carrier object or general Config value
enters the analysis input.

### Current bounded pointwise family

The portable route maps admitted Model occurrences once into a single CPU-private
`CpuPointwiseOpcode` vocabulary. The forty-eight opcodes are grouped by family rather than by
operation-specific lowerer, emitter, executable, or registry class:

| Family | Current admitted semantics and exact types |
|---|---|
| Binary arithmetic | Same-type `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` for FLOAT64, FLOAT32, INT32, and INT64; same-type `DIV` and direct Tensor/Tensor `POW` for FLOAT64 and FLOAT32 |
| Scalar arithmetic and range | Exact typed scalar `ADD`, `SUB`, `MUL`, `MIN`, and `MAX` for the same four numeric types; exact typed scalar `DIV` and `POW` plus first-class range `CLAMP` for FLOAT64 and FLOAT32 |
| Unary | All nineteen `UnaryElementwiseKind` values for same-typed FLOAT64/FLOAT32: `ABS`, `NEG`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`, `GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` |
| Classification | `IS_FINITE`, `IS_NAN`, and `IS_INF` for FLOAT64/FLOAT32 to BOOL |
| Comparison | All six ordered/equality comparisons for the four numeric types to BOOL |
| Logical | Canonical-BOOL `AND`, `OR`, and `NOT` |
| Selection | BOOL-conditioned `WHERE` with same-type FLOAT64 or FLOAT32 branches |
| Cast | Represented-value-preserving same-type `CAST` for all five executable types |

Scalar and parallel-scalar generated execution cover every row. Vector and parallel-vector require
one exact lane type, an eligible value/mask topology, and the completed contiguous-run access
checks:

| Lane/topology | Exact vector-eligible opcodes |
|---|---|
| FLOAT32 or FLOAT64 values | `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, eligible `SCALAR_POW`, `SCALAR_MIN`, `SCALAR_MAX`, `SCALAR_CLAMP`, `NEG`, `ABS`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `SIGN`, `RELU`, `TANH`, `GELU_EXACT`, and same-type `CAST` |
| INT32 or INT64 values | `ADD`, `SUB`, `MUL`, `MIN`, `MAX`, `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_MIN`, `SCALAR_MAX`, and same-type `CAST` |
| Canonical BOOL values | `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, and same-type `CAST` |
| FLOAT32 or FLOAT64 with virtual BOOL masks | Six comparisons and three classifications may produce virtual masks; `LOGICAL_AND`, `LOGICAL_OR`, and `LOGICAL_NOT` combine them; floating `WHERE` consumes a matching virtual mask or scalar/all-zero BOOL broadcast |

FLOAT32/FLOAT64 vector extrema retain NaN propagation and directional signed-zero selection;
`SCALAR_CLAMP` remains lower `MAX` followed by upper `MIN`; ReLU is `MAX(input, +0)`; and sign
preserves both zero signs and NaN while mapping every other negative or positive value, including
infinity, to exact `-1` or `+1`. INT32/INT64 vector addition, subtraction, and multiplication are
fixed-width modular operations, while their extrema use signed order. Same-type cast is an
identity for all five executable types and does not imply cross-type conversion.

`SCALAR_POW` is vector-realizable only for `POSITIVE_ONE`, `IDENTITY`, `SQUARE`, and
`RECIPROCAL`; `DIRECT` remains scalar. Materialized comparison/classification results, non-scalar
external BOOL conditions for floating `WHERE`, and mixed or otherwise unsafe mask topologies also
remain scalar. A chain containing one vector-ineligible instruction remains
one unit and selects scalar or parallel-scalar; it is not split and does not extract individual
lanes. Parallel-scalar remains available when cold orchestration selects more than one range. This
fallback is a cold strategy decision, not an execution failure or a universal vectorization
claim.

This remains the existing connected straight-line one-through-eight unit. General partition-DAG
decomposition and bounded vertical or horizontal fusion remain Draft CPU 0008A work.

Schema 40 adds one bounded body inside the existing scalar pointwise artifact for the frozen
FLOAT32 `[512,512]` `DIV -> SIGMOID -> MUL` chain. Its ordered carriers are `MemorySegment`,
`float[]`, `MemorySegment`, and `float[]`: the first and third inputs are dense, the second input
broadcasts across the last axis, and the output starts at offset 3 with strides `[1024,2]`.
Complete topology, carrier, geometry-length, ordered-range, start-coordinate/address, extent,
stride, and sentinel guards admit any legal half-open range. The proved loop derives row and
column from the integer ordinal, performs direct unaligned segment loads and the strided heap
store, and preserves FLOAT32 division and multiplication around the stable sigmoid. That sigmoid
uses its sign branch and binary64 `StrictMath.exp` work before one final narrowing to FLOAT32.
Every failed proof enters the unchanged typed general-long state machine in the same artifact.
This form changes no pointwise semantics, capability, lowering, declaration, materialization,
workspace, route, finalization, Runtime binding, or public API, and its retained timing evidence
supports only this frozen case and environment.

For example, a three-occurrence chain can add two FLOAT32 inputs, negate the virtual result, and
compare it with a third FLOAT32 input. Lowering derives the three external reads in first-use
order and appends the BOOL comparison result as the only materialized output:

```text
left -----\
           FLOAT32 ADD -> virtual sum -> FLOAT32 NEG -> virtual negated --\
right ----/                                                               > GREATER_THAN -> BOOL output
threshold ---------------------------------------------------------------/
```

The two intermediate values receive no CPU buffer declaration or Runtime slot. The generated
entry has four boundary arguments only because this example has three external reads and one
output; another legal chain derives a different count. The BOOL store is canonical byte `0` or
`1`, and cold binding validates every BOOL input boundary before hot invocation. This example
demonstrates family lowering, virtuality, and derived boundary cardinality; it does not add
cross-type conversion, general DAG fusion, or another operation family.

Complete-partition lowering admits one through eight stored occurrences only when the internal
dataflow is connected, acyclic in stored order, and straight-line. Each non-final result must have
one later internal consumer and no publication, fan-out, or cross-partition obligation. Side
inputs become external boundaries, later occurrences may right-broadcast a virtual result, and
exactly one final store is materialized. Disconnected subchains, multiple outputs, internal
fan-out, more than eight occurrences, or a partially supported partition fail before declaration
or artifact access.

Same-type CAST is intentionally narrow. Current Model construction represents all source/target
pairs but leaves cross-type numerical conversion—rounding, overflow, saturation, NaN, and BOOL
conversion—separately owned. CPU therefore executes only represented-value identity when input,
target attribute, and output type are identical. It does not remove the cast from the compiled
graph or imply a compiler canonicalization rule.

Partition lowering, fusion legality/profitability, canonical IR, access plans, materialization
accounting, numerical/determinism checks, and representation planning are common across routes.
Native provider adapters consume selected route facts and own only provider ABI/compatibility
mechanics. They do not reinterpret graphs, duplicate broadcasting or fusion, or own shared Runtime
resource lifetimes.

Numerical permission follows the same common path. An internal portable or vendor implementation
may use a fast exponential, hyperbolic tangent, or similar algorithm under the hood, but it is an
exact/default candidate only when it satisfies the ordinary operation's conformance contract. A
genuinely relaxed approximation requires explicit caller permission from the future backend-neutral
prepare configuration; hardware, an installed provider, workload size, a tuning objective, or a
benchmark result cannot grant it. Tuning and benchmarking compare only candidates already eligible
under that permission and contract.

### Current static affine view family

The portable affine family composes a closed set of existing Model view meanings during CPU
analysis:

- `CONTIGUOUS`, `RESHAPE`, and `EXPAND`;
- `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`;
- scalar-coordinate `SELECT`; and
- positive-step `SLICE`, including the current target-relative crop attribute form.

Every occurrence has exactly one input and one output of the same data type. Input, output, target,
and prefix Shapes used by the mapping must be fully static, and both tensor layouts must already be
resolved and exactly match the Model relationship. CPU does not bind symbolic dimensions, invent a
layout, normalize caller syntax, or admit the currently unresolved negative-step slice result.

One through eight occurrences may form one unit only when the chain has one external source and
one final result. Each intermediate must feed the next occurrence exactly once and must have no
publication, fan-out, or cross-partition obligation. Such an intermediate remains present in the
compiled graph and `LogicalMemoryPlan`, but CPU analysis declares no buffer or workspace for it;
there is no Runtime slot, generated instruction, or store for that value. This is CPU-private
virtuality, not shared storage aliasing. Because shared preparation does not assign two graph
values to one representation, the final result always receives its own buffer and one explicit
boundary copy.

Analysis walks the final result's logical coordinates, reverses the complete view chain, and
composes exact source and result element addresses. The result buffer keeps its resolved offset,
positive strides, zero strides, referenced element span, and view classification. An injective
result layout uses one write per logical element. A zero-stride or otherwise non-injective result
uses one deterministic write per distinct address, but only when all omitted repeated logical
coordinates select the same source value. Otherwise lowering fails closed. Scalar results contain
one address pair, and zero-element results contain none.

For example, start with a contiguous INT32 source of Shape `[2, 1]`, expand it to Shape `[2, 3]`
with result strides `[1, 0]`, and publish that expanded result. The six logical result coordinates
refer to only two result addresses:

```text
source logical values:                  [a, b]
expanded logical coordinates:           [a, a, a, b, b, b]
distinct source -> result address pairs: 0 -> 0, 1 -> 1
```

The generated copy writes `a` and `b` once each. This proves deterministic distinct-address
materialization; it does not attach shared aliasing to the expanded graph value or promise a dense
six-element result representation.

The copy transfers raw represented bits for FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL.
The seven generated carrier forms are `DOUBLE_ARRAY`, `FLOAT_ARRAY`, `SHORT_ARRAY`, `INT_ARRAY`,
`LONG_ARRAY`, `BYTE_ARRAY`, and `MEMORY_SEGMENT`. `SHORT_ARRAY` maps only the existing BFLOAT16
`CpuBufferArgument.Shorts` form. Array-to-array, array-to-segment, segment-to-array, and segment-to-
segment copies preserve floating NaN payloads and signed zeros, raw BFLOAT16 payloads, signed
integral patterns, and canonical BOOL bytes without conversion, promotion, arithmetic, or
canonicalization.

Affine bodies use scalar compute. The scalar and parallel-scalar strategies execute arbitrary
half-open ranges; parallel ranges are allowed only over disjoint distinct-address writes.
`VECTOR_IF_ELIGIBLE` therefore selects scalar compute for this family. There is no Vector API
affine load/store, gather, scatter, masked tail, or speed claim.

Lifecycle ownership remains unchanged. CPU analysis validates and composes the chain, chooses the
scalar orchestration, and declares exactly the source and final result. Shared Prepare assigns
those two slots without interpreting the affine plan. CPU finalization validates both assignments
before current schema-42 artifact access and constructs one immutable executable. Cold binding validates
the exact data type, carrier, byte size, alignment, accessibility, output writability, canonical
BOOL input bytes, and source/result non-overlap. Runtime then invokes only the prepared direct
carriers, address table, and `start`/`end` bounds; it receives no operation, graph node, Shape,
layout, affine IR, or route choice.

Static non-affine movement is the separate bounded family below. Index tensors, functional
scatter, overlap fold, ordering and top-K,
dynamic layouts, general partition-DAG decomposition/fusion, and
benchmarks remain outside this implemented family.

Schema 39 adds one guarded body inside that same affine artifact for the frozen raw-BFLOAT16
`[256,32,32]` PERMUTE followed by identity-range SLICE mapping. The input is a
`MemorySegment`, the output is `short[]`, and both layouts retain element offset 5 and their
proved positive non-unit strides. Complete geometry-length, ordered-range, and sentinel
address-pair guards establish the mapping for every legal half-open logical range before the body
narrows ordinals. Its loop derives coordinates with shifts and masks, performs a direct unaligned
short load, and stores the exact represented bits into the short array. A failed guard enters the
unchanged typed general-long address-pair body. This bounded form changes no view composition,
declaration, materialization, route, resource, finalization, or Runtime contract and makes no
general performance claim.

### Current static pad, tile, and composition movement family

The portable movement family accepts exactly one fully static, resolved-layout `PAD`, `TILE`,
`CONCAT`, or `STACK` occurrence. All inputs and the output have one exact common Model data type;
represented-bit movement therefore covers FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL
without conversion or numerical interpretation. PAD retains the exact typed fill bits, TILE
repeats complete logical input patterns, CONCAT selects ordered segments along an existing axis,
and STACK selects ordered inputs along one inserted axis.

Composition accepts one through sixteen semantic input occurrences. A repeated input remains a
repeated semantic segment, but CPU analysis declares that graph value only once. Unique input
boundaries appear in first-occurrence order, followed by one output declaration. The output must
have an injective resolved layout and a representation distinct from every input; even an
identity-like request materializes its result. Input layouts may be offset or strided because
cold geometry maps output coordinates directly to the selected input carrier.

The canonical movement IR records only the family, rank, represented type, unique structural
access forms, ordered occurrence-to-boundary mapping, one output store, and code-shaping padding
bits. Compatible extents, layout offsets and strides, normalized axes, padding widths, repeats,
composition segment prefixes, and slice placement remain compact immutable preparation geometry
outside artifact identity. CPU retains no address or selector entry per output element.

For example, concatenating semantic inputs `[left, middle, left]` along axis zero declares two
unique input buffers plus the output, while retaining occurrence map `[0, 1, 0]`:

```text
unique boundaries:  0 = left, 1 = middle
semantic segments:  left, middle, left
occurrence map:      0, 1, 0
```

Generated scalar bodies accept arbitrary half-open output ranges. Cold binding computes the
range-start coordinate and base addresses once. The hot loop advances coordinates with carry and
reset state; TILE wraps through the same mechanism, without per-element division or modulo.
PAD may branch between exact fill bits and its source, while CONCAT and STACK may branch among at
most sixteen ordered occurrences. There is no reflection, Model interpretation, map lookup,
callback, or per-element allocation in generated code.

Movement uses scalar compute and either single-thread or deterministic parallel orchestration.
Parallel chunks are safe because output injectivity proves disjoint writes. Vector preference
falls back to scalar for this family. CPU finalization realizes current schema-42 generated artifacts;
cold binding validates complete input/output spans and rejects every output/input overlap before
execution. The scalar reference consumes the same movement IR and compact geometry for
differential tests, not as a Runtime fallback.

Dense heap-array affine and movement entries use a cold-proved integer loop/address form. The
entry descriptor remains shape-polymorphic and accepts `long start, long end`; it narrows those
bounds, carrier bases, and compact movement geometry once before entering the loop. Dense affine
copies then advance direct source/result array indexes. PAD, TILE, CONCAT, STACK, UNFOLD_AXIS,
UNFOLD2D, and SLICE_UPDATE retain family-specific integer coordinate/address state. Schema 34
lets PAD, CONCAT, UNFOLD_AXIS, UNFOLD2D, and canonical-BOOL axis-zero STACK use bounded primitive geometry and address/coordinate
cursors when cold invocation checks prove every geometry value, range, and transition, regardless
of heap, segment, or mixed carrier choice. The original typed general-long form remains the
fallback whenever those checks cannot prove the bounded form. Target schema-37 classes embed this
work directly and contain no hidden Synaptik runtime call.

TILE wraps each source coordinate by its own input-axis extent. An outer source coordinate advances
only when the matching output axis carries; repeating the final axis therefore does not
incorrectly advance an outer source row. This rule applies to dense and general-long forms and to
arbitrary legal subranges. The generated TILE loop performs no per-element division, modulo,
allocation, or helper bridge.

CPU 0007A0A measured this bounded code-shape correction outside JUnit. Its pre-change
generated/direct ratios were 4.582 for dense FLOAT64 affine CONTIGUOUS, 1.928 for FLOAT32 TILE,
and 2.181 for INT32 SLICE_UPDATE. The corrected five-fork median-of-fork-medians ratios were
0.869, 1.107, and 1.132 respectively, all within the fixed `<= 1.15x` gate. The final TILE case
uses input Shape `[1024, 256]`, repeats the final axis four times, and writes 1,048,576 outputs.
The probe used Java 26.0.1 on macOS 26.5.2 aarch64, fixed one-gibibyte initial and maximum heaps,
five warmup batches, nine randomized measurement rounds, adaptive batches of at least 25 ms, and
exact verification. These local measurements validate the selected cases; they do not select a
production route or promise parity for every mapping or machine.

CPU 0007A1E applied the schema-33 bounded form to four fixed general-address cases. The worst of
five generated/direct fork ratios was `0.823098645x` for PAD, `1.143960287x` for CONCAT,
`0.896544010x` for UNFOLD_AXIS, and `0.825152484x` for UNFOLD2D; every fork passed the unchanged
`<= 1.15x` gate. These results cover only the retained cases and environment. The unchanged
20-row diagnostic probe still exits nonzero because twelve or thirteen deferred non-target rows
exceed the gate in each fork, so this evidence is not a full-probe performance pass.

CPU 0007A1F added a narrow canonical-BOOL STACK form for axis zero. After cold geometry proof, it
copies occurrences in semantic order through direct primitive loops, including the repeated
mapping `[0, 1, 0, 2]`, across mixed `byte[]` and `MemorySegment` inputs into `byte[]` output. The
typed general-long form remains the fallback. Its five generated/direct ratios were
`0.933040078x`, `0.878074544x`, `0.869558736x`, `0.932181070x`, and `0.986374612x`.
These measurements cover only that retained shape and environment.

### Current static window extraction

The movement family also accepts exactly one fully static, resolved-layout window occurrence.
`UNFOLD_AXIS` replaces one input axis by the floor-counted window-position extent and appends the
window size as the final result axis. For input Shape `[2, 3]`, axis `1`, size `2`, and step `1`,
the exact result Shape is `[2, 2, 2]`; output coordinate `[n, position, offset]` reads input
coordinate `[n, position + offset]`. This row copies represented bits for FLOAT64, FLOAT32,
BFLOAT16, INT32, INT64, and canonical BOOL without padding or numerical interpretation.

`UNFOLD2D` accepts rank-four NCHW input and produces canonical rank-three im2col columns in
`[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]` order. Kernel width and output
width are the fastest-changing coordinates. Kernel, stride, symmetric padding, dilation, and
floor/ceil output-grid calculations are checked during CPU analysis. Direct `Window2dAttrs` uses
represented positive zero outside the source. `Unfold2dAttrs` instead uses its exact matching
FLOAT64, FLOAT32, or BFLOAT16 padding bits, including signed zero and NaN payloads. Integral and
BOOL two-dimensional unfold remain unsupported because the Model contract is floating-only.

The generated artifact records the window family, rank/access structure, carrier/type facts, and
exact two-dimensional padding bits. Extents, layout magnitudes, axis and window parameters,
spatial grid, and arbitrary-range start coordinates remain compact cold geometry. Compatible
cold geometry can therefore reuse the same class bytes. Generated loops advance output and
window coordinates with carry/reset odometers; division and remainder are confined to cold range
initialization. One input and one distinct injective output are declared, with no workspace.

Window extraction does not implement value-dependent indices, scatter, fold, or overlap
accumulation, dynamic Shape binding, general mixed-family fusion, a native route, or a performance
claim. Functional slice update is a separate one-node movement form described below.

### Current gather and one-hot indexing family

The portable indexing family accepts exactly one fully static, resolved-layout occurrence of
`GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, or `ONE_HOT`. Gather data and output may use any of the
six current Model data types and copy represented bits without conversion. Indices are INT32 or
INT64. `ONE_HOT` writes canonical BOOL byte `0` or `1` and has no configurable on/off values,
alternate output type, ignored index, or default row.

The coordinate meanings remain Model-owned. `GATHER` replaces one data axis with the complete
indices Shape, `GATHER_ELEMENTS` aligns every indices coordinate with data except at the selected
axis, `GATHER_ND` uses the final indices axis as tuple components after any shared batch axes, and
`ONE_HOT` appends its positive depth. CPU capability and lowering recheck those exact signatures,
types, Shapes, static extents, resolved layouts, and output injectivity; they do not broaden or
reinterpret the public contracts.

For example, gather data Shape `[2, 3]` at axis `1` with indices Shape `[2]` and logical values
`[2, 0]`. The result Shape is `[2, 2]`. Each data row selects its third value and then its first:

```text
data:    [[10, 11, 12], [20, 21, 22]]
indices: [2, 0]
result:  [[12, 10],     [22, 20]]
```

The same indices vector is logically visited once during validation, even though both data rows
reuse it during output generation. This example demonstrates axis-gather coordinate mapping and
strict bounds. It does not imply embedding-specific behavior, negative-index normalization, or a
general multi-node indexing route.

Execution deliberately separates the value-dependent failure domain from output writes:

```text
direct bound INT32/INT64 index carrier
  -> scalar row-major validation of the complete logical index domain
  -> only after success, generated scalar or parallel-scalar output ranges
```

The invoking thread selects the first invalid scalar deterministically. Negative and upper-bound
indices throw `IndexOutOfBoundsException`; they never wrap, clamp, select a default, leave an
all-false one-hot row, or expose partially written CPU output bytes. Validation uses the exact
logical indices order rather than physical address order, so an offset, positive stride, or
read-zero stride does not change which failure is first. Generated writers may reload already
validated indices but contain no bounds branch. Worker submission begins only after the complete
validation pass succeeds.

Validation and output have independent zero domains. An empty index domain succeeds without a
load. A zero output caused by an unrelated data suffix still validates every supplied index. A
zero selected data extent rejects every encountered value because its valid interval is empty.
After successful validation, zero output invokes no generated entry and submits no worker work.
All cold carrier, span, alignment, accessibility, output-writability, canonical gathered-BOOL,
injectivity, and output/input non-overlap checks still run.

CPU analysis declares each distinct gather input `ValueId` once in semantic first-use order and
then one separate output; one-hot declares indices and output. Every indexing plan has one unit,
no materialization, no workspace, one current schema-42 generated class artifact, one prepared
executable, and one bound invocation. The generated class embeds carrier-, type-, family-, and
access-specialized output loops rather than delegating through a generic carrier bridge. Proved
dense heap arrays use integer loop/address state; segment, mixed-carrier, and general-layout forms
retain typed long-address traversal. The artifact owns only output writing. Compact CPU-private
geometry retains layout and coordinate facts without a per-index or per-output table, while the
bound executable owns run-value validation. Shared Prepare assigns declared buffers opaquely,
and Runtime sees only the prepared executable and direct carriers. Schema 24 introduced these
embedded indexing bodies. Schema 37 additionally admits guarded primitive cursor bodies for the
frozen mixed-carrier FLOAT64 GATHER row-copy and FLOAT32 GATHER_ND tuple/suffix geometries. The
full-range GATHER_ND body reads each tuple pair once and emits its proved 16-element suffix as a
generation-time straight-line primitive load/store sequence; partial subranges keep a bounded
suffix loop, and unproved geometry enters the existing typed general-long writer. This fixed body
is part of the one artifact and does not consume an additional fixed-shape or unrolled
specialization variant. Schema-36 and earlier artifacts are incompatible misses, with no migration
reader.

The frozen five-fork generated/direct ratios were `0.847429053x` through `0.874533105x` for
GATHER, with median `0.853338331x`, and `0.733345077x` through `0.772532795x` for GATHER_ND, with
median `0.745954874x`. The vector-segment, integral-mixed, and ARGSORT controls passed every fork
and median at `<= 1.15x`. This evidence closes only the two fixed indexing rows on the recorded
environment; five deferred diagnostic rows remain.

Current indexing support ends at these four one-node, fully static operations. Functional
scatter, fold accumulation, ordering/top-K, and explicit-state random work have separate current
families; dynamic
Shape/layout binding, vector gather, native routes, tuning, and general partition-DAG fusion
remain planned. The independent scalar reference is differential-test evidence, not a Runtime
fallback or a second artifact.

### Current functional slice update

The portable movement family accepts exactly one fully static, resolved-layout
`SliceKind.SLICE_UPDATE` occurrence. Ordered inputs are `[base, update]`; the output has the exact
base Shape and data type. All three descriptors have resolved layouts, the output layout is
injective, and cold binding rejects every output/input physical overlap. Input/input overlap is
allowed, including the deduplicated case where both occurrences are the same graph value and the
occurrence map is `[0, 0]`.

The semantic effect is copy-base-then-replace selected positions, although the generated body
realizes that effect in one output-domain pass: each result coordinate selects either the
corresponding base value or the unique mapped update value and writes once. Both inputs remain
unchanged. `SliceAttrs` supplies distinct normalized axes and finite coordinate sequences:

```text
baseCoordinate = start + updateCoordinate * step
0 <= updateCoordinate < length
```

Positive, negative, and non-unit steps are supported. A length-one sequence may legally retain
`Long.MIN_VALUE` because no second coordinate or absolute step is needed. Unselected axes use
start zero, the full base extent, and step one. `CropToShapeAttrs` instead uses every prefix extent
as a start, every update extent as a length, and step one. For example:

```text
base:    [10, 11, 12, 13, 14]
update:  [90, 80]
start:   4
length:  2
step:   -2
result:  [10, 11, 80, 13, 90]
```

The implementation copies represented bits for FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and
canonical BOOL through heap arrays, native-order `MemorySegment` carriers, or compatible mixed
patterns. It covers scalar and empty results or update regions. Arbitrary half-open result ranges,
including parallel chunks, cold-seed their own coordinate and signed-sequence cursors, so chunks
remain independent and deterministic. The hot loop performs no per-element allocation, division,
modulo, Model interpretation, or semantic dispatch.

CPU analysis declares each unique input once followed by one output, one execution unit, no
workspace or materialization, and one generated artifact. Schema 15 records the movement family,
output rank, occurrence map, access/type/carrier structure, and generated body compatibility;
exact starts, lengths, steps, extents, offsets, and stride magnitudes remain cold facts. Older
schemas are incompatible misses and have no migration reader.

This row remains functional slice replacement only. Index-valued functional scatter and
zero-initialized overlap fold have separate current CPU routes below.

### Current functional scatter family

The portable scatter family accepts exactly one fully static, resolved-layout current Model
occurrence of `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, or `SCATTER_ND`. Each consumes
ordered logical inputs `[data, indices, updates]` and produces a fresh, distinct result with the
exact data Shape and type. CPU never mutates an input. `SCATTER_ELEMENTS` uses same-rank aligned
indices and updates, `SCATTER_ADD` uses ordinary Gather's result Shape for updates and has
intrinsic `ADD`, and `SCATTER_ND` uses the final indices extent as tuple depth after its shared
batch Dimensions. Historical `SCATTER_AXIS_ADD` and the superseded reduced-rank Scatter Add Shape
are not current operations.

Indices are exactly INT32 or INT64. Replacement reduction `NONE` accepts FLOAT64, FLOAT32,
BFLOAT16, INT32, INT64, and canonical BOOL; `ADD`, `MUL`, `MIN`, and `MAX` accept the five numeric
types, while Gather-compatible `SCATTER_ADD` is fixed to numeric addition. Every untouched result
coordinate copies the exact base representation. `NONE` replaces an addressed coordinate with
its unique update. Other reductions include the base exactly once and every addressed update
exactly once, including duplicate contributions. Integral addition and multiplication are
fixed-width modular. Floating addition follows deterministic CPU row-major contribution order,
which is not a stronger Model guarantee. Floating extrema propagate NaN and apply the current
signed-zero rule. Floating multiplication computes the exact abstract unchanged-format product
and rounds once; it does not promise a NaN payload, source, signaling state, or sign.

Cold binding validates the complete logical index domain in deterministic row-major scalar order
before any output write or worker submission. Negative and out-of-range indices fail; CPU does not
normalize, wrap, clamp, ignore, or substitute a default. For `SCATTER_ELEMENTS + NONE` and
`SCATTER_ND + NONE`, a second complete pass rejects the first later update or tuple that repeats
an earlier complete target. Bounds always precede duplicate checking, including zero-output
cases, and any validation failure leaves the output bytes unchanged. `SCATTER_ADD` accumulates
duplicates and has no uniqueness pass.

The generated class embeds the selected scatter family, represented type, reduction, carrier
accesses, target mapping, and output store. For `NONE`, `ADD`, `MIN`, `MAX`, and integral `MUL`,
each invocation first copies the base values in its owned half-open output range, then traverses
the complete logical update-scalar domain once in row-major order. It derives each target's logical
output ordinal and applies the update only when that ordinal belongs to the range. A range of
length `R` with `U` update scalars therefore performs `O(R + U)` generated work; `W` disjoint
worker ranges perform `O(outputCount + W * U)` total work. Each worker still writes only its own
output coordinates, so duplicate reductions retain canonical update order without atomics, locks,
cross-range partials, or a merge step.

Cold-proved dense rank-one heap-array forms narrow universal bounds and bases once and retain
integer loop and address state. Segment, mixed-carrier, arbitrary-layout, large-range, and
otherwise unproved forms retain typed general long-address traversal. Both forms use the same
copy-then-update ownership rule. Neither contains a generic `Object` execution bridge or repeats
bounds and uniqueness validation inside generated work. A raw generated replacement entry is
deterministically last-assignment-wins if a caller bypasses the required uniqueness validator;
that is a Class-File test invariant, not supported public behavior. Resolved inputs may use
non-negative offsets and strides, including broadcast-zero strides; output layout must be writable
and injective. Heap carriers are
`double[]`, `float[]`, raw BFLOAT16 `short[]`, `int[]`, `long[]`, and canonical BOOL `byte[]`;
native-order `MemorySegment` and compatible mixed patterns are also supported. Exact repeated
input values deduplicate to one boundary, and input/input overlap is allowed. Output overlap with
any input is rejected.

Floating `MUL` is the deliberate algorithmic exception. Copy-then-update would either round an
intermediate product or require exact state for every interleaved target, but the established
resource contract provides only one exact accumulator slice per range. A floating-product range
therefore remains output-owned: for each owned output coordinate it scans the complete update
domain, groups that target's factors, finishes one once-rounded product, and reuses the same slice.
This is approximately `O(R * U)` work and is retained as a correctness/resource boundary, not
presented as whole-family linear complexity.

Only a floating `MUL` plan with non-empty output and contribution domains declares workspace. One
run-owned `CpuContiguousWorkspace` contains a checked, eight-byte-aligned fixed-capacity primitive-
limb scratch slice per selected range; slices are disjoint, reset and reused. The FLOAT64,
FLOAT32, or BFLOAT16 generated entry embeds its type-specialized factor classification,
sign/exponent/significand state, primitive-limb multiplication, and final ties-to-even rounding.
It uses no `BigInteger`, generic scatter helper, or per-factor/per-output allocation. Every other
scatter row declares no workspace, and scatter never selects input materialization. Schema 16
introduced scatter family, reduction, structural access/type/carrier facts, semantic occurrence
mapping, and the optional scratch entry. Schema 25 added the embedded
family/type/reduction/carrier/access body. Schema 31 adds the range-owned copy-then-update
loop shape for scratch-free rows while retaining grouped floating-product bytecode and the same
scratch signature. Schema 41 adds one completely guarded direct body for the frozen INT64
`SCATTER_ND + MIN` geometry with `[16384,16]` data/result, `[4096,1]` indices, and `[4096,16]`
updates. After complete cold guards, it copies the owned output interval linearly, loads each
tuple index once, visits its 16-element suffix, filters targets to the owned interval, and applies
signed `long` minimum in encounter order. It preserves duplicate tuples, arbitrary legal
half-open ranges, zero workspace, and the typed general-long fallback. Schema-40 and every earlier
envelope are incompatible misses with no migration reader. Concrete axes, batch/tuple values,
extents, layout magnitudes, ranges, workspace sizes and offsets, resource identities, and
validation results remain cold compatible facts.

The CPU 0007A1B local parity probe used five isolated JVM forks and six fixed adverse-ratio cases.
The final generated/direct median-of-fork-medians ratios were `1.034` for dense Elements `NONE`,
`1.022` for dense Elements `ADD`, `1.017` for dense Gather-compatible Scatter Add, `1.012` for
dense Scatter-ND `NONE`, `0.563` for a general mixed-carrier Elements `ADD`, and `0.850` for the
grouped exact floating product. All 30 individual ratios and all six aggregates were at most
`1.15`. Compared with preserved schema-30 class bytes, the five scratch-free generated medians
improved by `1795.9x`, `1966.0x`, `1925.0x`, `1388.1x`, and `6652.7x`; the unchanged-resource
exact-product split improved by `2.09x` and was exempt from the scratch-free `4x` gate. These are
writer-only results for the recorded Java 26/macOS arm64 cases and protocol. They exclude cold
validation, generation, class loading, preparation, binding, and allocation, and they are neither
a universal speed claim nor a production selection input.

This coverage changes only the CPU-private portable route. It does not change Model semantics,
Compiler capture or gradients, shared Prepare or Runtime contracts, public Tensor/API behavior,
native or vendor routes, Vector API scatter execution, dynamic or symbolic layout handling,
multi-node fusion, backend conformance, Engine composition, universal backend
support, or performance guarantees.

The fixed CPU 0007A1M evidence covers only that guarded INT64 case on the recorded OpenJDK 26.0.1
macOS arm64 environment. Its five generated/direct ratios are `0.984900063x`, `0.988888234x`,
`0.983823803x`, `0.978065816x`, and `0.992400680x`, with median `0.984900063x`. These values are
acceptance evidence for the generated shape, not a universal scatter-performance claim or a
production tuning input. The broader frozen comparison still has one unresolved row,
`X-MIN-MULTI`.

### Current overlap fold family

The portable fold family accepts exactly one fully static, resolved-layout `FOLD_AXIS` or
`FOLD2D` occurrence with one read-only input and one distinct writable output. It creates a fresh
result rather than mutating or aliasing the input. Every output coordinate begins at represented
positive zero, including coordinates that receive no contribution.

`FOLD_AXIS` removes the input's final window-size dimension and restores one explicit target
axis. If window position is `w`, final-dimension offset is `k`, and step is `s`, that input
position contributes at target coordinate `w * s + k`. Gaps and the trailing remainder left by
floor-counted windows remain positive zero. FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 are
supported.

`FOLD2D` accepts canonical columns `[N, C * KH * KW, OH * OW]` and writes explicit NCHW output
`[N, C, H, W]`. For each logical column position, CPU derives its batch, channel, kernel, and
window coordinates. It adds the value only when the corresponding unpadded height and width are
inside `[0, H)` and `[0, W)`. Leading or trailing symmetric-padding positions and terminal
ceil-grid positions outside the unpadded output are excluded geometrically; there is no padding
scalar to compare or add. This family supports FLOAT64, FLOAT32, and BFLOAT16 only.

For each output coordinate, scalar generated execution visits contributing logical input
positions in canonical flattened row-major order. FLOAT64 and FLOAT32 perform sequential
same-format addition. BFLOAT16 expands the represented accumulator and operand to binary32, adds,
and rounds back to BFLOAT16 after every contribution. INT32 and INT64 `FOLD_AXIS` use fixed-width
two's-complement modular addition. This order is the current CPU realization and gives bitwise
parity between scalar and parallel-scalar execution; it is not a cross-backend Model promise.

Input and output layouts may use non-negative offsets and strides, including repeated logical
input reads through zero strides. Heap arrays, native-order `MemorySegment`, and compatible mixed
carrier patterns are supported. The output layout must be injective, and cold binding rejects any
physical input/output overlap before a generated call or worker submission. A range owns a
disjoint half-open interval of flattened output coordinates, writes each coordinate once, and
shares no mutable state with another range.

Analysis declares exactly the input and output buffers, zero workspaces, zero materializations,
one computation unit, and one artifact. Scalar preference uses one direct scalar call; eligible
parallel orchestration calls the same scalar body over disjoint output ranges. Fold has no Vector
API body, atomics, partial sums, cross-range merge, hidden scratch, or input-domain parallelism.
CPU analysis and finalization keep concrete axes, windows, extents, layouts, carriers, and ranges
cold. Schema 17 introduced the fold family, represented type, boundary access/rank structure,
carrier pattern, execution mode, and canonical sequential-addition policy. Schema 31
embeds typed mapping, carrier access, and sequential addition; every older schema is an
incompatible miss with no migration reader.

Schema 35 adds one guarded form for the frozen A1G mixed-carrier FLOAT32 FOLD2D geometry: input
columns `[1, 9, 512]`, NCHW output `[1, 1, 16, 32]`, a `3 x 3` kernel, unit stride, symmetric
padding `2`, and dilation `2`. After complete cold proof, each output cell starts at `+0.0f`, then
visits kernel position `q` and column in canonical occurrence order, performs one sequential
FLOAT32 addition for each match, and stores once. The singleton batch and channel machinery is
removed only by that proof. The same entry accepts arbitrary legal half-open output subranges;
unproved geometry retains the typed general-long body.

The retained five-fork `F-FOLD2D` generated/direct ratios have median `0.232217638x` and maximum
`0.234704046x`. The low ratio reflects cold-proved constants and removed singleton/general
coordinate machinery around the same frozen direct comparator algorithm; it is not a different
semantic algorithm or evidence of universal fold parity.

This is exact current CPU route coverage, not broader Model, Compiler, Runtime, Engine, gradient,
native, fusion, dynamic-layout, reduction-framework, backend-conformance, cross-backend bitwise,
or performance support.

### Current stable ordering and selection family

The portable ordering family accepts exactly one fully static, resolved-layout `SORT`, `ARGSORT`,
or `TOP_K` occurrence. The input may be FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, or canonical
BOOL. SORT has ordered boundaries `[input, values]`; ARGSORT has `[input, indices]`; TOP_K has
`[input, values, indices]`, with values at output slot zero and INT64 indices at slot one. Every
output is distinct, writable, injective, and has the exact Model Shape. CPU creates no hidden
second SORT output and does not decompose TOP_K into separate executable occurrences.

Each logical slice varies along the normalized selected axis while all other coordinates remain
fixed. Floating non-NaNs use ordinary numerical order; ascending places negative zero before
positive zero and descending reverses that order. Every NaN remains after every non-NaN in either
direction. Equal numeric, integral, BOOL, and NaN keys retain increasing original axis index.
INT32 and INT64 use signed order, while BOOL uses `false < true` before direction reversal.
ARGSORT and TOP_K indices are zero-based logical-axis coordinates, not flattened offsets or
physical addresses.

TOP_K first selects the first `k` value/index pairs from that complete stable order. With
`sorted == true`, outputs retain selection order. With `sorted == false`, CPU keeps the same
selected set and orders it by increasing original axis index. SORT and ARGSORT preserve the axis
extent; TOP_K replaces it with `k`, where `0 <= k <= extent`. An empty selected SORT/ARGSORT axis,
an empty unselected dimension, or `k == 0` produces no generated call or worker submission after
cold validation. Scalar inputs remain unsupported because they have no axis.

Lowering records family, represented type, direction/output-order flags, structural boundary
accesses, output count, and the fixed two-index-region scratch policy in route-independent
ordering IR. Axis, `k`, concrete extents, offsets, stride magnitudes, and ranges remain cold.
Analysis declares one exact eight-byte-aligned run-owned workspace. Its checked size is twice the
input axis extent in INT64 elements for every selected range; each range receives a disjoint
region and never splits a slice. Scalar and parallel-scalar use the same ordering body and produce
bitwise-identical represented value outputs. Vector sorting, parallel merge, radix/network
sorting, approximate selection, and runtime algorithm choice are not implemented.

Cold binding validates carrier kind, byte size, alignment, accessibility, output writability,
canonical input BOOL bytes, complete buffer spans, and workspace geometry. It rejects every
input/output overlap and TOP_K output/output overlap before any scratch mutation, output write,
generated call construction, or worker submission. Heap arrays, native-order `MemorySegment`,
and compatible mixed carriers are supported; input reads may use dense, offset, positive-strided,
or zero-strided layouts, and outputs may use injective non-dense layouts. Values outputs copy the
selected raw FLOAT64/FLOAT32 NaN and signed-zero bits, raw BFLOAT16 payloads, integral patterns,
or canonical BOOL bytes without conversion.

Schema 18 records the ordering family, represented type, direction/output-order flags, ordered
boundary roles and carrier forms, output count, and scratch-bearing entry shape. It excludes
concrete axis, `k`, layout magnitudes, assigned slot/workspace identity, and slice ranges. One
TOP_K artifact and one bound invocation perform both stores. The scalar reference uses an
independent primitive-index insertion implementation for differential evidence; it is not a
Runtime fallback.

Schema 27 introduced the complete carrier-, represented-type-, family-, direction-, output-,
and access-specialized ordering body, which current schema 42 retains. It uses a stable bottom-up
merge over the two assigned
INT64 scratch regions, selecting the left logical index on equality. Dense heap-array forms use
cold-proved integer loop and address state. Arbitrary supported layouts and heap, segment, or
mixed carrier forms use typed long-address access; values and indices are stored without a
generic `Object` execution bridge or runtime family dispatch. Schema 27 and every earlier
envelope are incompatible misses and are regenerated under schema 37; no migration reader is
provided.

The retained fixed five-fork evidence compares dense FLOAT32 schema-27 generated entries with
equivalent direct primitive stable-merge bodies. Stable ascending SORT ratios were
`1.092722590x`, `1.077727976x`, `1.090165540x`, `1.049603676x`, and `1.089690637x`, with aggregate
`1.090165540x`. Sorted-largest two-output TOP_K ratios were `0.962180009x`, `0.969660236x`,
`0.936075945x`, `1.018861174x`, and `1.020688132x`, with aggregate `0.965951020x`. Every fork and
aggregate passed the task-local `<= 1.15x` gate. These observations cover only those fixed local
cases and do not claim machine-code shape, JIT behavior, other layouts or types, native routing,
or general performance parity.

This is CPU realization of the existing Model ordering meaning, not a change to Tensor semantics.
It adds no dynamic Shape or layout binding, custom/unstable ordering, native route, Vector API
sort, fusion, autotuning, public configuration, Engine integration, cross-backend bitwise promise,
or performance guarantee.

### Current explicit-state RNG and dropout family

The portable random family accepts exactly one fully static, resolved-layout occurrence. An
`INITIAL_STATE` occurrence has no input and one writable INT64 `Shape[2]` output. It writes the
exact raw key word to logical lane zero and counter word to lane one; it does not hash either
word, draw randomness, or advance the counter. A `DROPOUT` occurrence has ordered boundaries
`[value, state, output, keepMask, nextState]`. Value and output must be FLOAT64 or FLOAT32 with
the same Shape, the mask is canonical BOOL with that Shape, and both state values are INT64
`Shape[2]`. Every writable layout and the state input must be injective. BFLOAT16 dropout fails
closed: its raw `short` carrier does not establish the direct, correctly rounded scaling and
conversion rule required for numerical execution.

CPU realizes those Model semantics with the non-cryptographic, CPU-private configuration
`SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1`. All additions and multiplications below are modulo `2^64`;
`>>>` is an unsigned shift:

```text
KEY_BIAS = 0x9e3779b97f4a7c15
MIX_MULTIPLIER_1 = 0xbf58476d1ce4e5b9
MIX_MULTIPLIER_2 = 0x94d049bb133111eb

mix64(z):
  z = (z ^ (z >>> 30)) * MIX_MULTIPLIER_1
  z = (z ^ (z >>> 27)) * MIX_MULTIPLIER_2
  return z ^ (z >>> 31)

keyOffset(key) = mix64(key + KEY_BIAS)
word(key, counter, i) = mix64(counter + i + keyOffset(key))
uniform53(word) = (word >>> 11) * 0x1.0p-53
```

Here `i` is the zero-based row-major logical ordinal, independent of physical layout, carrier,
worker range, or scheduling. Each logical element consumes exactly one word. The element is kept
when `uniform53(word) >= probability`; equality therefore keeps. The comparison uses binary64
without first narrowing the draw or probability. Mask storage is byte `1` for kept and byte `0`
for dropped. A dropped value is represented positive zero. For a kept value, CPU computes
`denominator = 1.0d - probability` once per invocation. FLOAT64 performs one binary64 division.
FLOAT32 widens its exact input to binary64, performs the same division, then narrows once. This
fixed order forbids reciprocal multiplication and fixes parity for this CPU configuration.

The independent compatibility vectors are:

| Key | Counter | Word | Top 53 bits | `Double.toHexString(uniform53)` |
|---|---|---|---|---|
| `0000000000000000` | `0000000000000000` | `48218226ff3cd4bf` | `09043044dfe79a` | `0x1.2086089bfcf34p-2` |
| `0000000000000000` | `0000000000000001` | `ea8568d2e45fd6cb` | `1d50ad1a5c8bfa` | `0x1.d50ad1a5c8bfap-1` |
| `0000000000000001` | `0000000000000000` | `dce423fc82c0d5b8` | `1b9c847f90581a` | `0x1.b9c847f90581ap-1` |
| `ffffffffffffffff` | `ffffffffffffffff` | `e8ba9f99ca933538` | `1d1753f3395266` | `0x1.d1753f3395266p-1` |
| `0000000000001234` | `0000000000000007` | `3e4cf5a0c9489779` | `07c99eb4192912` | `0x1.f267ad064a448p-3` |

One generated `[0,0)` prologue writes state before element work. For dropout it retains the key
and writes `counter + N` modulo `2^64`, where `N` is the checked logical element count. A scalar
Shape consumes one draw. A zero extent consumes none, still writes unchanged next state, and
submits no worker work. Non-empty scalar and parallel-scalar ranges derive every word directly
from the global ordinal, so scheduling and chunk boundaries cannot alter output, mask, or state.
The prepared executable owns no mutable generator; separate runs obtain their ordinary isolated
state through their explicit input and distinct `RunState` resources.

Generation specializes family, represented type, probability or initializer words, each ordered
carrier role, and dense-versus-general access before invocation. Schema-28 dense heap-array
entries contain no member references. Segment entries refer only to the predefined native-order
unaligned primitive layouts and the matching typed `MemorySegment.get` or `set` methods required
by their roles. Generated work contains no generic `Object` bridge, carrier/type dispatch,
allocation, random service, route lookup, cache lookup, or Runtime interpretation. The universal
primitive `long start, long end` signature and `[0,0)` prologue sentinel remain unchanged.

Concrete state words, extents, element count, offsets, stride magnitudes, carrier instances,
assigned slots, addresses, worker ranges, and run identity remain invocation or lifecycle facts.
They do not enter class identity. Compatible geometries and state values therefore reuse the same
structural class, while probability bits, initializer bits, carrier pattern, represented type,
and emitted access form remain compatibility facts because they can change the descriptor or
class bytes.

Analysis declares one initializer boundary or five dropout boundaries, one computation unit, one
generated artifact, and zero workspaces, materializations, random-word buffers, replay buffers,
or per-thread generators. Cold binding validates carrier type, byte size, alignment, access,
complete spans, and writable layouts. Before the prologue or any worker submission, it rejects
each of the three output spans against both input spans and rejects all three output/output pairs.
Input/input overlap is allowed because both inputs remain read-only. A later generated or worker
failure may leave ordinary outputs partially written under the existing execution contract, but
there is no hidden random state to corrupt.

Schema 19 introduced the generator name, constants and mapping, uniform and threshold rules,
finite-precision scaling order, canonical mask and prologue policies, raw initializer or
probability bits, ordered boundary roles and carriers, and zero-scratch shape. Concrete layouts,
slots, carriers, workers, and ranges remain cold when they do not change emitted bytes. There is
no migration reader for old artifacts. Schema 28 is the first version whose random entries embed
the direct typed initializer and FLOAT64/FLOAT32 dropout bodies. A schema-27 envelope is an
incompatible safe miss: cold finalization regenerates and may publish current schema-42 bytes
after shared slot assignment rather than loading, converting, or aliasing the old bridge class.

Retained observational evidence for the fixed dense heap-array shape `[64,16384]`, probability
`0.25`, key `0x0123456789abcdef`, and counter `0xfedcba9876543210` used five fresh fixed-heap JVM
forks on OpenJDK 26.0.1+8-34, macOS 26.5.2 arm64. The FLOAT64 generated/direct fork ratios were
`0.242432568x`, `0.240238314x`, `0.245355311x`, `0.242999122x`, and `0.241854555x`, with aggregate
`0.241451022x`. The FLOAT32 ratios were `1.046430134x`, `1.063729192x`, `1.052403558x`,
`1.060950642x`, and `1.053413773x`, with aggregate `1.058174348x`. Every fixed-case fork and both
aggregates passed the task-local `<= 1.15x` gate. These results apply only to those exact cases,
host, JVM, direct bodies, and protocol; they are not a general speedup, other-layout, other-state,
parallel, native-route, or hardware performance claim and do not select production behavior.

Schema 35 also adds one guarded form for the frozen A1G rank-one `1 << 20` FLOAT32 mixed-carrier
dropout geometry. Complete cold checks permit primitive integer logical and physical cursors while
preserving the V1 key/counter/mix/uniform calculation, the binary64 `>= 0.25` threshold, canonical
BOOL mask, FLOAT32 widen/divide-by-`0.75`/narrow order, and the dedicated `[0,0)` next-state
prologue. It accepts arbitrary legal half-open subranges; unproved geometry retains the typed
general-long body. Its five-fork generated/direct ratios have median `1.052637855x` and maximum
`1.056324388x`. This evidence covers only the retained shape and protocol.

The same five A1G processes retained passing vector-segment, integral-mixed, and ARGSORT controls.
Each process still exited nonzero because nine or ten task-deferred diagnostic rows exceeded the
gate. A1G therefore closes only its two targets; it is not a 20-row performance pass and does not
establish that every generated kernel is optimal.

This configuration defines bounded replay only for the same CPU V1 implementation and numerical
order. It is not a public Model configuration, serialized RNG format, cross-backend or
cross-version bitstream, distinct-key disjointness guarantee, statistical certification, or
cryptographic generator. Model owns the explicit key/counter meaning, one abstract draw per
logical element, keep rule, and modulo advancement; CPU owns this private mapping and realization.

### Current cumulative scan family

The portable scan family accepts exactly one fully static, resolved-layout `CUM_SUM` or
`CUM_PROD` occurrence with one read-only input and one distinct writable output. Both descriptors
have the same non-scalar Shape and data type, the normalized axis is valid for that rank, and the
output layout is injective. Input reads may use dense, offset, positive-strided, interleaved,
transposed, or zero-strided layouts. Output writes may use any supported injective resolved
layout. Heap primitive arrays, native-order `MemorySegment` carriers, and compatible mixed input/
output patterns are supported.

The exact current type and execution matrix is:

| Type | `CUM_SUM` | `CUM_PROD` | Accumulator rule | Compute modes |
|---|---|---|---|---|
| FLOAT64 | yes | yes | One binary64 operation per visited value; retain the binary64 result | Scalar, parallel-scalar |
| FLOAT32 | yes | yes | One binary32 operation per visited value; retain the binary32 result | Scalar, parallel-scalar |
| BFLOAT16 | yes | yes | Widen represented operands to FLOAT32, operate once, then round back to BFLOAT16 after every value | Scalar, parallel-scalar |
| INT32 | yes | yes | Two's-complement 32-bit modular operation after every value | Scalar, parallel-scalar |
| INT64 | yes | yes | Two's-complement 64-bit modular operation after every value | Scalar, parallel-scalar |
| BOOL | no | no | Unsupported | None |

A logical scan slice fixes every coordinate except the selected axis. Slice ordinals follow
row-major order over those non-axis coordinates and do not depend on physical layout. Forward
mode visits axis coordinates from zero upward; reverse mode visits them from the last coordinate
downward while storing at their original coordinates. Inclusive mode incorporates the current
input before writing. Exclusive mode writes the current accumulator first. The represented
identities are positive zero for sum and positive one for product.

This concrete example uses one INT32 slice with input `[1, 2, 3]` and axis `0`:

| Mode | `CUM_SUM` result |
|---|---|
| Inclusive, forward | `[1, 3, 6]` |
| Exclusive, forward | `[0, 1, 3]` |
| Inclusive, reverse | `[6, 5, 3]` |
| Exclusive, reverse | `[5, 3, 0]` |

The example demonstrates placement and traversal order. It does not imply in-place execution,
cross-backend bitwise identity, a particular NaN payload, or a vector prefix algorithm.

Scalar execution covers every independent slice directly. Parallel-scalar execution partitions
only the slice-ordinal domain into deterministic disjoint ranges; it never divides one slice
between workers. Both modes therefore use the same sequential order for every output and are
bitwise identical for this CPU realization. When there is only one non-empty slice, execution is
scalar even if its selected axis is long. If the selected axis or any non-axis extent is zero,
the logical output is empty and no generated call or worker work occurs after cold validation.

Analysis declares exactly `[input, output]`, one computation unit, one generated artifact, zero
workspace, zero materialization, and no partial/carry/scratch buffer. Cold binding validates
type/carrier compatibility, complete byte sizes, alignment, access, output writability, and the
complete physical spans. Any input/output overlap fails before output mutation, generated-call
creation, or worker submission. The immutable prepared recipe owns no mutable accumulator or
shared state; each invocation range receives private packed coordinate state.

The generated two-boundary entry embeds a typed scan body selected from structural scan facts.
Dense rank-one heap-array scans use one-time integer base narrowing and a direct typed loop.
Schema 38 adds one completely guarded body for the fixed `[1024,1024]`, axis-one, exclusive
reverse INT64 `CUM_PROD` geometry when input and output are both `MemorySegment` carriers with
row stride 2,048 and axis stride 2. Invocation-specific carrier bases retain arbitrary legal
offsets, and the body accepts every legal half-open complete-slice range. It initializes one
input cursor and one output cursor at the last axis coordinate of each slice, stores the current
accumulator before loading the input, applies one same-width modular long multiplication, and
decrements both cursors by two elements. The proved loop calls direct
`JAVA_LONG_UNALIGNED` segment `get`/`set`; it performs no coordinate division or remainder,
packed-layout lookup, layout construction, generic dispatch, or helper call. The same generated
method retains the typed general-long body as its fallback whenever any kind, type, rank, axis,
mode, extent, stride, carrier, or range guard fails. That fallback reconstructs non-axis
coordinates once per slice and contains the method's remaining division/remainder pair. The
entry contains no generic `Object` carrier bridge or runtime data-type/kind dispatcher.

Schema 20 records scan kind, represented type, normalized axis role, inclusive/exclusive and
forward/reverse modes, ordered boundary roles and carrier forms, structural accesses, sequential
typed-rounding policy, scalar compute shape, and absence of scratch. Schema 22 added the embedded
typed body and dense heap-array loop-shape compatibility. Schema 30 additionally emits
BFLOAT16 conversion, arithmetic, and round-to-nearest-ties-to-even steps directly in the generated
body. It widens each represented operand to FLOAT32, performs one typed add or multiply, and rounds
back after every visited value without calling a Synaptik-owned runtime member.
Concrete extents, offsets, stride magnitudes, assigned slots, carrier objects,
addresses, worker identity, run identity, and selected range count remain cold when they do not
change emitted bytes. Schema 38 records the guarded scan body described above. Schema-37 and
earlier artifacts are incompatible safe misses under the current-only cache policy, and there is
no migration reader.

On the frozen OpenJDK 26.0.1, macOS 26.5.2 arm64 host and unchanged A1C probe, the accepted five
generated/direct ratios for this one scan case were `0.983678415x`, `0.947724365x`,
`0.992224731x`, `0.966227297x`, and `0.978732901x`, with median `0.978732901x`. This evidence
accepts only the recorded case and code shape; it is not a universal scan-performance claim or a
production tuning input. Two earlier five-fork samples were rejected in full because an unrelated
row exceeded the fixed gate in one fork, so neither rejected sample contributes accepted ratios.

Current coverage ends at this one-node static portable family. Ordinary aggregates and arg
extrema are the separate current families below. Masked, logarithmic, statistical, and norm
reductions remain Draft CPU 0007C–0007D work. Stable
`SOFTMAX`/`LOG_SOFTMAX` and normalization remain Draft CPU 0007E–0007F work. Multi-node scan
fusion, partial scans, cross-worker prefix combination, vector or native scan bodies, dynamic
Shape/layout binding, in-place/overlapping execution, public scan configuration, and a shared
Runtime scan primitive are not implemented.

### Current aggregate reduction family

The portable ordinary-aggregate family accepts exactly one fully static, resolved-layout
`SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, or `ANY` occurrence. `SUM`, `PROD`, `MIN`, and `MAX`
accept FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 and preserve that represented type. `MEAN`
accepts only FLOAT64, FLOAT32, and BFLOAT16. `ALL` and `ANY` accept and produce only canonical
one-byte BOOL. Integral MEAN, masked reductions, arg extrema, and later reduction families fail
closed before route selection.

The accepted attribute forms determine selected axes as follows:

| Attribute form | Selected domain | Output Shape |
|---|---|---|
| Exact parameterless form | Every input axis | Canonical scalar |
| One normalized axis | That axis | Axis removed, or retained with extent one |
| Distinct normalized axis list | Axis membership, independent of list order | Selected axes removed, or retained with extent one |
| Empty normalized axis list | One value at the corresponding input position | Input Shape, regardless of retention |

Binding-aware SUM-to-Shape is a fourth SUM attribute form, not another semantic kind or backend
route. CPU accepts it only when source, target, and output Shapes are fully static, both layouts
are resolved, output Shape equals the exact target, and output type equals input type. The target
is right-aligned with the source. Every unmatched leading source axis reduces and disappears; an
aligned unequal pair is valid only when the target extent is one, in which case that source axis
reduces while the target position remains; equal extents preserve the coordinate. Equality has
precedence, so source-one/target-one is preserved. CPU does not bind unresolved Dimensions.

For source Shape `[2, 3, 4]` and target Shape `[3, 1]`, source axis zero and aligned source axis
two reduce. Each output cell keeps its axis-one coordinate and sums `2 * 4` values. Source Shape
`[1, 3, 1]` to the identical target selects no reduction axis and copies each represented value
to the required distinct output. That copy preserves every floating bit, including NaN payload,
sign, signaling state, and signed zero; it does not classify or round the value as a one-element
SUM.

Each existing output cell owns its complete selected domain. A selected zero extent materializes
the kind's empty-domain result independently for every such cell: positive zero for SUM, positive
one for PROD, canonical positive quiet NaN for floating MEAN, the extrema identity, canonical true
for ALL, or canonical false for ANY. A zero extent on an unselected axis instead makes the output
empty, so there is no cell on which to write a result and no generated call or worker submission.

For a compact example, reduce this INT32 input of Shape `[2, 3]` along axis `1`:

```text
input rows: [[3, -2, 7], [4, 5, -1]]
MIN:        [-2, -1]
MAX:        [ 7,  5]
```

The two output cells are independent. For the first `MIN` cell, the selected-domain walk visits
`3`, then `-2`, then `7`; its result is `-2`. The second cell visits `4`, `5`, then `-1`; its
result is `-1`. Parallel-scalar execution may assign those two complete rows to different ranges,
but it never divides either three-value domain or combines partial results. This example shows
logical traversal and result placement; it does not promise vector reduction or a performance
result.

Floating domains use numerical order, with negative zero selected for `MIN` and positive zero for
`MAX`. Any NaN makes the result NaN. The CPU-local deterministic policy introduced with schema 21
copies the
represented bits of the first NaN in canonical logical input row-major traversal order; this is
not a Model, cross-backend, or future-schema payload promise. BFLOAT16 values widen only for
comparison, and the selected output copies the original 16 represented bits without arithmetic
or rounding.

Floating SUM uses the exact real sum of all finite selected values and rounds that value once to
BFLOAT16, FLOAT32, or FLOAT64 with round-to-nearest, ties-to-even. MEAN divides the exact sum by
the exact positive domain count before that single rounding; it does not first produce a rounded
SUM. PROD retains an exact integer significand and power-of-two exponent for all finite nonzero
factors, then rounds the exact product once with the same rule. "Exact" describes the internal
real-number target, not an arbitrary-precision output: the observable result remains the input
format.

Schema 29 makes special values deterministic for these numerical rows. Any NaN produces the
format's canonical positive quiet NaN. For SUM and nonempty MEAN, opposite-signed infinities also
produce that NaN; one infinity sign produces that infinity. Exact finite overflow rounds to signed
infinity. Empty SUM is positive zero, and a nonempty exact-zero SUM or MEAN is negative zero only
when every selected operand is negative zero. Empty MEAN is canonical NaN. For PROD, NaN wins;
zero together with infinity produces canonical NaN; otherwise infinity or zero uses the XOR of
all factor sign bits, and empty PROD is positive one. Subnormal inputs participate exactly, and
normal, subnormal, overflow, underflow, and underflow-to-signed-zero results all use the one final
ties-to-even conversion.

INT32 and INT64 SUM and PROD use same-width two's-complement modular arithmetic. Overflow wraps
modulo 2^32 or 2^64; no widening, saturation, floating accumulator, or boxed state is involved.

Analysis declares exactly `[input, output]`, one computation unit, and one generated artifact.
Floating SUM, MEAN, and PROD also declare one eight-byte-aligned CPU-private workspace before
shared slot assignment. It contains one disjoint exact-state slice for each simultaneously
executable output-cell range; zero output cells require no workspace. The workspace is allocated
and released with the run, not retained by the immutable prepared executable. Integral numerical,
extrema, and Boolean rows declare no workspace, and no aggregate row selects materialization.
Cold binding accepts the matching heap primitive arrays,
native-order `MemorySegment` carriers, and compatible mixed boundaries for arbitrary supported
non-negative resolved input layouts and injective output layouts. It validates canonical BOOL
input and rejects complete physical input/output overlap. Scratch-bearing rows additionally
validate workspace presence, exact size, alignment, accessibility, and complete physical
non-overlap with input and output. All checks finish before coordinate-pack mutation, output
initialization, a generated call, or worker submission. Empty output performs no generated work.
The same checks apply to SUM-to-Shape, including complete input/output physical-overlap rejection
for its raw-copy form. A selected empty floating domain with existing output cells still owns its
exact-state slices; integral reductions and every raw copy use no workspace.

The generated entry embeds a typed aggregate body selected from structural aggregate facts.
Primitive `start` and `end` values bound a contiguous range of flattened output-cell ordinals.
Full and eligible trailing-axis dense heap-array reductions use direct integer-address loops.
Other supported forms retain a typed long-address body that decodes output and selected-domain
coordinates only where arbitrary rank or strides require it. Floating numerical entries receive
their range's already assigned workspace slice and emit direct primitive-limb operations; they do
not call a numerical helper or runtime operation dispatcher. Every range receives invocation-
private coordinate and accumulator state, while the prepared recipe retains no mutable
accumulator, partial buffer, combine state, or run-shared reduction state.

Schema 21 records kind, represented type, ordinary form, canonical selected-axis membership,
retention, structural boundary access, first-logical-NaN/signed-zero policy, complete-output-cell
range meaning, carrier forms, and zero scratch. Schema 22 added embedded typed-body and dense
heap-array loop-shape compatibility. Schema 29 added numerical kind/type/form identity,
domain cardinality, exact-state limb and slice shape, scratch-bearing entry shape, and direct
SUM/MEAN/PROD bodies. Schema 30 additionally emits FLOAT64, FLOAT32, BFLOAT16, INT32, and
INT64 MIN/MAX selection and canonical BOOL ALL/ANY combination directly. Floating selection keeps
the first logical NaN bits and the established signed-zero choice, BFLOAT16 compares widened values
but copies the selected 16-bit representation, and integral comparison remains signed. Concrete
offsets, stride magnitudes, slots, carrier objects, addresses, workers, run identity, and selected
range count remain cold when they do not change emitted bytes or resource shape. Schema-29 and
earlier artifacts are incompatible safe misses; there is no migration reader or partial reuse.

Schema 34 adds one narrow rank-two canonical-BOOL ANY form with one zero-stride selected axis.
Cold checks prove its geometry before a direct primitive output-cell/domain loop visits every
logical selected position, combines bytes with `IOR`, and canonicalizes once at the store. It does
not short-circuit repeated reads; unproved geometry retains the typed general-long body. The fixed
five-fork generated/direct ratios were `0.307561645x`, `0.306641529x`, `0.304177560x`,
`0.306305081x`, and `0.307136502x`. This is evidence only for the retained shape and environment,
not every ANY reduction.

Schema 36 adds two runtime-guarded numerical forms without changing aggregate resources or
semantics. The frozen FLOAT32 MEAN form reduces axis one of Shape `[128, 2048]`; each output cell
visits exactly 2,048 factors, keeps the exact five-limb state in primitive locals, writes the
identical final state to its assigned 48-byte run-owned workspace slice, and performs the exact
division by `2^11` during extraction and ties-to-even rounding. The frozen BFLOAT16 PROD form
reduces axes `[0, 2]` of Shape `[4, 16384, 4]`; it preserves the exact product-state transitions
and four-by-four logical factor order while using primitive geometry cursors. Both forms accept
arbitrary legal output-cell subranges, and a failed complete geometry/carrier guard enters the
existing typed general-long body.

The five-fork generated/direct ratios for the MEAN form have median `0.973095585x` and maximum
`0.995106956x`; the PROD form has median `1.022472818x` and maximum `1.084500356x`. The same five
processes retained passing vector-segment, integral-mixed, and ARGSORT controls. Every process
remained nonzero only because seven explicitly deferred diagnostic rows exceeded the gate, so
this evidence closes the two fixed numerical rows rather than the complete twenty-row corpus.

Schema 42 adds one completely guarded BFLOAT16 `MIN` form for dense input Shape `[64,64,64]`,
selected axes `[0,2]`, kept output Shape `[1,64,1]`, a `MemorySegment` input, and an offset,
two-strided `short[]` output. Complete carrier, geometry, ordered-range, start-address, and sentinel
guards precede every specialized write. Each owned output-cell ordinal uses the direct primitive
axis-0-then-axis-2 traversal for 4,096 canonical visits, retains the first raw BFLOAT16 NaN,
chooses negative zero when both zero signs occur, otherwise selects the smaller represented value,
and writes its raw 16 bits once. Arbitrary legal complete-output-cell subranges are supported with
no workspace, materialization, partial/combine state, or selected-domain buffer; failed proofs use
the unchanged typed general-long body.

In the retained fixed environment, its five generated/direct ratios are `0.818775794x`,
`0.811182115x`, `0.811942840x`, `0.810720356x`, and `0.810918826x`, with median
`0.811182115x`. All twenty frozen rows and their medians passed five accepted forks at
`<= 1.15x`; one unrelated `M-CONCAT` whole sample at `2.537201532x` was retained and rejected.
This closes that bounded comparison and is neither a universal aggregate-performance claim nor a
production selection input.

Schema 43 adds fully bound SUM-to-Shape identity and generated execution. Its typed general body
uses long-address right-aligned target mapping for every supported heap-array, native-order
`MemorySegment`, or mixed carrier combination, including nonzero offsets, positive or zero input
strides, arbitrary injective output layouts, and arbitrary valid half-open output-cell ranges.
The generated raw-copy body handles the no-reduction form directly. The cold-guarded dense
`[64,128,256]` to `[128,1]` form uses direct primitive nested loops for all five numeric types;
its integral body has the same pre-tested one-accumulator algorithm, hot-loop work, and dataflow
shape as the optimal clean-Java implementation. Floating reductions reuse the exact SUM emitter
and one final round-to-nearest, ties-to-even result conversion; INT32 and INT64 addition remains
same-width modular arithmetic. Failed dense-form guards retain the direct typed general body.

The schema-43 entry partitions only complete output cells, never a selected domain, so scalar and
parallel-scalar execution remain bit-identical and one immutable prepared executable may be reused
with distinct run-owned state. Schema-42 envelopes are current-incompatible safe misses; the A1O
schema-42 ledger remains explicitly historical evidence rather than a current-schema claim.

The local CPU 0007A0 parity probe is evidence for the current generated code shape, not a
hardware-universal speed guarantee or production selector. On Java 26.0.1, macOS 26.5.2,
aarch64, with 128-bit preferred `DoubleVector` species and 1,048,576-element cases, the
median-of-fork-medians generated/direct ratios were 0.819 for scalar FLOAT64 ADD, 0.852 for
preferred-species FLOAT64 Vector ADD, 0.854 for scalar FLOAT64 ADD -> GELU_EXACT -> MUL, 1.001
for dense FLOAT64 full MIN, and 0.998 for inclusive forward FLOAT32 CUM_SUM. The probe used five
fresh JVM forks and kept timing outside ordinary unit tests.

The numerical implementation partitions only output cells. It never divides one reduction domain,
so scalar and every supported worker count visit each cell's inputs in the same logical row-major
order and produce bit-identical results. This is a deterministic CPU result guarantee, not a claim
that floating output arithmetic is mathematically associative or that another backend uses the
same NaN policy.

Current coverage ends at this one-node static family. Arg extrema are the separate current family
below. Masked and advanced reductions, softmax, normalization, multi-node reduction fusion,
within-domain parallelism, partial/combine trees, vector/native reduction bodies, dynamic
Shape/layout binding, in-place overlap, tuning, and broader performance guarantees remain outside
this increment. The task's fixed generated/direct measurements are acceptance evidence for its
selected dense cases, not a route selector or universal speed promise. No claim extends to
dynamic binding, a Vector API reduction body, native execution, or fusion.

### Current arg-extrema family

The portable arg-extrema family accepts exactly one fully static, resolved-layout `ARG_MIN` or
`ARG_MAX` occurrence. It has one numeric input and one distinct writable INT64 output. The input
type is exactly FLOAT64, FLOAT32, BFLOAT16, INT32, or INT64. CPU consumes the Model-normalized
non-negative axis and validates it against the resolved input rank; it does not renormalize the
axis or add a multi-axis form. The selected extent must be positive. A zero extent on an
unselected axis is legal and produces zero output cells, so execution makes no generated call,
submits no worker work, and reads or writes no value.

With `keepDimensions == false`, the selected Dimension is removed. With
`keepDimensions == true`, it remains in the same position with extent one. For an input Shape
`[2, 4]` reduced on axis `1`, those forms produce `[2]` and `[2, 1]`, respectively. Every result
is the zero-based logical coordinate along axis `1`, not a storage offset, flattened index, or
selected value.

This example shows extrema ordering and tie selection for two INT32 rows:

```text
input:             [[3, -1, -1, 2], [5, 7, 7, 4]]
ARG_MIN / FIRST:   [1, 3]
ARG_MIN / LAST:    [2, 3]
ARG_MAX / FIRST:   [0, 1]
ARG_MAX / LAST:    [0, 2]
```

The first row's equal minima occupy logical coordinates 1 and 2; FIRST chooses 1 and LAST chooses
2. The second row's equal maxima occupy coordinates 1 and 2. Physical offsets and strides do not
change those answers. For floating input, every represented NaN is preferred to every non-NaN in
both extrema directions, and multiple NaNs tie. Negative zero orders below positive zero, so
`ARG_MIN([-0.0, +0.0])` returns `0` and `ARG_MAX([-0.0, +0.0])` returns `1`. Infinities follow
ordinary numerical order. BFLOAT16 widens only its represented raw 16-bit value for comparison.

Input reads support arbitrary legal resolved non-negative layouts, including nonzero offsets,
positive strides, and zero-stride repeated reads. Output writes support arbitrary legal injective
layouts. Cold binding accepts matching primitive arrays, `MemorySegment` carriers, and mixed
input/output patterns; it validates accessibility, alignment, complete referenced subranges,
carrier compatibility, output injectivity, and complete physical input/output non-overlap before
any write or worker submission.

Scalar execution owns the complete output-cell range. Parallel-scalar execution divides only that
range into deterministic disjoint half-open chunks; each worker traverses the whole selected axis
for every cell it owns. No selected domain is split, and there is no partial result, combine step,
hidden workspace, or representation materialization. Packed coordinate state is invocation-
private, while the immutable prepared recipe borrows but never closes the configured worker group.

Schema 44 adds the separate arg-extrema intermediate representation, structural kind/type/axis/
Shape-form/tie/carrier identity, mixed numeric-input/INT64-output descriptor, and direct typed
generated body. The body decodes each output coordinate once, walks the selected axis in
increasing logical order, and stores one INT64 result per output cell. A structurally proved
unit-stride form and a guarded stride-two form remove repeated cold stride loads; every other legal
stride uses the direct typed arbitrary-stride fallback. The generated loop has the same algorithm,
traversal, comparison branches, address work, and store count as its optimal straightforward Java
oracle. It calls no Synaptik helper and performs no allocation, boxing, reflection, generic
dispatch, or Runtime semantic lookup.

The retained six-case performance matrix passed every ratio in five isolated fixed-heap forks at
`<= 1.15x`; the largest observed fork ratio was `1.038305072x`. This evidence covers only the
recorded classes, host, JVM, and protocol. It is not a universal speed guarantee, production
tuning input, vector/native claim, or evidence of broader CPU readiness.

Current coverage does not include multi-axis or runtime-axis arg extrema, values-plus-indices,
masked arg extrema, selected-domain vectorization, partial/combine execution, fusion, gradients,
compiler or Model changes, dynamic Shapes/layouts, native routes, or another backend. Those forms
remain fail-closed.

### Unary numerical closure

Every unary occurrence remains one Model node and one CPU instruction. FLOAT64 scalar emission
uses primitive arithmetic and permitted JDK primitive `Math` or `StrictMath` calls. Schema 30
embeds the selected ERF, sigmoid, exact and tanh-approximation GELU, and SiLU formulas directly in
the generated class. Where the JDK lacks a FLOAT32 overload, scalar emission widens the represented
binary32 input exactly, performs the selected binary64 calculation, and narrows once. This is a
private realization of a same-typed request, not a cross-type Model cast.

FLOAT32 `RSQRT` applies that rule to the complete `1.0d / StrictMath.sqrt(input)` expression: it
widens the input before the square root, performs the reciprocal in FLOAT64, and narrows only the
final reciprocal-square-root result. It does not round the square root to FLOAT32 between the two
operations.

The compute matrix separates semantic support from vector eligibility:

| Unary group | FLOAT32 scalar / parallel-scalar | FLOAT64 scalar / parallel-scalar | FLOAT32 and FLOAT64 vector / parallel-vector |
|---|---|---|---|
| `ABS`, `NEG`, `RECIPROCAL` | yes | yes | yes |
| `LOG`, `LOG1P`, `EXP`, `EXPM1`, `SQRT`, `RSQRT`, `TANH` | yes | yes | yes, direct Java 26 lane operators or typed division/square-root composition |
| `ERF`, `GELU` | yes | yes | yes, typed pure vector formulas |
| `SIGN`, `RELU` | yes | yes | yes |
| `FLOOR`, `CEIL` | yes | yes | no; scalar compute |
| `SIGMOID`, `GELU_TANH_APPROXIMATION`, `SILU` | yes | yes | no; scalar compute |

The Vector API math-library operators follow the corresponding Java accuracy and monotonicity
contracts, but this makes no hardware-intrinsic or speedup promise. Generated code does not
extract, call, and reinsert individual lanes. If any instruction in an otherwise admitted chain is
vector-ineligible, preparation selects scalar or parallel-scalar compute instead.

Exceptional values are part of the selected semantics. `ABS(-0)` is positive zero; `NEG(+0)` is
negative zero; reciprocal and reciprocal square root preserve the division-defined zero and
infinity signs. `LOG` rejects negative finite values, `LOG1P(-1)` is negative infinity, and values
below `-1` are NaN. `SQRT` and `RSQRT` reject negative finite values, while FLOOR, CEIL, SIGN, and
TANH preserve signed zero. ReLU is `Math.max(input, +0)`, so either zero sign produces positive
zero and NaN propagates. ERF maps infinities to signed one and preserves zero sign. Sigmoid uses a
sign-stable two-branch formula. GELU, its fixed tanh approximation, and SiLU explicitly map
negative infinity to negative zero rather than evaluating an indeterminate infinity-times-zero
product. NaN results have a classification promise only; payload and sign are unspecified.

Scalar `LOG`, `LOG1P`, `EXP`, and `EXPM1` retain the Java method's at-most-one-ulp contract;
`SQRT` is correctly rounded and scalar `TANH` permits at most 2.5 ulps. FLOAT32 vector
`LOG`/`LOG1P`/`EXP`/`EXPM1` and `RSQRT` permit at most 2 binary32 ulps against the completed scalar
reference; vector `SQRT` permits at most 1 binary32 ulp and vector `TANH` at most 5. FLOAT64 vector
`EXP`/`LOG`/`LOG1P`/`EXPM1` permit at most 2 ulps against the scalar oracle, and vector `TANH`
permits at most 5 ulps. ERF, sigmoid, both GELU kinds, and SiLU use
`max(2e-7, 2e-7 * abs(expected))` for FLOAT64 and
`max(2e-5, 2e-5 * abs(expected))` for FLOAT32. These are conformance limits for the fixed
algorithms, not relaxed-math permission or a hardware-acceleration claim.

The vector ERF formula retains the selected Cephes double-precision piecewise rational
approximation and coefficient order documented by the official [Cephes double-precision
reference](https://netlib.org/cephes/doubldoc.html). The FLOAT32 tables are source-stable
hexadecimal float literals: each is the one-time IEEE-754 binary32 rounding of its corresponding
retained binary64 coefficient. They are not per-call casts and do not substitute the different
Cephes `erff` polynomial family; the official [Cephes single-precision
reference](https://netlib.org/cephes/singldoc.html) is provenance review evidence. Directed,
deterministic-random, scalar-differential, and independent numerical-integration tests establish
the stated FLOAT32 bound. GELU calls the matching typed ERF formula and then evaluates
`0.5 * x * (1 + erf(x / sqrt(2)))`, including the explicit negative-infinity-to-negative-zero
correction.

Emission ownership follows the computation boundary. `CpuVectorInstructionEmitter` owns the
single closed opcode switch and emits one already-validated typed value-vector or virtual-mask
instruction into preallocated locals. `CpuVectorMath` has overloads for both floating precisions
and owns only pure multi-instruction formulas such as reciprocal, ERF, and GELU. It owns no `CodeBuilder`,
loop, carrier, route, or opcode selection. `CpuClassFileKernelGenerator` still owns class and loop
construction, local allocation, boundary loads/stores, scalar tails, byte verification, and hidden
class definition. Generated hot code contains no opcode, type, carrier, route, or shape dispatch.

The schema-30 self-containment audit is deliberately bounded to 28 representative generated
classes: ten scalar activation classes, four vector ERF/GELU classes, two BFLOAT16 scan classes,
ten numeric MIN/MAX classes, and two BOOL ALL/ANY classes. Scalar activation, scan, and aggregate
artifacts contain no Synaptik-owned call. Their remaining formula calls are the required primitive
JDK operations: `Math.abs`, `Math.sqrt`, `Math.copySign`, integral `Math.min`/`Math.max`,
`StrictMath.exp`, `StrictMath.tanh`, and primitive-wrapper raw-bit conversion or NaN
classification. The four vector artifacts reference only typed `CpuVectorMath.erf` or
`CpuVectorMath.gelu`, once per emitted vector instruction per chunk; their scalar tails contain no
Synaptik-owned call. None of the 28 classes contains a method handle, `invokedynamic`, dynamic
constant, or bootstrap-method entry. This is evidence for that audited matrix, not a claim about
every possible current or future generated kernel.

The matching 24-case direct-primitive-Java probe covered both precisions of all five changed
activations, both BFLOAT16 scan operations, numeric MIN/MAX for every supported represented type,
and BOOL ALL/ANY. All 120 isolated-fork results and all 24 aggregate ratios passed the `<= 1.15x`
gate on the recorded Java 26/macOS arm64 environment. The maximum fork ratio was `1.117147x` and
the maximum aggregate ratio was `1.111438x`, both for BOOL ANY. These measurements compare only
the fixed audited cases and do not select production behavior or establish a general speedup.

### Floating division, extrema, clamp, and power realization

CPU 0005F adds three distinct semantic opcodes. Binary `DIV` reads two tensor boundaries and uses
ordinary right-aligned broadcasting. Scalar `DIV` reads one tensor boundary and divides each
element by the exact same-typed `ScalarValueAttrs` denominator. Scalar `POW` also reads one tensor
boundary, but retains its exponent bits and semantic power opcode even when its selected
realization uses division. CPU 0005G adds direct Tensor/Tensor `POW`; it never applies scalar-
exponent analysis because the exponent is a runtime Tensor value.

| Semantic operation | FLOAT32 | FLOAT64 | Vector eligibility |
|---|---|---|---|
| Tensor/Tensor `DIV` | Primitive `left / right` | Primitive `left / right` | Both precisions, under the existing homogeneous-type, topology, access, and carrier gates |
| Tensor/scalar `DIV` | Primitive `input / denominator` | Primitive `input / denominator` | Both precisions, under the same gates |
| Tensor/scalar `POW` | Direct or one selected exact plan | Direct or one selected exact plan | Both precisions for the four special plans; direct power is scalar-compute only |
| Tensor/Tensor `POW` | Widen both represented binary32 values, call `StrictMath.pow`, narrow once | Direct `StrictMath.pow` | Scalar-compute only |

Division uses Java/IEEE-754 behavior in operand order. It preserves NaN classification,
infinities, signed zero, division by zero, overflow, underflow, and subnormal transitions without
throwing an integer-style divide-by-zero exception. CPU does not replace either DIV form with
multiplication by a reciprocal. FLOAT32 direct power widens the represented binary32 base and
exponent exactly, calls `StrictMath.pow`, and narrows once; FLOAT64 calls `StrictMath.pow`
directly.

Common lowering classifies each exact scalar-power exponent once:

| Exact exponent | Plan | Realization |
|---|---|---|
| positive or negative zero | `POSITIVE_ONE` | Store exact positive typed one without reading the base |
| positive one | `IDENTITY` | Forward the represented base |
| positive two | `SQUARE` | One typed multiply, `base * base` |
| negative one | `RECIPROCAL` | One typed division, `+1.0 / base` |
| every other finite value, infinity, or NaN | `DIRECT` | Direct power realization |

The reciprocal row is still `SCALAR_POW(-1)`, not binary or scalar DIV. Its opcode, exact
immediate bits, realization, canonical IR identity, specialization metadata, and cold manifest
remain power facts. Exponents such as `0.5`, `3`, and `-2` stay direct: square-root substitution,
multiply chains, reciprocal chains, and exponentiation by squaring do not have the required
universal rounding and exceptional-value proof.

All three opcodes are ordinary members of the existing one-to-eight connected pointwise fusion.
For example, a binary broadcast DIV may feed a scalar DIV and then scalar POW inside one
three-instruction unit. The two intermediate graph values remain virtual, instruction order is
preserved, and only the last result is stored. This example explains lowering structure; it does
not add a public execution facade or broaden Tensor/Tensor power support.

CPU 0005G also realizes represented-value extrema and first-class range clamp. FLOAT32/FLOAT64
use `Math.min`/`Math.max` semantics so either NaN propagates and opposite signed zeros select
negative zero for minimum or positive zero for maximum. INT32/INT64 use exact signed order.
`CLAMP(input, lower, upper)` remains one Model occurrence and one `SCALAR_CLAMP` instruction whose
calculation is ordered exactly as `MIN(MAX(input, lower), upper)`.

For a concrete signed-zero example, the input values `[-1.0, -0.0, +0.0, +1.0]` with bounds
`[-0.0, +0.0]` produce `[-0.0, -0.0, +0.0, +0.0]`. The lower `MAX` stage runs first, then the
upper `MIN` stage. This result demonstrates directional zero selection and ordered clamp
evaluation; it does not promise which NaN payload is retained. Reversing the accepted zero bounds
to `[+0.0, -0.0]` produces `-0.0` for every non-NaN input.

Logical `AND` and `OR` use ordinary right-aligned broadcasting; `NOT` preserves Shape. Cold
binding validates external BOOL bytes as canonical `0` or `1`, and materialized logical results
stay canonical. The BOOL vector `NOT` masks the bitwise complement back to byte `0` or `1`; it
never exposes `0xFF`. Generated loops perform no numeric-truthiness conversion or repeated BOOL
validation. Virtual floating predicate masks instead remain matching `VectorMask` locals until
logical combination or floating `WHERE` consumption.

The selected realization is a preparation-time code-shaping fact. It participates, with the
semantic opcode and exact scalar bits, in canonical IR, specialization and artifact compatibility,
and the optional lowering manifest. Generator schema 10 includes exact FLOAT32, FLOAT64, INT32,
INT64, or BOOL preferred-species specialization plus virtual floating-mask topology in addition
to the forty-eight-opcode vocabulary and earlier instruction shapes. It rejects schema 9 and every
older stored artifact as incompatible, with no migration reader. Generated
code performs no exponent comparison or policy lookup, and Runtime only
invokes the already-prepared executable.

The implemented access regimes are dense linear, scalar/all-zero broadcast,
last-axis bias, block/outer broadcast with a contiguous inner loop, and the complete general
positive-strided odometer fallback. Bytecode emits offset/carry arithmetic directly with no hot
cursor, virtual call, or per-element division/modulo. A cold binding computes the starting
coordinates/address and exact accessed half-open span for its requested range. CPU 0005C
vectorizes dense linear, scalar-broadcast, last-axis, and block/outer access only when every
non-scalar boundary has a complete preferred-species contiguous run. General odometer access and
too-short runs select scalar compute rather than rejecting the partition. There is no vector
gather. Cost-gated contiguous materialization may replace one eligible non-dense input access, but
complete access semantics do not promise universal vectorization.

Vector classes use the Java 26 preferred species for their validated lane type:
`FloatVector.SPECIES_PREFERRED`, `DoubleVector.SPECIES_PREFERRED`,
`IntVector.SPECIES_PREFERRED`, `LongVector.SPECIES_PREFERRED`, or
`ByteVector.SPECIES_PREFERRED`. The exact species bit size and ordered boundary types participate
in generated specialization and cache identity; schema 10 also reflects opcode/mask topology
through the lowering fingerprint, and no duplicate lane-type field is needed. Each generated class
contains only its selected scalar or vector body and one direct entry signature for the prepared derived-boundary
carrier pattern. Vector loads and stores call the exact primitive array or `MemorySegment` form
selected during generation; segment access uses native byte order. Proved dense heap-array loops
narrow universal `long` bounds and bases once to integer locals. Each contiguous vector run uses
one precomputed bound, unmasked complete vectors, and then the existing scalar body for every
remainder. This scalar-tail rule covers arbitrary starts and worker chunk ends. General
odometers, masked tails, gather access, and vector-ineligible chains select scalar compute. The
JVM may internally
scalarize a Vector API operation, so this route and its tests do not promise hardware intrinsics
or speedup.

### Current contiguous materialization decision

Before shared assignment, CPU analysis enumerates candidates in stable order: direct access,
then at most the first three eligible FLOAT64 read boundaries. It admits only one-input copies whose source
is non-scalar, non-dense, consumed by the unit, within the additional-memory limit, and usable as
canonical contiguous FLOAT64 segment access. Direct wins every tie. For each admitted input,
analysis derives use count from the lowered unit and compares these dimensionless cold estimates:

```text
direct = expected runs * uses * direct kernel cost
copied = expected runs * (copy cost + uses * contiguous kernel cost)
net benefit = direct - copied
```

Selection also requires positive benefit and the configured absolute and basis-point thresholds.
Checked `long` arithmetic fails analysis on overflow. These inputs are supplied evidence, not
nanoseconds, and CPU analysis never measures a model or searches a tuning cache.

A selected `CpuMaterializationPlan` retains the original source boundary and access binding plus a
canonical dense consumer binding. All derived graph-value buffer declarations remain first and
unchanged. CPU analysis appends exactly one workspace declaration with analysis-local ID `0`,
`elementCount * Double.BYTES` bytes, and `Double.BYTES` alignment. The workspace has no `ValueId`;
the Model graph, `LogicalMemoryPlan`, boundary identities, and two virtual intermediates remain
unchanged. Shared Prepare assigns the workspace without interpreting CPU copy or cost facts.

Finalization resolves all boundary buffer assignments and the optional workspace assignment before
the one artifact lookup. Cold binding validates the original source and the run-owned aligned
`CpuContiguousWorkspace`. The original carrier pattern still describes the Runtime buffers;
the adjusted generated pattern replaces only the copied input with `MemorySegment`. The generated
entry therefore retains the derived boundary count—materialization adds no workspace argument.
The invoking thread copies logical elements in canonical order once per bound invocation, then an
inline consumer or every selected worker reads the completed workspace. A copy failure prevents
consumer execution, and an empty execution range touches neither source nor workspace.

Shared Prepare remains blind to CPU units and fusion. Its narrow declaration hardening checks only
cross-planned-partition values: the producer partition, when present, and every distinct external
consumer partition must each declare the value. A value whose producer and consumers are confined
to one partition may remain undeclared. Existing bindable-input, constant, and publication checks
remain unchanged.

The final portable strategy vocabulary is exactly scalar, vector, parallel-scalar, and
parallel-vector: scalar/vector is the compute axis and single-thread/parallel is orchestration.
Generated kernels always take `start` and `end`; workers dispatch chunks outside the inner loop.
CPU analysis bounds usable parallelism by the configured maximum and an available-parallelism
snapshot, then limits the selected range count by the positive minimum elements per worker. These
are explicit CPU-private cold inputs, not process properties, Runtime decisions, or tuning values.
Zero elements always select scalar/single-thread. Cold binding divides a requested non-empty range
into deterministic ascending contiguous chunks that cover it exactly once; a one-chunk invocation
runs inline, while two or more chunks run synchronously through one borrowed worker group.

Composition or test code creates and closes `CpuWorkerGroup`. It owns a fixed positive number of
named daemon platform workers. Finalization requires an open group large enough for a selected
parallel plan, and the finalizer, prepared executable, and bound invocation borrow it without ever
closing it. Every selected segment must be accessible to every worker before binding succeeds.
Nested multi-chunk submission and close from an owned worker fail before nested work starts.
Concurrent external submissions remain isolated and share only the fixed worker capacity.

On worker failure, unclaimed chunks are cancelled, started chunks quiesce, and the lowest failing
range index supplies the primary unchecked failure; later distinct failures are suppressed in
ascending range order. Interruption cancels unclaimed work, joins started work, restores interrupt
status, and reports a CPU-private coordination exception. A racing close rejects new work,
cancels unclaimed chunks, joins started chunks, and terminates every owned worker. No write
rollback is promised; Runtime retains its existing failed-executable output-validity behavior.
CPU 0005D completes this materialization and persistence/specialization evidence gate.

For example, iteration Shape `[2, 4, 3]` can combine a dense `[2, 4, 3]` input, a right-aligned
`[3]` bias, and a contiguous `[2, 1, 3]` input. Their access regimes are respectively
`DENSE_LINEAR`, `LAST_AXIS_BIAS`, and `BLOCK_OUTER`; the output may use another injective resolved
layout. With a final contiguous run of only three elements, this exact example selects scalar
compute whenever the preferred FLOAT64 species has more than three lanes. It explains access and
fallback only: it does not add another operation topology or a materialization policy.

The reset replaces the flat execution package with unsupported `.internal` packages for memory,
prepare, lowering, IR, portable code generation/emission, `route.portable`, cache, executable, and
reference responsibilities. Java subpackages are not friends, so only the minimum collaboration
contracts are technically public below `.internal`; `CpuCapabilityProvider` remains the sole
supported public CPU API. CPU 0005A creates no native placeholder package. Later concrete tasks own
`route.nativeblas` leaves for OpenBLAS/Accelerate/oneMKL/AOCL and `route.nativeops` leaves for
vDSP/vForce/VML/oneDNN/AOCL-LibM/ZenDNN.

### Generated class-byte persistence and evidence

Generated-class persistence is optional cold-path policy. Without a trusted root, CPU emits,
verifies, and defines deterministically in memory. With an explicit normalized trusted root, one
realization performs one weak-intern lookup and at most one bounded current-schema envelope lookup.
The envelope is limited to 2 MiB, with at most 64 KiB of compatibility metadata and 1 MiB of class
bytes. Schema, structural key, metadata, lengths, checksum, trailing data, class shape, and entry
descriptor must all verify. Any absence, incompatibility, corruption, malformed class, or file or
security failure is a safe miss; verified generation remains correct, and optional publication
uses a forced temporary file plus atomic move. There is one current schema, no migration reader,
expiry, eviction, background service, or hostile-byte authentication claim.

A hit reuses verified Java Virtual Machine (JVM) class bytes only. It still defines a fresh hidden
class that the JVM may independently interpret or just-in-time (JIT) compile and profile. Neither
class bytes nor JIT machine code/profile are the future workload tuning cache: that separate cache
will record compatible route/configuration decisions before finalization.

The opt-in CPU 0005D development evidence suite compared complete no-root generation against
verified trusted-root hits on Oracle JDK 26.0.1, macOS 26.5.2, aarch64, with 16 available
processors. Six scalar/vector and segment/heap/mixed fixtures each used seven fresh JVM forks per
mode, 20 warmups and 50 samples per fork, for 350 samples per mode. All 2,100 hit samples were
verified persistence hits with no fallback. Median hit ratios ranged from `0.747375` to `0.863274`;
median absolute savings ranged from 83,625 ns to 150,500 ns. Every fixture missed the required
200,000 ns absolute median saving, and several also missed the 0.80 median-ratio threshold. The
recorded verdict is therefore `KEEP_DISABLED` (report hash
`0977061bba616421a23f69a7819ba0a85de9af072956d98c21a934af13ba6453`). The default finalizer and
`CpuPartitionAnalysisInputs.DEFAULT` remain persistence-free.

Ordinary prepare and Runtime never run this evidence suite, benchmark, or compare timing. Ordinary
explicit-root use only performs bounded lookup/verification and generate/optional-store on a miss.
Rerun the evidence only after a material generator, generator schema, JDK, persistence-policy
change, or deliberate performance evaluation—not per model and not as part of the ordinary CPU
suite.

The replacement removed the old portable candidate, preparer, finalizer, executable, pointwise-ADD,
unused worker/vector, and flat-package types without aliases or adapters. Historical CPU 0001–0005
specifications remain preserved as Superseded evidence.

## Current low-level OpenBLAS foundation

[`OpenBlasLibrary`](../glossary.md#openblas-library-handle--openblaslibrary) is the current public
lifetime boundary. A caller opens exactly one supplied operating-system library name or one
absolute path:

The first example proves only that the supplied library loaded, exported the complete required
symbol set, and remained open inside one caller-owned scope. It assumes that the loader resolves
the exact name `openblas` to a compatible 32-bit-`blasint` build.

```java
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;

try (OpenBlasLibrary library = OpenBlasLibrary.open("openblas")) {
    if (!library.isOpen()) {
        throw new IllegalStateException("unexpected closed OpenBLAS handle");
    }
}
```

`open("openblas")` passes that name unchanged to the JDK lookup and returns only after all four
required symbols bind. `isOpen()` then reports this owner's local lifecycle state. Exiting the
try-with-resources block closes the lookup lifetime, including when the body fails. The final
observable state is a closed Java owner; this proves loading, binding, and cleanup only, not a
GEMM result or viable CPU route.

The input is explicit: the provider does not choose a platform filename, inspect configuration,
read an environment variable or system property, search directories, or decide fallback. The
path overload requires an absolute path and passes it unchanged to the JDK lookup.

Opening succeeds only after the provider binds this complete ordered set:

1. `cblas_sgemm`;
2. `cblas_dgemm`;
3. `openblas_set_num_threads`; and
4. `openblas_get_num_threads`.

The bindings use the ordinary OpenBLAS C interface with 32-bit `blasint`. A library using the
64-bit integer interface is outside the current contract. Missing symbols fail the whole open in
the order above, the partial lookup lifetime is closed, and the caller receives
`OpenBlasLoadException` with the original failure as its cause.

Each successful open returns a fresh Java owner backed by a shared Foreign Function and Memory
(FFM) arena. `close()` is safe for repeated or concurrent close attempts and ends that owner's
lookup lifetime. It does not promise physical unloading, because the JDK, operating system, or
another owner may retain the same process library. A later native call must not race closure.

The handle exposes no native address or Foreign Function and Memory (FFM) handle. It now exposes
`sgemm` and `dgemm` for one already-normalized product:

```text
C[m,n] = alpha * (A[m,k] x B[k,n]) + beta * C[m,n]
```

`A`, `B`, and `C` begin at byte offset zero in caller-owned native segments. They use dense
row-major, non-transposed geometry; `A` and `B` may be read-only, while `C` must be writable and
must not overlap either required input range. The provider validates dimensions, segment
lifetime/access, alignment, required spans, and overlap before it either calls OpenBLAS or returns
for an output-empty product. It forwards scalar bits unchanged and does not define OpenBLAS
numerical accuracy, exceptional-value, determinism, or performance behavior.

A positive-output call with `k == 0` still invokes OpenBLAS so `beta` can apply to `C`. Calls with
`m == 0 || n == 0` make no native call only after complete validation. Caller memory remains
borrowed for the call: the provider does not allocate, copy, retain, reinterpret, or close it.
Concurrent calls require caller-managed nonconflicting segment access, and callers must not race
`close()` with invocation.

This is an invocation boundary, not a CPU route. A later CPU prepare implementation must still
decide whether OpenBLAS is eligible, normalize MATMUL and higher-level operations into this exact
geometry, materialize transpose or layout conversions, pack when required, allocate and bind storage,
construct prepared execution, choose and safely coordinate threads, and provide scalar or other
fallback. Direct provider thread control does not perform any of those CPU decisions.

### Direct thread control

`threadCount()` returns the exact positive count reported by OpenBLAS. `setThreadCount(int)`
accepts a positive 32-bit count and returns after the native setter invocation completes; it does
not confirm an effective later value. Synaptik conservatively treats the setting as mutable
library/process state. Two owners of one loaded binary may observe one another's mutations, but
the provider makes no shared-state claim across independently loaded copies, loader namespaces,
or arbitrary native consumers.

The following pattern shows caller-owned temporary restoration. It assumes the caller has already
excluded concurrent OpenBLAS work and writers for the complete scope:

```java
try (OpenBlasLibrary library = OpenBlasLibrary.open("openblas")) {
    int original = library.threadCount();
    try {
        library.setThreadCount(1);
        // Perform coordinated OpenBLAS work while the requested count is installed.
    } finally {
        library.setThreadCount(original);
    }
}
```

The intermediate requested value is `1`; the final requested value is the captured positive
`original` count. This pattern proves only explicit caller coordination and restoration attempts.
A competing native writer can still change the value, query/set sequences are not atomic,
concurrent setters have no provider-defined winner, and thread mutation is not serialized with
GEMM. The owner must remain open through restoration, and `close()` neither restores the count nor
safely races a query, setter, or GEMM call.

### Low-level invocation example

This example supplies one `A[2,4] x B[4,3] -> C[2,3]` geometry. It requires the same compatible
installed library and native-access permission as the loading example; it demonstrates the Java
contract and call boundary, not numerical correctness.

```java
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

try (Arena matrices = Arena.ofConfined();
        OpenBlasLibrary library = OpenBlasLibrary.open("openblas")) {
    MemorySegment a = matrices.allocate(2L * 4L * Float.BYTES, Float.BYTES);
    MemorySegment b = matrices.allocate(4L * 3L * Float.BYTES, Float.BYTES);
    MemorySegment c = matrices.allocate(2L * 3L * Float.BYTES, Float.BYTES);

    library.sgemm(2, 3, 4, 1.0f, a, b, 0.0f, c);
}
```

The three allocations provide 32, 48, and 24 bytes respectively: exactly the dense row-major
ranges for 8, 12, and 6 FLOAT32 elements. The call requests `alpha = 1` and `beta = 0`; the
provider checks the open owner and all matrix preconditions, derives leading dimensions 4, 3,
and 3, then invokes the bound SGEMM symbol. Normal return means that validation and native
invocation completed without a reported failure. It does not establish particular values in
`c`, because installed-library numerical validation belongs to the provider checkpoint.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| Loading fails immediately for a short name | The operating-system loader cannot resolve the exact supplied name. | Supply an installed name the loader recognizes or use the absolute-path overload. |
| Opening reports missing required symbols | The selected binary is incompatible or incomplete for the four-symbol contract. | Supply a compatible OpenBLAS C library; do not treat a partial binding as available. |
| A caller expects scalar fallback after `OpenBlasLoadException` | Fallback policy was placed mentally in the leaf provider. | Handle policy in later CPU/composition code; the provider only reports loading failure. |
| A caller passes batched, transposed, strided, offset, or tensor-shaped data directly to `sgemm`/`dgemm` | The low-level call was mistaken for CPU normalization. | Normalize and materialize the exact dense row-major product in future CPU prepare/execution code. |
| A caller expects the provider to allocate or return `C` | The borrowed in-place ABI boundary was mistaken for a storage API. | Supply a writable, sufficiently large native `C` segment and retain its ownership. |
| Two Java handles appear to have independent thread counts | Both may refer to the same loaded binary and mutable library/process state. | Conservatively coordinate their thread mutations together; do not infer sharing across independent copies or namespaces. |
| A temporary thread setting remains after the Java owner closes | The provider owns only local lookup lifetime and does not retain or restore a prior value. | Capture a positive count, exclude competing native work, and restore explicitly through a still-open owner. |

## Toolchain and resources

The project baseline is JDK 26. The Vector API remains an incubator module and is not enabled
globally; a focused CPU task must configure and validate it. Real OpenBLAS loading requires a
compatible installed library and deployment JVM permission for restricted native access. The
ordinary provider unit tests do not require an installed OpenBLAS library.

## Native checkpoint and validation

The test-source `OpenBlasNativeCheckpoint` accepts exactly one caller-supplied absolute compatible-
library path. In an isolated process with native access explicitly enabled, it opens that path
twice, requests count `1`, checks observation through both owners, runs one fixed SGEMM and DGEMM
case, and restores and verifies the captured count in `finally`. It performs no discovery,
download, packaging, fallback, or environment/property lookup.

The ordinary provider suite passed 5 suites and 50 tests with no skips, failures, or errors using
deterministic fake handles. The isolated real-native checkpoint subsequently passed against the
supplied compatible arm64 OpenBLAS 0.3.33 library, including shared thread-count observation,
fixed SGEMM/DGEMM cases, and restoration of the original thread count. The ordered repository and
architecture capability checkpoint then passed, and the provider milestone is complete.

The current CPU foundation provides the bounded fully static pointwise matrix, static
resolved-layout affine family, and one-node static movement, indexing, functional-scatter,
overlap-fold, stable ordering, explicit-state random, cumulative-scan, aggregate, and arg-extrema
families
described above. Scalar
execution covers every admitted row; parallel-scalar orchestration is available for disjoint
affine, movement, scatter, fold, ordering, random-element, whole-scan-slice, whole-aggregate-
output-cell, and whole-arg-extrema-output-cell ranges; and the
pointwise family retains its exact typed value-vector and virtual-mask parity matrix. Generator
schema 44 distinguishes pointwise,
affine, movement, indexing, scatter, fold, ordering, random, scan, aggregate, and arg-extrema
structures,
including movement occurrence order,
unequal-rank access, exact
padding bits, functional slice-update rank/map identity, scatter reduction/scratch signature and
embedded typed output/contribution body,
plus fold family/addition-policy identity and embedded typed mapping/addition body, ordering direction/output-order/output-count/scratch-
entry identity and embedded typed stable-merge/output body, explicit-state mapping/scaling
identity and direct typed state/mapping/threshold/value/mask bodies, cumulative kind/axis/mode/typed-rounding
identity, ordinary aggregate form/axis/selection/range identity, embedded typed scan/aggregate
bodies, schema-29 exact floating aggregate state and direct numerical bodies, schema-30 direct
covered scalar activation/BFLOAT16 scan/extrema/Boolean formulas, proved dense heap-array integer
pointwise/affine/movement/indexing/scatter/fold loop forms, and the affine
mapping/write domain and all seven carrier forms. Schema 35 adds the guarded frozen-shape FOLD2D
and dropout forms described above. Schema 36 adds the guarded FLOAT32 MEAN and BFLOAT16 PROD forms,
with exact run-owned state and typed general-long fallbacks preserved. Schema 37 adds the guarded
FLOAT64 GATHER and FLOAT32 GATHER_ND cursor forms, including the fixed 16-element full-range suffix
body described above, while preserving validation-before-write, arbitrary legal subranges, and
typed general-long fallbacks. Schema 38 adds the completely guarded fixed `[1024,1024]` axis-one
exclusive reverse INT64 product segment-cursor form described above, with direct unaligned long
access, arbitrary legal complete-slice subranges, unchanged zero resources, and a typed general-
long fallback. Schema 39 adds the guarded raw-BFLOAT16 affine PERMUTE/SLICE form described above,
with arbitrary legal half-open ranges and the unchanged typed general-long fallback. Schema 40
adds the completely guarded frozen FLOAT32 mixed-carrier pointwise ordinal loop described above,
including arbitrary legal half-open ranges, stable binary64 exponential/sign/final-narrowing
semantics, and the unchanged typed general-long fallback. No excluded pointwise form is enabled
by that bounded body. Schema 41 adds the guarded frozen INT64 `SCATTER_ND + MIN` direct copy and
tuple/suffix loops with arbitrary legal subranges, duplicate encounter order, zero workspace, and
the unchanged typed general-long fallback. Schema 42 adds the completely guarded frozen BFLOAT16
axes-zero-and-two MIN traversal described above, with arbitrary legal complete-output-cell ranges,
raw first-NaN and negative-zero selection, zero workspace/materialization, and the unchanged typed
general-long fallback. Schema 43 adds the bound SUM-to-Shape mapping, exact-state reduction,
modular integral, represented-copy, and guarded dense forms described above. Schema 44 adds the
separate mixed-input/output arg-extrema identity and direct typed unit-stride, guarded stride-two,
and arbitrary-stride bodies described above. No excluded
aggregate/scatter form or later semantic family,
general BFLOAT16 pointwise or dropout numerical operation,
cross-type CAST, dynamic layout, vector affine/scatter/fold/ordering execution, native fallback, backend-conformance
result, public Engine integration, hardware-intrinsic guarantee, or performance result is
implemented or promised.
Ordinary provider tests
prove Java validation and exact ABI forwarding, not installed-library numerical correctness. The
native checkpoint proves only its selected binary and fixed cases.
Future CPU work must compare optimized routes with a scalar reference through backend-conformance
tests and keep benchmarks reproducible.

See the [CPU master plan](../planning/backends/cpu/master-plan.md), [kernel routes](kernel-routes.md), and [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md).
