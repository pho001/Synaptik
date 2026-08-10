# CPU backend

## Outcome and status

This guide defines the CPU integration boundary and helps contributors avoid treating CPU routes
as separate backends. The current CPU module accepts two bounded, fully static portable families.
A pointwise partition is one supported occurrence or one connected straight-line chain of at most
eight occurrences. A static affine partition is one connected one-input/one-output chain of at
most eight resolved-layout view occurrences. Either family lowers to one computation unit, one
route-independent canonical kernel intermediate representation (IR), one generated Java 26 class,
and one partition-level prepared executable. Internal single-use results remain graph and logical-
memory values but are virtual in the unit. Pointwise analysis declares its derived external
boundaries and sole final output; affine analysis declares exactly the original source and final
result.

Generated scalar and Java 26 Vector API loops accept primitive `start` and `end` bounds.
Compatible concrete extents bind on the cold path and share identical class bytes and one
process-local loaded compatibility identity. The pointwise semantic matrix uses exactly five
executable types—FLOAT64, FLOAT32, INT32, INT64, and BOOL—and one forty-eight-opcode CPU-private
vocabulary. Affine copies accept all six current Model data types and transfer represented bits
without conversion. BFLOAT16 uses the existing raw `short[]` representation or native-order
two-byte segment access only; this is not BFLOAT16 arithmetic or numerical support. The access
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
non-affine/index/scatter/order/random work, general partition-DAG fusion, and later operation
families remain Draft.

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
two compatible extents of the same fused topology. Fixed-shape or unrolled variants remain later,
budgeted, evidence-selected specializations. The current hard ceiling admits four complete
analysis candidates and realizes one artifact; fixed-shape and unrolled variant budgets are both
zero.

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
before schema-11 artifact access and constructs one immutable executable. Cold binding validates
the exact data type, carrier, byte size, alignment, accessibility, output writability, canonical
BOOL input bytes, and source/result non-overlap. Runtime then invokes only the prepared direct
carriers, address table, and `start`/`end` bounds; it receives no operation, graph node, Shape,
layout, affine IR, or route choice.

Non-affine movement, index tensors, functional scatter and overlap-fold, ordering and top-K,
explicit-state random work, dynamic layouts, general partition-DAG decomposition/fusion, and
benchmarks remain outside this implemented family.

### Unary numerical closure

Every unary occurrence remains one Model node and one CPU instruction. FLOAT64 scalar emission
uses primitive arithmetic, `Math`, `StrictMath`, or the shared pure error-function and activation
helpers. Where the JDK lacks a FLOAT32 overload, scalar emission widens the represented binary32
input exactly, performs the selected binary64 calculation, and narrows once. This is a private
realization of a same-typed request, not a cross-type Model cast.

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
selected during generation; segment access uses native byte order. Each
contiguous run uses unmasked complete vectors, then the existing scalar body for every remainder.
This scalar-tail rule covers arbitrary starts and worker chunk ends. General odometers, masked
tails, gather access, and vector-ineligible chains select scalar compute. The JVM may internally
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

The current CPU foundation provides the bounded fully static pointwise matrix and the static
resolved-layout affine family described above. Scalar execution covers every admitted row;
parallel-scalar orchestration is available for disjoint affine write ranges; and the pointwise
family retains its exact typed value-vector and virtual-mask parity matrix. Generator schema 11
distinguishes pointwise and affine structures, the affine mapping/write domain, and all seven
carrier forms. No excluded pointwise or later semantic family, BFLOAT16 numerical operation,
cross-type CAST, dynamic layout, vector affine execution, native fallback, backend-conformance
result, public Engine integration, hardware-intrinsic guarantee, or performance result is
implemented or promised.
Ordinary provider tests
prove Java validation and exact ABI forwarding, not installed-library numerical correctness. The
native checkpoint proves only its selected binary and fixed cases.
Future CPU work must compare optimized routes with a scalar reference through backend-conformance
tests and keep benchmarks reproducible.

See the [CPU master plan](../planning/backends/cpu/master-plan.md), [kernel routes](kernel-routes.md), and [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md).
