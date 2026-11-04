# CPU integration and native ownership

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

The public `backends:cpu` integration surface contains four supported types:

- `CpuCapabilityProvider`, the stateless fail-closed CPU capability provider; and
- `CpuBackendIntegration`, the closeable lifecycle SPI intended for Engine composition rather
  than ordinary application code;
- `CpuLocalWorkloadTuning`, the artifact-free Phase-1 route-candidate collaboration; and
- `CpuCompletePlanTuning`, the artifact-free Phase-2 retained-plan collaboration.

One `CpuBackendIntegration.open()` call creates the fixed profile-free CPU composition. It
reports the immutable `cpu/host` availability fact, exposes occurrence-specific capability,
one artifact-free positional `PartitionPreparation` and
one retained schedule assembler for exactly one non-empty maximal CPU-owned partition, and provides
artifact-free local and complete-plan tuning collaborations. CPU retains qualified portable,
generated, OpenBLAS and explicitly configured oneDNN routes with non-profile plan,
generated-artifact, tuning and cache identity. Engine owns the `CompileArtifacts`, obtains the exact
partition projection from `GraphPreparation`, and passes every ordinary, tuned, selected, or
fallback preparation back through `GraphPreparation`. During complete preparation, shared Prepare
identifies eligible fully static source-only published splat constants and asks the CPU assembler
for their physical geometry. The integration also borrows intrinsically compatible
`HostTensorStorage` as a non-owning Runtime buffer representation. Its
`copyToCanonicalHostBytes(representation, descriptor, maximumBytes)` operation copies one exact
current CPU publication representation into a fresh caller-owned mutable `byte[]`. This is
Engine-facing SPI, not an ordinary application result API; the caller pairs the representation
with the final descriptor from the same publication occurrence while the Runtime result lease is
open. The adapter also attempts bounded automatic OpenBLAS discovery and qualification. That
attempt may add an eligible native candidate; the portable route remains available when discovery
or qualification fails, and OpenBLAS is neither guaranteed to be present nor guaranteed to be
selected.

`CpuBackendIntegration.openOneDnn(absolutePath)` instead adds a required exact external
**oneDNN** native kernel library, defaulting to the nested `OneDnnApi.AUTO` preference. The
overload accepts `AUTO`, `PRIMITIVES` or `GRAPH`. The primitive and Graph APIs both consume
common CPU lowering; Graph does not mean automatic differentiation and node count does not
force it. A preference limits eligible native APIs, not provider selection: the cost comparison
may retain portable or OpenBLAS units. No provider priority or Runtime fallback chain is added.
The CPU backend identity and `cpu/host` availability remain the same.

Explicit opening requires an absolute path to oneDNN 3.14.0, little-endian LP64, sequential CPU
runtime (SEQ, code 1), with Graph support; the JDK Foreign Function and Memory (FFM) API needs
native-access permission. Loading/version/runtime/symbol failure throws rather than becoming
optional discovery. No oneDNN binary is bundled, and default `open()`/`Engine.standard()` behavior
is unchanged. The [CPU guide's complete configuration example](../../backend-guide/cpu-backend.md#complete-public-configuration-example)
shows `Engine.builder().takeOwnership(configuredCpu).build()`, the JVM flag, exact external build
pin and resource closure.

The native F32 surface now includes qualified common arithmetic, predicates, aggregates,
coordinate-proved movement and literal MSE/BatchNorm/padded-pooling realizations, alongside bare
MATMUL with K > 1. Each API/implementation/geometry is independently admitted; the
[CPU qualification table](../../backend-guide/cpu-backend.md#numerical-qualification-and-training)
defines the exact boundary. Unsupported occurrences, symbolic/Scan orchestration and opaque
convolution or fused activation replacements retain preplanned CPU execution. Literal native
pipelines preserve the existing common unit's stores and arithmetic sites; they are neither a
second graph nor a second AD system. Saved/public stores and current-weight visibility remain.

The oneDNN comparison uses conservative dimensionless whole-plan heuristics, not measured
latencies or promised speedups. Decisions account for arithmetic/remainder, control/dispatch,
preparation, memory, ingress and scratch. OpenBLAS Phase-1 decisions and selected Phase-2
topologies remain frozen; Phase-2 schema 4 includes the oneDNN policy/load-session identity and
rejects stale decisions. Selected plans are private; public timings alone cannot prove provider
selection. See the [guide's qualification, lease and benchmark details](../../backend-guide/cpu-backend.md#explicit-onednn-composition).

The copy requires a fully static Shape, a present resolved layout, a non-negative byte limit, and
one of the two exact current CPU representation implementations. It traverses logical coordinates
in canonical row-major order, with the final axis changing fastest, through the descriptor's
non-negative element offset and positive or zero strides. A rank-zero descriptor reads its offset
element. A Shape with any zero extent returns a fresh empty array without source-element access.
Repeated physical addresses from a zero stride are valid and produce repeated logical values.

Segment-backed source elements are interpreted using the CPU's native-order unaligned primitive
access, but the detached output format is always big-endian. `FLOAT64` and `FLOAT32` preserve raw
NaN payloads and signed zeros; `BFLOAT16` preserves its exact stored 16 bits; `INT64` and `INT32`
preserve their two's-complement bit patterns; and `BOOL` accepts only the exact stored byte `0` or
`1`. No data-type conversion or BOOL normalization occurs. The checked logical byte count must
not exceed either `maximumBytes` or `Integer.MAX_VALUE`; arithmetic overflow and a genuine JVM
allocation failure remain distinct failures.

The backend-local CPU complete-assembler entry remains deliberately exact. A pure zero-node
constant or pass-through graph is rejected because it requires one non-empty maximal CPU
partition; direct use also rejects mixed-owner or multiple-partition artifacts. Ordinary Engine
mixed-owner preparation instead invokes CPU through its partition finalizer and physical
contributor, then composes all owners centrally. For a backend-local accepted projection, the CPU
assembler describes creation of fresh run-owned buffers and workspaces, initialization of
projected constants, one prepared CPU execution occurrence, and publications in Prepare-supplied
order. A source-only constant gains no executable schedule step. Assembly performs none
of that physical work: `PreparedExecution` retains immutable recipes only, and creation of every
fresh `RunState` invokes each initialized-buffer recipe exactly once. Sequential or concurrent
runs therefore own distinct initialized representations.

The adapter owns any qualified OpenBLAS coordinator and explicitly loaded oneDNN library.
Callers must keep it open for preparation and new run admission. oneDNN prepared-resource leases
retain persistent kernels; each issued run workspace retains the needed kernels and library
until its own close. Stream, memory/tensor handles, ABI tables and user scratch are created by the
run workspace factory before binding, not in hot execution. Data/padding/scratch requirements are
declared before assignment; ingress copies refresh current inputs/weights once per shared source
span on every run. SEQ execution reserves one shared CPU-budget permit and waits synchronously.
Closure rejects new admission without invalidating already-issued native workspaces.
Borrowing transfers no storage ownership, and closing a borrowed wrapper does not close caller
storage. The borrow operation validates only intrinsic storage consistency and current access; it
cannot validate an expected logical input type, required span, or write role because those target
facts are absent. A successful host-byte copy likewise does not close, retain, mutate, transfer,
or change validity of its source. It is synchronous: the caller must keep the adapter, Runtime
result lease, and representation open, keep storage accessible to the calling thread, and prevent
source mutation or closure until return. Independent valid calls may run concurrently, but racing
mutation has no atomic-snapshot guarantee. Typed logical input binding, a simpler end-user execute
surface, and ordinary typed or host-materialized results remain Engine work.

The adapter does not expose CPU internals, discover backends for Runtime, run tuning policy, or
turn OpenBLAS or oneDNN into another backend identity. CPU has no production Compiler or Engine
dependency; its compiler-backed fixtures are tests only. Metal retains its independently
qualified restrictions and has no FP8 capability.


---

[Previous topic](../compiler/artifact-values.md) · [Next topic](../runtime/contracts.md)
