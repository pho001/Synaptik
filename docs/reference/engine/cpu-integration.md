# [CPU](../../glossary.md#term-cpu) integration and native [ownership](../../glossary.md#term-ownership-meanings)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

An integration is the CPU [backend](../../glossary.md#backend)'s [lifecycle](../../glossary.md#lifecycle) connection to [Engine](../../glossary.md#term-engine): capability answers,
preparation collaborators, input [borrowing](../../glossary.md#term-borrowing) and output copying. A **route** is an implementation
choice inside that same backend, such as portable Java or a [native library](../../glossary.md#term-native-library). Opening [OpenBLAS](../../glossary.md#term-openblas) or
[oneDNN](../../glossary.md#term-onednn) does not create a new [backend identity](../../glossary.md#backend-identity--backendid) or a second automatic-differentiation system.

The public `backends:cpu` integration surface contains four supported types:

- `CpuCapabilityProvider`, the stateless [fail-closed](../../glossary.md#term-fail-closed) CPU capability provider; and
- `CpuBackendIntegration`, the closeable lifecycle SPI intended for Engine composition rather
  than ordinary application code;
- `CpuLocalWorkloadTuning`, the [artifact](../../glossary.md#term-artifact)-free Phase-1 route-candidate collaboration; and
- `CpuCompletePlanTuning`, the artifact-free Phase-2 retained-plan collaboration.

One `CpuBackendIntegration.open()` call creates the fixed profile-free CPU composition. It
reports the [immutable](../../glossary.md#term-immutable) `cpu/host` availability fact, exposes [occurrence](../../glossary.md#term-occurrence)-specific capability,
one artifact-free positional `PartitionPreparation` and
one retained schedule assembler for exactly one non-empty maximal CPU-owned [partition](../../glossary.md#partition), and provides
artifact-free local and complete-plan tuning collaborations. CPU retains qualified portable,
generated, OpenBLAS and explicitly configured oneDNN routes with non-profile plan,
generated-artifact, tuning and [cache](../../glossary.md#term-cache) [identity](../../glossary.md#term-identity-meanings). Engine owns the `CompileArtifacts`, obtains the exact
partition projection from [`GraphPreparation`](../../glossary.md#graph-preparation--graphpreparation), and passes every ordinary, tuned, selected, or
fallback preparation back through `GraphPreparation`. During complete preparation, shared [Prepare](../../glossary.md#prepare)
identifies eligible fully static source-only published splat [constants](../../glossary.md#term-constant) and asks the CPU assembler
for their physical geometry. The integration also borrows intrinsically compatible
`HostTensorStorage` as a non-owning [Runtime](../../glossary.md#term-runtime) [buffer](../../glossary.md#term-buffer) representation. Its
`copyToCanonicalHostBytes(representation, descriptor, maximumBytes)` [operation](../../glossary.md#operation) copies one exact
current CPU [publication](../../glossary.md#term-publication) representation into a fresh caller-owned [mutable](../../glossary.md#term-mutable) `byte[]`. This is
Engine-facing [SPI](../../glossary.md#term-service-provider-interface-spi), not an ordinary application result [API](../../glossary.md#term-application-programming-interface-api); the caller pairs the representation
with the final [descriptor](../../glossary.md#term-descriptor) from the same [publication occurrence](../../glossary.md#publication-occurrence) while the Runtime result [lease](../../glossary.md#term-lease) is
open. The adapter also attempts bounded automatic OpenBLAS discovery and [qualification](../../glossary.md#term-qualification). That
attempt may add an eligible native candidate; the portable route remains available when discovery
or qualification fails, and OpenBLAS is neither guaranteed to be present nor guaranteed to be
selected.

`CpuBackendIntegration.openOneDnn(absolutePath)` instead adds a required exact external
**oneDNN** native [kernel](../../glossary.md#kernel) library, defaulting to the nested `OneDnnApi.AUTO` preference. The
overload accepts `AUTO`, `PRIMITIVES` or [`GRAPH`](../../glossary.md#graph). The primitive and Graph APIs both consume
common CPU [lowering](../../glossary.md#lowering); Graph does not mean [automatic differentiation](../../glossary.md#autograd) and [node](../../glossary.md#node) count does not
force it. A preference limits eligible native APIs, not provider selection: the cost comparison
may retain portable or OpenBLAS units. No provider priority or Runtime fallback chain is added.
The CPU backend identity and `cpu/host` availability remain the same.

Explicit opening requires an absolute path to oneDNN 3.14.0, [little-endian](../../glossary.md#text-encoding-and-file-boundaries) [LP64](../../glossary.md#term-lp64), sequential CPU
runtime (SEQ, code 1), with Graph support; the [JDK](../../glossary.md#term-java-development-kit-jdk) [Foreign Function and Memory](../../glossary.md#application-binary-interface-and-native-bridge) ([FFM](../../glossary.md#term-ffm)) API needs
native-access permission. Loading/version/runtime/[symbol](../../glossary.md#term-symbol) failure throws rather than becoming
optional discovery. No oneDNN [binary](../../glossary.md#term-binary-descriptor-and-metadata) is bundled, and default `open()`/`Engine.standard()` behavior
is unchanged. The [CPU guide's complete configuration example](../../backend-guide/cpu-backend.md#complete-public-configuration-example)
shows `Engine.builder().takeOwnership(configuredCpu).build()`, the [JVM](../../glossary.md#term-java-virtual-machine-jvm) flag, exact external build
pin and resource closure.

The native [F32](../../glossary.md#floating-point-precision-and-rounding) surface now includes qualified common arithmetic, predicates, aggregates,
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix)-proved movement and literal [MSE](../../glossary.md#mean-squared-error--mse)/BatchNorm/padded-[pooling](../../glossary.md#term-pooling) realizations, alongside bare
[MATMUL](../../glossary.md#matrix-multiplication--matmul) with K > 1. Each API/implementation/geometry is independently admitted; the
[CPU qualification table](../../backend-guide/cpu-backend.md#numerical-qualification-and-training)
defines the exact boundary. Unsupported occurrences, symbolic/[Scan](../../glossary.md#term-cumulative-scan) orchestration and [opaque](../../glossary.md#term-opaque)
[convolution](../../glossary.md#term-channels-first-convolution-layers) or [fused](../../glossary.md#specialization-and-fusion) [activation](../../glossary.md#term-activation) replacements retain preplanned CPU execution. Literal native
pipelines preserve the existing common unit's stores and arithmetic sites; they are neither a
second graph nor a second [AD](../../glossary.md#autograd) system. Saved/public stores and current-[weight](../../glossary.md#term-weight) visibility remain.

The oneDNN comparison uses conservative dimensionless whole-plan heuristics, not measured
latencies or promised speedups. Decisions account for arithmetic/remainder, control/[dispatch](../../glossary.md#term-dispatch),
preparation, memory, ingress and scratch. OpenBLAS Phase-1 decisions and selected Phase-2
topologies remain frozen; Phase-2 [schema](../../glossary.md#term-schema) 4 includes the oneDNN policy/load-session identity and
rejects stale decisions. Selected plans are private; public timings alone cannot prove provider
selection. See the [guide's qualification, lease and benchmark details](../../backend-guide/cpu-backend.md#explicit-onednn-composition).

The copy requires a fully static [Shape](../../glossary.md#shape), a present resolved [layout](../../glossary.md#layout), a non-negative byte limit, and
one of the two exact current CPU representation implementations. It traverses logical coordinates
in [canonical](../../glossary.md#term-canonical-form) [row-major](../../glossary.md#term-row-major) order, with the final [axis](../../glossary.md#term-axis) changing fastest, through the descriptor's
non-negative [element](../../glossary.md#term-element) offset and positive or zero [strides](../../glossary.md#term-stride). A [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) descriptor reads its offset
element. A Shape with any zero [extent](../../glossary.md#term-extent) returns a fresh empty array without source-element access.
Repeated physical addresses from a zero stride are valid and produce repeated logical values.

Segment-backed source elements are interpreted using the CPU's native-order unaligned primitive
access, but the [detached](../../glossary.md#detached-model-results) output format is always [big-endian](../../glossary.md#text-encoding-and-file-boundaries). [`FLOAT64`](../../glossary.md#term-float64) and [`FLOAT32`](../../glossary.md#term-float32) preserve raw
[NaN](../../glossary.md#term-nan) payloads and signed zeros; [`BFLOAT16`](../../glossary.md#term-bfloat16) preserves its exact stored 16 bits; [`INT64`](../../glossary.md#data-type--datatype) and [`INT32`](../../glossary.md#data-type--datatype)
preserve their [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bit patterns; and [`BOOL`](../../glossary.md#data-type--datatype) accepts only the exact stored byte `0` or
`1`. No data-type conversion or BOOL [normalization](../../glossary.md#term-normalization) occurs. The checked logical byte count must
not exceed either `maximumBytes` or `Integer.MAX_VALUE`; arithmetic [overflow](../../glossary.md#term-overflow) and a genuine JVM
allocation failure remain distinct failures.

The backend-local CPU complete-assembler entry remains deliberately exact. A pure zero-node
constant or pass-through graph is rejected because it requires one non-empty maximal CPU
partition; direct use also rejects mixed-[owner](../../glossary.md#term-owner) or multiple-partition artifacts. Ordinary Engine
mixed-owner preparation instead invokes CPU through its partition finalizer and physical
contributor, then composes all owners centrally. For a backend-local accepted projection, the CPU
assembler describes creation of fresh [run](../../glossary.md#term-run)-owned buffers and workspaces, initialization of
projected constants, one prepared CPU execution occurrence, and publications in Prepare-supplied
order. A source-only constant gains no executable schedule step. Assembly performs none
of that physical work: [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) retains immutable recipes only, and creation of every
fresh [`RunState`](../../glossary.md#run-state--runstate) invokes each initialized-buffer recipe exactly once. Sequential or concurrent
runs therefore own distinct initialized representations.

The adapter owns any qualified [OpenBLAS coordinator](../../glossary.md#openblas-coordinator--cpuopenblascoordinator) and explicitly loaded oneDNN library.
Callers must keep it open for preparation and new run [admission](../../glossary.md#term-admission). oneDNN prepared-resource leases
retain persistent [kernels](../../glossary.md#kernel); each issued run workspace retains the needed kernels and library
until its own close. Stream, memory/[tensor](../../glossary.md#tensor) handles, [ABI](../../glossary.md#term-application-binary-interface-abi) tables and user scratch are created by the
run workspace factory before [binding](../../glossary.md#term-binding), not in hot execution. Data/[padding](../../glossary.md#padding)/scratch requirements are
declared before assignment; ingress copies refresh current inputs/weights once per shared source
span on every run. SEQ execution reserves one shared CPU-budget permit and waits synchronously.
Closure rejects new admission without invalidating already-issued native workspaces.
Borrowing transfers no storage ownership, and closing a [borrowed](../../glossary.md#ownership-borrowing-and-lease) wrapper does not close caller
storage. The borrow operation validates only intrinsic storage consistency and current access; it
cannot validate an expected logical input type, required span, or write role because those [target](../../glossary.md#term-target)
facts are absent. A successful [host](../../glossary.md#term-host)-byte copy likewise does not close, retain, mutate, transfer,
or change validity of its source. It is [synchronous](../../glossary.md#term-synchronous): the caller must keep the adapter, Runtime
result lease, and representation open, keep storage accessible to the calling thread, and prevent
source mutation or closure until return. Independent valid calls may run concurrently, but racing
mutation has no [atomic](../../glossary.md#resource-transaction-and-rollback)-[snapshot](../../glossary.md#term-snapshot) guarantee. Typed logical input binding, a simpler end-user execute
surface, and ordinary typed or host-materialized results remain Engine work.

The adapter does not expose CPU internals, discover backends for Runtime, run tuning policy, or
turn OpenBLAS or oneDNN into another backend identity. CPU has no production [Compiler](../../glossary.md#term-compiler) or Engine
[dependency](../../glossary.md#term-dependency); its compiler-backed fixtures are tests only. [Metal](../../glossary.md#term-metal) retains its independently
qualified restrictions and has no [FP8](../../glossary.md#term-fp8) capability.


---

[Previous topic](../compiler/artifact-values.md) · [Next topic](../runtime/contracts.md)
