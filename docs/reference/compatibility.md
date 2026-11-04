<a id="public-api-status"></a>
# Public API status and compatibility

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

<a id="purpose-and-status"></a>
## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual lifecycle examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation contains the
selected public model foundation, tensor-expression metadata surface, common trace-event envelope
plus model-correlation identifiers, and the first backend-neutral planning capability contracts.
It also contains backend and backend-scoped device identity values plus a coarse
CPU-versus-accelerator device classification. Planning also contains internal per-query hard-
eligibility evaluation and cost-free baseline owner comparison. Those two steps remain internal;
public `BackendOwnerPlanning` composes them for one operation occurrence. Public stateless
`MaximalSameOwnerPartitioning` and `LogicalMemoryPlanning` generate the current immutable
partition and logical-memory recipes from compiler-owned graph-wide inputs. Package-private
Compiler orchestration now consumes those three operations and uses the public immutable
`FunctionalGradientRequest`, `GradientPublicationBinding`, `DerivativeGraphMetadata`,
`CompileArtifacts`, `PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` contracts.
Public `GraphCompilationPort` exposes that complete constant-free pipeline as a narrow
cross-module integration service-provider interface (SPI). The CPU backend exposes the supported
`CpuBackendIntegration` lifecycle SPI, including bounded canonical host-byte materialization. The Metal backend exposes
`MetalBackendConfiguration`, `MetalBackendIntegration`, `MetalTraceObserver`, and a closed
occurrence-qualified domain. The [current production support matrix](../backend-guide/production-support.md)
owns the detailed operation/type/layout/derivative inventory, including attention, normalization
and Scan. Metal retains seven storage carriers and its own predicate domain; it has no FP8
support. CPU low-precision parity does not widen Metal admission.
Direct BFLOAT16/FLOAT16 mixed-low execution remains outside the documented Metal authority;
use explicit FLOAT32 casts where required by that matrix.

Low-containing Metal partitions retain their qualified custom-program route rather than an
alternate CPU route. Explicit casts are graph operations, not implicit conversion or fallback.
Backend ABI, program schema and cache identities are implementation compatibility boundaries;
use the backend guide and authenticated current artifacts rather than pinning copies of those
version strings in application guidance.

Canonical typed host ingress/publication and CPU/Metal transfer support Metal's seven carriers.
Cross-owner values require the documented static rank-0..16 canonical or approved positive-stride
non-overlapping layouts and checked physical spans. Transfer preserves carrier values and
performs no conversion or fallback.
`Engine.builder()` is the public explicit composition root for
opened CPU and Metal integrations. It freezes their Planning inputs in registration order and
supports a complete non-empty plan only when every partition has one exact registered owner. The
`Engine.standard()` convenience still constructs one fresh CPU-only composition through that same
path. Both forms expose owner-bound compiled graphs and one reusable closeable PreparedExecution.
Engine.run binds exact Tensor IDs in arbitrary order; Engine.runValues binds fresh fully static
values in compiled order (exact descriptors for static slots, qualified binding for named B).
Both use synchronous execution, independent publication
leases and explicit detached host materialization.
The four one-shot `compute(...)` overloads automatically discover reachable provenance-free input
leaves, then compose a fresh forward-only compile, prepare, run, complete publication and aggregate-
byte preflight, ordered materialization, and cleanup under one Engine admission. The one-shot
`backward(objective, targets, maximumTotalBytes)` call applies automatic input discovery to one
scalar floating gradient-eligible objective, fixes Compiler's absent positive-one scalar seed and
disconnected-target `ERROR` policy, and returns a detached objective plus immutable target-aligned
gradients. Public capability matrices and a public graph-wide Planning workflow remain planned.
Prepare analysis, finalization, and complete graph-preparation contracts plus the initial Runtime
geometry, prepared representation creation, per-run resource/validity, executable and transfer
cold-binding, prepared publication and result leasing, and the ordered schedule contracts are
current and now form that deliberately low-level lifecycle through the CPU integration. The backend
contract also contains an immutable caller-supplied availability snapshot; it is data, not a
discovery or liveness API. A sealed requirement family can now name one hard eligibility target.
The current config module can record that hard-target optionality, requested graph scope,
permission for optional semantics-preserving compiler optimization, and one optional soft coarse
device-class preference. It can also hold one explicit immutable model-autotuning request as
declarative data. The internal baseline consumes the compile preference after hard eligibility.
The package-private complete Compiler entry consumes all four compile-config leaves directly, and
`GraphCompilationPort` supplies them through its public integration call. No current
`CompileConfig`, public capability-matrix or eligibility surface, or numeric scoring evaluator is
callable. The ordinary Engine supplies fixed compile settings rather than those configurable
surfaces. The model-autotuning request is current through `Engine.prepareTuned(...)` for the
bounded single-owner CPU workflow and exact singleton-NEG Metal workflow described below. Generic
graph/plan tuning remains planned. APIs may change through the ordered planning process.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module boundaries, not source or binary
compatibility.


---

[Next topic](extensions/overview.md)
