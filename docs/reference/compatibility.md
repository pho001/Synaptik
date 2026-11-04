<a id="public-api-status"></a>
# Public [API](../glossary.md#term-application-programming-interface-api) status and compatibility

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

Here **public** means Java callers can name an API; **supported** also depends on the documented
workflow and constraints. **Compatibility** asks whether callers or stored data keep working across
changes, not whether two [backends](../glossary.md#backend) compute the same result. A declared name or marker alone is
not evidence of implemented execution.

<a id="purpose-and-status"></a>
## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual [lifecycle](../glossary.md#lifecycle) examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation contains the
selected public [model](../glossary.md#term-model) foundation, [tensor](../glossary.md#tensor)-expression [metadata](../glossary.md#term-metadata) surface, common [trace](../glossary.md#trace)-event envelope
plus model-correlation identifiers, and the first [backend](../glossary.md#backend)-neutral [planning](../glossary.md#term-planning) capability contracts.
It also contains backend and backend-scoped [device](../glossary.md#term-device) [identity](../glossary.md#term-identity-meanings) values plus a coarse
[CPU](../glossary.md#term-cpu)-versus-[accelerator](../glossary.md#term-accelerator) device classification. Planning also contains internal per-query hard-
eligibility [evaluation](../glossary.md#term-evaluation) and cost-free baseline [owner](../glossary.md#term-owner) comparison. Those two steps remain internal;
public `BackendOwnerPlanning` composes them for one [operation](../glossary.md#operation) [occurrence](../glossary.md#term-occurrence). Public stateless
`MaximalSameOwnerPartitioning` and `LogicalMemoryPlanning` generate the current [immutable](../glossary.md#term-immutable)
[partition](../glossary.md#partition) and logical-memory recipes from [compiler](../glossary.md#term-compiler)-owned [graph](../glossary.md#graph)-wide inputs. Package-private
Compiler orchestration now consumes those three operations and uses the public immutable
`FunctionalGradientRequest`, `GradientPublicationBinding`, `DerivativeGraphMetadata`,
`CompileArtifacts`, `PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` contracts.
Public `GraphCompilationPort` exposes that complete [constant](../glossary.md#term-constant)-free pipeline as a narrow
cross-[module](../glossary.md#term-module) integration service-provider interface ([SPI](../glossary.md#term-service-provider-interface-spi)). The CPU backend exposes the supported
`CpuBackendIntegration` lifecycle SPI, including bounded [canonical](../glossary.md#term-canonical-form) [host](../glossary.md#term-host)-byte [materialization](../glossary.md#materialization). The [Metal](../glossary.md#term-metal) backend exposes
`MetalBackendConfiguration`, `MetalBackendIntegration`, [`MetalTraceObserver`](../glossary.md#metal-trace-observer--metaltraceobserver), and a closed
occurrence-qualified domain. The [current production support matrix](../backend-guide/production-support.md)
owns the detailed operation/type/[layout](../glossary.md#layout)/[derivative](../glossary.md#term-derivative) inventory, including attention, [normalization](../glossary.md#term-normalization)
and Scan. Metal retains seven storage carriers and its own predicate domain; it has no [FP8](../glossary.md#term-fp8)
support. CPU low-[precision](../glossary.md#term-precision) parity does not widen Metal [admission](../glossary.md#term-admission).
Direct [BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) mixed-low execution remains outside the documented Metal authority;
use explicit [FLOAT32](../glossary.md#term-float32) casts where required by that [matrix](../glossary.md#term-matrix-meanings).

Low-containing Metal partitions retain their qualified custom-program route rather than an
alternate CPU route. Explicit casts are graph operations, not implicit conversion or fallback.
Backend [ABI](../glossary.md#term-application-binary-interface-abi), program [schema](../glossary.md#term-schema) and [cache](../glossary.md#term-cache) identities are implementation compatibility boundaries;
use the backend guide and authenticated current [artifacts](../glossary.md#term-artifact) rather than pinning copies of those
version strings in application guidance.

Canonical typed host ingress/[publication](../glossary.md#term-publication) and CPU/Metal transfer support Metal's seven carriers.
Cross-owner values require the documented static [rank](../glossary.md#term-rank)-0..16 canonical or approved positive-[stride](../glossary.md#term-stride)
non-overlapping layouts and checked physical spans. Transfer preserves [carrier](../glossary.md#term-storage-carrier) values and
performs no conversion or fallback.
`Engine.builder()` is the public explicit [composition root](../glossary.md#term-composition-root) for
opened CPU and Metal integrations. It freezes their Planning inputs in registration order and
supports a complete non-empty plan only when every partition has one exact registered owner. The
`Engine.standard()` convenience still constructs one fresh CPU-only composition through that same
path. Both forms expose owner-bound compiled graphs and one reusable closeable [PreparedExecution](../glossary.md#prepared-execution--preparedexecution).
[Engine](../glossary.md#term-engine).[run](../glossary.md#term-run) binds exact Tensor IDs in arbitrary order; [Engine.runValues](../glossary.md#fresh-value-binding--enginerunvalues) binds fresh fully static
values in compiled order (exact [descriptors](../glossary.md#term-descriptor) for static slots, qualified [binding](../glossary.md#term-binding) for named B).
Both use [synchronous](../glossary.md#term-synchronous) execution, independent publication
[leases](../glossary.md#term-lease) and explicit [detached](../glossary.md#detached-model-results) host materialization.
The four one-shot `compute(...)` overloads automatically discover reachable [provenance](../glossary.md#provenance)-free input
[leaves](../glossary.md#term-leaf), then compose a fresh forward-only [compile](../glossary.md#compile), [prepare](../glossary.md#prepare), run, complete publication and aggregate-
byte [preflight](../glossary.md#term-preflight), ordered materialization, and cleanup under one Engine admission. The one-shot
`backward(objective, targets, maximumTotalBytes)` call applies automatic input discovery to one
[scalar](../glossary.md#term-scalar) floating [gradient](../glossary.md#term-gradient)-eligible [objective](../glossary.md#term-objective), fixes Compiler's absent positive-one scalar [seed](../glossary.md#term-seed) and
disconnected-[target](../glossary.md#term-target) `ERROR` policy, and returns a detached objective plus immutable target-aligned
gradients. Public capability [matrices](../glossary.md#term-review-matrix) and a public graph-wide Planning workflow remain planned.
Prepare analysis, finalization, and complete graph-preparation contracts plus the initial [Runtime](../glossary.md#term-runtime)
geometry, prepared representation creation, per-run resource/validity, executable and transfer
cold-binding, [prepared publication](../glossary.md#prepared-publication--preparedpublication) and result leasing, and the ordered schedule contracts are
current and now form that deliberately low-level lifecycle through the CPU integration. The backend
contract also contains an immutable caller-supplied availability [snapshot](../glossary.md#term-snapshot); it is data, not a
discovery or [liveness](../glossary.md#term-liveness) [API](../glossary.md#term-application-programming-interface-api). A [sealed](../glossary.md#term-sealed) requirement family can now name one hard eligibility target.
The current config module can record that hard-target optionality, requested graph [scope](../glossary.md#term-scope),
permission for optional semantics-preserving compiler optimization, and one optional soft coarse
device-class preference. It can also hold one explicit immutable [model-autotuning request](../glossary.md#model-autotuning-request--preparation-result) as
declarative data. The internal baseline consumes the compile preference after hard eligibility.
The package-private complete Compiler entry consumes all four compile-config leaves directly, and
`GraphCompilationPort` supplies them through its public integration call. No current
`CompileConfig`, public capability-matrix or eligibility surface, or numeric scoring evaluator is
callable. The ordinary Engine supplies fixed compile settings rather than those configurable
surfaces. The model-autotuning request is current through `Engine.prepareTuned(...)` for the
bounded single-owner CPU workflow and exact [singleton](../glossary.md#term-singleton)-NEG Metal workflow described below. Generic
graph/plan tuning remains planned. APIs may change through the ordered planning process.
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module boundaries, not source or [binary](../glossary.md#term-binary-descriptor-and-metadata)
compatibility.


---

[Next topic](extensions/overview.md)
