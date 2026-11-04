# Explicit CPU and Metal composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

Explicit composition means the caller opens backend integrations and transfers them into one
Engine. A backend **owner** is the implementation assigned a graph partition, not the owner of
the caller's input memory. A **cross-owner transfer** moves a value between those implementations;
it is planned before a run, not an automatic retry on another backend.

<a id="current-explicit-cpu-and-metal-composition"></a>
## Current explicit CPU and Metal composition

The public composition shape is:

```java
MetalBackendConfiguration metalConfiguration =
        new MetalBackendConfiguration(nativeLibrary.toAbsolutePath());
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(CpuBackendIntegration.open());
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    try (Engine engine = builder.build()) {
        CompiledGraph graph = engine.compile(List.of(output));
        try (PreparedExecution prepared = engine.prepare(graph);
                RunResult result = engine.run(prepared, List.of(input))) {
            HostTensorValue value =
                    result.materialize(result.publications().getFirst(), maximumBytes);
        }
    }
}
```

`Engine.Builder` is a public single-use `AutoCloseable` construction owner. Its concrete
`takeOwnership(CpuBackendIntegration)` and
`takeOwnership(MetalBackendIntegration)` overloads transfer a non-null integration at method
entry, including when later registration validation fails. Callers must not close or reuse an
integration after invoking the overload. A successful `build()` atomically transfers the ordered
backend set to the returned Engine. Builder close and failed build release accepted integrations
in reverse registration order; failed build is terminal.

Because the overload parameters are backend-owned public types, Engine publishes CPU and Metal as
API-visible dependencies. Backend types remain confined to explicit construction and do not appear
in compile, prepare, run, handle, or result signatures.

Registration captures one `BackendId`, stable capability provider, and immutable point-in-time
availability snapshot. Their backend IDs must be equal, and an equal duplicate ID is rejected
without replacing the first entry. The snapshot remains fixed for that Engine's lifetime.
Registration order supplies deterministic compile-time Planning input; it is not a runtime
fallback order.

Compilation chooses owner identities from that fixed inventory. Cold preparation requires a
non-empty plan, resolves every distinct partition owner, and validates every required directed
cross-owner transfer before backend analysis. A missing owner, zero-partition plan, or unsupported
transfer fails without implicit CPU fallback, retry, owner substitution, or conversion.

Shared Prepare analyzes and finalizes partitions in plan order, assigns one logical slot and one
deterministic representation position per participating owner, and obtains exact backend physical
creators. Engine assembles execution occurrences in partition order, inserting one immutable
transfer per logical value and distinct destination owner immediately before that owner's first
consumer, then appends the dense publication suffix. Current CPU/Metal transfer is direct native
host-staged in either direction for all seven carriers over exact rank-0..16 static canonical or
positive-stride non-overlapping layouts with checked physical spans.

One outward `PreparedExecution` retains direct non-owning adapters in caller-input and publication occurrence order
beside the inward Runtime execution. Run uses each captured input adapter for caller-host ingress,
and the result uses each publication's captured adapter for backend-owned host materialization.
These outer calls do not re-query the registry. Engine shutdown closes retained results first,
then ordinary and advanced prepared handles in reverse preparation-publication
order, then integrations. Closing a prepared owner rejects later runs without closing an
already returned result; an already leased Runtime run may finish, and its final lease release
performs deferred prepared-resource cleanup.

`MetalBackendConfiguration` and `MetalBackendIntegration` are public Metal-owned types. Metal
validates and snapshots the explicit absolute native-library path and acquires one system-default-
device native context and command queue before the integration is transferred. The configuration
has no physical-device selector. Engine neither parses the native-library path nor owns a duplicate
Metal configuration. The integration supplies partition preparation, retained
`localWorkloadTuning()` and `completePlanTuning()` collaborations, physical creation contribution,
exact transfer endpoints, host ingress, materialization, execution, and close for that exact
context in addition to capability.

One Engine can own at most one Metal integration because its registry rejects a duplicate Metal
`BackendId`. `BackendDeviceId(metal, "default")` is the sole abstract availability slot, not a
stable physical-device fingerprint or selector; Planning proves eligibility and retains the
backend owner rather than a selected Metal device. Separate Engines may open separate default
contexts, but the API provides no enumeration, explicit selection, multi-device scheduling,
cross-device migration/coherence, failover, or hot-plug behavior.

`MetalTraceObserver` is a public functional interface whose sole abstract method receives
`TraceEvent<? extends TracePayload>`. Passing a non-null caller-owned thread-safe observer to
`MetalBackendIntegration.open(configuration, observer)` enables one integration-local trace
stream. Metal retains but never closes the observer; callbacks may be concurrent. The ordinary
one-argument open overload remains the no-trace path and performs no trace allocation, ID, clock,
or callback work.

The traced path reports one final `PREPARE` outcome per enabled prepared unit and one `RUN` outcome
per enabled native invocation. Stream-local event, prepared-unit, and invocation IDs use
independent non-negative sequences; backend/device correlations are both zero. Preparation reports
the neutral custom-kernel or graph-executable route and `NOT_QUERIED`; invocation
reports the same immutable facts and exact mapped native status when one exists. An observer
`RuntimeException`, ID exhaustion, or trace-object construction failure disables later tracing
without changing backend behavior or outward exceptions. An observer `Error` from success
reporting propagates normally; during failure reporting, the existing finalization or run failure
remains primary and receives a distinct acyclic observer `Error` as a suppressed failure without
losing earlier rollback suppression. Events contain only the bounded trace DTO fields and expose
no native path, handle, Tensor value, shape, exception, free-form string, or generic map.

For a low-precision prepared unit, `LowPrecisionTraceMetadata` identifies the selected custom
route and ordered feed-then-target logical dtype tuple. It has no numerical-profile, accumulator
or working-type field; the Model's FLOAT32 low-arithmetic rule is a separate semantic guarantee.
The DTO contains no hypothetical candidate, certificate, accuracy, determinism, or environment
state. Metal exposes no generic trace viewer, serializer, or exporter, so callers consume these
typed DTOs directly through `MetalTraceObserver`.

Metal retains a custom route for an eligible singleton `NEG`. A supported
partition containing any Task-0052 node, exact custom movement/replacement/fold/aggregate node,
non-direct MATMUL, ordering/top-K/arg-extrema node, or Task-0064 convolution/pooling node uses the
fixed shared custom whole-program resource with declared run-owned buffers and one synchronous
Java/native invocation. Existing MPSGraph-compatible nodes may remain nested in that recipe; there
is no per-node fallback. Other supported partitions use one typed whole-partition MPSGraph
executable. The general qualified FLOAT32 MATMUL domain also executes Compiler-generated
explicitly seeded gradients for both operands through authenticated local last-two-axis
transposes. Primitive-closed static FLOAT32 Conv2d/Conv3d graphs execute selected or joint
input, weight and bias cotangents through the shared convolution rule. FOLD2D/FOLD3D
overlap-add is FLOAT32-only, deterministic and output-owned, with no atomics, scratch or host
fallback/repair; FLOAT64/BFLOAT16/FLOAT16 image folds and FOLD_AXIS retain non-overlap.
Pooling gradients close only where every generated occurrence is admitted; their low image
folds remain non-overlapping. Attention and convolution transpose still reject before preparation.
See the [production support matrix](../../backend-guide/production-support.md#metal-production-target)
for actual GPU Conv3d all-role and NN training/persistence evidence. No implicit seeding,
Metal Scan support, software FLOAT64, or unrestricted training support follows.

`MetalLocalWorkloadTuning` publicly exposes opaque candidates, session compatibility, identities,
decision encode/decode, and trial/selected `PartitionPreparation` for the exact two-route
singleton-NEG batch. `MetalCompletePlanTuning` authenticates that Phase-1 decision and exposes the
same opaque operations for exactly one fixed-route complete-plan candidate. Both are retained
integration collaborations, reject foreign/stale/closed use, and perform no execution,
measurement, selection, fallback, or cache input/output themselves. Public `prepareTuned(...)`
adapts them only for one Metal partition; MPSGraph-only, multiple-partition, and mixed-owner plans
remain ineligible. An allowed safe fallback stays with the selected adapter and never changes
owner.

`Engine.standard()` remains the CPU-only convenience and uses this builder ownership path
internally. `AdvancedEngine.takeOwnership(CpuBackendIntegration)` remains CPU-only; explicit
multi-backend construction belongs to ordinary `Engine.Builder`.

See the
[authoritative Engine composition contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition),
[ADR 0015](../../design/decisions/0015-explicit-engine-backend-composition.md), and
[ADR 0016](../../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).


---

[Previous topic](advanced.md) · [Next topic](limitations.md)
