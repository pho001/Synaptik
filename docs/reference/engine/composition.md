# Explicit [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

Explicit composition means the caller opens [backend](../../glossary.md#backend) integrations and transfers them into one
[Engine](../../glossary.md#term-engine). A backend **[owner](../../glossary.md#term-owner)** is the implementation assigned a [graph](../../glossary.md#graph) [partition](../../glossary.md#partition), not the owner of
the caller's input memory. A **cross-owner transfer** moves a value between those implementations;
it is planned before a [run](../../glossary.md#term-run), not an automatic retry on another backend.

<a id="current-explicit-cpu-and-metal-composition"></a>
## Current explicit [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) composition

The public composition [shape](../../glossary.md#shape) is:

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

`Engine.Builder` is a public single-use `AutoCloseable` construction [owner](../../glossary.md#term-owner). Its concrete
`takeOwnership(CpuBackendIntegration)` and
`takeOwnership(MetalBackendIntegration)` overloads transfer a non-null integration at method
entry, including when later registration [validation](../../glossary.md#term-validation) fails. Callers must not close or reuse an
integration after invoking the overload. A successful `build()` atomically transfers the ordered
[backend](../../glossary.md#backend) set to the returned [Engine](../../glossary.md#term-engine). Builder close and failed build [release](../../glossary.md#term-release) accepted integrations
in reverse registration order; failed build is terminal.

Because the overload [parameters](../../glossary.md#term-parameter) are backend-owned public types, Engine publishes CPU and Metal as
[API](../../glossary.md#term-application-programming-interface-api)-visible dependencies. Backend types remain confined to explicit construction and do not appear
in [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [run](../../glossary.md#term-run), handle, or result [signatures](../../glossary.md#term-signature).

Registration [captures](../../glossary.md#term-capture-meanings) one [`BackendId`](../../glossary.md#backend-identity--backendid), stable capability provider, and [immutable](../../glossary.md#term-immutable) point-in-time
availability [snapshot](../../glossary.md#term-snapshot). Their backend IDs must be equal, and an equal duplicate ID is rejected
without replacing the first entry. The snapshot remains fixed for that Engine's [lifetime](../../glossary.md#term-lifetime).
Registration order supplies deterministic compile-time [Planning](../../glossary.md#term-planning) input; it is not a [runtime](../../glossary.md#term-runtime)
fallback order.

[Compilation](../../glossary.md#compiler-engine-and-execution-stage-names) chooses owner [identities](../../glossary.md#term-identity-meanings) from that fixed inventory. Cold preparation requires a
non-empty plan, resolves every distinct [partition](../../glossary.md#partition) owner, and validates every required directed
cross-owner transfer before backend analysis. A missing owner, zero-partition plan, or unsupported
transfer fails without implicit CPU fallback, retry, owner substitution, or conversion.

Shared Prepare analyzes and finalizes partitions in plan order, assigns one logical slot and one
deterministic representation position per participating owner, and obtains exact backend physical
creators. Engine assembles execution occurrences in partition order, inserting one immutable
transfer per logical value and distinct destination owner immediately before that owner's first
consumer, then appends the dense [publication](../../glossary.md#term-publication) suffix. Current CPU/Metal transfer is direct native
[host](../../glossary.md#term-host)-staged in either direction for all seven carriers over exact [rank](../../glossary.md#term-rank)-0..16 static [canonical](../../glossary.md#term-canonical-form) or
positive-[stride](../../glossary.md#term-stride) non-overlapping [layouts](../../glossary.md#layout) with checked physical spans.

One outward [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) retains direct non-owning adapters in caller-input and [publication occurrence](../../glossary.md#publication-occurrence) order
beside the inward Runtime execution. Run uses each captured input adapter for caller-host ingress,
and the result uses each publication's captured adapter for backend-owned host [materialization](../../glossary.md#materialization).
These outer calls do not re-query the [registry](../../glossary.md#term-registry). Engine shutdown closes retained results first,
then ordinary and advanced prepared handles in reverse preparation-publication
order, then integrations. Closing a prepared owner rejects later runs without closing an
already returned result; an already leased Runtime run may finish, and its final [lease](../../glossary.md#term-lease) release
performs deferred prepared-resource cleanup.

`MetalBackendConfiguration` and `MetalBackendIntegration` are public Metal-owned types. Metal
validates and snapshots the explicit absolute native-library path and acquires one system-default-
[device](../../glossary.md#term-device) native context and [command queue](../../glossary.md#term-command-queue) before the integration is transferred. The configuration
has no physical-device selector. Engine neither parses the native-library path nor owns a duplicate
Metal configuration. The integration supplies partition preparation, retained
`localWorkloadTuning()` and `completePlanTuning()` collaborations, physical creation contribution,
exact transfer endpoints, host ingress, materialization, execution, and close for that exact
context in addition to capability.

One Engine can own at most one Metal integration because its registry rejects a duplicate Metal
`BackendId`. `BackendDeviceId(metal, "default")` is the sole abstract availability slot, not a
stable physical-device [fingerprint](../../glossary.md#term-fingerprint) or selector; Planning proves eligibility and retains the
backend owner rather than a selected Metal device. Separate Engines may open separate default
contexts, but the API provides no enumeration, explicit selection, multi-device scheduling,
cross-device migration/coherence, failover, or hot-plug behavior.

[`MetalTraceObserver`](../../glossary.md#metal-trace-observer--metaltraceobserver) is a public functional interface whose sole abstract method receives
`TraceEvent<? extends TracePayload>`. Passing a non-null caller-owned [thread-safe](../../glossary.md#term-thread-safe) [observer](../../glossary.md#term-observer) to
`MetalBackendIntegration.open(configuration, observer)` enables one integration-local [trace](../../glossary.md#trace)
stream. Metal retains but never closes the observer; callbacks may be concurrent. The ordinary
one-[argument](../../glossary.md#term-argument) open overload remains the no-trace path and performs no trace allocation, ID, clock,
or [callback](../../glossary.md#term-callback) work.

The traced path reports one final `PREPARE` outcome per enabled prepared unit and one `RUN` outcome
per enabled native invocation. Stream-local event, prepared-unit, and invocation IDs use
independent non-negative sequences; backend/device correlations are both zero. Preparation reports
the neutral custom-[kernel](../../glossary.md#kernel) or [graph](../../glossary.md#graph)-executable route and `NOT_QUERIED`; invocation
reports the same immutable facts and exact mapped native status when one exists. An observer
`RuntimeException`, ID exhaustion, or trace-object construction failure disables later tracing
without changing backend behavior or outward exceptions. An observer `Error` from success
reporting propagates normally; during failure reporting, the existing finalization or run failure
remains primary and receives a distinct acyclic observer `Error` as a suppressed failure without
losing earlier [rollback](../../glossary.md#term-rollback) suppression. Events contain only the bounded trace [DTO](../../glossary.md#data-transfer-object--dto) fields and expose
no native path, handle, [Tensor](../../glossary.md#tensor) value, shape, exception, free-form string, or generic map.

For a low-[precision](../../glossary.md#term-precision) prepared unit, `LowPrecisionTraceMetadata` identifies the selected custom
route and ordered feed-then-[target](../../glossary.md#term-target) logical [dtype](../../glossary.md#data-type--datatype) tuple. It has no numerical-profile, [accumulator](../../glossary.md#accumulator-type)
or working-type field; the [Model](../../glossary.md#term-model)'s [FLOAT32](../../glossary.md#term-float32) low-arithmetic rule is a separate semantic guarantee.
The DTO contains no hypothetical candidate, [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), accuracy, determinism, or environment
state. Metal exposes no generic trace viewer, serializer, or exporter, so callers consume these
typed DTOs directly through `MetalTraceObserver`.

Metal retains a custom route for an eligible [singleton](../../glossary.md#term-singleton) `NEG`. A supported
partition containing any Task-0052 [node](../../glossary.md#node), exact custom movement/replacement/fold/aggregate node,
non-direct [MATMUL](../../glossary.md#matrix-multiplication--matmul), ordering/top-K/arg-[extrema](../../glossary.md#saturation-pole-and-extremum) node, or Task-0064 [convolution](../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../glossary.md#term-pooling) node uses the
fixed shared custom whole-program resource with declared run-owned [buffers](../../glossary.md#term-buffer) and one [synchronous](../../glossary.md#term-synchronous)
Java/native invocation. Existing [MPSGraph](../../glossary.md#term-mpsgraph)-compatible nodes may remain nested in that recipe; there
is no per-node fallback. Other supported partitions use one typed whole-partition MPSGraph
executable. The general qualified FLOAT32 MATMUL domain also executes [Compiler](../../glossary.md#term-compiler)-generated
explicitly seeded [gradients](../../glossary.md#term-gradient) for both operands through authenticated local last-two-[axis](../../glossary.md#term-axis)
transposes. Primitive-closed static FLOAT32 [Conv2d](../../glossary.md#convolution--conv2d)/[Conv3d](../../glossary.md#conv3d--ncdhw) graphs execute selected or joint
input, [weight](../../glossary.md#term-weight) and [bias](../../glossary.md#term-bias) cotangents through the shared convolution rule. FOLD2D/FOLD3D
overlap-add is FLOAT32-only, deterministic and output-owned, with no atomics, scratch or host
fallback/repair; [FLOAT64](../../glossary.md#term-float64)/[BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) image folds and FOLD_AXIS retain non-overlap.
Pooling gradients close only where every generated [occurrence](../../glossary.md#term-occurrence) is admitted; their low image
folds remain non-overlapping. Attention and convolution [transpose](../../glossary.md#term-transpose) still reject before preparation.
See the [production support matrix](../../backend-guide/production-support.md#metal-production-target)<span data-glossary-navigation="true"> ([support matrix glossary](../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../../glossary.md#term-review-matrix))</span>
for actual [GPU](../../glossary.md#term-gpu) Conv3d all-role and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) training/[persistence](../../glossary.md#term-persistence) evidence. No implicit seeding,
Metal [Scan](../../glossary.md#term-cumulative-scan) support, software FLOAT64, or unrestricted training support follows.

`MetalLocalWorkloadTuning` publicly exposes [opaque](../../glossary.md#term-opaque) candidates, session compatibility, identities,
decision encode/decode, and trial/selected `PartitionPreparation` for the exact two-route
singleton-NEG [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). `MetalCompletePlanTuning` authenticates that Phase-1 decision and exposes the
same opaque operations for exactly one fixed-route [complete-plan candidate](../../glossary.md#complete-plan-candidate). Both are retained
integration collaborations, reject foreign/stale/closed use, and perform no execution,
measurement, selection, fallback, or [cache](../../glossary.md#term-cache) input/output themselves. Public `prepareTuned(...)`
adapts them only for one Metal partition; MPSGraph-only, multiple-partition, and mixed-owner plans
remain ineligible. An allowed safe fallback stays with the selected adapter and never changes
owner.

`Engine.standard()` remains the CPU-only convenience and uses this builder [ownership](../../glossary.md#term-ownership-meanings) path
internally. `AdvancedEngine.takeOwnership(CpuBackendIntegration)` remains CPU-only; explicit
multi-backend construction belongs to ordinary `Engine.Builder`.

See the
[authoritative Engine composition contract](../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition),
[ADR 0015](../../design/decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>, and
[ADR 0016](../../design/decisions/0016-cpu-metal-mixed-owner-schedule.md).


---

[Previous topic](advanced.md) · [Next topic](limitations.md)
