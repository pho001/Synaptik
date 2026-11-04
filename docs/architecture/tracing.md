# Tracing

This document explains the typed [trace](../glossary.md#trace) [model](../glossary.md#term-model) required by
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The current `modules/trace` implementation provides the common event envelope, trace-local model
and [backend](../glossary.md#backend)-execution correlation identifiers, and neutral backend preparation/invocation outcome
payloads. [CPU](../glossary.md#term-cpu) preparation and ordinary [Engine](../glossary.md#term-engine) tuning also expose bounded [producer](../glossary.md#term-producer)-specific
payloads through the open marker. Generic [serialization](../glossary.md#term-serialization), storage and viewing remain unimplemented.

A **trace** is a structured diagnostic record of selected facts, not another execution path.
The **producer** supplies a fact; the **[payload](../glossary.md#artifact-envelope-and-atomic-publication)** is its typed data; the **envelope** adds
identification and stage information. A **correlation identifier** lets a reader associate
related records without retaining the live objects. These descriptions can outlive resources
because a **[detached](../glossary.md#detached-model-results)** record does not borrow their storage.

## [Mental model](../glossary.md#term-mental-model)

```text
producer-owned fact
  -> producer translates it into a detached typed TracePayload
  -> TraceEvent adds event ID, lifecycle phase, level, and monotonic time
  -> a later diagnostic consumer inspects the typed DTO

producer-owned node / value / Tensor / backend / device / prepared-unit / invocation identity
  -> producer assigns the corresponding trace-local correlation value
  -> typed payloads carry that value without importing producer types
```

The [producer](../glossary.md#term-producer) owns the fact, [identity](../glossary.md#term-identity-meanings) assignment, clock, and eventual emission. The [trace](../glossary.md#trace) [module](../glossary.md#term-module)
owns only typed data-transfer objects (DTOs), so it does not traverse [graphs](../glossary.md#graph), execute work, import
producer-layer domain types, or control a [lifecycle](../glossary.md#lifecycle) stage.
Neutral envelopes, identifiers and shared payloads live in Trace. A producer-specific passive [DTO](../glossary.md#data-transfer-object--dto)
may live in its owning module without introducing a reverse Trace [dependency](../glossary.md#term-dependency). [CPU](../glossary.md#term-cpu) preparation,
[Engine](../glossary.md#term-engine) tuning failure resolution and [Metal](../glossary.md#term-metal) [Scan](../glossary.md#term-cumulative-scan) [validation](../glossary.md#term-validation) use that extension point. These
records retain no graph, storage, [Tensor](../glossary.md#tensor), native handle or Throwable.

[Runtime profiling](../glossary.md#runtime-profiling) is one possible producer activity: it passively observes actual execution and
translates selected facts into trace-owned DTOs. Trace remains observational. It does not compare
tuning candidates, select [parameters](../glossary.md#term-parameter), or mutate prepared or [runtime](../glossary.md#term-runtime) state.

## Current event foundation

The implemented public foundation consists of:

- [`TraceEventId`](../glossary.md#trace-event-identity--traceeventid), a non-negative producer-assigned identity whose uniqueness domain is defined by
  the producer;<span data-glossary-navigation="true"> ([identity glossary](../glossary.md#term-identity-meanings))</span>
- [`TracePhase`](../glossary.md#trace-phase--tracephase), with exactly [`COMPILE`](../glossary.md#compile), [`PREPARE`](../glossary.md#prepare), and [`RUN`](../glossary.md#term-run);
- [`TraceLevel`](../glossary.md#trace-level--tracelevel), with [`TRACE`](../glossary.md#trace), `DEBUG`, `INFO`, `WARN`, and `ERROR` classification values;
- [`TracePayload`](../glossary.md#trace-payload--tracepayload), an open method-free marker for [immutable](../glossary.md#term-immutable) typed diagnostic DTOs;
- `TraceEvent<T extends TracePayload>`, the immutable generic envelope;
- the trace-local correlation identifiers described below; and
- `BackendPreparationOutcome` and `BackendInvocationOutcome`, the first concrete payloads.

The current envelope has the following Java record components. “Components” means named
record fields, not a Tensor's axis extents or storage layout:

```java
public record TraceEvent<T extends TracePayload>(
        // The producer supplies correlation and stage facts; this record emits nothing.
        TraceEventId id,
        TracePhase phase,
        TraceLevel level,
        long monotonicNanos,
        // Retain detached diagnostic data, not a live backend resource or Tensor.
        T payload
) {}
```

The [producer](../glossary.md#term-producer) supplies every component. `monotonicNanos` is a monotonic-clock reading in
nanoseconds, not a wall-clock or [epoch](../glossary.md#term-time-epoch) timestamp. Every `long` bit pattern is retained, and only
differences interpreted within the producer's documented clock domain are meaningful. The
envelope does not allocate IDs, read a clock, normalize timestamps, or establish ordering between
different clock domains.

A [monotonic clock](../glossary.md#term-monotonic-clock) measures elapsed intervals in its own clock domain; it does not give a
calendar date. Two different producers' readings are not automatically comparable timestamps.

The record retains its component references without copying them. Its state is shallowly
immutable; because `TracePayload` is open, [payload](../glossary.md#artifact-envelope-and-atomic-publication) implementations must honor the documented
immutability contract themselves. The foundation defines no [serialization](../glossary.md#term-serialization), filtering, storage,
sink, logging, or emission behavior.

## Current [model](../glossary.md#term-model)-correlation identifiers

The `io.github.pho001.synaptik.trace.id` package contains three [immutable](../glossary.md#term-immutable) correlation values:

| [Trace](../glossary.md#trace)-owned type | Correlates | Deliberately does not identify |
|---|---|---|
| [`TraceNodeId`](../glossary.md#trace-node-correlation-identity--tracenodeid) | one computation [occurrence](../glossary.md#term-occurrence) | [operation](../glossary.md#operation) semantics, an output value, or a [runtime](../glossary.md#term-runtime) unit |
| [`TraceValueId`](../glossary.md#trace-value-correlation-identity--tracevalueid) | logical [graph](../glossary.md#graph) data | a [node](../glossary.md#node), public [Tensor](../glossary.md#tensor), storage location, [buffer](../glossary.md#term-buffer), or runtime slot |
| [`TraceTensorId`](../glossary.md#trace-tensor-correlation-identity--tracetensorid) | public Tensor state | a graph node/value, storage address, [device](../glossary.md#term-device) allocation, or runtime [residency](../glossary.md#residency) |

The table separates three [identity](../glossary.md#term-identity-meanings) domains that may share the same numeric value but must not be
substituted for one another. Each type is a one-component record containing a non-negative
`long`; zero through `Long.MAX_VALUE` are valid, and ordinary record equality applies only within
the same [nominal](../glossary.md#term-nominal) type.

“Nominal” means the identifier's type is part of its meaning. A node ID and a value ID that both
contain `0` are no more interchangeable than row number zero and column number zero.

These identifiers are trace-local. The [producer](../glossary.md#term-producer) defines the trace stream or correlation domain
in which a value is meaningful and owns allocation, uniqueness, [lifetime](../glossary.md#term-lifetime), and translation from
its own identity. Translation may preserve a producer ID's numeric value or choose a different
one; numeric equality is not part of the contract. The trace [module](../glossary.md#term-module) provides no allocator,
translator, [registry](../glossary.md#term-registry), mapping table, or producer-object reference.

The records are correlation vocabulary for later typed payloads. They do not themselves [carry](../glossary.md#term-carry) a
diagnostic fact, implement [`TracePayload`](../glossary.md#trace-payload--tracepayload), emit an event, or define [serialization](../glossary.md#term-serialization).

## Current [backend](../glossary.md#backend)-execution diagnostics

Four additional one-component, non-negative `long` records correlate backend execution facts:

| [Trace](../glossary.md#trace)-owned type | Correlates |
|---|---|
| [`TraceBackendId`](../glossary.md#trace-backend-correlation-identity--tracebackendid) | one backend within a [producer](../glossary.md#term-producer)-defined trace stream |
| [`TraceDeviceId`](../glossary.md#trace-device-correlation-identity--tracedeviceid) | one [device](../glossary.md#term-device) within that stream |
| [`TracePreparedUnitId`](../glossary.md#trace-prepared-unit-correlation-identity--tracepreparedunitid) | one finalized backend prepared unit |
| [`TraceInvocationId`](../glossary.md#trace-invocation-correlation-identity--traceinvocationid) | one invocation of a prepared unit |

These are nominally distinct from one another and from the [model](../glossary.md#term-model)-correlation IDs. The producer
owns allocation and the association with its backend objects; no ID contains a backend name,
device [token](../glossary.md#term-token), path, pointer, handle, or producer object.

Among the [immutable](../glossary.md#term-immutable) [`TracePayload`](../glossary.md#trace-payload--tracepayload) records in `io.github.pho001.synaptik.trace.payload`, the two
backend outcome records are:

- `BackendPreparationOutcome` correlates backend, device, and prepared unit; it carries a final
  outcome, neutral route kind, cache fact, and optional native status.
- `BackendInvocationOutcome` additionally correlates one invocation and carries the same outcome,
  route, and optional native status, without a cache claim.

`LowPrecisionTraceMetadata` is a separate [PREPARE](../glossary.md#prepare) [payload](../glossary.md#artifact-envelope-and-atomic-publication) for a selected low-[precision](../glossary.md#term-precision) custom
route. It reports that route and the ordered boundary-feed-then-[target](../glossary.md#term-target) logical [dtype](../glossary.md#data-type--datatype) names. It
has no numerical-profile, [accumulator](../glossary.md#accumulator-type), or working-type field: [FLOAT32](../glossary.md#term-float32) low-arithmetic
working values and accumulators belong to Model semantics, while raw-preserving operations copy
represented words. It carries no [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions) or candidate evidence.

The closed vocabulary is deliberately narrow. Outcomes are `SUCCEEDED` or `FAILED`; routes are
`CUSTOM_KERNEL` or `GRAPH_EXECUTABLE`; and the only
current [cache](../glossary.md#term-cache) fact is `NOT_QUERIED`. `NOT_QUERIED` means the producer performed no cache [lookup](../glossary.md#term-lookup) and
must not be read as a [cache miss](../glossary.md#cache-and-memoization). Route kinds describe the first producer's stable mechanisms, not
a universal [registry](../glossary.md#term-registry) for every backend.

[`TraceNativeStatus`](../glossary.md#trace-native-status--tracenativestatus) retains an exact signed [native code](../glossary.md#native-numerical-library-and-instruction-architecture) with a trace-owned neutral kind. Code
`0` is valid exactly with `SUCCESS`; every nonzero signed code is retained unchanged with a
non-success kind. Codes remain meaningful only in their producing backend stream, so Trace
defines no global numeric mapping. A successful outcome requires present native success. A failed
outcome permits an empty native status when failure occurs before a native status return, or a
present non-success status.

The outcome records contain no free-form text, generic map, `Throwable`, path, pointer, handle,
[tensor](../glossary.md#tensor) value, [shape](../glossary.md#shape), byte [extent](../glossary.md#term-extent), [fingerprint](../glossary.md#term-fingerprint), or producer object. Trace defines the immutable data
only; it does not allocate these IDs or produce, emit, store, or consume the events.

## [Lifecycle](../glossary.md#lifecycle) phase and [backend](../glossary.md#backend) diagnostics

[`TracePhase`](../glossary.md#trace-phase--tracephase) answers when a fact occurred:

```text
COMPILE  -> capture, validation, transformation, ownership, and logical planning
PREPARE  -> backend preparation, route selection, and executable-state construction
RUN      -> invocation, execution, transfer, materialization, and publication
```

Backend is not a fourth phase. A backend may produce facts while preparing a [partition](../glossary.md#partition) and while
executing it, so a backend event uses [`PREPARE`](../glossary.md#prepare) or [`RUN`](../glossary.md#term-run) according to when that fact occurred.
Keeping lifecycle stage separate from [producer](../glossary.md#term-producer) role preserves the decision boundary that the
event describes.

[`TraceLevel`](../glossary.md#trace-level--tracelevel) classifies detail or severity only. Its order does not define a filtering threshold,
sink policy, logging integration, failure response, or process-exit behavior.

## Remaining planned [payload](../glossary.md#artifact-envelope-and-atomic-publication) families

The implemented [backend](../glossary.md#backend) outcomes and selected structural/low-[precision](../glossary.md#term-precision) diagnostics are
intentionally bounded. The following broader payload families remain conceptual:

- **[Compile](../glossary.md#compile) payloads** for graph capture, transformations, ownership scoring, partition creation,
  logical memory, and publication planning.<span data-glossary-navigation="true"> ([graph capture glossary](../glossary.md#capture-graph-ports-and-reconstruction); [ownership glossary](../glossary.md#term-ownership-meanings); [partition glossary](../glossary.md#partition); [publication glossary](../glossary.md#term-publication); [planning glossary](../glossary.md#term-planning))</span>
- **Broader [prepare](../glossary.md#prepare) payloads** beyond current outcomes and structural/low-precision facts.
- **Broader [run](../glossary.md#term-run) payloads** beyond current outcomes and invocation-plan facts, including transfers,
  materialization, step boundaries, and publication.<span data-glossary-navigation="true"> ([materialization glossary](../glossary.md#materialization))</span>
- **Broader backend payloads** for availability, capability, kernels, storage, and other
  backend-owned diagnostic facts during the applicable lifecycle phase.<span data-glossary-navigation="true"> ([kernels glossary](../glossary.md#kernel); [lifecycle glossary](../glossary.md#lifecycle))</span>

These families must use [detached](../glossary.md#detached-model-results) typed DTOs rather than expose [producer](../glossary.md#term-producer) objects. Neutral shared
DTOs belong to [Trace](../glossary.md#trace); producer-specific DTOs remain in their owning [module](../glossary.md#term-module).

## Planned correlation and attributes

[`TraceEventId`](../glossary.md#trace-event-identity--traceeventid), the three [model](../glossary.md#term-model)-correlation IDs, and the [backend](../glossary.md#backend), [device](../glossary.md#term-device), prepared-unit, and
invocation IDs are current. [Trace](../glossary.md#trace)-local identifiers for partitions, schedules, runs, and any other
later domain remain planned until their [producer](../glossary.md#term-producer) contracts are stable. Like the current
correlations, later IDs must avoid importing [identities](../glossary.md#term-identity-meanings) or object references from [planning](../glossary.md#term-planning),
[runtime](../glossary.md#term-runtime), backend-contract, or backend [modules](../glossary.md#term-module).

Typed backend-specific attributes also remain planned. They will be a constrained escape hatch
for facts that a shared [payload](../glossary.md#artifact-envelope-and-atomic-publication) cannot predict, not the primary event model.

## Why not `Map<String,String>`

An unstructured string map as the primary [trace](../glossary.md#trace) [model](../glossary.md#term-model) would hide required fields, discard numeric
and [boolean](../glossary.md#data-type--datatype) types, push parsing into every consumer, and make [schema](../glossary.md#term-schema) changes difficult to
validate. Typed payloads preserve meaning and let consumers handle known diagnostic categories
explicitly. A later typed attribute escape hatch will complement those payloads without replacing
them.

## Diagnostic scenario

A [producer](../glossary.md#term-producer) can wrap `BackendPreparationOutcome` in a `TraceEvent` with phase [`PREPARE`](../glossary.md#prepare) after a
prepared unit is finalized, then wrap each `BackendInvocationOutcome` with phase [`RUN`](../glossary.md#term-run). The shared
DTOs preserve the prepared-unit and invocation correlations, selected neutral route,
and exact native outcome without exposing producer state. [Trace](../glossary.md#trace) itself still emits nothing: the
producer supplies the event ID, monotonic timestamp, correlations, and event delivery. A single
string such as `"backend failure"` would lose these typed facts and could incorrectly suggest that
[runtime](../glossary.md#term-runtime) changed a prepare-time decision.

## Current [Metal](../glossary.md#term-metal) [producer](../glossary.md#term-producer)

`MetalBackendIntegration.open(configuration, observer)` is the first concrete producer of the
[backend](../glossary.md#backend) outcome DTOs. One traced integration defines one stream with fixed backend/[device](../glossary.md#term-device)
correlations `0`, independent non-negative event/prepared-unit/invocation sequences beginning at
zero, and `System.nanoTime()` timestamps. The caller owns the [`MetalTraceObserver`](../glossary.md#metal-trace-observer--metaltraceobserver); callbacks may
be concurrent and Metal never closes it. The existing one-[argument](../glossary.md#term-argument) `open(configuration)` path
creates no producer, [payload](../glossary.md#artifact-envelope-and-atomic-publication), ID, clock read, or [callback](../glossary.md#term-callback).

After Metal fixes a route, it emits one structural [`PREPARE`](../glossary.md#prepare) event before native
finalization. A low-[precision](../glossary.md#term-precision) program also emits `LowPrecisionTraceMetadata` before the
`PREPARE` outcome while tracing remains enabled. Each route-specific
native invocation emits one bounded invocation-plan [`RUN`](../glossary.md#term-run) event before native execution and one
`RUN` outcome afterward. Preparation always reports `NOT_QUERIED` because Metal performs no [cache](../glossary.md#term-cache)
[lookup](../glossary.md#term-lookup), including when an outer [owner](../glossary.md#term-owner) supplied a decision. A returned native status is translated
to the closed [trace](../glossary.md#trace) vocabulary while retaining its exact signed code; a Java-side failure before
native status return leaves native status empty.

A structural PREPARE or invocation-plan callback failure, including [observer](../glossary.md#term-observer) `Error`, permanently
disables tracing and is contained without aborting native finalization or execution. ID exhaustion,
event/[DTO](../glossary.md#data-transfer-object--dto) construction
failure, or an observer `RuntimeException` from outcome reporting likewise disables later events
without changing backend work, results, [rollback](../glossary.md#term-rollback), or outward exceptions. An observer `Error`
propagates only from successful outcome reporting. During failure outcome reporting, the existing
backend/finalization/run failure remains primary; a distinct observer `Error` is attached as an
acyclic suppressed failure without disabling tracing or disturbing prior rollback suppression.
Events expose no library path, session/device [token](../glossary.md#term-token), address or handle, [Tensor](../glossary.md#tensor)/storage value,
[scalar](../glossary.md#term-scalar), per-[buffer](../glossary.md#term-buffer) byte [extent](../glossary.md#term-extent), cache or workload [fingerprint](../glossary.md#term-fingerprint), thread [identity](../glossary.md#term-identity-meanings), exception,
free-form string, or generic map. The producer adds no [Engine](../glossary.md#term-engine) production behavior, native
[ABI](../glossary.md#term-application-binary-interface-abi)/[schema](../glossary.md#term-schema)/export, capability, route, or [lifecycle](../glossary.md#lifecycle) change.

## Why [trace](../glossary.md#trace) stays a [dependency leaf](../glossary.md#term-dependency-leaf)

Trace [producers](../glossary.md#term-producer) exist throughout the architecture. If `modules/trace` depended on [model](../glossary.md#term-model),
[planning](../glossary.md#term-planning), [compiler](../glossary.md#term-compiler), [prepare](../glossary.md#prepare), [runtime](../glossary.md#term-runtime), [engine](../glossary.md#term-engine), or concrete [backends](../glossary.md#backend), using trace types could
introduce reverse dependencies or cycles.

Keeping `modules/trace` as a DTO-only dependency leaf means Trace needs no higher execution
layer. This is different from a Tensor leaf, which has no producing expression. Producers
construct diagnostic records and consumers retain those records, while resource ownership and
business logic remain in the producing layer. The record retains component references rather
than copying arbitrary payload objects; producers must supply immutable detached payloads.
Architecture tests should enforce this dependency boundary; see
[Dependency Rules](dependency-rules.md#trace-is-a-dependency-leaf) and
[ADR 0003: Typed trace DTOs](../design/decisions/0003-typed-trace-dtos.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.
