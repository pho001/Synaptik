# Tracing

This document explains the typed trace model required by
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The current `modules/trace` implementation provides the common event envelope, trace-local model
and backend-execution correlation identifiers, and neutral backend preparation/invocation outcome
payloads. Other compile, prepare, run, and backend payload families, typed backend attributes,
serialization, and event emission remain planned.

## Mental model

```text
producer-owned fact
  -> producer translates it into a trace-owned TracePayload
  -> TraceEvent adds event ID, lifecycle phase, level, and monotonic time
  -> a later diagnostic consumer inspects the typed DTO

producer-owned node / value / Tensor / backend / device / prepared-unit / invocation identity
  -> producer assigns the corresponding trace-local correlation value
  -> typed payloads carry that value without importing producer types
```

The producer owns the fact, identity assignment, clock, and eventual emission. The trace module
owns only typed data-transfer objects (DTOs), so it does not traverse graphs, execute work, import
producer-layer domain types, or control a lifecycle stage.

Runtime profiling is one possible producer activity: it passively observes actual execution and
translates selected facts into trace-owned DTOs. Trace remains observational. It does not compare
tuning candidates, select parameters, or mutate prepared or runtime state.

## Current event foundation

The implemented public foundation consists of:

- `TraceEventId`, a non-negative producer-assigned identity whose uniqueness domain is defined by
  the producer;
- `TracePhase`, with exactly `COMPILE`, `PREPARE`, and `RUN`;
- `TraceLevel`, with `TRACE`, `DEBUG`, `INFO`, `WARN`, and `ERROR` classification values;
- `TracePayload`, an open method-free marker for immutable typed diagnostic DTOs;
- `TraceEvent<T extends TracePayload>`, the immutable generic envelope;
- the trace-local correlation identifiers described below; and
- `BackendPreparationOutcome` and `BackendInvocationOutcome`, the first concrete payloads.

The current envelope has this exact component shape:

```java
public record TraceEvent<T extends TracePayload>(
        TraceEventId id,
        TracePhase phase,
        TraceLevel level,
        long monotonicNanos,
        T payload
) {}
```

The producer supplies every component. `monotonicNanos` is a monotonic-clock reading in
nanoseconds, not a wall-clock or epoch timestamp. Every `long` bit pattern is retained, and only
differences interpreted within the producer's documented clock domain are meaningful. The
envelope does not allocate IDs, read a clock, normalize timestamps, or establish ordering between
different clock domains.

The record retains its component references without copying them. Its state is shallowly
immutable; because `TracePayload` is open, payload implementations must honor the documented
immutability contract themselves. The foundation defines no serialization, filtering, storage,
sink, logging, or emission behavior.

## Current model-correlation identifiers

The `io.github.pho001.synaptik.trace.id` package contains three immutable correlation values:

| Trace-owned type | Correlates | Deliberately does not identify |
|---|---|---|
| `TraceNodeId` | one computation occurrence | operation semantics, an output value, or a runtime unit |
| `TraceValueId` | logical graph data | a node, public Tensor, storage location, buffer, or runtime slot |
| `TraceTensorId` | public Tensor state | a graph node/value, storage address, device allocation, or runtime residency |

The table separates three identity domains that may share the same numeric value but must not be
substituted for one another. Each type is a one-component record containing a non-negative
`long`; zero through `Long.MAX_VALUE` are valid, and ordinary record equality applies only within
the same nominal type.

These identifiers are trace-local. The producer defines the trace stream or correlation domain
in which a value is meaningful and owns allocation, uniqueness, lifetime, and translation from
its own identity. Translation may preserve a producer ID's numeric value or choose a different
one; numeric equality is not part of the contract. The trace module provides no allocator,
translator, registry, mapping table, or producer-object reference.

The records are correlation vocabulary for later typed payloads. They do not themselves carry a
diagnostic fact, implement `TracePayload`, emit an event, or define serialization.

## Current backend-execution diagnostics

Four additional one-component, non-negative `long` records correlate backend execution facts:

| Trace-owned type | Correlates |
|---|---|
| `TraceBackendId` | one backend within a producer-defined trace stream |
| `TraceDeviceId` | one device within that stream |
| `TracePreparedUnitId` | one finalized backend prepared unit |
| `TraceInvocationId` | one invocation of a prepared unit |

These are nominally distinct from one another and from the model-correlation IDs. The producer
owns allocation and the association with its backend objects; no ID contains a backend name,
device token, path, pointer, handle, or producer object.

The `io.github.pho001.synaptik.trace.payload` package contains two immutable `TracePayload`
records:

- `BackendPreparationOutcome` correlates backend, device, and prepared unit; it carries a final
  outcome, numerical profile, neutral route kind, cache fact, and optional native status.
- `BackendInvocationOutcome` additionally correlates one invocation and carries the same outcome,
  profile, route, and optional native status, without a cache claim.

The closed vocabulary is deliberately narrow. Outcomes are `SUCCEEDED` or `FAILED`; profiles are
`STRICT_IEEE` or `ACCELERATOR`; routes are `CUSTOM_KERNEL` or `GRAPH_EXECUTABLE`; and the only
current cache fact is `NOT_QUERIED`. `NOT_QUERIED` means the producer performed no cache lookup and
must not be read as a cache miss. Route kinds describe the first producer's stable mechanisms, not
a universal registry for every backend.

`TraceNativeStatus` retains an exact signed native code with a trace-owned neutral kind. Code
`0` is valid exactly with `SUCCESS`; every nonzero signed code is retained unchanged with a
non-success kind. Codes remain meaningful only in their producing backend stream, so Trace
defines no global numeric mapping. A successful outcome requires present native success. A failed
outcome permits an empty native status when failure occurs before a native status return, or a
present non-success status.

These records contain no free-form text, generic map, `Throwable`, path, pointer, handle, tensor
value, shape, byte extent, fingerprint, or producer object. Trace defines the immutable data only;
it does not allocate these IDs or produce, emit, store, or consume the events.

## Lifecycle phase and backend diagnostics

`TracePhase` answers when a fact occurred:

```text
COMPILE  -> capture, validation, transformation, ownership, and logical planning
PREPARE  -> backend preparation, route selection, and executable-state construction
RUN      -> invocation, execution, transfer, materialization, and publication
```

Backend is not a fourth phase. A backend may produce facts while preparing a partition and while
executing it, so a backend event uses `PREPARE` or `RUN` according to when that fact occurred.
Keeping lifecycle stage separate from producer role preserves the decision boundary that the
event describes.

`TraceLevel` classifies detail or severity only. Its order does not define a filtering threshold,
sink policy, logging integration, failure response, or process-exit behavior.

## Remaining planned payload families

The implemented backend preparation and invocation outcomes are intentionally bounded. The
following broader payload families remain conceptual:

- **Compile payloads** for graph capture, transformations, ownership scoring, partition creation,
  logical memory, and publication planning.
- **Broader prepare payloads** beyond one backend prepared-unit finalization outcome.
- **Broader run payloads** beyond one backend invocation outcome, including transfers,
  materialization, step boundaries, and publication.
- **Broader backend payloads** for availability, capability, kernels, storage, and other
  backend-owned diagnostic facts during the applicable lifecycle phase.

These families must continue to use trace-owned DTOs rather than expose producer objects. Their
exact fields remain deferred until the relevant producer-layer contracts are stable.

## Planned correlation and attributes

`TraceEventId`, the three model-correlation IDs, and the backend, device, prepared-unit, and
invocation IDs are current. Trace-local identifiers for partitions, schedules, runs, and any other
later domain remain planned until their producer contracts are stable. Like the current
correlations, later IDs must avoid importing identities or object references from planning,
runtime, backend-contract, or backend modules.

Typed backend-specific attributes also remain planned. They will be a constrained escape hatch
for facts that a shared payload cannot predict, not the primary event model.

## Why not `Map<String,String>`

An unstructured string map as the primary trace model would hide required fields, discard numeric
and boolean types, push parsing into every consumer, and make schema changes difficult to
validate. Typed payloads preserve meaning and let consumers handle known diagnostic categories
explicitly. A later typed attribute escape hatch will complement those payloads without replacing
them.

## Diagnostic scenario

A producer can wrap `BackendPreparationOutcome` in a `TraceEvent` with phase `PREPARE` after a
prepared unit is finalized, then wrap each `BackendInvocationOutcome` with phase `RUN`. The shared
DTOs preserve the prepared-unit and invocation correlations, selected profile and neutral route,
and exact native outcome without exposing producer state. Trace itself still emits nothing: the
producer supplies the event ID, monotonic timestamp, correlations, and event delivery. A single
string such as `"backend failure"` would lose these typed facts and could incorrectly suggest that
runtime changed a prepare-time decision.

## Current Metal producer

`MetalBackendIntegration.open(configuration, observer)` is the first concrete producer of the
backend outcome DTOs. One traced integration defines one stream with fixed backend/device
correlations `0`, independent non-negative event/prepared-unit/invocation sequences beginning at
zero, and `System.nanoTime()` timestamps. The caller owns the `MetalTraceObserver`; callbacks may
be concurrent and Metal never closes it. The existing one-argument `open(configuration)` path
creates no producer, payload, ID, clock read, or callback.

After Metal fixes a profile and route, finalization emits one `PREPARE` outcome while tracing
remains enabled. Each route-specific native invocation emits one `RUN` outcome. Preparation always
reports `NOT_QUERIED` because Metal performs no cache lookup, including when an outer owner supplied
a decision. A returned native status is translated to the closed trace vocabulary while retaining
its exact signed code; a Java-side failure before native status return leaves native status empty.

ID exhaustion, event/DTO construction failure, or an observer `RuntimeException` permanently
disables later events for that integration without changing backend work, results, rollback, or
outward exceptions. An observer `Error` from success reporting propagates normally. During failure
reporting, the existing backend/finalization/run failure remains primary; a distinct observer
`Error` is attached as an acyclic suppressed failure without disabling tracing or disturbing prior
rollback suppression. Events expose no library path, session/device token, address or handle,
Tensor/storage value, scalar, shape, byte extent, cache or workload fingerprint, thread identity,
exception, free-form string, or generic map. The producer
adds no Engine production behavior, native ABI/schema/export, capability, route, or lifecycle
change.

## Why trace stays a dependency leaf

Trace producers exist throughout the architecture. If `modules/trace` depended on model,
planning, compiler, prepare, runtime, engine, or concrete backends, using trace types could
introduce reverse dependencies or cycles.

Keeping `modules/trace` as a DTO-only leaf lets later layers share diagnostic contracts while
ownership and business logic remain in the producing layer. Architecture tests should enforce
this leaf boundary; see [Dependency Rules](dependency-rules.md) and
[ADR 0003: Typed trace DTOs](../design/decisions/0003-typed-trace-dtos.md).
