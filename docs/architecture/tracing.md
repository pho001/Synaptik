# Tracing

This document explains the typed trace model required by
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The current `modules/trace` implementation provides the common event envelope and trace-local
node, logical-value, and public-Tensor correlation identifiers. Concrete compile, prepare, run,
and backend payload families, other correlation domains, typed backend attributes, serialization,
and event emission remain planned.

## Mental model

```text
producer-owned fact
  -> producer translates it into a trace-owned TracePayload
  -> TraceEvent adds event ID, lifecycle phase, level, and monotonic time
  -> a later diagnostic consumer inspects the typed DTO

producer-owned node / value / Tensor identity
  -> producer assigns the corresponding trace-local correlation value
  -> later typed payloads can carry that value without importing model types
```

The producer owns the fact, identity assignment, clock, and eventual emission. The trace module
owns only typed data-transfer objects (DTOs), so it does not traverse graphs, execute work, import
producer-layer domain types, or control a lifecycle stage.

## Current event foundation

The implemented public foundation consists of:

- `TraceEventId`, a non-negative producer-assigned identity whose uniqueness domain is defined by
  the producer;
- `TracePhase`, with exactly `COMPILE`, `PREPARE`, and `RUN`;
- `TraceLevel`, with `TRACE`, `DEBUG`, `INFO`, `WARN`, and `ERROR` classification values;
- `TracePayload`, an open method-free marker for immutable typed diagnostic DTOs; and
- `TraceEvent<T extends TracePayload>`, the immutable generic envelope.

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

## Planned payload families

The following payload families remain conceptual; no concrete payload record is implemented yet:

- **Compile payloads** for graph capture, transformations, ownership scoring, partition creation,
  logical memory, and publication planning.
- **Prepare payloads** for backend preparation, selected routes, prepared partitions and units,
  prepared memory, and prepared schedules.
- **Run payloads** for invocation, execution, transfers, materialization, step boundaries, and
  publication.
- **Backend payloads** for availability, capability, routes, kernels, storage, and other
  backend-owned diagnostic facts during the applicable lifecycle phase.

These families will use trace-owned DTOs rather than expose producer objects. Their exact fields
must follow the later producer-layer contracts and are deliberately not selected by the current
envelope task.

## Planned correlation and attributes

`TraceEventId`, `TraceNodeId`, `TraceValueId`, and `TraceTensorId` are current. Trace-local
identifiers for partitions, backends, devices, prepared units, schedules, runs, and any other
later domain remain planned until their producer contracts are stable. Like the current model
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

## Conceptual diagnostic scenario

The following sequence is conceptual because its concrete payload records and emission APIs are
not implemented. A compile payload could record that a trace-local node was assigned to CPU. A
later CPU prepare payload could record the selected route with phase `PREPARE`, and a run payload
could record execution of the prepared unit with phase `RUN`. This sequence preserves which
lifecycle stage made each decision. A single string such as `"cpu fallback"` would lose those
typed facts and could incorrectly suggest that runtime changed compile-time ownership.

## Why trace stays a dependency leaf

Trace producers exist throughout the architecture. If `modules/trace` depended on model,
planning, compiler, prepare, runtime, engine, or concrete backends, using trace types could
introduce reverse dependencies or cycles.

Keeping `modules/trace` as a DTO-only leaf lets later layers share diagnostic contracts while
ownership and business logic remain in the producing layer. Architecture tests should enforce
this leaf boundary; see [Dependency Rules](dependency-rules.md) and
[ADR 0003: Typed trace DTOs](../design/decisions/0003-typed-trace-dtos.md).
