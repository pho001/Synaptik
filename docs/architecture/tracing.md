# Tracing

This document explains the typed trace model required by
[`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The current `modules/trace` implementation provides the common event envelope only. Concrete
compile, prepare, run, and backend payload families, trace-local correlation identifiers beyond
the event ID, typed backend attributes, serialization, and event emission remain planned.

## Mental model

```text
producer-owned fact
  -> producer translates it into a trace-owned TracePayload
  -> TraceEvent adds event ID, lifecycle phase, level, and monotonic time
  -> a later diagnostic consumer inspects the typed DTO
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

`TraceEventId` is current. Additional trace-local IDs for nodes, values, tensors, partitions,
backends, devices, and prepared units remain planned. Their purpose is to let later events
correlate related facts without importing identities or object references from model, planning,
runtime, or backend modules. Producers will translate their identities into those trace-owned
forms.

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
