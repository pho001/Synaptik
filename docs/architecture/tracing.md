# Tracing

This document explains the typed trace model required by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

## Typed diagnostic DTOs

`modules/trace` contains diagnostic data-transfer objects only. A trace producer in the compiler, prepare layer, runtime, or a backend maps its local state into trace DTOs; the trace module does not traverse graphs, execute work, or import producer-layer domain types.

Typed DTOs make event schemas explicit, keep consumers independent of producer internals, and allow diagnostic data to be serialized and evolved deliberately.

## Event envelope

Every event has a common envelope around a typed payload. Conceptually:

```java
public record TraceEvent<T extends TracePayload>(
        TraceEventId id,
        TracePhase phase,
        TraceLevel level,
        long monotonicNanos,
        T payload
) {}
```

The envelope provides event identity, phase, severity, and ordering time. The payload carries the phase-specific facts.

## Payload families

The primary payload families are:

- **Compile payloads** for graph capture, optimization passes and rewrites, ownership scoring, partition creation, logical memory, and publication planning.
- **Prepare payloads** for backend preparation, prepared partitions and units, selected backend routes, prepared memory, and prepared schedules.
- **Run payloads** for execution and step boundaries, transfers, materialization, and publication.
- **Backend payloads** for backend availability, capability, routes, kernels, and storage details.

The families distinguish lifecycle phases without forcing trace consumers to interpret arbitrary strings.

## Trace-local identifiers

Trace events use trace-local identifiers such as `TraceEventId`, `TraceNodeId`, `TraceValueId`, `TraceTensorId`, `TracePartitionId`, `TraceBackendId`, and `TraceUnitId`.

These identifiers let events correlate related facts without importing IDs or object references from model, planning, runtime, or backend modules. Producers translate their own identifiers into the trace representation.

## Backend-specific attributes

Some backend diagnostics cannot be predicted by the shared schema. Typed `TraceAttributes` provides a constrained escape hatch for those details:

```java
public record TraceAttributes(
        Map<String, TraceAttributeValue> values
) {}
```

`TraceAttributeValue` is typed, with variants such as strings, numbers, booleans, and string lists. This escape hatch complements the typed payload model; it does not replace it.

## Why not `Map<String,String>`

An unstructured string map as the primary trace model would hide required fields, discard numeric and boolean types, push parsing into every consumer, and make schema changes difficult to validate. Typed payloads preserve meaning and let code handle compile, prepare, run, and backend events explicitly.

Maps are therefore limited to backend-specific `TraceAttributes`, whose values remain typed.

## Why trace stays a dependency leaf

Trace producers exist throughout the architecture. If the trace module depended on model, planning, compiler, prepare, runtime, engine, or concrete backends, using trace types could introduce reverse dependencies or cycles.

Keeping `modules/trace` as a DTO-only leaf means all layers can emit common diagnostics while ownership and business logic remain in the producing layer. Architecture tests should enforce this leaf boundary; see [Dependency Rules](dependency-rules.md).
