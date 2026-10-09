# Debugging with traces

## What you will learn

This guide explains the implemented typed trace DTO boundary and the current Metal observer. There
is no repository-wide trace capture command, serializer, viewer, or export format.

## Mental model

```text
producer-owned state -> typed trace payload -> caller-owned observer -> human diagnosis
```

The trace module owns data-transfer objects (DTOs), not graph traversal or execution. A producer
translates local objects into trace-local identifiers and typed fields. Metal callers opt in through
`MetalBackendIntegration.open(configuration, observer)`; the ordinary overload performs no trace
work.

## Metal low-precision investigation

For any BFLOAT16 or FLOAT16 occurrence, inspect `LowPrecisionTraceMetadata` emitted during PREPARE.
`selectedRoute` is always the route actually fixed by preparation and must be `CUSTOM_KERNEL`.
`logicalDtypeTuple` orders boundary feeds before boundary targets. The payload has no numerical
profile, accumulator, or working-type field: FLOAT32 arithmetic is a
separate Model guarantee, and raw-preserving programs copy represented words exactly. It contains
no hypothetical candidate or certificate state.

Preparation and invocation outcome payloads correlate through the prepared-unit ID and report the
same route. Metal performs no executable or preparation cache lookup, so its preparation cache
status is `NOT_QUERIED`.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A low event is displayed as MPSGraph. | A removed alternate route was assumed. | Treat the DTO's required `CUSTOM_KERNEL` route as authoritative. |
| A consumer expects `accumulatorDtype` or `numericalProfile`. | An old diagnostic field was mistaken for a numerical-policy requirement. | Read route and logical dtype tuple from the event; use the Model contract for arithmetic semantics. |
| Consumers parse numeric facts from strings. | The primary payload is unstructured. | Add an appropriate typed field or typed trace attribute. |
| Trace imports model/runtime/backend objects. | DTOs depend on producer domains. | Translate to trace-local identifiers and values. |
| Enabling trace changes execution decisions. | Diagnostics became business logic. | Keep emission observational and producer-owned. |

## Limitations

No generic capture command, file format, viewer, or export exists. The public surface is the typed
event stream delivered to the caller-owned observer. See
[Tracing architecture](../architecture/tracing.md),
[typed trace ADR](../design/decisions/0003-typed-trace-dtos.md), and the
[trace master plan](../planning/modules/trace/master-plan.md).
