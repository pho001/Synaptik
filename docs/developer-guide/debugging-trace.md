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

For a BFLOAT16 or FLOAT16 raw-preserving occurrence, inspect `LowPrecisionTraceMetadata` emitted
during PREPARE. `selectedRoute` is the route actually fixed by preparation, not every available
candidate. A custom selection has `NOT_CERTIFIED` status and empty certificate-key, accuracy, and
determinism optionals. An exactly certified MPSGraph selection has `GRAPH_EXECUTABLE`, the complete
certificate key, the accuracy record, and the independently associated determinism record. The key
itself identifies `MPSGRAPH_CERTIFIED_RAW_V1`; classic MPS and MPP have no selectable trace
identity.

Preparation and invocation outcome payloads correlate through the prepared-unit ID and report the
same route. Metal performs no executable or preparation cache lookup, so its preparation cache
status is `NOT_QUERIED`.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A custom event is displayed with an MPSGraph certificate. | Candidate evidence was confused with the selected route. | Treat empty certificate optionals on `NOT_CERTIFIED` as authoritative. |
| Accuracy and reproducibility are collapsed into one verdict. | The certificate fields were flattened. | Display `accuracy` and `determinism` independently. |
| Consumers parse numeric facts from strings. | The primary payload is unstructured. | Add an appropriate typed field or typed trace attribute. |
| Trace imports model/runtime/backend objects. | DTOs depend on producer domains. | Translate to trace-local identifiers and values. |
| Enabling trace changes execution decisions. | Diagnostics became business logic. | Keep emission observational and producer-owned. |

## Limitations

No generic capture command, file format, viewer, or export exists. The public surface is the typed
event stream delivered to the caller-owned observer. See
[Tracing architecture](../architecture/tracing.md),
[typed trace ADR](../design/decisions/0003-typed-trace-dtos.md), and the
[trace master plan](../planning/modules/trace/master-plan.md).
