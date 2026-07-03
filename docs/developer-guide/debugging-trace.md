# Debugging with traces (planned workflow)

## What you will learn

This guide explains how typed traces will help locate a problem in compile, prepare, or run. The trace DTOs, emitters, sinks, and tooling are not implemented yet.

## Mental model

```text
producer-owned state -> typed trace payload -> sink/tool -> human diagnosis
```

The trace module owns data-transfer objects (DTOs), not graph traversal or execution. A producer translates local objects into trace-local identifiers and typed fields.

## Planned investigation

Suppose a graph region unexpectedly receives CPU ownership. A useful trace sequence would show capability candidates, typed scoring factors, the selected backend identity, partition creation, CPU prepare route, and run steps. Follow the same trace-local node or partition identifier across events.

The interpretation boundary matters: a compile event can explain ownership, a prepare event can explain a selected CPU route, and a run event can explain invocation timing. A run event must not claim that runtime performed ownership scoring.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| Consumers parse numeric facts from strings | The primary payload is unstructured. | Add an appropriate typed field or typed trace attribute. |
| Trace imports model/runtime/backend objects | DTOs depend on producer domains. | Translate to trace-local identifiers and values. |
| Enabling trace changes execution decisions | Diagnostics became business logic. | Keep emission observational and producer-owned. |

## Limitations

No capture command, file format, schema version, or UI exists yet. See [Tracing architecture](../architecture/tracing.md), [typed trace ADR](../design/decisions/0003-typed-trace-dtos.md), and the [trace master plan](../planning/modules/trace/master-plan.md).
