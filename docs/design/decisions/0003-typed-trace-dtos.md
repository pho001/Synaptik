# ADR 0003: Typed trace DTOs

## Status

Accepted — required by the current architecture contract. The original decision date is not recorded.

## Context

Compile, prepare, runtime, and backends need common diagnostics without creating reverse dependencies or forcing consumers to parse arbitrary strings. Trace data must remain observational and serializable rather than become producer business logic.

## Decision drivers

- explicit schemas and machine-readable value types;
- a trace module usable by every producer as a dependency leaf;
- correlation without importing model or runtime identifiers; and
- controlled extensibility for backend-specific facts.

## Options considered

No historical deliberation is available. The contract explicitly contrasts typed payload families with a primary `Map<String,String>` model. It also permits typed trace attributes as a limited backend-specific escape hatch.

## Decision

Use a common event envelope with typed compile, prepare, run, and backend payloads. Use trace-local identifiers. Keep `modules/trace` free of graph traversal, execution, business logic, runtime state, and producer-layer dependencies. Do not use `Map<String,String>` as the primary model.

## Rationale

Typed fields preserve numeric and boolean meaning, make required data visible, and allow consumers and schema tests to evolve deliberately. Trace-local IDs prevent diagnostic correlation from creating module coupling.

## Consequences

Producers must translate local state into DTOs, and adding a shared diagnostic fact may require a schema change. Consumers gain safer handling and do not need producer objects. Backend-specific attributes remain possible but must use typed values and cannot replace primary payloads.

## Related documentation

- [Tracing](../../architecture/tracing.md)
- [Debugging with traces](../../developer-guide/debugging-trace.md)
- [Trace master plan](../../planning/modules/trace/master-plan.md)
