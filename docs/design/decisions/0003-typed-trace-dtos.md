# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0003: Typed [trace](../../glossary.md#trace) DTOs


## Editorial reading context

A producer here is Compiler, Prepare, Runtime or a backend emitting a diagnostic event. A consumer is the application or diagnostic collector reading it. A data transfer object (DTO) carries typed facts rather than live producer objects; Trace is a dependency leaf because it imports no other project layer. That is different from a Tensor leaf with no producing expression.

Use the [observer setup and event interpretation](../../developer-guide/debugging-trace.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — required by the current [architecture contract](../../glossary.md#architecture-contract). The original decision date is not recorded.

## Context

[Compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [runtime](../../glossary.md#term-runtime), and [backends](../../glossary.md#backend) need common diagnostics without creating reverse dependencies or forcing consumers to parse arbitrary strings. [Trace](../../glossary.md#trace) data must remain observational and serializable rather than become [producer](../../glossary.md#term-producer) business logic.

## Decision drivers

- explicit [schemas](../../glossary.md#term-schema) and machine-readable value types;
- a [trace](../../glossary.md#trace) [module](../../glossary.md#term-module) usable by every [producer](../../glossary.md#term-producer) as a [dependency leaf](../../glossary.md#term-dependency-leaf);
- correlation without importing [model](../../glossary.md#term-model) or [runtime](../../glossary.md#term-runtime) identifiers; and
- controlled extensibility for [backend](../../glossary.md#backend)-specific facts.

## Options considered

No historical deliberation is available. The contract explicitly contrasts typed [payload](../../glossary.md#trace-payload--tracepayload) families with a primary `Map<String,String>` model. It also permits typed [trace](../../glossary.md#trace) attributes as a limited [backend](../../glossary.md#backend)-specific escape hatch.

## Decision

Use a common event envelope with typed [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [run](../../glossary.md#term-run), and [backend](../../glossary.md#backend) payloads. Use [trace](../../glossary.md#trace)-local identifiers. Keep `modules/trace` free of [graph](../../glossary.md#graph) traversal, execution, business logic, [runtime](../../glossary.md#term-runtime) state, and [producer](../../glossary.md#term-producer)-layer dependencies. Do not use `Map<String,String>` as the primary model.

## Rationale

Typed fields preserve numeric and [boolean](../../glossary.md#data-type--datatype) meaning, make required data visible, and allow consumers and [schema](../../glossary.md#term-schema) tests to evolve deliberately. [Trace](../../glossary.md#trace)-local IDs prevent diagnostic correlation from creating [module](../../glossary.md#term-module) coupling.

## Consequences

[Producers](../../glossary.md#term-producer) must translate local state into DTOs, and adding a shared diagnostic fact may require a [schema](../../glossary.md#term-schema) change. Consumers gain safer handling and do not need [producer](../../glossary.md#term-producer) objects. [Backend](../../glossary.md#backend)-specific attributes remain possible but must use typed values and cannot replace primary payloads.

## Related documentation

- [Tracing](../../architecture/tracing.md)
- [Debugging with traces](../../developer-guide/debugging-trace.md)
- [Trace master plan](../../planning/modules/trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../glossary.md#trace))</span>
