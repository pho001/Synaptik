# Trace Master Plan

## Goal

Define typed, serializable diagnostic DTOs shared by compile, prepare, run, and backend trace producers.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- trace envelopes and phases
- typed payload families
- trace-local identifiers
- typed backend attribute escape hatch

## Out of scope

- graph traversal
- business logic
- runtime state
- backend execution

## Module invariants

- Trace remains a DTO-only dependency leaf.
- `Map<String,String>` is never the primary trace model.
- Producers translate local state into trace-local types.

## Allowed dependencies

- JDK standard library only.

## Forbidden dependencies

- model, planning, compiler, runtime, prepare, engine, and concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.trace/
  <root>       shared public event envelope, event identity, lifecycle phase, level, and payload marker
  id/          later trace-local correlation identifiers translated from producer-owned identities
  payload/     later typed compile, prepare, run, and backend diagnostic DTO families
  attribute/   later typed backend-specific attribute escape hatch
```

The root package is deliberately small and contains only contracts needed by every producer and
consumer. Concrete payload families do not accumulate in the root package. No package may expose
or import producer-domain types.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Core trace event envelope](tasks/0001-core-trace-event-envelope.md) | Complete | Model milestone complete | Replaced the placeholder with the caller-supplied event identity, lifecycle phase, diagnostic level, open typed-payload marker, and immutable generic event envelope. |
| 0002 | Trace-local correlation identifiers | Draft | 0001 | Add backend-neutral trace identities for node, value, tensor, partition, backend, and prepared-unit correlation without importing producer IDs. |
| 0003 | Typed trace attributes | Draft | 0001 | Add the constrained typed backend-specific attribute escape hatch without making a string map the primary model. |
| 0004 | Compile payload family | Draft | 0001–0002 | Define typed capture, transformation, ownership, partition, logical-memory, and publication diagnostic payloads after compiler/planning facts stabilize. |
| 0005 | Prepare payload family | Draft | 0001–0003 | Define typed preparation, route, prepared-memory, partition, unit, and schedule diagnostics after prepare contracts stabilize. |
| 0006 | Run payload family | Draft | 0001–0003 | Define typed invocation, execution, transfer, materialization, step, and publication diagnostics after runtime contracts stabilize. |
| 0007 | Backend payload family | Draft | 0001–0003 | Define typed availability, capability, route, kernel, storage, and backend-detail diagnostics without backend implementation dependencies. |
| 0008 | Serialization and schema validation | Draft | 0001–0007 | Select and validate a stable external encoding only after all shared DTO families are concrete. |


## Milestones

- Core envelope and identifiers
- Lifecycle payload families
- Serialization and schema validation

## Current status

In progress. Task 0001 is Complete and supplies the common event foundation. Task 0002 is the next
ordered frontier but remains Draft without a detailed specification. It and later rows remain
Draft until their producer-layer facts are stable enough to select exact schemas.

## Open questions

- The external serialization format and compatibility/versioning policy remain intentionally open
  until task 0008; task 0001 adds no Java-native serialization promise.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Lifecycle phase classifies `COMPILE`, `PREPARE`, or `RUN`. Backend is a payload family and
  producer role, not a fourth lifecycle phase.
- The initial payload marker is open and method-free so later trace-owned DTO families can be
  added incrementally without a central registry or premature permits list.
- Event identity and monotonic time are supplied by producers. Trace owns neither an allocator nor
  a clock.
- The event record is shallowly immutable. Its open payload bound documents an immutable DTO
  obligation but cannot enforce payload implementation immutability at runtime.

## Risks

- Allowing producer-domain types or unstructured string maps into shared trace contracts.
- Treating caller-supplied monotonic timestamps as wall-clock instants or globally comparable
  values.
- Selecting a serialization mechanism before the shared payload schemas exist.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 passed 12 focused tests across two suites, final trace Javadoc generation, repository
Markdown validation, and exact fifteen-path scope validation. The implementation introduced no
dependency, architecture, build, backend, or cross-module behavior change.
