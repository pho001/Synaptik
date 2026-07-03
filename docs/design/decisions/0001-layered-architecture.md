# ADR 0001: Layered architecture

## Status

Accepted — reflected in the current architecture contract. The original decision date and deliberation record are not present in the repository.

## Context

Synaptik must preserve computational meaning while supporting multiple backends, compile-time optimization, reusable preparation, and per-run mutable state. Combining those responsibilities would let backend details leak into the public model, make runtime repeat compilation decisions, and create cyclic dependencies around composition.

## Decision drivers

- one backend-independent semantic model;
- explicit compile, prepare, and run state boundaries;
- a runtime hot path free of compiler graph objects;
- concrete backends that can evolve without owning public APIs; and
- enforceable acyclic dependency direction.

## Options considered

No historical option list is available. The current contract distinguishes these architecturally relevant alternatives:

1. Separate model, planning, compiler, prepare, backend, runtime, engine, and trace responsibilities.
2. Combine compile-time, backend, and runtime responsibilities in broader modules or service abstractions.

The second alternative is represented by explicit contract prohibitions; this ADR does not claim it reproduces the original discussion.

## Decision

Use the layered ownership and dependency model in [`ARCHITECTURE.md`](../../../ARCHITECTURE.md). Model owns semantics, planning owns backend-neutral ownership, compiler owns graph transformation, prepare owns shared transition contracts, concrete backends own implementation, runtime executes prepared work, engine composes components, and trace remains a typed DTO leaf.

## Rationale

Each stage receives only the state needed for its question. This prevents physical allocation during compile, kernel selection in planning, backend discovery in runtime, and engine dependencies from concrete backends.

## Consequences

Positive consequences are clearer ownership, testable boundaries, reusable prepared state, and independently evolvable backends. Costs include more explicit handoff types, validators, composition, and architecture tests. Cross-layer shortcuts require redesign rather than a convenient import.

## Related documentation

- [Architecture overview](../../architecture/overview.md)
- [Module boundaries](../../architecture/module-boundaries.md)
- [Dependency rules](../../architecture/dependency-rules.md)
