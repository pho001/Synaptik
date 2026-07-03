# ADR 0006: No runtime service locator

## Status

Accepted — required by the current architecture contract. The original decision date is not recorded.

## Context

Runtime should execute a prepared schedule with predictable dependencies and low hot-path overhead. Looking up backends or kernel registries during execution would hide composition, defer implementation choice, and permit backend discovery or fallback after preparation.

## Decision drivers

- explicit and testable composition;
- no backend discovery, lowering, or kernel choice in runtime;
- concrete backends independent of engine; and
- reusable prepared executable contracts.

## Options considered

The repository contains no historical option discussion. The current contract contrasts explicit engine registration with a runtime service locator and with reflective or `ServiceLoader` discovery as the default core mechanism.

## Decision

Engine is the composition root and registers concrete backends explicitly before compilation and preparation. Runtime receives prepared schedules and executables and does not query a global service registry. Reflective plugin discovery is not the core backend mechanism.

## Rationale

Explicit composition exposes availability and dependencies at construction time. Preparation resolves implementation choices once, keeping runtime focused on slots, state, transfers, execution, and publication.

## Consequences

Applications or engine builders must register components deliberately. Tests can construct exact backend sets. Optional plugin convenience would require the architecture update named by the contract and must stay outside the hot path.

## Related documentation

- [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Writing a backend](../../backend-guide/writing-a-backend.md)
- [Dependency rules](../../architecture/dependency-rules.md)
