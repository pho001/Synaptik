# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0006: No [runtime](../../glossary.md#term-runtime) [service locator](../../glossary.md#term-service-locator)

## Status

Accepted — required by the current [architecture contract](../../glossary.md#architecture-contract). The original decision date is not recorded.

## Context

[Runtime](../../glossary.md#term-runtime) should execute a [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule) with predictable dependencies and low hot-path overhead. Looking up [backends](../../glossary.md#backend) or [kernel](../../glossary.md#kernel) registries during execution would hide composition, defer implementation choice, and permit backend discovery or fallback after preparation.

## Decision drivers

- explicit and testable composition;
- no [backend](../../glossary.md#backend) discovery, [lowering](../../glossary.md#lowering), or [kernel](../../glossary.md#kernel) choice in [runtime](../../glossary.md#term-runtime);
- concrete backends independent of [engine](../../glossary.md#term-engine); and
- reusable [prepared executable](../../glossary.md#prepared-executable--preparedexecutable) contracts.

## Options considered

The repository contains no historical option discussion. The current contract contrasts explicit [engine](../../glossary.md#term-engine) registration with a [runtime](../../glossary.md#term-runtime) [service locator](../../glossary.md#term-service-locator) and with reflective or `ServiceLoader` discovery as the default core mechanism.

## Decision

[Engine](../../glossary.md#term-engine) is the [composition root](../../glossary.md#term-composition-root) and registers concrete [backends](../../glossary.md#backend) explicitly before [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) and preparation. [Runtime](../../glossary.md#term-runtime) receives prepared schedules and executables and does not query a global service [registry](../../glossary.md#term-registry). Reflective [plugin](../../glossary.md#term-plugin) discovery is not the core backend mechanism.

## Rationale

Explicit composition exposes availability and dependencies at construction time. Preparation resolves implementation choices once, keeping [runtime](../../glossary.md#term-runtime) focused on slots, state, transfers, execution, and [publication](../../glossary.md#term-publication).

## Consequences

Applications or [engine](../../glossary.md#term-engine) builders must register components deliberately. Tests can construct exact [backend](../../glossary.md#backend) sets. Optional [plugin](../../glossary.md#term-plugin) convenience would require the architecture update named by the contract and must stay outside the [hot path](../../glossary.md#term-hot-path).

## Related documentation

- [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [prepare glossary](../../glossary.md#prepare); [backend glossary](../../glossary.md#backend))</span>
- [Writing a backend](../../backend-guide/writing-a-backend.md)
- [Dependency rules](../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
- [ADR 0015: Explicit Engine backend composition](0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [Engine glossary](../../glossary.md#term-engine); [backend composition glossary](../../glossary.md#backend-composition))</span>
