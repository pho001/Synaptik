# ADR 0002: Backend-owned lowering

## Status

Accepted — reflected in the current architecture contract. The original decision date and historical option discussion are not documented.

## Context

After planning assigns a graph region to a backend, that region still needs backend-specific decomposition, fusion, specialization, route selection, storage decisions, and executable construction. Those decisions depend on one backend's execution model and native facilities.

## Decision drivers

- keep planning backend-neutral;
- keep lowering close to route and storage knowledge;
- prevent a shared layer from importing concrete implementations; and
- complete implementation selection before runtime.

## Options considered

The repository has no historical comparison record. The current contract addresses three structural alternatives: concrete backends own lowering; a shared `backend.lowering` module owns it; or runtime performs lowering on demand.

## Decision

Each concrete backend lowers its assigned `PlannedPartition` during prepare and constructs its own `PreparedExecutable`. There is no shared backend-lowering module. Compiler passes may perform backend-neutral canonicalization; planning selects ownership only; runtime invokes prepared work only.

## Rationale

Fusion, specialization, kernel choice, workspace, and storage constraints are mutually dependent. Keeping them together avoids abstractions that either leak backend types into shared modules or hide the real owner of a decision.

## Consequences

Backends implement more complete vertical slices and must pass conformance tests. Shared prepare remains smaller and validates coverage and schedules. Some lowering logic may look structurally similar across backends, but it is not centralized without a demonstrated backend-neutral semantic transformation.

## Related documentation

- [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Partition preparer](../../backend-guide/partition-preparer.md)
- [Module boundaries](../../architecture/module-boundaries.md)
