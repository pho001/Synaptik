# ADR 0002: Backend-owned lowering

## Status

Accepted — reflected in the current architecture contract. The original decision date and
historical option discussion are not documented. The generated CPU-kernel mechanism clarification
was synchronized on 2026-08-03.

## Context

After planning assigns a graph region to a backend, that region still needs backend-specific decomposition, fusion, specialization, route selection, storage decisions, and executable construction. Those decisions depend on one backend's execution model and native facilities.

The architecture previously named ASM as a CPU route and separately required an architecture
update before bytecode-generated CPU fused kernels. CPU planning now selects the Java 26
Class-File API as its current implementation direction. The owning architecture decision must
permit that direction without turning either ASM or a specific JDK builder API into a permanent
cross-layer contract.

## Decision drivers

- keep planning backend-neutral;
- keep lowering close to route and storage knowledge;
- prevent a shared layer from importing concrete implementations; and
- complete implementation selection before runtime; and
- let a backend replace its internal code-generation API without changing module ownership or
  lifecycle placement.

## Options considered

The repository has no historical comparison record. The current contract addresses three
structural alternatives: concrete backends own lowering; a shared `backend.lowering` module owns
it; or runtime performs lowering on demand. Within backend-owned CPU lowering, the mechanism can
be fixed to ASM, fixed to the Java 26 Class-File API, or left implementation-neutral while the
generated JVM-bytecode result and lifecycle remain explicit.

## Decision

Each concrete backend lowers its assigned `PlannedPartition` during prepare and constructs its own
`PreparedExecutable`. There is no shared backend-lowering module. Compiler passes may perform
backend-neutral canonicalization; planning selects ownership only; runtime invokes prepared work
only.

Generated JVM-bytecode CPU computation kernels are one permitted backend-internal lowering result.
CPU analysis owns their lowering, specialization, route choice, and exact resource declarations;
CPU finalization owns generation, definition, and compatible generated-artifact reuse after shared
slot assignment. This decision deliberately does not select a bytecode-generation library or JDK
builder API.

## Rationale

Fusion, specialization, kernel choice, workspace, and storage constraints are mutually dependent.
Keeping them together avoids abstractions that either leak backend types into shared modules or
hide the real owner of a decision. Naming the generated JVM-bytecode result while leaving its
builder implementation-neutral preserves the lifecycle boundary without freezing a replaceable
backend detail.

## Consequences

Backends implement more complete vertical slices and must pass conformance tests. Shared prepare
remains smaller and validates coverage and schedules. A CPU backend may change its internal JVM-
bytecode generation API without changing this decision, provided ownership, dependencies,
prepared lifecycle, and observable semantics remain intact. Some lowering logic may look
structurally similar across backends, but it is not centralized without a demonstrated backend-
neutral semantic transformation.

## Related documentation

- [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Partition preparer](../../backend-guide/partition-preparer.md)
- [Module boundaries](../../architecture/module-boundaries.md)
- [CPU backend guide](../../backend-guide/cpu-backend.md)
- [CPU backend master plan](../../planning/backends/cpu/master-plan.md)
