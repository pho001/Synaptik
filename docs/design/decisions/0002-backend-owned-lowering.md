# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0002: [Backend-owned lowering](../../glossary.md#backend-owned-lowering)

## Status

Accepted — reflected in the current [architecture contract](../../glossary.md#architecture-contract). The original decision date and
historical option discussion are not documented. The generated [CPU](../../glossary.md#term-cpu)-[kernel](../../glossary.md#kernel) mechanism clarification
was synchronized on 2026-08-03.

## Context

After [planning](../../glossary.md#term-planning) assigns a [graph](../../glossary.md#graph) region to a [backend](../../glossary.md#backend), that region still needs backend-specific [decomposition](../../glossary.md#term-decomposition), [fusion](../../glossary.md#term-fusion), [specialization](../../glossary.md#term-specialization), route selection, storage decisions, and executable construction. Those decisions depend on one backend's execution [model](../../glossary.md#term-model) and native facilities.

The architecture previously named ASM as a [CPU](../../glossary.md#term-cpu) route and separately required an architecture
update before [bytecode](../../glossary.md#term-bytecode)-generated CPU [fused](../../glossary.md#specialization-and-fusion) [kernels](../../glossary.md#kernel). CPU planning now selects the Java 26
[Class-File API](../../glossary.md#term-class-file-api) as its current implementation direction. The owning architecture decision must
permit that direction without turning either ASM or a specific [JDK](../../glossary.md#term-java-development-kit-jdk) builder [API](../../glossary.md#term-application-programming-interface-api) into a permanent
cross-layer contract.

## Decision drivers

- keep [planning](../../glossary.md#term-planning) [backend](../../glossary.md#backend)-neutral;
- keep [lowering](../../glossary.md#lowering) close to route and storage knowledge;
- prevent a shared layer from importing concrete implementations; and
- complete implementation selection before [runtime](../../glossary.md#term-runtime); and
- let a backend replace its internal code-generation API without changing module ownership or
  lifecycle placement.<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [module glossary](../../glossary.md#term-module); [ownership glossary](../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../glossary.md#lifecycle))</span>

## Options considered

The repository has no historical comparison record. The current contract addresses three
structural alternatives: concrete [backends](../../glossary.md#backend) own [lowering](../../glossary.md#lowering); a shared `backend.lowering` [module](../../glossary.md#term-module) owns
it; or [runtime](../../glossary.md#term-runtime) performs lowering on demand. Within backend-owned [CPU](../../glossary.md#term-cpu) lowering, the mechanism can
be fixed to ASM, fixed to the Java 26 [Class-File API](../../glossary.md#term-class-file-api), or left implementation-neutral while the
generated [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../../glossary.md#term-bytecode) result and [lifecycle](../../glossary.md#lifecycle) remain explicit.

## Decision

Each [concrete backend](../../glossary.md#concrete-backend) lowers its assigned `PlannedPartition` during [prepare](../../glossary.md#prepare) and constructs its own
[`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable). There is no shared [backend](../../glossary.md#backend)-[lowering](../../glossary.md#lowering) [module](../../glossary.md#term-module). [Compiler](../../glossary.md#term-compiler) passes may perform
backend-neutral [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form); [planning](../../glossary.md#term-planning) selects [ownership](../../glossary.md#term-ownership-meanings) only; [runtime](../../glossary.md#term-runtime) invokes prepared work
only.

Generated [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../../glossary.md#term-bytecode) [CPU](../../glossary.md#term-cpu) computation [kernels](../../glossary.md#kernel) are one permitted backend-internal lowering result.
CPU analysis owns their lowering, [specialization](../../glossary.md#term-specialization), route choice, and exact resource declarations;
CPU finalization owns generation, definition, and compatible generated-[artifact](../../glossary.md#term-artifact) reuse after shared
slot assignment. This decision deliberately does not select a bytecode-generation library or [JDK](../../glossary.md#term-java-development-kit-jdk)
builder [API](../../glossary.md#term-application-programming-interface-api).

## Rationale

[Fusion](../../glossary.md#term-fusion), [specialization](../../glossary.md#term-specialization), [kernel](../../glossary.md#kernel) choice, workspace, and storage constraints are mutually dependent.
Keeping them together avoids abstractions that either leak [backend](../../glossary.md#backend) types into shared [modules](../../glossary.md#term-module) or
hide the real [owner](../../glossary.md#term-owner) of a decision. Naming the generated [JVM](../../glossary.md#term-java-virtual-machine-jvm)-[bytecode](../../glossary.md#term-bytecode) result while leaving its
builder implementation-neutral preserves the [lifecycle](../../glossary.md#lifecycle) boundary without freezing a replaceable
backend detail.

## Consequences

[Backends](../../glossary.md#backend) implement more complete vertical slices and must pass [conformance](../../glossary.md#term-conformance) tests. Shared [prepare](../../glossary.md#prepare)
remains smaller and validates coverage and schedules. A [CPU](../../glossary.md#term-cpu) backend may change its internal [JVM](../../glossary.md#term-java-virtual-machine-jvm)-
[bytecode](../../glossary.md#term-bytecode) generation [API](../../glossary.md#term-application-programming-interface-api) without changing this decision, provided [ownership](../../glossary.md#term-ownership-meanings), dependencies,
prepared [lifecycle](../../glossary.md#lifecycle), and observable semantics remain intact. Some [lowering](../../glossary.md#lowering) logic may look
structurally similar across backends, but it is not centralized without a demonstrated backend-
neutral semantic transformation.

## Related documentation

- [Runtime/prepare/backend boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [prepare glossary](../../glossary.md#prepare); [backend glossary](../../glossary.md#backend))</span>
- [Partition preparer](../../backend-guide/partition-preparer.md)<span data-glossary-navigation="true"> ([Partition glossary](../../glossary.md#partition))</span>
- [Module boundaries](../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [CPU backend guide](../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [CPU backend master plan](../../planning/backends/cpu/master-plan.md)
