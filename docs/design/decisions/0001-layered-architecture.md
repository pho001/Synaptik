# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0001: Layered architecture


## Editorial reading context

Follow the responsibility chain: the application supplies Tensor expressions to Engine; Engine coordinates Compiler, Prepare and Runtime. Compiler describes computation, Prepare makes reusable executable recipes, and Runtime supplies isolated state for each invocation. “Owner” here first means the component allowed to make a decision, not automatically an object that needs closing.

Use the [lifecycle walkthrough](../../architecture/lifecycle.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — reflected in the current [architecture contract](../../glossary.md#architecture-contract). The original decision date and deliberation record are not present in the repository.

## Context

Synaptik must preserve computational meaning while supporting multiple [backends](../../glossary.md#backend), [compile](../../glossary.md#compile)-time optimization, reusable preparation, and per-[run](../../glossary.md#term-run) [mutable](../../glossary.md#term-mutable) state. Combining those responsibilities would let backend details leak into the public [model](../../glossary.md#term-model), make [runtime](../../glossary.md#term-runtime) repeat [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) decisions, and create cyclic dependencies around composition.

## Decision drivers

- one [backend](../../glossary.md#backend)-independent semantic [model](../../glossary.md#term-model);
- explicit [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), and [run state](../../glossary.md#run-state--runstate) boundaries;
- a [runtime](../../glossary.md#term-runtime) [hot path](../../glossary.md#term-hot-path) free of [compiler](../../glossary.md#term-compiler) [graph](../../glossary.md#graph) objects;
- concrete backends that can evolve without owning public APIs; and
- enforceable acyclic [dependency direction](../../glossary.md#java-platform-and-build-tools).

## Options considered

No historical option list is available. The current contract distinguishes these architecturally relevant alternatives:

1. Separate [model](../../glossary.md#term-model), [planning](../../glossary.md#term-planning), [compiler](../../glossary.md#term-compiler), [prepare](../../glossary.md#prepare), [backend](../../glossary.md#backend), [runtime](../../glossary.md#term-runtime), [engine](../../glossary.md#term-engine), and [trace](../../glossary.md#trace) responsibilities.
2. Combine [compile](../../glossary.md#compile)-time, backend, and runtime responsibilities in broader [modules](../../glossary.md#term-module) or service abstractions.

The second alternative is represented by explicit contract prohibitions; this [ADR](../../glossary.md#term-architecture-decision-record-adr) does not claim it reproduces the original discussion.

## Decision

Use the layered [ownership](../../glossary.md#term-ownership-meanings) and [dependency](../../glossary.md#term-dependency) model in [`ARCHITECTURE.md`](../../../ARCHITECTURE.md). Model owns semantics, [planning](../../glossary.md#term-planning) owns [backend](../../glossary.md#backend)-neutral ownership, [compiler](../../glossary.md#term-compiler) owns [graph](../../glossary.md#graph) transformation, [prepare](../../glossary.md#prepare) owns shared transition contracts, concrete backends own implementation, [runtime](../../glossary.md#term-runtime) executes prepared work, [engine](../../glossary.md#term-engine) composes components, and [trace](../../glossary.md#trace) remains a typed [DTO](../../glossary.md#data-transfer-object--dto) [leaf](../../glossary.md#term-dependency-leaf).

## Rationale

Each stage receives only the state needed for its question. This prevents physical allocation during [compile](../../glossary.md#compile), [kernel](../../glossary.md#kernel) selection in [planning](../../glossary.md#term-planning), [backend](../../glossary.md#backend) discovery in [runtime](../../glossary.md#term-runtime), and [engine](../../glossary.md#term-engine) dependencies from concrete backends.

## Consequences

Positive consequences are clearer [ownership](../../glossary.md#term-ownership-meanings), testable boundaries, reusable prepared state, and independently evolvable [backends](../../glossary.md#backend). Costs include more explicit [handoff](../../glossary.md#term-handoff) types, validators, composition, and architecture tests. Cross-layer shortcuts require redesign rather than a convenient import.

## Related documentation

- [Architecture overview](../../architecture/overview.md)
- [Module boundaries](../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Dependency rules](../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
