# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0004: [Backend](../../glossary.md#backend)-neutral [partition](../../glossary.md#partition) scoring

## Status

Accepted — required by the current [architecture contract](../../glossary.md#architecture-contract). The original decision date and weighting discussion are not recorded.

## Context

More than one [backend](../../glossary.md#backend) may support a [node](../../glossary.md#node) or segment. Synaptik needs a [compile](../../glossary.md#compile)-time [ownership](../../glossary.md#term-ownership-meanings) decision that accounts for [graph](../../glossary.md#graph) context and boundary costs without importing concrete [kernels](../../glossary.md#kernel) or [mutable](../../glossary.md#term-mutable) [runtime](../../glossary.md#term-runtime) resources.

## Decision drivers

- decide [ownership](../../glossary.md#term-ownership-meanings) before [backend](../../glossary.md#backend) preparation;
- compare candidates with backend-neutral [compile](../../glossary.md#compile)-time facts;
- avoid fragmented plans and unnecessary transfers; and
- keep implementation selection in concrete backends.

## Options considered

No historical option record exists. The contract distinguishes [backend](../../glossary.md#backend)-neutral [ownership](../../glossary.md#term-ownership-meanings) scoring from fixed single-backend assignment, backend-specific [kernel](../../glossary.md#kernel) scoring in [planning](../../glossary.md#term-planning), and [runtime](../../glossary.md#term-runtime) selection based on current [residency](../../glossary.md#residency).

## Decision

[Planning](../../glossary.md#term-planning) scores valid [ownership](../../glossary.md#term-ownership-meanings) candidates using [compile](../../glossary.md#compile)-time facts such as intent, capabilities, [operation](../../glossary.md#operation) and [shape](../../glossary.md#shape) [metadata](../../glossary.md#term-metadata), estimated transfer/[materialization](../../glossary.md#materialization)/boundary costs, and [immutable](../../glossary.md#term-immutable) profiles. The output is a [backend identity](../../glossary.md#backend-identity--backendid). Maximal adjacent same-[owner](../../glossary.md#term-owner) work forms partitions afterward.

## Rationale

[Ownership](../../glossary.md#term-ownership-meanings) needs whole-[graph](../../glossary.md#graph) context, while concrete route choice needs [backend](../../glossary.md#backend) implementation knowledge. Separating the two lets [planning](../../glossary.md#term-planning) reason about boundaries without coupling to [OpenBLAS](../../glossary.md#term-openblas), [MPSGraph](../../glossary.md#term-mpsgraph), [CUDA](../../glossary.md#term-cuda) [kernels](../../glossary.md#kernel), or prepared executables.

## Consequences

Scoring policies need deterministic tests and diagnostics. Estimates may be imperfect and later profile-guided enhancements require an architecture update where specified. [Backends](../../glossary.md#backend) retain freedom to choose routes during [prepare](../../glossary.md#prepare), but they must realize the capabilities they advertised.

## Related documentation

- [Partition scoring](../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../glossary.md#partition))</span>
- [Backend selection](../../user-guide/backend-selection.md)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [Planning master plan](../../planning/modules/planning/master-plan.md)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span>
