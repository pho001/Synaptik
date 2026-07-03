# ADR 0004: Backend-neutral partition scoring

## Status

Accepted — required by the current architecture contract. The original decision date and weighting discussion are not recorded.

## Context

More than one backend may support a node or segment. Synaptik needs a compile-time ownership decision that accounts for graph context and boundary costs without importing concrete kernels or mutable runtime resources.

## Decision drivers

- decide ownership before backend preparation;
- compare candidates with backend-neutral compile-time facts;
- avoid fragmented plans and unnecessary transfers; and
- keep implementation selection in concrete backends.

## Options considered

No historical option record exists. The contract distinguishes backend-neutral ownership scoring from fixed single-backend assignment, backend-specific kernel scoring in planning, and runtime selection based on current residency.

## Decision

Planning scores valid ownership candidates using compile-time facts such as intent, capabilities, operation and shape metadata, estimated transfer/materialization/boundary costs, and immutable profiles. The output is a backend identity. Maximal adjacent same-owner work forms partitions afterward.

## Rationale

Ownership needs whole-graph context, while concrete route choice needs backend implementation knowledge. Separating the two lets planning reason about boundaries without coupling to OpenBLAS, MPSGraph, CUDA kernels, or prepared executables.

## Consequences

Scoring policies need deterministic tests and diagnostics. Estimates may be imperfect and later profile-guided enhancements require an architecture update where specified. Backends retain freedom to choose routes during prepare, but they must realize the capabilities they advertised.

## Related documentation

- [Partition scoring](../../architecture/partition-scoring.md)
- [Backend selection](../../user-guide/backend-selection.md)
- [Planning master plan](../../planning/modules/planning/master-plan.md)
