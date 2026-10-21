# ADR 0042 Data extension boundary

This architecture decision record explains the Data-only host-preparation boundary. The
[architecture root](../../../ARCHITECTURE.md#dependency-rules) and its
[owning scoped contract](../../architecture/contracts/extensions-training.md#extensionsdata)
are authoritative; this record preserves the rationale.

## Status

Accepted — 2026-10-20. The user approved Data0001 implementation with “pust se di toho”,
releasing the previous Draft architecture hold. Acceptance covers Data only, not Text or Vision.

## Context

At decision time, no Data project was registered, although numeric sample and sequence batching
belongs outside NN model computation. Future batch values will expose Model Tensor values.
Establishing that boundary first allows later batching tasks to work without inventing execution
or storage owners.
The initial capability is only the empty module foundation and its dependency enforcement.

## Decision drivers

- Keep Model Tensor semantics and host storage ownership unchanged.
- Give consumers the Model types needed by future batch signatures through Data alone.
- Avoid speculative production types and joint approval of unrelated Text/Vision capabilities.
- Preserve the distinction between host valid-length metadata and runtime input binding.

## Options considered

1. Keep the status quo with no Data module. This avoids a new boundary but leaves numeric batching
   without its selected owner.
2. Add an empty Data module exporting Model. This establishes the narrow boundary without
   selecting a batching API or runtime lifecycle.
3. Use an implementation-only Model dependency. This hides the Model types that future batch
   signatures need consumers to see.
4. Approve Data, Text and Vision together, or add placeholder Data types to demonstrate the edge.
   This expands approval beyond numeric preparation or creates unsupported production contracts.

## Decision

Choose option 2: Gradle `:extensions:data`, physical `extensions/data`, namespace
`io.github.pho001.synaptik.data`, and exactly `api(project(":modules:model"))` as the sole project
edge. No additional edge is allowed even in tests, and no inward module or existing extension
may depend on Data. Use the existing root build conventions without a new plugin or library.

The foundation has no production type, placeholder API or reserved subpackage. Leave production
sources empty by default; add package documentation only for a genuine package contract.
The existing extensions/training scoped contract owns Data; no seventh normative file is added.
Text/Vision modules and edges retain their own architecture gates.

## Rationale

Exporting Model matches the planned public Tensor boundary without requiring an invented Data
type. A compile-only consumer that depends solely on Data can name Model Tensor declarations;
that proves transitive API visibility, not a batching API or execution capability. Test-local
positive and negative fixtures can validate namespace/import rules even while Data has no sources.

Future host lengths `[5,3,1]` for time extent 5 describe valid prefixes, with derived padding
lengths `[0,2,4]`. They are not the scan's rank-one non-gradient `INT64` runtime input. A later
materialization/binding decision must connect those lifecycles explicitly, without storing a
second canonical padding-length or dense-mask representation. Current static NN length APIs,
runtime scan semantics, Model ownership and Training behavior remain unchanged.

## Consequences

The boundary can be built and tested independently of NN, Text and Vision. Consumers inherit
Model API visibility, which makes Model an intentional public dependency of future Data APIs.
An empty module adds build/enforcement maintenance without delivering batching behavior; later
capabilities must still select their own APIs and validation.

No existing production API needs migration. Data0001 implements registration, exact-edge and
reverse-edge protection, namespace/import fixtures and the isolated consumer in architecture
tests. Implementation, review and validation evidence belongs in the task brief; the decision
itself is not executable evidence. Independent Class C review and the recorded integration gate
are required before Data0001 is Complete.

## Related documentation

- [Data0001 implementation brief](../../planning/extensions/data/tasks/0001-data-extension-boundary.md)
- [Data plan and staged valid lengths](../../planning/extensions/data/master-plan.md#valid-length-representation-decision)
- [Module responsibilities](../../architecture/module-boundaries.md#extensionsdata)
- [Dependency enforcement](../../architecture/dependency-rules.md#architecture-test-enforcement)
- [Repository layout](../../developer-guide/repository-layout.md)
- [Architecture testing requirements](../../../ARCHITECTURE.md#testing-requirements)
