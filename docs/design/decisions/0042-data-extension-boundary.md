# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0042 Data extension boundary


## Editorial reading context

The application prepares host batches through Data before calling a model. Data depends on generic Model Tensor descriptions, not NN layers or Engine execution. The empty-module foundation below establishes dependency permission only; later batch APIs have their own acceptance. Use the live repository map when locating the implementation rather than inferring features from this original foundation.

Use the [Data and Text project locations](../../developer-guide/repository-layout.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

This [architecture decision record](../../glossary.md#adr-ci-and-qualification) explains the Data-only [host](../../glossary.md#term-host)-preparation boundary. The
[architecture root](../../../ARCHITECTURE.md#dependency-rules) and its
[owning scoped contract](../../architecture/contracts/extensions-training.md#extensionsdata)
are authoritative; this record preserves the rationale.

## Status

Accepted — 2026-10-20. The user approved Data0001 implementation with “pust se di toho”,
releasing the previous Draft architecture hold. Acceptance covers Data only, not Text or Vision.

## Context

At decision time, no Data project was registered, although numeric [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) and sequence batching
belongs outside [NN model](../../glossary.md#model-module-buffer-and-scan-distinctions) computation. Future [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) values will expose [Model](../../glossary.md#term-model) [Tensor](../../glossary.md#tensor) values.
Establishing that boundary first allows later batching tasks to work without inventing execution
or storage [owners](../../glossary.md#term-owner).
The initial capability is only the empty [module](../../glossary.md#term-module) foundation and its [dependency](../../glossary.md#term-dependency) enforcement.

## Decision drivers

- Keep [Model](../../glossary.md#term-model) [Tensor](../../glossary.md#tensor) semantics and [host storage](../../glossary.md#host-storage) [ownership](../../glossary.md#term-ownership-meanings) unchanged.
- Give consumers the Model types needed by future [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [signatures](../../glossary.md#term-signature) through Data alone.
- Avoid speculative production types and joint approval of unrelated Text/Vision capabilities.
- Preserve the distinction between [host](../../glossary.md#term-host) valid-length [metadata](../../glossary.md#term-metadata) and [runtime](../../glossary.md#term-runtime) input [binding](../../glossary.md#term-binding).

## Options considered

1. Keep the status quo with no Data module. This avoids a new boundary but leaves numeric batching
   without its selected owner.<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module); [owner glossary](../../glossary.md#term-owner))</span>
2. Add an empty Data module exporting Model. This establishes the narrow boundary without
   selecting a batching API or runtime lifecycle.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [API glossary](../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../glossary.md#term-runtime); [lifecycle glossary](../../glossary.md#lifecycle))</span>
3. Use an implementation-only Model dependency. This hides the Model types that future batch
   signatures need consumers to see.<span data-glossary-navigation="true"> ([dependency glossary](../../glossary.md#term-dependency); [batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters); [signatures glossary](../../glossary.md#term-signature))</span>
4. Approve Data, Text and Vision together, or add placeholder Data types to demonstrate the edge.
   This expands approval beyond numeric preparation or creates unsupported production contracts.

## Decision

Choose option 2: [Gradle](../../glossary.md#term-gradle) `:extensions:data`, physical `extensions/data`, namespace
`io.github.pho001.synaptik.data`, and exactly `api(project(":modules:model"))` as the sole project
edge. No additional edge is allowed even in tests, and no inward [module](../../glossary.md#term-module) or existing extension
may depend on Data. Use the existing root build conventions without a new [plugin](../../glossary.md#term-plugin) or library.

The foundation has no production type, placeholder [API](../../glossary.md#term-application-programming-interface-api) or reserved subpackage. Leave production
sources empty by default; add package documentation only for a genuine package contract.
The existing extensions/training scoped contract owns Data; no seventh [normative](../../glossary.md#term-normative) file is added.
Text/Vision modules and edges retain their own architecture gates.

## Rationale

Exporting [Model](../../glossary.md#term-model) matches the planned public [Tensor](../../glossary.md#tensor) boundary without requiring an invented [Data
type](../../glossary.md#data-type--datatype). A [compile](../../glossary.md#compile)-only consumer that depends solely on Data can name Model Tensor declarations;
that proves transitive [API](../../glossary.md#term-application-programming-interface-api) visibility, not a batching API or execution capability. Test-local
positive and negative fixtures can validate namespace/import rules even while Data has no sources.

Future [host](../../glossary.md#term-host) lengths `[5,3,1]` for time [extent](../../glossary.md#term-extent) 5 describe valid prefixes, with derived [padding](../../glossary.md#padding)
lengths `[0,2,4]`. They are not the [scan](../../glossary.md#term-cumulative-scan)'s [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) non-[gradient](../../glossary.md#term-gradient) [`INT64`](../../glossary.md#data-type--datatype) [runtime](../../glossary.md#term-runtime) input. A later
[materialization](../../glossary.md#materialization)/[binding](../../glossary.md#term-binding) decision must connect those lifecycles explicitly, without storing a
second [canonical](../../glossary.md#term-canonical-form) padding-length or dense-[mask](../../glossary.md#term-boolean-mask) representation. Current static [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) length APIs,
runtime scan semantics, Model [ownership](../../glossary.md#term-ownership-meanings) and Training behavior remain unchanged.

## Consequences

The boundary can be built and tested independently of [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Text and Vision. Consumers inherit
[Model](../../glossary.md#term-model) [API](../../glossary.md#term-application-programming-interface-api) visibility, which makes Model an intentional public [dependency](../../glossary.md#term-dependency) of future Data APIs.
An empty [module](../../glossary.md#term-module) adds build/enforcement maintenance without delivering batching behavior; later
capabilities must still select their own APIs and [validation](../../glossary.md#term-validation).

No existing production API needs migration. Data0001 implements registration, exact-edge and
reverse-edge protection, namespace/import fixtures and the isolated consumer in architecture
tests. Implementation, review and validation evidence belongs in the task brief; the decision
itself is not executable evidence. Independent Class C review and the recorded integration gate
are required before Data0001 is Complete.

## Related documentation

- [Data0001 implementation brief](../../planning/extensions/data/tasks/0001-data-extension-boundary.md)
- [Data plan and staged valid lengths](../../planning/extensions/data/master-plan.md#module-invariants-and-dependencies)
- [Module responsibilities](../../architecture/module-boundaries.md#extensionsdata)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Dependency enforcement](../../architecture/dependency-rules.md#architecture-test-enforcement)<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
- [Repository layout](../../developer-guide/repository-layout.md)
- [Architecture testing requirements](../../../ARCHITECTURE.md#testing-requirements)
