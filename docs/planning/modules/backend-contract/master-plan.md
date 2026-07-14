# Backend Contract Master Plan

## Goal

Define minimal backend identities, device identities, availability snapshots, and declarative requirements.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- backend and device identifiers
- availability snapshots
- declarative backend requirements
- device classes

## Out of scope

- operation support logic
- kernel registries
- prepare services
- runtime storage or physical buffers

## Module invariants

- Compile-time ownership uses backend identity, not live backend services.
- Contracts remain declarative and implementation-free.

## Allowed dependencies

- JDK standard library only.

## Forbidden dependencies

- compiler, planning, runtime, prepare, engine, concrete backend implementation, and Tensor API dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.contract/
  <root>  current backend and backend-scoped device identities; later minimal declarative contracts
```

The root package remains a small public shared-contract surface. It contains no concrete backend,
live service, discovery, registration, preparation, or execution behavior.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend and device identifiers](tasks/0001-backend-and-device-identifiers.md) | Complete | Completed model milestone; trace 0001–0002 roadmap foundation | Replaced the placeholder with open backend identity and backend-scoped device identity values without registration, discovery, availability, or live services. |
| 0002 | [Device classification](tasks/0002-device-classification.md) | Complete | 0001 | Added the minimal CPU-versus-accelerator device category needed by later availability and requirements without enumerating concrete devices or routes. |
| 0003 | [Backend availability snapshot](tasks/0003-backend-availability-snapshot.md) | Complete | 0001–0002 | Added an immutable caller-supplied map of one backend's currently available device identities to coarse classes, without discovery or live backend objects. |
| 0004 | [Declarative backend requirements](tasks/0004-declarative-backend-requirements.md) | Complete | 0001–0003 | Added the minimal sealed exact-backend, exact-device, and device-class hard-eligibility vocabulary without evaluation, preference, or kernel selection. |

## Milestones

- Identity value types
- Availability and requirement DTOs
- Contract validation

## Current status

Complete. The explicit roadmap interleave after the stable trace foundation delivered tasks
0001–0004 in order: identities, coarse classification, caller-supplied availability, and the
sealed hard-requirement vocabulary. Module validation, independent documentation review, and the
repository capability checkpoint passed. No later backend-contract task is specified or Ready.

## Open questions

- No open questions remain for the completed selected milestone. Later requirement consumers own
  their own still-open policy and evaluation decisions.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Backend names remain an open nonblank string vocabulary rather than a closed CPU/Metal/CUDA
  enum.
- Device identity is scoped by its owning backend identity plus an opaque nonblank backend-defined
  token. It does not prove discovery, availability, or resource access.
- Device classification uses exactly `CPU` and `ACCELERATOR`. Current configuration and scoring
  language needs that distinction but provides no concrete need for GPU, NPU, FPGA, memory, or
  location subcategories.
- A class describes a reported device; it is not stored inside `BackendDeviceId`, does not classify
  execution routes, and carries no scoring order or availability guarantee.
- Availability is represented by one immutable snapshot per backend: an explicit backend identity
  plus a structurally immutable map from same-backend device identities to their classes. An empty
  map reports no currently available device and no ordering or separate status flag is implied.
- Requirements form one sealed, method-free family with exact-backend, exact-device, and
  device-class record variants. They are hard eligibility targets only; config owns optionality
  and intent, while planning later owns evaluation and no-match failure.
- The trace project area is deliberately interleaved after its stable foundation because later
  trace payload and attribute schemas require concrete producer-owned contracts.

## Risks

- Growing the module into a backend service or execution abstraction.
- Treating identity as evidence that a backend or device is registered, available, or supported.
- Growing the coarse device class into speculative hardware taxonomy or treating enum order as a
  planning preference.
- Treating a supplied availability snapshot as discovery, liveness monitoring, engine
  registration, capability support, or a device-selection policy.
- Expanding requirements into preferences, composition, predicates, capability language,
  scoring, or consumer-layer behavior.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 added only two immutable identity values and package documentation. Its focused six-test
run and final six-test/one-suite module run passed. The separate documentation pass finalized the
identity Javadocs and explanations and passed backend-contract Javadoc, repository Markdown,
exact twelve-path, status, and whitespace validation without rerunning Java tests. It added no
dependency, availability query, registry, service, discovery, capability, planning, prepare,
runtime, or concrete backend behavior.

Task 0002 added only the two-value declarative classification, one focused exact-shape test, and
the associated Javadoc/status documentation. Its focused three-test development runs and final
nine-test/two-suite module run passed. The separate documentation pass finalized the enum and
package Javadocs plus the public API, capability-provider guide, glossary, task, master plan, and
roadmap, then passed backend-contract Javadoc, repository Markdown, exact nine-path, status, and
whitespace validation without rerunning Java tests. Availability association, requirements,
configuration policy, scoring, and execution remain in their later owning tasks.

Task 0003 is limited to the caller-supplied immutable availability fact. Discovery, refresh,
capabilities, requirements, ownership selection, preparation, and execution remain outside the
snapshot and in their later owning layers.

Task 0003 added only the single-backend immutable availability snapshot, one focused exact-
contract test, and the associated Javadoc/status documentation. Its two focused seven-test runs
and final 16-test/three-suite module run passed. The separate documentation pass finalized the
snapshot, package, and `DeviceClass` Javadocs plus the public API, capability-provider guide,
glossary, task, master plan, and roadmap, then passed backend-contract Javadoc, repository
Markdown, exact ten-path,
status, and whitespace validation without rerunning Java tests. Discovery, registration,
refresh/liveness, requirements, capability evaluation, ownership selection, preparation,
execution, and trace translation remain in their later owning tasks.

Task 0004 is limited to a sealed hard-eligibility vocabulary. Its implementation, final
22-test/four-suite module run, and independent documentation stabilization succeeded. The single
final repository capability checkpoint then passed 1,055 tests across 135 suites with no
failures, errors, or skips, closing the selected backend-contract milestone.

The post-checkpoint reassessment selected config task 0001 as the next implementation frontier.
Its current `BackendIntent` now consumes `BackendRequirement` through an immutable optional hard-
intent value without reopening or changing this completed module. Config task 0001 and its
repository dependency checkpoint are Complete.
