# [Backend](../../../glossary.md#backend) Contract Master Plan

## Goal

Define minimal [backend](../../../glossary.md#backend) [identities](../../../glossary.md#term-identity-meanings), [device](../../../glossary.md#term-device) identities, availability snapshots, and declarative requirements.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [backend](../../../glossary.md#backend) and [device](../../../glossary.md#term-device) identifiers
- availability snapshots
- declarative backend requirements
- device classes

## Out of [scope](../../../glossary.md#term-scope)

- [operation](../../../glossary.md#operation) support logic
- [kernel](../../../glossary.md#kernel) registries
- [prepare](../../../glossary.md#prepare) services
- [runtime](../../../glossary.md#term-runtime) storage or physical [buffers](../../../glossary.md#term-buffer)

## [Module](../../../glossary.md#term-module) invariants

- [Compile](../../../glossary.md#compile)-time [ownership](../../../glossary.md#term-ownership-meanings) uses [backend identity](../../../glossary.md#backend-identity--backendid), not live [backend](../../../glossary.md#backend) services.
- Contracts remain declarative and implementation-free.

## Allowed dependencies

- [JDK](../../../glossary.md#term-java-development-kit-jdk) standard library only.

## Forbidden dependencies

- [compiler](../../../glossary.md#term-compiler), [planning](../../../glossary.md#term-planning), [runtime](../../../glossary.md#term-runtime), [prepare](../../../glossary.md#prepare), [engine](../../../glossary.md#term-engine), [concrete backend](../../../glossary.md#concrete-backend) implementation, and [Tensor](../../../glossary.md#tensor) [API](../../../glossary.md#term-application-programming-interface-api) dependencies

## Package structure

```text
io.github.pho001.synaptik.backend.contract/
  <root>  current backend and backend-scoped device identities; later minimal declarative contracts
```

The root package remains a small public shared-contract surface. It contains no [concrete backend](../../../glossary.md#concrete-backend),
live service, discovery, registration, preparation, or execution behavior.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend and device identifiers](tasks/0001-backend-and-device-identifiers.md) | Complete | Completed [model](../../../glossary.md#term-model) milestone; [trace](../../../glossary.md#trace) 0001–0002 roadmap foundation | Replaced the placeholder with open [backend identity](../../../glossary.md#backend-identity--backendid) and [backend](../../../glossary.md#backend)-scoped [device](../../../glossary.md#term-device) [identity](../../../glossary.md#term-identity-meanings) values without registration, discovery, availability, or live services. |
| 0002 | [Device classification](tasks/0002-device-classification.md) | Complete | 0001 | Added the minimal [CPU](../../../glossary.md#term-cpu)-versus-[accelerator](../../../glossary.md#term-accelerator) device category needed by later availability and requirements without enumerating concrete devices or routes. |
| 0003 | [Backend availability snapshot](tasks/0003-backend-availability-snapshot.md)<span data-glossary-navigation="true"> ([Backend availability snapshot glossary](../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot))</span> | Complete | 0001–0002 | Added an [immutable](../../../glossary.md#term-immutable) caller-supplied map of one backend's currently available device identities to coarse classes, without discovery or live backend objects. |
| 0004 | [Declarative backend requirements](tasks/0004-declarative-backend-requirements.md) | Complete | 0001–0003 | Added the minimal [sealed](../../../glossary.md#term-sealed) exact-backend, exact-device, and device-class hard-eligibility vocabulary without [evaluation](../../../glossary.md#term-evaluation), preference, or [kernel](../../../glossary.md#kernel) selection. |

## Milestones

- [Identity](../../../glossary.md#term-identity-meanings) value types
- Availability and requirement DTOs
- Contract [validation](../../../glossary.md#term-validation)

## Current status

Complete. The explicit roadmap interleave after the stable [trace](../../../glossary.md#trace) foundation delivered tasks
0001–0004 in order: [identities](../../../glossary.md#term-identity-meanings), coarse classification, caller-supplied availability, and the
[sealed](../../../glossary.md#term-sealed) hard-requirement vocabulary. [Module](../../../glossary.md#term-module) [validation](../../../glossary.md#term-validation), independent documentation review, and the
repository capability [checkpoint](../../../glossary.md#term-checkpoint) passed. No later [backend](../../../glossary.md#backend)-contract task is specified or [Ready](../../../glossary.md#term-ready).

## Open questions

- No open questions remain for the completed selected milestone. Later requirement consumers own
  their own still-open policy and evaluation decisions.<span data-glossary-navigation="true"> ([evaluation glossary](../../../glossary.md#term-evaluation))</span>

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.
- Backend names remain an open nonblank string vocabulary rather than a closed CPU/Metal/CUDA
  enum.<span data-glossary-navigation="true"> ([Backend glossary](../../../glossary.md#backend); [Metal glossary](../../../glossary.md#term-metal); [CUDA glossary](../../../glossary.md#term-cuda))</span>
- Device identity is scoped by its owning backend identity plus an opaque nonblank backend-defined
  token. It does not prove discovery, availability, or resource access.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [backend identity glossary](../../../glossary.md#backend-identity--backendid); [opaque glossary](../../../glossary.md#term-opaque); [token glossary](../../../glossary.md#term-token))</span>
- [Device](../../../glossary.md#term-device) classification uses exactly [`CPU`](../../../glossary.md#term-cpu) and [`ACCELERATOR`](../../../glossary.md#term-accelerator). Current configuration and scoring
  language needs that distinction but provides no concrete need for GPU, NPU, FPGA, memory, or
  location subcategories.<span data-glossary-navigation="true"> ([GPU glossary](../../../glossary.md#term-gpu))</span>
- A class describes a reported device; it is not stored inside [`BackendDeviceId`](../../../glossary.md#backend-device-identity--backenddeviceid), does not classify
  execution routes, and carries no scoring order or availability guarantee.
- Availability is represented by one immutable snapshot per backend: an explicit backend identity
  plus a structurally immutable map from same-backend device identities to their classes. An empty
  map reports no currently available device and no ordering or separate status flag is implied.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [snapshot glossary](../../../glossary.md#term-snapshot))</span>
- Requirements form one sealed, method-free family with exact-backend, exact-device, and
  device-class record variants. They are hard eligibility targets only; config owns optionality
  and intent, while planning later owns evaluation and no-match failure.<span data-glossary-navigation="true"> ([sealed glossary](../../../glossary.md#term-sealed); [targets glossary](../../../glossary.md#term-target); [planning glossary](../../../glossary.md#term-planning); [evaluation glossary](../../../glossary.md#term-evaluation))</span>
- The trace project area is deliberately interleaved after its stable foundation because later
  trace payload and attribute schemas require concrete producer-owned contracts.<span data-glossary-navigation="true"> ([trace glossary](../../../glossary.md#trace); [trace payload glossary](../../../glossary.md#trace-payload--tracepayload); [schemas glossary](../../../glossary.md#term-schema); [producer glossary](../../../glossary.md#term-producer))</span>

## Risks

- Growing the [module](../../../glossary.md#term-module) into a [backend](../../../glossary.md#backend) service or execution abstraction.
- Treating [identity](../../../glossary.md#term-identity-meanings) as evidence that a backend or [device](../../../glossary.md#term-device) is registered, available, or supported.
- Growing the coarse device class into speculative hardware taxonomy or treating enum order as a
  planning preference.<span data-glossary-navigation="true"> ([device class glossary](../../../glossary.md#device-class--deviceclass); [planning glossary](../../../glossary.md#term-planning))</span>
- Treating a supplied availability snapshot as discovery, liveness monitoring, engine
  registration, capability support, or a device-selection policy.<span data-glossary-navigation="true"> ([snapshot glossary](../../../glossary.md#term-snapshot); [liveness glossary](../../../glossary.md#term-liveness); [engine glossary](../../../glossary.md#term-engine))</span>
- Expanding requirements into preferences, composition, predicates, capability language,
  scoring, or consumer-layer behavior.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 added only two [immutable](../../../glossary.md#term-immutable) [identity](../../../glossary.md#term-identity-meanings) values and package documentation. Its focused six-test
[run](../../../glossary.md#term-run) and final six-test/one-suite [module](../../../glossary.md#term-module) run passed. The separate documentation pass finalized the
identity Javadocs and explanations and passed [backend](../../../glossary.md#backend)-contract [Javadoc](../../../glossary.md#term-javadoc), repository Markdown,
exact twelve-path, status, and whitespace [validation](../../../glossary.md#term-validation) without rerunning Java tests. It added no
[dependency](../../../glossary.md#term-dependency), availability query, [registry](../../../glossary.md#term-registry), service, discovery, capability, [planning](../../../glossary.md#term-planning), [prepare](../../../glossary.md#prepare),
[runtime](../../../glossary.md#term-runtime), or [concrete backend](../../../glossary.md#concrete-backend) behavior.

Task 0002 added only the two-value declarative classification, one focused exact-[shape](../../../glossary.md#shape) test, and
the associated Javadoc/status documentation. Its focused three-test development runs and final
nine-test/two-suite module run passed. The separate documentation pass finalized the enum and
package Javadocs plus the public [API](../../../glossary.md#term-application-programming-interface-api), capability-provider guide, glossary, task, master plan, and
roadmap, then passed backend-contract Javadoc, repository Markdown, exact nine-path, status, and
whitespace validation without rerunning Java tests. Availability association, requirements,
configuration policy, scoring, and execution remain in their later owning tasks.

Task 0003 is limited to the caller-supplied immutable availability fact. Discovery, refresh,
capabilities, requirements, [ownership](../../../glossary.md#term-ownership-meanings) selection, preparation, and execution remain outside the
[snapshot](../../../glossary.md#term-snapshot) and in their later owning layers.

Task 0003 added only the single-backend immutable availability snapshot, one focused exact-
contract test, and the associated Javadoc/status documentation. Its two focused seven-test runs
and final 16-test/three-suite module run passed. The separate documentation pass finalized the
snapshot, package, and [`DeviceClass`](../../../glossary.md#device-class--deviceclass) Javadocs plus the public API, capability-provider guide,
glossary, task, master plan, and roadmap, then passed backend-contract Javadoc, repository
Markdown, exact ten-path,
status, and whitespace validation without rerunning Java tests. Discovery, registration,
refresh/[liveness](../../../glossary.md#term-liveness), requirements, capability [evaluation](../../../glossary.md#term-evaluation), ownership selection, preparation,
execution, and [trace](../../../glossary.md#trace) translation remain in their later owning tasks.

Task 0004 is limited to a [sealed](../../../glossary.md#term-sealed) hard-eligibility vocabulary. Its implementation, final
22-test/four-suite module run, and independent documentation stabilization succeeded. The single
final repository capability [checkpoint](../../../glossary.md#term-checkpoint) then passed 1,055 tests across 135 suites with no
failures, errors, or skips, closing the selected backend-contract milestone.

The post-checkpoint reassessment selected config task 0001 as the next implementation frontier.
Its current [`BackendIntent`](../../../glossary.md#backend-intent--backendintent) now consumes [`BackendRequirement`](../../../glossary.md#backend-requirement--backendrequirement) through an immutable optional hard-
intent value without reopening or changing this completed module. Config task 0001 and its
repository dependency checkpoint are Complete.
