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
| 0002 | Device classification | Draft | 0001 | Define the minimal declarative device classes needed by availability and requirements without enumerating concrete devices or routes. |
| 0003 | Backend availability snapshot | Draft | 0001–0002 | Define immutable caller-supplied backend and device availability facts without discovery or live backend objects. |
| 0004 | Declarative backend requirements | Draft | 0001–0003 | Define backend-neutral requirement values used by later config and planning without operation support logic or kernel selection. |

## Milestones

- Identity value types
- Availability and requirement DTOs
- Contract validation

## Current status

In progress through an explicit roadmap interleave after the completed trace envelope and stable
model-correlation foundation. Task 0001 is Complete. No backend-contract task is Ready; tasks
0002–0004 remain Draft without detailed specifications.

## Open questions

- Exact device classes remain open until task 0002 planning identifies the smallest vocabulary
  required by availability and declarative requirements.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Backend names remain an open nonblank string vocabulary rather than a closed CPU/Metal/CUDA
  enum.
- Device identity is scoped by its owning backend identity plus an opaque nonblank backend-defined
  token. It does not prove discovery, availability, or resource access.
- The trace project area is deliberately interleaved after its stable foundation because later
  trace payload and attribute schemas require concrete producer-owned contracts.

## Risks

- Growing the module into a backend service or execution abstraction.
- Treating identity as evidence that a backend or device is registered, available, or supported.
- Prematurely freezing device classes or requirement language before their consumers are concrete.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 added only two immutable identity values and package documentation. Its focused six-test
run and final six-test/one-suite module run passed. The separate documentation pass finalized the
identity Javadocs and explanations and passed backend-contract Javadoc, repository Markdown,
exact twelve-path, status, and whitespace validation without rerunning Java tests. It added no
dependency, availability query, registry, service, discovery, capability, planning, prepare,
runtime, or concrete backend behavior.
