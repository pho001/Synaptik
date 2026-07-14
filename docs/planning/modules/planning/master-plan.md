# Planning Master Plan

## Goal

Make backend-neutral compile-time ownership, partitioning, capability, and logical memory decisions.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- backend intent propagation
- capability query contracts and matrices
- backend-neutral ownership scoring
- maximal same-owner partitioning
- logical materialization and memory requirements

## Out of scope

- fusion implementation
- concrete kernel or route selection
- physical allocation
- prepared execution and runtime residency

## Module invariants

- Planning answers where work runs, not which implementation runs it.
- Scoring uses compile-time information only.
- Scoring output is backend ownership.

## Allowed dependencies

- modules/model, with public visibility when planning signatures expose model types
- modules/config
- modules/backend-contract, with public visibility when planning signatures expose backend
  identity types
- modules/trace

## Forbidden dependencies

- runtime, prepare, engine, and concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.planning/
  capability/  public operation capability query and provider contracts; later matrix and
               hard-eligibility planning may refine this package map before becoming Ready
  ownership/   later backend-neutral candidates, scoring, and ownership decisions
  partition/   later maximal same-owner partition contracts
  memory/      later logical materialization and memory requirements
```

The module root is not a catch-all facade. Task 0001 opens only `capability` with the immutable
operation-occurrence question and inward-facing provider collaboration. It adds no matrix,
candidate, scoring, partition, memory, registry, or provider implementation.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Operation capability-query foundation](tasks/0001-operation-capability-query-foundation.md) | Complete | Completed model milestone, backend-contract 0001–0004, config 0001–0002, and trace foundation | Added one immutable operation-occurrence query and one backend capability-provider contract without a matrix, eligibility evaluation, diagnostics result, device query, or provider implementation. |
| 0002 | Capability matrix and hard eligibility | Draft | 0001, config hard intent, and backend-contract supplied availability | Build backend capability facts and combine them with hard eligibility without scoring or ownership selection. |
| 0003 | Ownership candidates and scoring | Draft | 0002, config partition-scoring configuration and profiles | Define backend-neutral candidates, compare eligible ownership choices, and produce ownership rather than implementation routes. |
| 0004 | Maximal same-owner partitioning | Draft | 0003 | Group adjacent work with the same selected backend owner without backend lowering or executable construction. |
| 0005 | Logical materialization and memory requirements | Draft | 0004 | Describe backend-neutral logical requirements without physical allocation, runtime residency, or prepared memory. |
| 0006 | Planning contract closure | Draft | 0001–0005 | Audit capability, ownership, partition, logical-memory, documentation, and dependency boundaries before compiler planning orchestration. |

## Milestones

- Intent and capability
- Ownership scoring
- Partitioning and logical memory

## Current status

In progress after completing the explicit narrow interleave before config task 0003. Task 0001 is
Complete with its typed operation-capability query/provider boundary, focused and repository test
evidence, and independent documentation pass. No planning task is Ready. Tasks 0002–0006 remain
ordered Draft work without detailed specifications. Config task 0003 is Complete with one
optional coarse device-class preference as soft scoring input; it adds no planning evaluator,
candidate, score, owner, or profile contract. No global task is Ready until a separate frontier
reassessment.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- The current capability contract asks whether one named backend can semantically own an
  immutable operation occurrence. It is distinct from availability, hard-eligibility evaluation,
  scoring, routing, preparation, and execution.
- Concrete backends may later implement the planning-owned inward-facing provider contract;
  planning does not depend on them and task 0001 added no implementation.
- Planning task 0001 intentionally interleaves before config task 0003 so the typed question is
  stable before a scoring preference is exposed. The frontier then returns to config 0003.
- Complete config task 0003 keeps hard eligibility separate from soft ranking through one
  `Optional<DeviceClass>` preference. Planning will later own interpretation after its candidate
  and profile prerequisites are stable.

## Risks

- Leaking backend route selection or runtime residency into compile-time scoring.
- Growing the foundation query into graph identity, availability, scoring, device selection, or
  rejection diagnostics before those consumers are concrete.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 adds only the query/provider contracts and package documentation, the two required
public dependency-visibility changes, focused planning and architecture tests, and synchronized
explanatory/planning documentation. Its implementation context passed 13 tests across four suites
with no failures, errors, or skips. The independent documentation context reused that evidence,
changed no executable behavior, and finalized the affected Javadocs, examples, terminology,
current-versus-planned boundaries, glossary entries, and planning records. The single final root
suite passed 1,079 tests across 141 suites with no skips, failures, or errors. A separate planning
step then made config task 0003 Ready; it is now Complete. Planning task 0002 remains Draft without
a detailed specification. Reassess the global frontier rather than assuming planning task 0002 or
0003 can advance while profile data and other prerequisites remain Draft.

Config task 0003 is now Complete after its focused config validation and independent
documentation pass. It changed no planning Java, provider contract, module dependency, or scoring
behavior. Planning tasks 0002–0006 remain Draft without detailed specifications, and the next
global frontier is intentionally deferred to a separate reassessment.
