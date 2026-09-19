# Tuning Master Plan

## Goal

Coordinate one explicit model-autotuning workflow that reuses compatible local workload results,
measures bounded complete plan candidates, and writes explicit persistent artifacts before runtime.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Performance evidence and model autotuning](../../../architecture/performance-evidence-and-tuning.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Runtime, prepare, and backend boundary](../../../architecture/runtime-prepare-backend-boundary.md)

## Scope

- model-guided extraction of actual tunable workloads and routes
- canonical workload signatures and identical-signature deduplication with occurrence weight and
  context retention
- compatible workload-cache reuse and miss-only local measurement
- bounded end-to-end comparison of complete valid graph and prepared-plan candidates
- explicit workload tuning cache and model-plan cache or prepared-plan record persistence
- rich measurement evidence and compact cache inspection

## Out of scope

- fixed-workload observational benchmark reporting
- runtime hot-path decisions, search, cache access, or graph inspection
- compiler graph semantics or transformations
- planning ownership policy
- backend lowering, route logic, candidate vocabulary, or kernel implementation
- hidden global state, Java object serialization, or a generic backend configuration language
- backend-owned generated-code caching, hidden-class lifetime, class definition, or persistent
  generated class-byte storage

## Module invariants

- One workflow owns coordinated local workload tuning and bounded graph/plan tuning.
- The graph/plan phase reuses local cache results and does not repeat route-parameter search.
- Model tuning is optional for correctness; cache-only preparation and safe backend heuristics
  remain valid fallbacks.
- Shared tooling sees backend candidates opaquely and does not interpret private fields.
- Cache load precedes measurement; compatible hits are reused, misses may be tuned and atomically
  persisted, and incompatible or corrupt entries fail safely.
- Rich evidence remains separate from compact cache state.
- Persistent tuning caches select compatible candidates; they are distinct from a concrete
  backend's bounded in-memory generated-artifact cache.

## Allowed dependencies

- modules/config for later stable immutable user request inputs
- modules/compiler and modules/planning through candidate contracts owned by those modules
- modules/prepare through the current public opaque candidate/decision roles and handoff
- modules/engine through a later public lifecycle orchestration boundary
- other explicitly required public lifecycle and trace contracts

## Forbidden dependencies

- private backend internals
- runtime service lookup or runtime hot-path integration

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Exact/default model-guided workload tuning and reusable cache](tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md) | Complete | CPU 0010E; Prepare 0004; caller-supplied stable identity, typed opaque collaboration, and operational cold measurement | Added the generic caller-supplied cold workload tuner, exact compatible-occurrence deduplication, bounded miss-only measurement, deterministic median selection, reusable bounded persistent cache, and separate rich evidence. Tool-local request inputs precede any later Config facade. |
| 0002 | Bounded graph and plan tuning | Blocked | 0001; Complete CPU 0010J; mandatory fresh re-audit of the implemented producer; unresolved truthful Phase-2 budget, correctness oracle, model-plan record/cache, tools-only scope, and later Engine composition | Measure a budget-bounded set of complete valid candidates end to end, reuse local results without repeating local search, and select an explicit model-plan record. CPU 0010J now supplies the first producer, but 0002 remains Blocked with no specification until a fresh evidence-based re-audit resolves every remaining owner and contract. |
| 0003 | Cache and plan inspection | Draft | 0001–0002, stable artifact schemas | Inspect compatibility, provenance summaries, invalidation reasons, selected plans, and separate measurement evidence without executing payloads or mutating runtime state. |

## Milestones

- Canonical workload reuse
- Bounded model-plan selection
- Persistent artifact validation and inspection

## Current status

Task 0001, Config 0006A, CPU 0010I, Engine 0006A–0008, and Prepare 0004 are Complete. The current
public Engine composition is operational but deliberately bounded to CPU-local Phase 1: it maps
at most one eligible workload to occurrence 0 in partition 0 with weight 1, measures complete
fresh trial executions, and freshly prepares the selected or safe-heuristic production recipe.
The generic cache-first workflow remains independent of CPU internals.

Task 0002 is blocked and has no detailed specification. Complete
[CPU 0010J supported complete-plan candidate and decision producer](../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
now supplies the first truthful owner-produced Phase-2 batch: retained CPU 0008D/0008E legal
fusion/split and materialization alternatives form one bounded supported set of opaque complete
one-partition candidates. The producer authenticates and reuses the exact Phase-1 result, exposes
session-scoped compatibility and decisions, and freshly prepares a selected recipe without
measuring it. It does not add Compiler graph alternatives, Planning ownership alternatives,
multiple partitions, mixed backends, model-plan persistence, or Engine composition.

A fresh evidence-based audit is now mandatory before task 0002 can become Ready or receive a
detailed specification. That audit must resolve a truthful Phase-2 candidate/execution budget,
the correctness oracle for comparing distinct complete plans, the model-plan record and cache
compatibility/persistence contract, a genuinely tools/tuning-only generic scope, and the separate
later Engine composition boundary. The mere existence of the CPU producer resolves none of those
questions. Config 0006A currently provides
`maximumDistinctCacheMisses`, `maximumCandidatesPerMiss`, `warmupCount`, and
`timedSampleCount` for Phase 1. It has no maximum-total-executions field, model-plan-cache path,
or Phase-2 candidate-count meaning, so a later Config change may follow only after the Phase-2
consumer contract is stable.

No compact model-plan record is implementable yet. CPU 0010J supplies only a session-scoped
owner-produced decision for the current sole-partition CPU slice; there is still no canonical
cross-compilation model fingerprint or persistent complete-plan compatibility schema.
Any later record must remain separate from rich evidence, carry explicit schema/owner/target/model/
profile/objective/constraint compatibility, reject corruption and incompatibility safely, publish
atomically, and trigger fresh preparation on a hit; it must never serialize `PreparedExecution`
or executable payloads.

## Open questions

- Prepared-executable serialization remains deliberately unresolved.
- The future Phase-2 request still needs an owner for maximum plan-candidate count, maximum total
  plan executions, and an explicit model-plan-cache path after the consumer stabilizes.
- Engine's current representative session proves lifecycle completion and cleanup, not numerical
  equivalence among distinct whole plans; the later Engine composition task must define its
  correctness oracle or fail closed before measurement.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Benchmarking is report-only and does not populate tuning caches.
- Running this same workflow over a representative model corpus may pre-seed the same workload
  cache for a target; there is no separate calibration abstraction or profile.
- Operation family selects a candidate generator but is not a universal cache key. Canonical
  signatures include exact semantic, data, layout, policy, and target-compatibility facts.
- Concrete backend prepare supplies the complete typed candidates for each operation occurrence
  or partition and workload. Candidate compatibility includes platform/provider availability,
  operation attributes, data type, `Shape`, layout, numerical/determinism requirements, and
  resource validity. Tuning never substitutes one fixed global vendor priority.
- Exact/default numerical semantics remove incompatible routes before measurement or cache
  comparison. A future explicit numerical policy is required before any vendor relaxed- or
  fast-math routine can enter a candidate set.
- Native-call overhead remains part of workload evidence: a small workload may retain a scalar or
  JDK Vector API candidate even when a compatible native provider is installed.
- Compiler, planning, prepare, and concrete backends generate complete valid candidates for their
  own decisions. Tuning coordinates measurement and selection only.
- Task 0001 accepts one typed caller collaboration parameterized by the method-free Prepare batch
  and decision roles. The collaborator enumerates a backend-owned complete batch, supplies opaque
  compatibility and candidate identities, constructs decisions, and encodes/decodes them. The
  tool never reflects on, downcasts, or imports a concrete backend value.
- Task 0001 accepts stable model and representative-profile fingerprints as explicit immutable
  tool-local request values. They identify evidence and do not create a Model or Engine identity
  contract. Workload compatibility remains supplied by the backend-aware collaborator so the
  workload cache can reuse identical results across models.
- The task-0001 cold measurement action performs one complete comparable candidate execution per
  invocation. Tuning owns warmup/sample counts, monotonic timing, elapsed-sample validation,
  median comparison, deterministic encounter-order tie breaking, and fail-before-measurement
  budget checks. Runtime and report-only benchmarks are not used.
- The first compact workload cache uses one bounded versioned binary format with length-prefixed
  opaque keys and decisions, fixed numeric objective/sampling fields, compact summaries, a whole-
  file checksum, deterministic ordering, and same-directory forced temporary-file plus atomic-
  move publication. Rich per-candidate samples remain only in the returned evidence.
- Backend candidate generators are typed, version-controlled, tested, and colocated with routes.
  They do not use `Map<String,Object>`, string dispatch, reflection annotations, a central knob
  registry, a generic config language, or arbitrary vector-lane promises.
- A model author supplies the model, target, representative input or shape profiles, objective,
  budget, constraints, and explicit caches. Backend authors define backend candidate spaces.
- Future workload and model-plan artifacts are explicit files with schema and backend candidate-
  schema versions, fingerprints, objective and constraints, and a measurement summary. Loads
  invalidate incompatible data and reject corruption safely; misses may be atomically persisted.
- A CPU tuning-cache hit selects a compatible route and specialization candidate; it does not
  contain or substitute for a loaded generated class. CPU backend finalization separately reuses
  or creates that candidate's executable through the CPU-owned in-memory artifact cache.
- Persistent generated class bytes are not part of the initial tuning workflow. They remain
  deferred until strict Synaptik-build, generator-schema, classfile/JDK, Vector API, target,
  validation, and corruption-handling compatibility is designed by the owning lifecycle.
- Config 0006A owns the current declarative request vocabulary only. Later outer composition owns
  any Phase-2 extension after the consumer stabilizes; it must not reinterpret Phase-1 miss and
  per-miss candidate bounds as plan or total-execution bounds.
- A generic Phase-2 tools foundation is not created before one current owner can supply complete
  valid plan candidates. A caller-supplied abstraction without a production producer would be
  speculative and would not prove Phase-1 reuse.
- Complete CPU 0010J is the first truthful producer because CPU already retains bounded executable
  fusion/split and materialization alternatives. Compiler and Planning remain architecture-only
  future candidate roles until they expose more than their current single deterministic result.
- Phase 2 may reuse a local selection only through an explicit typed handoff into complete-plan
  generation. Current Engine selected preparation is not that handoff and must not be described
  as reuse.

## Risks

- Duplicating local route search in the graph/plan phase.
- Treating operation family as a family-wide configuration cache key.
- Letting shared orchestration interpret private backend fields or invent a generic knob language.
- Allowing implicit caches, unsafe deserialization, or stale compatibility matches.
- Expanding an end-to-end search beyond the explicit tuning budget.
- Persisting an ad hoc caller label as though it were a canonical model/plan compatibility key.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/`
and follow [the planning guide](../../planning-guide.md).
