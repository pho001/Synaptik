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
| 0002 | Bounded graph and plan tuning | Draft | 0001, compiler graph candidates, planning ownership/partition candidates, complete prepare candidates, and operational engine paths | Measure a budget-bounded set of complete valid candidates end to end, reuse local results without repeating local search, and select an explicit model plan or prepared artifact. |
| 0003 | Cache and plan inspection | Draft | 0001–0002, stable artifact schemas | Inspect compatibility, provenance summaries, invalidation reasons, selected plans, and separate measurement evidence without executing payloads or mutating runtime state. |

## Milestones

- Canonical workload reuse
- Bounded model-plan selection
- Persistent artifact validation and inspection

## Current status

Task 0001 is Complete. CPU 0010E and Prepare 0004 remain Complete. The current repository has no
Engine facade or general operational model-execution path, so task 0001 uses a narrow typed generic cold
collaboration supplied explicitly by composition: the caller supplies stable model/profile
identity, opaque candidate enumeration and decision encoding/decoding, and one complete
candidate execution action; tuning owns timing, validation, deduplication, cache coordination,
selection, evidence, and atomic persistence. The implementation passed its 24-test module gate,
clean Javadoc and rendered-text inspection, documentation checks, and the 3,032-test repository
checkpoint. This makes the local workflow operational without a
concrete CPU dependency, shared interpretation, Runtime work, or an invented Engine facade.

The initial slice remains exact/default and FLOAT32/FLOAT64 only. It does not wait for the blocked
BFLOAT16 side branch or unfinished relaxed numerical configuration. Config 0006A is now the next
Draft task and may add a public immutable request facade around this stable consumer. Tuning 0002
and 0003 remain Draft without detailed specifications.

## Open questions

- Exact public Config vocabulary and Engine composition remain deferred until their consumers and
  lifecycle paths are implemented.
- Prepared-executable serialization remains deliberately unresolved.
- Task 0002's graph/plan budget and end-to-end Engine measurement boundary remain unresolved.

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

## Risks

- Duplicating local route search in the graph/plan phase.
- Treating operation family as a family-wide configuration cache key.
- Letting shared orchestration interpret private backend fields or invent a generic knob language.
- Allowing implicit caches, unsafe deserialization, or stale compatibility matches.
- Expanding an end-to-end search beyond the explicit tuning budget.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/`
and follow [the planning guide](../../planning-guide.md).
