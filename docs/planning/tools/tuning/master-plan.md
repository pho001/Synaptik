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

## Module invariants

- One workflow owns coordinated local workload tuning and bounded graph/plan tuning.
- The graph/plan phase reuses local cache results and does not repeat route-parameter search.
- Model tuning is optional for correctness; cache-only preparation and safe backend heuristics
  remain valid fallbacks.
- Shared tooling sees backend candidates opaquely and does not interpret private fields.
- Cache load precedes measurement; compatible hits are reused, misses may be tuned and atomically
  persisted, and incompatible or corrupt entries fail safely.
- Rich evidence remains separate from compact cache state.

## Allowed dependencies

- modules/config for stable immutable user request inputs
- modules/compiler and modules/planning through candidate contracts owned by those modules
- modules/prepare and modules/engine through the future public opaque orchestration boundary
- other explicitly required public lifecycle and trace contracts

## Forbidden dependencies

- private backend internals
- runtime service lookup or runtime hot-path integration

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Model-guided workload tuning and reusable cache | Draft | Stable model/target identity, prepare orchestration, concrete backend typed candidate generators, and artifact compatibility contracts | Extract actual workloads, form canonical signatures, deduplicate with occurrence context, reuse compatible explicit cache hits, measure only misses, and atomically persist compact results while retaining separate rich evidence. |
| 0002 | Bounded graph and plan tuning | Draft | 0001, compiler graph candidates, planning ownership/partition candidates, complete prepare candidates, and operational engine paths | Measure a budget-bounded set of complete valid candidates end to end, reuse local results without repeating local search, and select an explicit model plan or prepared artifact. |
| 0003 | Cache and plan inspection | Draft | 0001–0002, stable artifact schemas | Inspect compatibility, provenance summaries, invalidation reasons, selected plans, and separate measurement evidence without executing payloads or mutating runtime state. |

## Milestones

- Canonical workload reuse
- Bounded model-plan selection
- Persistent artifact validation and inspection

## Current status

Draft. The former broad platform-calibration row is retired. No tuning task is Ready, and no
detailed task specification exists. These rows wait for the actual compiler, planning, prepare,
engine, concrete backend candidate, and artifact-lifecycle contracts.

## Open questions

- Exact canonical signature, target fingerprint, model fingerprint, objective, constraint,
  measurement-summary, and candidate-schema fields wait for their stable producers and consumers.
- The physical file encoding and prepared-executable serialization remain deliberately unresolved.
- The budget policy and end-to-end measurement boundary wait for operational lifecycle paths.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Benchmarking is report-only and does not populate tuning caches.
- Running this same workflow over a representative model corpus may pre-seed the same workload
  cache for a target; there is no separate calibration abstraction or profile.
- Operation family selects a candidate generator but is not a universal cache key. Canonical
  signatures include exact semantic, data, layout, policy, and target-compatibility facts.
- Compiler, planning, prepare, and concrete backends generate complete valid candidates for their
  own decisions. Tuning coordinates measurement and selection only.
- Backend candidate generators are typed, version-controlled, tested, and colocated with routes.
  They do not use `Map<String,Object>`, string dispatch, reflection annotations, a central knob
  registry, a generic config language, or arbitrary vector-lane promises.
- A model author supplies the model, target, representative input or shape profiles, objective,
  budget, constraints, and explicit caches. Backend authors define backend candidate spaces.
- Future workload and model-plan artifacts are explicit files with schema and backend candidate-
  schema versions, fingerprints, objective and constraints, and a measurement summary. Loads
  invalidate incompatible data and reject corruption safely; misses may be atomically persisted.

## Risks

- Duplicating local route search in the graph/plan phase.
- Treating operation family as a family-wide configuration cache key.
- Letting shared orchestration interpret private backend fields or invent a generic knob language.
- Allowing implicit caches, unsafe deserialization, or stale compatibility matches.
- Expanding an end-to-end search beyond the explicit tuning budget.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/`
and follow [the planning guide](../../planning-guide.md).
