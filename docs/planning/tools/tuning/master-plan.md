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

## Package structure

```text
io.github.pho001.synaptik.tools.tuning/
  WorkloadTuning*              current public Phase-1 request, workflow, result, and evidence
  BackendWorkloadTuning        current typed Phase-1 backend collaboration
  ColdCandidateMeasurement     current one-complete-local-candidate action
  WorkloadCacheFile            current package-private workload-cache persistence
  CompletePlanTuning*          current public Phase-2 request, workflow, result, and evidence
  BackendCompletePlanTuning    current typed complete-plan backend collaboration
  CompletePlanCorrectness      current opaque-reference exact comparison collaboration
  CompleteCandidateMeasurement current one-complete-plan action
  ModelPlanCacheFile           current package-private model-plan cache persistence
  TuningInspection             current public read-only cache/evidence inspection namespace
```

The root package remains the deliberate small public surface for both coordinated phases. Task
0002 added no backend, Engine, Runtime, Config, cache, internal, utility, or registry package.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Exact/default model-guided workload tuning and reusable cache](tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md) | Complete | CPU 0010E; Prepare 0004; caller-supplied stable identity, typed opaque collaboration, and operational cold measurement | Added the generic caller-supplied cold workload tuner, exact compatible-occurrence deduplication, bounded miss-only measurement, deterministic median selection, reusable bounded persistent cache, and separate rich evidence. Tool-local request inputs precede any later Config facade. |
| 0002 | [Bounded complete-plan tuning and model-plan cache](tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md) | Complete | 0001; Complete CPU 0010J and Engine 0008A evidence | Added the generic tools-only complete-plan transaction with checked execution budgets, opaque exact correctness, deterministic timing selection, strict SESSION no-I/O, authenticated PERSISTENT reuse, a compact selected-plan record/cache, and separate rich evidence. Later Config and Engine composition remain out of scope. |
| 0003 | [Read-only cache, plan, and evidence inspection](tasks/0003-read-only-cache-plan-and-evidence-inspection.md) | Complete | 0001–0002, stable artifact schemas | Added bounded Path/byte inspection for both compact formats, redacted provenance and selected-candidate summaries, typed structural and exact-key mismatch reasons, and separate summaries of existing rich evidence without decoder, execution, or mutation. |

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

Task 0002 is Complete. It implements the generic tools-only Phase-2 transaction selected by the
required fresh post-0008A source audit. Complete
[CPU 0010J supported complete-plan candidate and decision producer](../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
now supplies the first truthful owner-produced Phase-2 batch: retained CPU 0008D/0008E legal
fusion/split and materialization alternatives form one bounded supported set of opaque complete
one-partition candidates. The producer authenticates and reuses the exact Phase-1 result, exposes
session-scoped compatibility and decisions, and freshly prepares a selected recipe without
measuring it. It does not add Compiler graph alternatives, Planning ownership alternatives,
multiple partitions, mixed backends, model-plan persistence, or Engine composition.

The mandatory fresh post-CPU-0010J audit selected
[Engine 0008A representative complete-plan correctness oracle](../../modules/engine/tasks/0008a-representative-complete-plan-correctness-oracle.md)
as the missing owning prerequisite. Engine 0008A is now Complete. Its package-private
representative-session primitive preflights the ordered publication boundary and aggregate byte
limit, captures one opaque same-session reference from a fresh complete execution, and reports
only exact `MATCH` or `MISMATCH` for later fresh complete executions after every copy and result
cleanup succeed. It preserves aliases, empty values, order, signed zero, and NaN payload bits and
does not alter current Phase-1 completion-only measurement.

The fresh post-0008A audit verified the generic callback boundary against implemented source,
reconciled CPU 0010J association/lifetime with it, and confirmed the complete budget, evidence,
compact-record/cache, Config, and later Engine-composition sequence. The completed
[task 0002](tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md) remains bounded to
`tools/tuning`: CPU and Engine are producer evidence, not tool dependencies.

Fresh post-0008A planning/documentation audit context
`01a0b964-81a3-7cb2-a1dc-ec44487865d1` created the task specification and passed the planning-only
Markdown, scope, diff, and staging validation recorded in its completion report.

The resolved Phase-2 budget is fail-closed and known before measurement. For `N` cache-miss
candidates, `W` warmups, and `S` timed samples, correctness uses one separate execution per
candidate: the first creates the reference and `N - 1` compare with it. The exact total is
`N * (1 + W + S)`. A later tool-local request must validate a positive candidate ceiling,
non-negative `W`, positive odd `S`, a positive maximum-total-executions ceiling, and every checked
addition/multiplication before the first execution. All `N` correctness runs must match and clean
up before the first warmup or timed sample. A compatible cache hit performs zero correctness,
warmup, or timed executions. Preflight failure, mismatch, or execution failure publishes no
partial evidence or cache state.

The correctness callback/result is implemented. Tuning invokes a caller-supplied opaque
reference capture for the first encounter-ordered candidate and a comparison action for every
later candidate. It receives only a typed `MATCH` or `MISMATCH`, never Tensor descriptors,
publication bytes, Runtime representations, or backend fields. Engine 0008A owns exact represented-
byte equality across every ordered publication occurrence, including aliases, signed zero, and
NaN payloads; it owns the reference, materialization, and cleanup. Tolerance and relaxed numerical
policies are outside the first slice.

The model-plan artifact contract is implemented independently of CPU 0010J's current persistence
limit. One bounded versioned compact entry keys an opaque producer/decision-codec identity,
exact caller-supplied model, representative-profile, and target fingerprints; backend
compatibility schema/bytes and reuse scope; objective; constraint/correctness-policy identity; and
sampling fields. It stores only selected candidate identity, backend-encoded decision bytes,
and a compact timing summary. Rich raw samples and
correctness evidence remain only in the returned evidence. It uses bounded length-prefixed
parsing, a whole-file checksum, canonical ordering, and forced same-directory temporary-file plus
atomic-replacement publication. Missing or incompatible entries are misses; corrupt, oversized,
unsupported, duplicate, or structurally invalid artifacts fail before execution and remain
unmodified. Every hit is authenticated by the current backend decoder and ends in fresh selected
preparation; no `PreparedExecution` or executable state is serialized.

CPU 0010J honestly returns `SESSION`. That does not require another CPU target-fingerprint task:
the generic transaction skips filesystem lookup and publication for a `SESSION` batch, always
measure it, and return only an in-memory compact record plus rich evidence. The same generic
format may persist a future `PERSISTENT` producer. Caller-supplied stable model/profile/target
fingerprints label and key the explicit artifact without inventing a canonical Compiler or Engine
model identity; backend compatibility and compatible decision decoding remain reuse authority.

The implemented tools-only boundary is narrow: tuning validates and snapshots opaque candidates and
identities, preflight the total budget, order correctness then warmup/timed actions, select the
lowest integer-middle median with encounter-order ties, construct backend decisions, coordinate
the compact artifact, and return rich evidence. It must not import Engine or CPU, interpret
private candidate fields, bind representative inputs, materialize publications, own cleanup or
fallback policy, mutate Runtime, or rerun Phase-1 route ranking. Config 0006A currently provides
`maximumDistinctCacheMisses`, `maximumCandidatesPerMiss`, `warmupCount`, and
`timedSampleCount` for Phase 1. It has no maximum-total-executions field, model-plan-cache path,
or Phase-2 candidate-count meaning, so a later Config change may follow only after the Phase-2
consumer contract is stable.

Task 0003 read-only cache, plan, and evidence inspection is Complete. One field-free tools-only
namespace now inspects bounded Path or byte snapshots of both stable schema-1 formats and summarizes
existing rich evidence without a backend decoder, Engine/CPU import, execution, or mutation. It
reports exact key matches as requiring decoder authentication, treats `SESSION` as persistently
ineligible, redacts opaque values to schema/length/SHA-256 summaries, and preserves task-0001/
task-0002 bytes and operational loader behavior. Separately planned Config extension and
Engine-owned CPU composition remain follow-ups for public Phase-2 use, not inspection prerequisites.

## Open questions

- Prepared-executable serialization remains deliberately unresolved.
- No Phase-2 tools blocker remains. Complete Engine 0008A proves the package-private exact-output
  reference/comparison seam through current publication, host-copy, result-cleanup, and
  representative-session contracts; completed task 0002 consumes only a caller-adapted opaque
  reference and typed match/mismatch result.
- The later Config follow-up must add distinct Phase-2 maximum-plan-candidate,
  maximum-total-plan-execution, maximum-correctness-byte, and explicit model-plan-cache inputs
  without changing the four Phase-1 budget meanings. The later Engine composition then owns CPU
  0010J adaptation, evidence translation, fresh selected preparation, and public fallback.

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
- Workload and model-plan artifacts are explicit files with schema and backend candidate-
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
- The generic Phase-2 tools foundation was created only after one current owner supplied complete
  valid plan candidates. CPU 0010J provides that production evidence without becoming a tools
  dependency.
- Complete CPU 0010J is the first truthful producer because CPU already retains bounded executable
  fusion/split and materialization alternatives. Compiler and Planning remain architecture-only
  future candidate roles until they expose more than their current single deterministic result.
- Phase 2 may reuse a local selection only through an explicit typed handoff into complete-plan
  generation. Current Engine selected preparation is not that handoff and must not be described
  as reuse.
- The first Phase-2 correctness policy is exact canonical represented-byte equality over every
  ordered publication occurrence. The first candidate supplies the Engine-owned reference; every
  candidate receives one separate correctness execution before any warmup or timed sample for
  that candidate set. Tolerance and relaxed numerical policies require later explicit contracts.
- Phase-2 budget preflight uses `N * (1 + W + S)` with checked arithmetic and a distinct maximum-
  total-executions ceiling. All correctness runs precede timing; cache hits execute zero trials.
  No preflight failure or correctness mismatch may publish partial evidence or cache state.
- Current CPU 0010J `SESSION` compatibility is sufficient for the generic tool: it disables
  persistent lookup/publication for that transaction without blocking in-memory selection or a
  future `PERSISTENT` producer. A new CPU target-fingerprint task is not required before 0002.
- A compact model-plan record keys an opaque producer/decision-codec identity, caller-supplied
  model/profile/target fingerprints, backend compatibility, objective, constraint/correctness
  policy, and sampling facts, and stores only candidate identity, encoded decision, and compact
  summary. Rich evidence remains separate; backend decoding and fresh preparation authenticate
  every hit.
- Complete Engine 0008A is the smallest upstream prerequisite because output semantics, canonical
  copying, Runtime result cleanup, and the reference lifecycle are Engine-owned. The completed
  generic task accepts only opaque actions and typed match/mismatch results and imports no Engine or CPU
  type; later Engine composition supplies the adapters.

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
