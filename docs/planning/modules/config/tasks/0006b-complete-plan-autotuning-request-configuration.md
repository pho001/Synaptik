# Task 0006B: Complete-Plan Autotuning Request Configuration

## Status

Complete

## Goal

Extend the existing immutable `ModelAutotuningConfig` request with only the caller-owned limits
and explicit cache location required by the completed tools/tuning 0002 complete-plan transaction:

```text
existing Phase-1 request inputs
  + independent complete-plan timing/resource policy
  + explicit model-plan-cache path
  -> one Config-owned immutable request
  -> later Engine composition translates it into the tools/tuning Phase-2 request
```

Preserve every Config 0006A component and its meaning. This task declares no candidates, performs
no correctness execution or measurement, reads or writes no cache, selects no plan, and adds no
Engine workflow.

This is a second explicit staged ordering exception around Draft Config 0004–0006. Completed
tools/tuning 0002 now supplies the stable consumer contract that did not exist when Config 0006A
was planned. The work stays in the separate `config.tuning` package, does not consume the cost or
numerical-policy concepts owned by Draft tasks 0004–0006, and is the smallest truthful Config
follow-up before later Engine composition.

## Scope

- Append exactly two components to the existing `ModelAutotuningConfig` record after its five
  existing components, preserving the existing prefix order and semantics:
  1. `CompletePlanBudget completePlanBudget`; and
  2. `Path modelPlanCache`.
- Add exactly one nested public final record, `CompletePlanBudget`, with these components in order:
  1. positive `int maximumPlanCandidates`;
  2. non-negative `int warmupCount`;
  3. positive odd `int timedSampleCount`;
  4. positive `long maximumTotalPlanExecutions`; and
  5. non-negative `long maximumAggregateCorrectnessBytes`.
- The outer canonical constructor rejects null `completePlanBudget` and `modelPlanCache` values
  with component-named `NullPointerException` messages. It rejects a `modelPlanCache` whose
  `toString()` is empty with `IllegalArgumentException`. It retains the exact immutable budget and
  exact `Path` references.
- `CompletePlanBudget` validates the exact tools/tuning 0002 primitive ranges. It does not
  precompute a candidate count or repeat the transaction's checked total-execution formula because
  actual enumeration belongs to the later consumer.
- The new values are required and have no defaults. A compatibility constructor would need to
  invent resource and path policy that the stable consumer does not define, so downstream test
  fixtures must supply explicit inert values until later composition consumes them.
- Preserve the existing `objective`, Phase-1 `budget`, `representativeProfile`, `fallbackPolicy`,
  and `workloadCache` meanings exactly. Do not reuse `maximumCandidatesPerMiss` as the complete-
  plan candidate ceiling. The Phase-1 `Budget.warmupCount` and `Budget.timedSampleCount` continue
  to govern Phase 1 only. The same-named `CompletePlanBudget` components independently govern
  Phase 2, avoiding observable coupling between the transactions.
- Document that later Phase-2 preflight uses the independent complete-plan timing counts in the
  checked formula `N * (1 + W + S)`, where `N` is the actual complete-plan candidate count, `W`
  is `CompletePlanBudget.warmupCount`, and `S` is
  `CompletePlanBudget.timedSampleCount`.
- Document `maximumAggregateCorrectnessBytes` as a caller-enforced aggregate ceiling in bytes for
  the canonical publication payload used during correctness capture/comparison. Zero is valid. It
  is not a cache-file limit, a per-result allocation promise, or permission for Config to publish.
- Document `modelPlanCache` as an explicit requested location retained without normalization,
  resolution, existence checks, directory creation, or I/O. A later session-scoped transaction
  supplies the path but does not access it; a persistent producer may cause the tuning owner to
  use it under that owner's validation and publication rules.
- Preserve ordinary record equality, hashing, and diagnostic text. Equality includes the new
  nested record structurally and both `Path` values according to `Path.equals`; no canonical path
  or filesystem identity is inferred.
- Update existing Engine and integration test fixtures only as required to construct the extended
  Config record with explicit valid values. Do not change their assertions, production paths,
  tuning behavior, fallback behavior, or public Engine API.
- Finalize the affected Config Javadocs, package documentation, public API guide, benchmarking and
  tuning guide, glossary impact, and planning status in a separate clean documentation-focused
  context in the same overall change.

### Explicit translation to the stable Phase-2 consumer

Later Engine composition may depend on Config and tools/tuning and translate these independent
types. No translation code belongs in this task.

| Config value or source | tools/tuning 0002 input | Resolution |
|---|---|---|
| `completePlanBudget.maximumPlanCandidates` | `maximumPlanCandidates` | Copy the positive `int` exactly. |
| `completePlanBudget.warmupCount` | `warmupCount` | Copy the non-negative Phase-2 warmup count exactly; do not read the Phase-1 budget. |
| `completePlanBudget.timedSampleCount` | `timedSampleCount` | Copy the positive odd Phase-2 sample count exactly; do not read the Phase-1 budget. |
| `completePlanBudget.maximumTotalPlanExecutions` | `maximumTotalPlanExecutions` | Copy the positive `long` exactly; the tool checks actual enumeration and the independent Phase-2 timing counts. |
| `completePlanBudget.maximumAggregateCorrectnessBytes` | `maximumAggregateCorrectnessBytes` | Copy the non-negative byte count exactly; Engine's correctness collaboration enforces it. |
| `modelPlanCache` | `modelPlanCache` | Pass the exact path. The tool accesses it only for producer-declared persistent reuse. |
| Existing `objective` | `Objective` | Continue exhaustive enum translation; never use names or ordinals. |
| Existing `budget` | No Phase-2 request input | Preserve all four fields as Phase-1 policy only; no value controls Phase 2. |
| Existing `representativeProfile` | `ProfileFingerprint` | Continue explicit construction of a tool-local snapshot. |
| Actual model and target | `ModelFingerprint` and `TargetFingerprint` | Derive later from the actual lifecycle request and target/environment evidence. |
| Candidate constraints and producer contract | `PolicyIdentity`, compatibility, producer and codec identities | Derive later from composition and the backend-owned producer. |
| Exact correctness policy | `EXACT_CANONICAL_BYTES` | Fixed by the supported Engine 0008A/tools 0002 slice; no new Config enum is justified. |
| Candidate batch and decision-empty handoff | `handoff` | Supplied by the complete-plan producer after Phase 1; never stored in Config. |

## Out of scope

- Config 0004 cost-profile work, Config 0005 compile aggregation, Config 0006 numerical or
  determinism permission, and any change to their Draft status
- a Config dependency on tools/tuning, Engine, CPU, Runtime, Prepare, Compiler, Planning, Model, a
  concrete backend, or a provider
- target, policy, producer, codec, compatibility, candidate, decision, model, or handoff identities
- representative Tensor values, Shapes, resources, input bindings, publication descriptors,
  canonical bytes, correctness references, callbacks, or execution actions
- candidate generation or enumeration, measurement, timing, correctness execution, selection,
  fallback orchestration, selected-plan application, preparation, or Runtime execution
- cache lookup, cache-hit authentication, parsing, schemas, mutation, publication, locking,
  migration, eviction, inspection, default paths, directory creation, or filesystem probing
- alternative objectives, correctness-policy vocabulary, deadlines, adaptive sampling,
  tolerances, relaxed numerics, or generic parameter maps
- a detailed Engine task, Engine production changes, tools/tuning changes, CPU changes, Prepare
  changes, Runtime changes, Gradle changes, architecture tests, backend conformance, or new
  integration behavior
- architecture contract, architecture explanation, ADR, or dependency-rule changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle boundaries](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Completed Config 0006A](0006a-model-autotuning-request-configuration.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Completed tuning 0002](../../../tools/tuning/tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md)
- [Completed tuning 0003](../../../tools/tuning/tasks/0003-read-only-cache-plan-and-evidence-inspection.md)
- [Engine master plan](../../engine/master-plan.md)
- [Completed Engine 0008A](../../engine/tasks/0008a-representative-complete-plan-correctness-oracle.md)
- [Completed CPU 0010J](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
- [Public API](../../../../api/public-api.md)
- [Benchmarking and tuning guide](../../../../developer-guide/benchmarking.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Config owns immutable declarative inputs only. It owns no search, runner, mutable evidence,
  cache behavior, live discovery, backend vocabulary, service, executable, or Runtime state.
- Config must use only JDK and Config-owned values for this extension and must not add or expose a
  tools/tuning, Engine, CPU, Runtime, or Prepare dependency.
- The existing Phase-1 fields remain stable. Similar names do not authorize combining the
  per-miss candidate limit with the complete-plan candidate limit or letting either Phase-1 timing
  count control Phase 2. `CompletePlanBudget` owns separate Phase-2 timing counts so each phase can
  evolve without changing the other's observable policy.
- Candidate legality, compatibility, reuse scope, ordering, producer/codec identities, and
  decisions remain backend-owned. Current CPU `SESSION` reuse is a producer fact, not a Config
  option and not grounds to omit the generic consumer's explicit cache path.
- Correctness execution, canonical publication copying, aggregate-byte enforcement, resource
  cleanup, and fallback remain with Engine/later composition. Config stores the ceiling only.
- Tools/tuning retains checked budget enforcement, exact comparison sequencing, timing,
  selection, cache authentication, persistence, compact records, and evidence.
- Runtime never tunes. Later Engine composition may depend on both modules and translate
  exhaustively; the reverse dependency is forbidden.
- If implementation needs a default, compatibility constructor, another component or top-level
  type, new identity vocabulary, dependency, behavior, architecture decision, or path beyond the
  ceiling below, stop and return the task to planning.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.config.tuning` — extend its one immutable caller request and package
  documentation; add no package and no second top-level production type.

Type placement:

- `ModelAutotuningConfig` remains the sole top-level request facade.
- `ModelAutotuningConfig.CompletePlanBudget` is nested because its five values form one bounded
  Phase-2 policy used only by this request.
- `modelPlanCache` remains an outer component parallel to `workloadCache`: both are explicit
  caller locations, while all cache mechanics remain outside Config.

The outer component order becomes exactly:

```text
objective
budget
representativeProfile
fallbackPolicy
workloadCache
completePlanBudget
modelPlanCache
```

The first five form the exact existing prefix. No existing nested declaration changes shape.

## Affected files

Expected production and test paths:

- `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfig.java`
- `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/package-info.java`
- `modules/config/src/test/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfigTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/ModelAutotuningCompositionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineModelAutotuningIntegrationTest.java`

Expected documentation paths:

- `docs/api/public-api.md`
- `docs/developer-guide/benchmarking.md`
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction requires returning to planning:
architecture files, ADRs, Gradle files, tools/tuning source and tests, Engine production, CPU,
Prepare, Runtime, other Config types/tests/tasks, Engine planning, architecture tests,
backend-conformance tests, other integration tests, and unrelated documentation.

## Maximum scope

At most the exact twelve paths above:

- 2 Config production/Javadoc paths;
- 1 focused Config test path;
- 3 downstream construction-fixture test paths;
- 3 explanatory documentation paths; and
- 3 planning paths.

The downstream tests may change only their `ModelAutotuningConfig` construction arguments. The
task adds one nested public record and two outer components, but no top-level production type,
dependency, default, overload, factory, or behavior. If another path or broader change is needed,
stop and return the task to planning.

## Acceptance criteria

1. `ModelAutotuningConfig` remains one public final record and has exactly the seven ordered
   components listed above; the original five retain their types and meanings.
2. `CompletePlanBudget` is the sole new nested declaration and is a public static final record with
   exactly the five ordered primitive components in Scope.
3. The complete-plan budget accepts `(1, 0, 1, 1L, 0L)` and rejects non-positive plan candidates,
   negative warmups, non-positive or even timed-sample counts, non-positive total plan executions,
   and negative aggregate correctness bytes with field-specific `IllegalArgumentException`
   messages.
4. Both new outer components are required. Nulls fail with component-named
   `NullPointerException`; an empty-string model-plan path fails with `IllegalArgumentException`.
   No default or compatibility constructor is added.
5. Valid construction retains the exact budget and `Path` references and performs no path
   normalization, resolution, discovery, existence check, directory creation, or I/O.
6. Ordinary record equality, hashing, and text include all five complete-plan policy primitives
   and the exact model-plan path without filesystem-derived identity.
7. Every existing `Budget` component remains Phase-1-only. `CompletePlanBudget.maximumPlanCandidates`
   remains distinct from Phase-1 `maximumCandidatesPerMiss`; its `warmupCount` and
   `timedSampleCount` independently control Phase 2; and `maximumTotalPlanExecutions` is the whole
   Phase-2 transaction ceiling checked against `N * (1 + W + S)` using those Phase-2 counts.
8. `maximumAggregateCorrectnessBytes` uses bytes, permits zero, and is documented as a ceiling
   enforced by the later correctness collaboration over canonical publication data.
9. Javadocs state that `modelPlanCache` is explicit and required but a session-scoped transaction
   does not access it. No current CPU persistent-reuse claim is made.
10. No model, target, policy, producer, codec, compatibility, candidate, decision, handoff,
    correctness-policy, backend, cache-schema, measurement, evidence, or execution type enters
    Config.
11. Public signatures use only JDK and Config-owned declarations. Config adds no dependency,
    reflection, string dispatch, serialization, generic parameter bag, callback, service, adapter,
    or filesystem behavior.
12. The focused Config test locks the exact outer/nested public shape, every validation boundary,
    including non-negative Phase-2 warmups and positive odd Phase-2 timed samples, no-I/O path
    behavior, exact-reference retention, record behavior, and forbidden surfaces. It proves the
    existing Phase-1 `Budget` shape and meanings remain unchanged.
13. The three downstream test files change only explicit constructor fixtures and retain all
    existing Engine/integration assertions. No Engine production or behavior changes.
14. Package Javadoc, public API, benchmarking/tuning guide, and glossary distinguish declarative
    Phase-2 inputs from later composition and operational cache/correctness behavior.
15. Config 0004–0006 and 0007–0008 remain Draft. Config 0006B alone is Ready until implementation,
    and its second staged exception is synchronized in this task, the Config master, and roadmap.
16. Architecture files, ADRs, Gradle, tools/tuning, Engine production, CPU, Prepare, Runtime,
    architecture tests, and unrelated files remain unchanged.
17. A separate clean documentation-focused context finalizes affected Javadocs and explanatory
    documentation and records reasoned no-change conclusions before this task becomes Complete.
18. Focused and affected tests, Config Javadoc/rendering, repository validation, Markdown
    validation, exact twelve-path and dependency/public-shape checks, empty staging, and
    `git diff --check` pass.

## Tests / validation

During implementation, run the focused Config test, then affected consumers:

```bash
./gradlew :modules:config:test --tests io.github.pho001.synaptik.config.tuning.ModelAutotuningConfigTest
./gradlew :modules:engine:test --tests io.github.pho001.synaptik.engine.ModelAutotuningCompositionTest --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineModelAutotuningIntegrationTest
```

After Java stabilizes, run one repository validation because the public constructor change requires
fixtures in three project areas:

```bash
./gradlew test
```

Hand the final diff and Java evidence to a distinct clean documentation-focused context. That
context applies General, API/Javadoc, and Planning styles, avoids repeating successful Java suites
without a concrete reason, and runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short -uall
git diff --cached --stat
```

Render and inspect the generated package, outer type, and nested `CompletePlanBudget` pages. Check
changed Markdown links/anchors, unique headings, fences, LF/final newlines, whitespace, exact
status synchronization and scope, empty staging, dependency absence, and public shape.

The focused test must independently cover all five `CompletePlanBudget` components, their exact
order and primitive types, boundary-valid values, every invalid range including even timed-sample
counts, ordinary record equality/hash/text behavior, and the unchanged four-field Phase-1
`Budget`. Javadoc inspection must confirm distinct Phase-1 and Phase-2 timing semantics and the
checked Phase-2 `N * (1 + W + S)` relationship.

No performance run, real provider run, cache-I/O test, architecture-test change,
backend-conformance test, or new end-to-end behavior test is required. Existing integration
testing runs only because its explicit Config construction must remain compilable.

## Dependencies

- Config 0006A — Complete; owns the facade and stable Phase-1 meanings this task extends.
- tools/tuning 0002 — Complete; supplies the implemented and tested Phase-2 consumer contract.
- tools/tuning 0003 — Complete; confirms inspection does not move cache ownership into Config.
- CPU 0010J — Complete; supplies the current session-scoped producer and proves why the path
  cannot be documented as current CPU persistence.
- Engine 0008A — Complete; supplies the correctness owner and establishes the later composition
  boundary. The completed tuning 0002 contract supplies the exact five-field Phase-2 policy.
- Explicit user authorization for this staged ordering exception around Draft Config 0004–0006.

Config 0004–0006 are not dependencies because this task adds no planning-cost input, compile
aggregate, or relaxed-numerical permission. Engine composition is a follow-up, not a dependency.

## Follow-up tasks

- Plan one later Engine composition task only after Config 0006B is Complete. It should construct
  the tool-local complete-plan request, derive model/target/policy identities, adapt CPU 0010J and
  Engine 0008A collaborations, apply the selected decision through fresh preparation, and preserve
  fallback and cleanup semantics. Do not create that detailed task here.
- Persistent CPU reuse remains a separate producer/backend concern. Current CPU `SESSION` scope
  must continue to cause zero model-plan-cache I/O even though the caller supplied a path.
- Config 0004 remains Draft pending a stable cost-bearing consumer; 0005 remains Draft behind it;
  0006 remains Draft until a relaxed-numerics consumer exists; 0007–0008 remain Draft.

## Architecture impact

Expected impact: None.

The architecture already assigns stable declarative model-autotuning inputs to Config after their
consumer exists. This task realizes that boundary with no dependency or ownership change. If
implementation needs an architecture rule, ADR, Config-to-tool edge, backend vocabulary, cache
behavior, or public Engine workflow, stop without editing architecture files.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Config task 0006B. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read root AGENTS.md, ARCHITECTURE.md, the focused architecture documents named by the task,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the Config/tuning/Engine master plans,
Config 0006A, tools/tuning 0002–0003, CPU 0010J, Engine 0008A, and
docs/planning/modules/config/tasks/0006b-complete-plan-autotuning-request-configuration.md in full.
Inspect the current affected source and tests. Implement exactly the Ready task within its
twelve-path ceiling. Preserve all Phase-1 field meanings, add only CompletePlanBudget and the
explicit model-plan-cache path. CompletePlanBudget must contain, in order, maximum plan
candidates, its own Phase-2 warmup count, its own positive odd Phase-2 timed-sample count, maximum
total plan executions, and maximum aggregate correctness bytes. Stop for any architecture,
dependency, identity-ownership, default, cache-behavior, public-shape, or scope conflict.

After Java and affected tests stabilize, hand the exact diff and evidence to a distinct clean
documentation-focused context in the same overall change. That context must follow
documentation-rules.md, independently inspect final source/tests, finalize Javadocs and the three
explanatory documents, record no-change conclusions, and avoid repeating successful Java tests
absent a concrete reason. Neither context may commit or push. Do not mark 0006B Complete until
every specified gate succeeds.
```

## Local decisions

- Use task ID 0006B because this is the second adjacent extension of the Config 0006 tuning
  request line, not the cost profile, compile aggregate, or numerical permission in Draft 0004–0006.
- Append the two components so the existing five-component prefix and Phase-1 meanings remain
  visible and unchanged.
- Group the five independent Phase-2 timing/resource inputs in `CompletePlanBudget`; keep the
  cache path separate, matching the existing outer workload-cache pattern.
- Use `maximumAggregateCorrectnessBytes`, the implemented consumer name. “Correctness/publication
  bytes” describes the canonical publication payload, not a second field or cache-publication limit.
- Keep Phase-2 warmup and timed-sample counts separate from the same-named Phase-1 fields. This
  avoids coupling the two transactions and lets the Phase-2 counts feed its own checked
  `N * (1 + W + S)` execution ceiling without broadening any Config 0006A meaning.
- Require explicit values and reject an empty model-plan path. The consumer defines no safe
  resource or location defaults, and current session-only CPU behavior does not make it optional.
- Exclude model/target/policy identities. Model and target derive from the actual lifecycle
  request; policy derives from candidate constraints and producer facts.
- Limit downstream test changes to constructor fixtures instead of adding a compatibility
  constructor or prematurely implementing Engine translation.
- Record the exception without reopening or detailing Draft Config 0004–0006.

## Open questions

None for implementation. Later Engine planning must define model, target, and policy identity
derivations and behavior when CPU 0010J supplies no complete candidate batch. Those questions do
not widen or block this declarative Config task.

## Known limitations

- The new values are not consumed by current Engine production until the follow-up task.
- Current CPU complete-plan compatibility is session-scoped, so the path is not accessed.
- The request still supports exact canonical-byte correctness and the existing elapsed-time
  objective only; Phase 1 and Phase 2 have separate fixed-count timing policies.
- No executable or prepared-plan persistence is introduced.

## Planning evidence

- Planning/readiness audit context: `01a0ba6a-9d58-7d30-b703-24adbbb94cf8`.
- The audit began from a clean worktree and empty staging area and changed no production, tests,
  Gradle, architecture contract, or behavior.
- Source inspection confirmed the request types, units, ranges, no-I/O construction, checked
  execution formula, correctness-byte forwarding, and session/persistent path semantics.
- Current Config source/tests confirm the five-component Phase-1 facade. Current Engine and
  integration tests identify exactly three construction fixtures required by the constructor
  extension.
- CPU source and CPU 0010J confirm `ReuseScope.SESSION`; tuning source/tests confirm zero
  filesystem operations for that scope despite the required path.
- Engine 0008A confirms Engine owns correctness execution and records the later Config/cache
  follow-up boundary without authorizing a detailed Engine composition task. The completed tuning
  0002 consumer contract and tuning master refine the complete Config policy to include its own
  Phase-2 warmup and timed-sample counts.
- Completed Config 0006A fixes the five-component Phase-1 prefix and its four-field `Budget`.
  Completed tuning 0002 fixes the distinct positive `maximumPlanCandidates`, non-negative
  `warmupCount`, positive odd `timedSampleCount`, positive `maximumTotalPlanExecutions`,
  non-negative `maximumAggregateCorrectnessBytes`, and required explicit non-empty-string
  `modelPlanCache` path.
- Tuning 0002's checked transaction formula is
  `candidateCount * (1 + warmupCount + timedSampleCount)`. Giving Phase 2 its own two timing
  counts avoids coupling while keeping its candidate and total-execution ceilings internally
  coherent; no Phase-1 `Budget` field gains Phase-2 meaning.
- The broad tuning-task follow-up mention of target and policy identities is refined by the
  implemented request, CPU producer, and Engine oracle: those are evidence derived by later
  composition, not stable caller policy with a Config-owned schema.
- The architecture already authorizes Config-owned immutable request data and forbids the
  dependencies and behaviors excluded above. No missing architecture decision or ADR was found.

## Validation evidence

- Implementation context `01a0ba84-f0eb-76b3-8af4-4fc070f16b19` supplied the final executable
  evidence. The focused Config run passed 14/14 tests; the affected Engine run passed 36/36; the
  affected integration run passed 1/1; and one final `./gradlew test` passed 506 suites and 3,277
  tests with 28 skipped and zero failures or errors. Documentation context
  `01a0ba8e-5310-77f3-b6d1-459ca207e315` changed no executable Java or tests, found no concrete
  stale-evidence risk, and therefore did not repeat those successful suites.
- `./gradlew :modules:config:javadoc` passed with `BUILD SUCCESSFUL`; 3 actionable tasks ran, 2
  executed and 1 was up-to-date, with no Javadoc warning or error. Pandoc rendered and the
  documentation pass inspected the generated package summary, `ModelAutotuningConfig`, and
  `ModelAutotuningConfig.CompletePlanBudget` pages. The rendered contracts show all seven outer
  components, the five Phase-2 primitives, distinct Phase-1/Phase-2 timing policy, numeric units
  and ranges, the checked `N * (1 + W + S)` relationship, exact path retention/no I/O, current CPU
  session-only non-access, return semantics, and caller-visible failures.
- `python3 /tmp/validate_synaptik_markdown.py` passed for all six changed Markdown paths. It
  validated local targets and heading anchors, renderer-effective heading uniqueness, balanced
  backtick/tilde fences, LF endings, final newlines, and trailing whitespace.
- `javap -public` confirmed the outer record's exact seven-component canonical constructor and
  accessors and the nested complete-plan record's exact five primitive components and accessors.
  Source/dependency scans confirmed only JDK and Config-owned values, no Config dependency on
  tools/tuning, Engine, Runtime, Prepare, or a concrete backend, and no filesystem API or path
  normalization behavior. The focused automated shape test remains the durable exact declaration,
  validation, forbidden-surface, and unchanged Phase-1 contract check.
- Final scope inspection found exactly the twelve permitted worktree paths: the two Config
  production/Javadoc files, four implementation-owned tests, three explanatory documents, and
  three planning documents. No file was staged. `git diff --check`, `git status --short -uall`,
  and `git diff --cached --stat` passed with an empty cached diff.
- Status inspection confirmed Config 0006B is Complete in this task, the Config master plan, and
  the roadmap. Config 0004–0006 and 0007–0008 remain Draft. No later Engine task was detailed or
  marked Ready by this change.

## Implementation notes

- Appended exactly `CompletePlanBudget completePlanBudget` and `Path modelPlanCache` after the
  existing five outer components. Added exactly one nested public final record with the specified
  five validated Phase-2 primitives. No default, overload, compatibility constructor, top-level
  type, dependency, I/O, cache behavior, candidate vocabulary, or execution behavior was added.
- Preserved `Budget` as the unchanged four-component Phase-1 policy. Final Javadocs and
  explanatory documentation now state directly that its timing counts do not control Phase 2 and
  that the later complete-plan transaction uses its independent counts and actual candidate count
  for checked preflight.
- Updated both Public API constructor examples for the required seven-component record. The API,
  benchmarking guide, glossary, package documentation, and generated Javadocs distinguish current
  declarative Config ownership from later Engine translation, correctness execution, candidate
  production, tuning, cache mechanics, and fresh selected preparation.
- Architecture files and ADRs required no change because this task realizes the existing Config
  ownership of immutable declarative tuning inputs without changing a rule or dependency.
  Architecture tests required no change because module edges and forbidden dependencies are
  unchanged. Gradle required no change because the API uses only existing JDK/Config types.
- Tools/tuning required no change because its completed Phase-2 request, checked preflight,
  correctness/timing order, cache authentication, persistence, and evidence remain the consumer
  contract rather than Config behavior. Engine production and planning required no change because
  current `prepareTuned(...)` remains Phase-1-only; later Phase-2 composition is explicitly
  deferred. The three Engine/integration tests changed only constructor fixtures.
- CPU required no change because its current complete-plan producer remains `SESSION`; the
  explicit path is supplied by Config but is not accessed. Prepare and Runtime required no change
  because no handoff, prepared recipe, execution state, or hot-path behavior changed. Backend
  conformance required no change because no backend behavior changed.
- Other Config tasks and types required no change because Config 0004–0006 and 0007–0008 remain
  Draft and this extension preserves the existing compile package and Phase-1 declarations.
  Unrelated documentation required no change because the Public API, focused tuning guide,
  glossary, task, Config master, and roadmap cover the complete caller-visible and planning impact.

## Completion summary

- Completed changes: extended the declarative request with the independent validated Phase-2
  budget and exact model-plan-cache path; finalized type, constructor, component/accessor and
  package Javadocs; updated the Public API, benchmarking guide, glossary, and synchronized
  planning status without adding operational behavior.
- Files changed or created: exactly the twelve paths listed under Affected files; no other path
  changed and no new file was created.
- Tests and validation: reused the implementation context's passing 14-test Config, 36-test
  Engine, 1-test integration, and 3,277-test repository evidence; final Config Javadoc/rendering,
  six-file Markdown, public-shape/dependency, exact-scope, staging, and whitespace checks passed.
- Documentation-agent review: clean documentation context
  `01a0ba8e-5310-77f3-b6d1-459ca207e315` independently reviewed the implementation diff, affected
  source/tests, contracts, guides, glossary, and planning records and changed no executable Java.
- Documentation impact: Phase 1 and Phase 2 now have explicit independent mental models, limits,
  and current-versus-planned boundaries in caller and developer documentation.
- Javadoc review: all affected public records, canonical constructors, components/accessors,
  constraints, units, nullability, exact-reference ownership, no-I/O behavior, returns, and
  failures are documented and rendered successfully.
- Glossary impact: expanded the existing request-facade term rather than adding a competing term;
  it now covers both budgets, both explicit paths, operational exclusions, and current CPU
  session-only path behavior.
- Unresolved issues: None within task 0006B scope.
- Follow-up required: None for this task. Later Engine Phase-2 composition and persistent producer
  reuse remain separately planned capabilities.

Status: Complete
