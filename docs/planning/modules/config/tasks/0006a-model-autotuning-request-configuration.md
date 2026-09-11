# Task 0006A: Model-Autotuning Request Configuration

## Status

Complete

## Goal

Add the smallest public immutable Config facade for the user-owned policy and identity inputs
that are stable after completed tools/tuning task 0001:

```text
objective + bounded sampling budget + representative-profile identity
          + fallback policy + explicit workload-cache path
  -> ModelAutotuningConfig
  -> later composition translates Config values into tool-local request values
```

The facade describes a request. It does not enumerate candidates, run measurements, interpret
backend values, read or write a cache, select a decision, prepare an executable, or orchestrate
Engine or Runtime behavior.

This is an explicit staged ordering exception. Config tasks 0004–0006 remain Draft because their
cost-bearing and relaxed-numerics consumers are not stable. Task 0006A may proceed now because
completed tuning 0001 supplies its exact/default consumer evidence, this task uses the separate
`config.tuning` package and exact paths below, and it neither depends on nor overlaps those Draft
tasks. The exception changes planning order only, not architecture dependency direction.

## Pre-implementation behavior and evidence

- `modules/config` currently exposes four immutable leaves only under `config.compile`; it has no
  model-autotuning package, facade, objective, budget, representative-profile identity, fallback
  policy, or cache-location value.
- Completed tools/tuning 0001 exposes `WorkloadTuningRequest` with the sole objective
  `MIN_MEDIAN_ELAPSED_NANOS`, a four-field bounded `Budget`, a caller-defined
  `ProfileFingerprint`, a caller-defined `ModelFingerprint`, ordered Prepare handoff occurrences,
  and one explicit workload-cache `Path`.
- The tuning consumer owns measurement, candidate comparison, decision reuse, evidence, cache
  parsing, and atomic persistence. It intentionally accepts tool-local inputs and has no Config
  dependency.
- Tuning 0001 validation passed 24 tool tests and the recorded repository checkpoint. Its public
  contract is therefore stable enough to justify a Config-owned user request vocabulary without
  making Config depend on the outer tool.
- The initial implemented tuning slice admits only exact/default FLOAT32/FLOAT64 candidates.
  Relaxed numerical permission remains Draft Config 0006 work and is not required by this task.

## Scope

- Add one public final record
  `io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig` with exactly these five
  components, in this order:
  1. `Objective objective`;
  2. `Budget budget`;
  3. `RepresentativeProfileIdentity representativeProfile`;
  4. `FallbackPolicy fallbackPolicy`; and
  5. `Path workloadCache`.
- The canonical constructor rejects a null component with `NullPointerException` naming that
  component. It retains the exact immutable nested values and exact `Path` reference. It performs
  no filesystem access, path normalization, directory creation, cache validation, discovery, or
  default selection.
- Nest exactly four public policy/value declarations inside `ModelAutotuningConfig`:
  - `Objective` is an enum with the sole constant `MIN_MEDIAN_ELAPSED_NANOS`.
  - `Budget` is a record with, in order, positive `maximumDistinctCacheMisses`, positive
    `maximumCandidatesPerMiss`, non-negative `warmupCount`, and positive odd
    `timedSampleCount`. Its constructor validates those conditions before any consumer work.
  - `RepresentativeProfileIdentity` is a public final immutable class with a positive
    `schemaVersion` and non-empty opaque `byte[]` identity. Construction snapshots the array;
    `bytes()` returns a fresh copy; equality and hashing compare schema version and byte content;
    diagnostic text reports the schema version and byte length without exposing bytes. Its exact
    public surface is constructor `RepresentativeProfileIdentity(int schemaVersion, byte[] bytes)`,
    accessors `int schemaVersion()` and `byte[] bytes()`, plus ordinary value
    `equals(Object)`, `hashCode()`, and `toString()` overrides; it exposes no other method.
  - `FallbackPolicy` is an enum with exactly `REQUIRE_TUNED_RESULT` and
    `ALLOW_SAFE_HEURISTIC`.
- Define fallback semantics narrowly:
  - `REQUIRE_TUNED_RESULT` tells later composition to report failure if the requested tuning
    transaction cannot produce a complete selected result.
  - `ALLOW_SAFE_HEURISTIC` permits later composition to abandon an unavailable or failed tuning
    transaction and continue through the ordinary safe heuristic preparation path. It does not
    turn corrupt/incompatible cache data into a hit, grant candidate eligibility, accept partial
    selections, suppress unrelated preparation failures, or require Config to catch exceptions.
- Add no enabled/disabled flag or implicit default. Possessing a `ModelAutotuningConfig` means
  tuning was requested; absence and aggregation belong to a later composition owner.
- Use one representative-profile identity because the implemented tuning request accepts one
  `ProfileFingerprint`. Do not store actual Tensor values, shapes, input maps, suppliers,
  callbacks, resources, or descriptive strings. The later Engine/composition boundary owns
  representative inputs and proves equivalent complete candidate execution.
- Keep the model fingerprint out of Config. Later composition derives or supplies the tool-local
  model evidence identity from the actual model/lifecycle request; Config has no stable canonical
  Model identity contract to duplicate.
- Keep numerical/determinism constraints out of this type. The current tuning candidate set is
  fixed to exact/default eligibility, while Draft Config 0006 remains the sole future owner of
  explicit relaxed numerical permission.
- Add package documentation that distinguishes immutable request data from tuning, cache,
  backend, preparation, Engine, and Runtime behavior.
- Add one focused test class covering exact public shape, constructor validation, immutable byte
  snapshots, equality/hash/text behavior, ordinary outer-record behavior, exact enum values,
  exact component order/types, and absence of backend/tool/measurement/persistence/lifecycle
  surfaces.
- Finalize all new Javadocs and the focused public API, tuning guide, glossary, and planning
  status in a separate clean documentation-focused context in the same overall change.

### Explicit translation to the stable consumer

Later composition may depend on both Config and tools/tuning and must translate explicitly. No
translation code belongs in this task:

| Config value | tools/tuning 0001 input | Resolution |
|---|---|---|
| `Objective.MIN_MEDIAN_ELAPSED_NANOS` | `WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS` | One exhaustive enum switch; never `valueOf`, string dispatch, or a Config-to-tool dependency. |
| Four `Budget` components | Same four `WorkloadTuningRequest.Budget` components | Copy one-for-one after Config validation; the tool repeats its own boundary validation. |
| `RepresentativeProfileIdentity(schemaVersion, bytes)` | `WorkloadTuningRequest.ProfileFingerprint(schemaVersion, bytes)` | Construct a tool-local snapshot from the Config accessor copy. |
| `workloadCache` | `WorkloadTuningRequest.workloadCache` | Pass the exact requested path; tuning remains responsible for cache I/O and validation. |
| `FallbackPolicy` | No `WorkloadTuningRequest` field | Composition decides whether failure ends the request or enters ordinary safe heuristic preparation; tuning 0001 remains unchanged. |
| Actual model identity | `WorkloadTuningRequest.ModelFingerprint` | Deferred to composition because it owns the actual model request and evidence identity. |
| Tunable occurrences, weights, and contexts | `WorkloadTuningRequest.Occurrence` list | Deferred to Prepare/backend-aware composition; Config owns no handoff or backend candidate type. |
| Exact/default numerical eligibility | Candidate batches supplied to tuning | Already enforced by current candidate producers; future relaxed permission remains Config 0006 and requires a later consumer update. |
| Representative input values/resources | `ColdCandidateMeasurement` behavior | Deferred to Engine/composition; no live input or execution resource enters Config. |

The table is a boundary map, not a second API. It shows that a later outer composition layer can
translate stable values without adding the forbidden `modules/config -> tools/tuning` dependency.

## Out of scope

- a Config dependency on tools/tuning, Prepare, Runtime, Engine, Model, Planning, Compiler, a
  concrete backend, or a provider
- `WorkloadTuningRequest` changes, a translation adapter, Engine facade, compile/prepare/run
  aggregate, dependency injection, registration, discovery, or orchestration
- model fingerprints, model objects, Tensor values, Shapes, input maps, input suppliers,
  callbacks, execution actions, resources, or representative-input validation
- Prepare handoffs, occurrences, occurrence contexts or weights, candidate batches, candidates,
  decisions, compatibility identities, candidate identities, backend routes, thread counts,
  vector species, layouts, materialization choices, kernels, or executable artifacts
- candidate generation, eligibility, timing, warmup execution, measurement, comparison, median
  calculation, selection, correctness checking, evidence, profiling, tracing, or Runtime behavior
- cache loading, parsing, compatibility, corruption handling, mutation, persistence, atomic move,
  locking, eviction, migration, inspection, file-format/schema ownership, model-plan cache, or
  prepared-executable serialization
- alternative objectives, wall-clock deadlines, adaptive sampling, confidence thresholds,
  energy/memory objectives, multi-objective ranking, or arbitrary constraint/parameter maps
- relaxed or fast-math permission, determinism configuration, backend-specific numerical policy,
  Config 0004–0006 implementation, or changing their Draft status
- Config 0007–0008, tuning 0002–0003, another task specification, or any code outside
  `modules/config`
- architecture contract, ADR, dependency rule, Gradle/build, architecture-test,
  backend-conformance, integration-test, or unrelated documentation changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/config`, Concrete backend modules, Performance evidence and optimization tooling,
  Runtime service locator, and Dependency rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [ADR 0008: Separate performance evidence and model-autotuning concerns](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Completed tuning 0001 consumer](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Public API](../../../../api/public-api.md)
- [Benchmarking and tuning guide](../../../../developer-guide/benchmarking.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Config owns immutable declarative request inputs only. It owns no runner, search, cache
  mutation, measurement algorithm, live discovery, mutable evidence, service, executable, or
  Runtime state.
- Model autotuning completes before Runtime hot-path execution. Runtime profiling remains passive
  and cannot select settings.
- Backend authors retain candidate vocabulary, generation, compatibility, decisions, and route
  semantics. No backend field or generic parameter bag enters Config.
- Tuning retains measurement, comparison, selection, evidence, cache parsing, and persistence.
  Similar field names do not transfer those responsibilities to Config.
- The Config API uses JDK and Config-owned values only. `modules/config` must not depend on
  tools/tuning or add another module/Gradle edge.
- Later outer composition may depend on both modules and use exhaustive typed translation. Config
  must not call outward or mirror tool implementation mechanics beyond the stable user-owned
  values identified in the translation table.
- Exact/default eligibility remains a producer/backend fact for the current slice. An objective
  favoring elapsed time and `ALLOW_SAFE_HEURISTIC` never grant relaxed mathematics or candidate
  eligibility.
- No hidden global cache, default cache location, filesystem side effect, automatic discovery,
  service locator, reflection, string dispatch, or Java serialization is authorized.
- If implementation requires another top-level public type, component, factory, dependency,
  package, cache behavior, lifecycle object, or architecture change, stop and return the task to
  planning.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.config.compile` — remains unchanged; its conventions establish
  explicit immutable leaves, constructor validation, meaningful Javadoc, and shape-locking tests.

Packages added or changed:

- `io.github.pho001.synaptik.config.tuning` — new public package containing only the immutable
  model-autotuning request facade and its package documentation.

Type placement:

- `io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig` — Config owns this one
  declarative request aggregate; its nested values are meaningful only as parts of that request
  and therefore do not widen the top-level package surface.
- `ModelAutotuningConfig.Objective` — sole stable selection objective consumed by tuning 0001.
- `ModelAutotuningConfig.Budget` — bounded workload/candidate and sampling policy consumed by
  tuning 0001.
- `ModelAutotuningConfig.RepresentativeProfileIdentity` — user-owned opaque identity for the one
  representative profile attached to tuning evidence.
- `ModelAutotuningConfig.FallbackPolicy` — outer-composition policy for required tuning versus
  ordinary safe heuristic fallback.

Tests mirror the production package. No `api`, `cache`, `profile`, `input`, `internal`, `util`,
`common`, backend, service, or adapter subpackage is added.

## Affected files

Expected production and test paths:

- add `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfig.java`
- add `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/package-info.java`
- add `modules/config/src/test/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfigTest.java`

Expected documentation paths:

- `docs/api/public-api.md` — add the current Config facade and keep tuning execution/composition
  status explicit
- `docs/developer-guide/benchmarking.md` — distinguish the new declarative request from the
  existing operational tool-local workflow and later composition
- `docs/glossary.md` — define the request facade, representative-profile identity, and fallback
  distinction only where reusable terminology requires it

Expected planning paths:

- this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/tools/tuning/master-plan.md` — synchronize the consumer plan's current Config
  status and preserve the deferred composition boundary
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `AGENTS.md`,
`ARCHITECTURE.md`, focused architecture explanations, ADR 0008, other Config source/tests/build,
completed Config tasks, tools/tuning source/tests/build and completed task 0001, Prepare and CPU
tuning contracts, other APIs/guides/glossary entries, Config 0004–0006 and 0007–0008 rows, tuning
0002–0003, all Gradle files, architecture tests, backend-conformance tests, integration tests,
and other modules.

## Maximum scope

At most the exact ten paths above:

- 2 Config production/Javadoc paths;
- 1 focused Config test path;
- 3 explanatory documentation paths; and
- 4 planning paths.

The single top-level record and its four nested declarations are one cohesive request vocabulary.
Splitting them into independent top-level types would widen the public package without another
consumer, while omitting any of them would fail the stable requested policy boundary. If another
path, top-level production type, dependency, package, module behavior, or public component is
needed, stop and return the task to planning.

## Acceptance criteria

1. `ModelAutotuningConfig` is the sole new public top-level production type and is a public final
   record with exactly the five components, order, types, and names in Scope.
2. The record has only its canonical constructor, explicitly documented component accessors,
   ordinary record object methods, and exactly the four specified nested public declarations; it
   adds no builder, static factory, overload, adapter, service, callback, or serialization type.
3. Null outer components fail immediately with component-named `NullPointerException` messages;
   valid construction retains exact nested values and the exact `Path` reference without I/O or
   normalization.
4. `Objective` has exactly `MIN_MEDIAN_ELAPSED_NANOS`; no tool enum, ordinal, name conversion,
   scoring callback, or alternative objective is exposed.
5. `Budget` has exactly the four ordered primitive components and rejects every non-positive
   maximum, negative warmup, and non-positive or even sample count. Boundary-valid values retain
   their exact primitives and preserve ordinary record equality/hash/text behavior.
6. `RepresentativeProfileIdentity` rejects a non-positive schema version, null bytes, and empty
   bytes; snapshots input and output arrays; compares by schema version and byte content; and does
   not expose or retain a mutable array through an accessor, equality, hashing, or diagnostic text.
7. `FallbackPolicy` has exactly `REQUIRE_TUNED_RESULT` and `ALLOW_SAFE_HEURISTIC`, with Javadoc
   that preserves the strict semantics in Scope and grants no eligibility or cache-hit meaning.
8. Public signatures use only `java.nio.file.Path`, primitives, and the enclosing Config-owned
   declarations. Apart from the ordinary value `equals(Object)` override, there is no application
   payload typed as `Object`; no import, signature, dependency, reflection, string dispatch,
   unchecked generic, backend vocabulary, or arbitrary map connects Config to tuning mechanics.
9. No model fingerprint, actual representative input, Tensor/Shape, Prepare handoff, occurrence,
   candidate, decision, compatibility value, evidence, cache schema, or execution resource enters
   the facade.
10. The translation table remains exact: objective/budget/profile/cache map as specified;
    fallback, model fingerprint, occurrences, eligibility, and representative inputs remain with
    later composition or their named owners. No translation implementation is added.
11. The test locks the complete public shape and validates immutability/value behavior and
    forbidden-surface absence so future widening fails automatically.
12. Package and type Javadocs meaningfully document purpose, units, constraints, nullability,
    ownership/snapshot behavior, return semantics, expected failures, fallback boundaries, and
    current-versus-planned behavior for every public declaration and member.
13. Public API, benchmarking/tuning guide, and glossary describe only the implemented Config
    request data and clearly leave execution, mapping, cache operation, Engine integration, and
    later graph/plan tuning outside it.
14. Config 0004–0006 and 0007–0008 remain Draft without detailed specifications. Tuning 0001
    remains Complete; tuning 0002–0003 remain Draft without specifications. The recorded
    out-of-order exception and non-overlapping path/package rationale remain visible.
15. Architecture contracts and explanations, ADRs, Gradle dependencies, existing Config and
    tuning Java/tests, architecture tests, backend-conformance tests, integration tests, and
    other modules remain unchanged.
16. A separate clean documentation-focused context independently reviews the final source/tests,
    finalizes every affected Javadoc and the three explanatory documents, records glossary and
    no-change conclusions, and does not rerun successful Java tests unless executable behavior
    changes or a concrete stale-evidence risk is recorded.
17. Focused/final Config tests, Config Javadoc and generated-page inspection, all seven changed-
    Markdown files' links/anchors/headings/fences/newlines/whitespace, exact ten-path/package/
    public-shape/status/order checks, and `git diff --check` pass before completion.

## Tests / validation

Run focused tests while implementing. After executable Java stabilizes, run one final module
command:

```bash
./gradlew :modules:config:test
```

The focused test must perform the public-shape checks rather than relying on a repeated manual
reflection audit. It must cover defensive byte copying, content equality/hash behavior, valid
boundaries, every invalid primitive/null/empty condition, exact enum members, exact outer/nested
shape, and the absence of extra public or forbidden lifecycle surfaces.

Hand the exact diff and final Java-test evidence to a distinct clean documentation-focused
context in the same overall change. That context applies General style with API/Javadoc as the
primary profile, Planning style for status synchronization, and the relevant developer-guide
rules. It independently inspects source and tests, finalizes Javadocs and the three documentation
paths, records glossary impact and reasoned no-change conclusions, then runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short -uall
```

Render and inspect the generated `ModelAutotuningConfig` and package pages, including every
nested declaration and accessor. Validate all changed Markdown targets and heading anchors,
unique headings, balanced backtick/tilde fences, LF/final newlines, trailing whitespace,
terminology, exact scope, package placement, public shape, dependency absence, and synchronized
statuses.

Repository-wide validation is deferred to Config task 0008's configuration-contract closure or
continuous integration. This task changes one module, adds no Gradle edge or architecture rule,
and changes no backend or end-to-end behavior. No architecture-test, backend-conformance,
integration-test, tuning-test, real OpenBLAS, performance, cache-I/O, or Runtime command is
required.

## Dependencies

- Config tasks 0001–0003 — Complete; establish current immutable Config conventions and package
  discipline.
- tools/tuning task 0001 — Complete; supplies the stable exact/default local-workload consumer
  and its validated objective, budget, profile-fingerprint, and explicit-cache inputs.
- CPU 0010E and Prepare 0004 — Complete indirect producer/handoff evidence used by tuning 0001;
  they are reviewed but not imported or modified by this Config task.
- Explicit user authorization for the recorded staged ordering exception around Draft Config
  0004–0006.

Config 0004–0006 are not dependencies for this exact/default request facade. Config 0006 becomes
a dependency only for a later consumer that admits relaxed or fast-math candidates. Engine,
Model, Prepare, Runtime, concrete backends, and tuning 0002–0003 are not task dependencies.

## Follow-up tasks

- Config 0004 remains Draft pending a stable backend-neutral cost-bearing Planning consumer.
- Config 0005 remains Draft and later composes the compile leaves after Config 0004.
- Config 0006 remains Draft and later owns the smallest backend-neutral numerical/determinism
  permission before any relaxed candidate consumer exists.
- Config 0007 remains Draft for run/publication inputs; Config 0008 remains the closure checkpoint.
- A later Engine/composition task may translate this facade into tool-local values and supply the
  model fingerprint, occurrences, representative execution, and fallback control flow. Do not
  create that specification here.
- Tuning 0002–0003 remain Draft for bounded graph/plan tuning and artifact inspection. Their
  future needs do not widen this task's one-workload-cache surface.

## Architecture impact

Expected impact: None.

This task realizes Config's already-authorized ownership of immutable declarative
model-autotuning inputs after the consumer stabilized. It adds no module dependency or lifecycle
behavior. If implementation requires a Config-to-tool/backend/execution edge, a new ownership
rule, a second cache role, or any architecture change, stop and report the conflict without
editing architecture files.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik Config task 0006A. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the Config master plan, and
docs/planning/modules/config/tasks/0006a-model-autotuning-request-configuration.md in full, plus
the directly referenced tuning 0001 contract and current affected Config source/tests. Implement
exactly the Ready task within its ten-path ceiling. Stop for any architecture, dependency,
package, public-shape, lifecycle, cache-ownership, candidate-vocabulary, or scope conflict instead
of inventing a boundary.

After executable code and the final Config test stabilize, hand the exact diff and recorded test
evidence to a distinct clean documentation-focused context in the same overall change. That
context must follow docs/developer-guide/documentation-rules.md, independently inspect source and
tests, finalize affected Javadocs/package prose, Public API, benchmarking/tuning guide, glossary,
and planning evidence, and avoid repeating successful Java tests unless executable behavior
changes or a concrete stale-evidence risk is recorded. Neither context may commit or push. Do not
mark 0006A Complete until that pass and every specified gate succeeds.
```

## Local decisions

- Use one top-level `ModelAutotuningConfig` record with four nested declarations. They form one
  request vocabulary, share one consumer, and do not justify five independent package-level APIs.
- Use Config-owned types even where names and fields correspond to tuning 0001. Config cannot
  import the outer tool; later composition performs an exhaustive typed translation.
- Keep the sole objective name identical to the stable tool objective because its unit and
  selection meaning are user-visible and exact. Translation still uses an explicit switch rather
  than enum names or ordinals.
- Match the stable four budget fields exactly. Do not add a wall-clock or total-execution budget
  that the current complete-batch consumer cannot honor deterministically.
- Store one opaque representative-profile identity, not representative inputs or an informal
  description. The stable consumer needs identity for evidence, while actual input ownership,
  shape binding, resources, equivalence, and execution remain unresolved outer-lifecycle work.
- Exclude model identity because Config has no canonical Model fingerprint contract and the
  existing tool value is a caller/composition evidence assertion, not tuning policy.
- Require an explicit workload-cache path because tuning 0001 requires one and architecture
  forbids hidden global caches. Defer the model-plan cache because tuning 0002 has not stabilized
  that artifact role.
- Use `REQUIRE_TUNED_RESULT` and `ALLOW_SAFE_HEURISTIC` as the complete fallback vocabulary.
  Cache-only operation is not implemented by tuning 0001, and candidate or error-specific
  fallback maps would move orchestration into Config.
- Do not add a generic constraints object. Current exact/default eligibility is fixed outside the
  request, and Draft Config 0006 remains the future owner of relaxed numerical permission.
- Record the staged ordering exception explicitly: the stable tuning consumer now justifies
  0006A, while its separate package and paths do not overlap Draft Config 0004–0006.

## Open questions

None for implementation. If later composition cannot derive a stable model evidence identity or
cannot honor `ALLOW_SAFE_HEURISTIC` without weakening cache/candidate failure semantics, plan that
composition boundary separately; do not widen this Config task.

## Known limitations

- The facade supports one representative-profile identity, one workload-cache path, one elapsed-
  time objective, and fixed-count sampling only.
- It carries no representative values, shapes, model identity, target identity, occurrence data,
  numerical relaxation, model-plan cache, or executable artifact.
- `ALLOW_SAFE_HEURISTIC` is declarative until an outer composition consumer exists. It does not
  change current tuning 0001 behavior or catch its failures inside Config.
- Possessing the facade records that tuning was requested, but the value does not execute tuning;
  no current Engine aggregate or supported CPU adapter consumes it.

## Validation evidence

- This final targeted planning/documentation correction ran in a distinct clean context that was
  not given a visible context identifier. It authorized and audited the tuning master as the
  necessary tenth path because that consumer plan still called completed Config 0006A the next
  Draft task and still deferred its now-current public vocabulary. No Java, test, Javadoc,
  explanatory API/guide, or glossary file changed in this correction context.
- This mandatory distinct clean documentation-focused review context was not given a visible
  context identifier. It applied General style with API/Javadoc as the primary profile,
  Developer-guide style for the benchmarking explanation, Planning style for status records, and
  the repository example format.
- Reused the implementation context's final Java evidence because this pass changed no executable
  Java or tests and found no concrete stale-evidence risk: the focused
  `ModelAutotuningConfigTest` run passed 11 tests, and final
  `./gradlew :modules:config:test` passed 5 suites and 28 tests with 0 failures, 0 errors, and 0
  skipped. The retained XML reports independently show suite counts of 5, 2, 5, 5, and 11 with
  zero failures, errors, or skips. The initial failed private-synthetic-enum-method inspection is
  historical development evidence; the corrected public-surface test is the retained passing
  gate.
- `./gradlew :modules:config:javadoc` passed after final Javadoc review with `BUILD SUCCESSFUL`,
  3 actionable tasks, and no Javadoc warnings.
- Pandoc rendered the generated package summary, `ModelAutotuningConfig`, `Objective`, `Budget`,
  `RepresentativeProfileIdentity`, and `FallbackPolicy` pages to text. Inspection confirmed the
  package/type purpose, all five component and constructor contracts, all eleven explicit accessors,
  units and numeric constraints, non-null and exact-reference semantics, byte snapshot/copy and
  content-value behavior, failure conditions, fallback exclusions, and current-versus-planned
  composition boundary. No graphical browser was available, so generated HTML rendered to text
  is the recorded visual-inspection limitation.
- `python3 /tmp/validate_synaptik_markdown.py` passed for the three explanatory documents and four
  planning files: 7 Markdown files validated for local targets and heading anchors, unique
  headings, balanced backtick/tilde fences, LF endings, final newlines, and trailing whitespace.
- `javap -public` over the outer type and all four nested declarations confirmed one public final
  record with the exact five-component constructor/accessor surface, one one-value objective enum,
  one four-primitive Budget record, one five-method final profile-identity class, and one two-value
  fallback enum. Source/test inspection confirmed exactly one new public top-level type and no
  factory, builder, adapter, service, callback, serialization contract, generic parameter bag,
  backend vocabulary, or application payload typed as `Object` beyond `equals(Object)`.
- Source, `modules/config/build.gradle.kts`, and path inspection confirmed the new package uses only
  Config-owned declarations and JDK `Path`/primitive/array values. Config retains only its existing
  public Backend Contract dependency and adds no tools/tuning, Model, Planning, Compiler, Prepare,
  Runtime, Engine, provider, or concrete-backend dependency.
- Exact-scope inspection confirmed precisely the ten authorized worktree paths: two new
  production/Javadoc files, one new reviewed-only test, three explanatory documents, and four
  planning files. The test and both production/Javadoc files remained unchanged during this
  corrective documentation pass.
- Status/specification inspection confirmed Config 0006A is Complete; Config 0004–0006 and
  0007–0008 remain Draft without detailed specifications; tuning 0001 remains Complete; tuning
  0002–0003 remain Draft without specifications; and the historical non-overlapping staged
  ordering rationale remains recorded. Closing 0006A selected no next task and made no task Ready.
- `git diff --check` passed on the finalized ten-path change.
- Repository-wide tests were not run: this is a single-module capability with no Gradle,
  dependency, architecture, backend, or end-to-end behavior change, so Config 0008 or continuous
  integration remains the recorded repository-wide checkpoint.

## Implementation notes

- Added exactly one public final record and its four nested declarations in the planned
  `config.tuning` package. Constructor checks and immutable snapshots realize the planned value
  contracts without adding an aggregate default, enabled flag, filesystem action, or consumer.
- The focused test locks outer/nested declaration shape, validation, exact reference retention,
  defensive copying, content equality/hash/text behavior, enum vocabularies, and forbidden
  lifecycle/dependency surfaces.
- The documentation pass retained the implementation's already complete Javadocs after an
  independent source/test/render audit; no Javadoc-only edit was needed. It added the current
  request mental model and five inputs to Public API and the benchmarking guide and defined the
  facade, representative-profile identity, and strict-versus-safe fallback distinction in the
  glossary.
- Architecture, focused architecture explanations, ADR 0008, Gradle, architecture tests,
  backend-conformance tests, integration tests, tuning implementation, existing Config APIs/tests,
  and other modules required no change. The facade realizes already-authorized declarative Config
  ownership and changes no dependency direction, backend behavior, or end-to-end lifecycle.
- The final corrective documentation context added the tuning master as the authorized tenth path
  and synchronized its current status, open questions, and decisions. This was necessary because
  leaving that consumer plan outside the task ceiling preserved a direct current-status
  contradiction after 0006A became Complete.

## Completion summary

- Completed changes: added the exact immutable five-input `ModelAutotuningConfig` request facade,
  its four nested values, package documentation, focused contract test, explanatory documentation,
  glossary definitions, and synchronized planning status, including the corrective tuning-master
  update, without executable orchestration.
- Files changed or created: exactly the ten paths listed under Affected files.
- Tests and validation: reused the final focused 11-test and Config 28-test implementation
  evidence and clean Javadoc/render evidence; seven-file Markdown checks,
  public-shape/dependency/status/scope checks, and final whitespace validation passed.
- Documentation-agent review: the mandatory distinct clean documentation-focused pass and this
  final targeted corrective clean context independently audited the combined source/test/diff and
  completed every documentation gate without changing executable Java or tests.
- Documentation impact: Public API and benchmarking now describe only the current five-input
  request data, its relationship to operational tuning 0001, explicit later composition mapping,
  fallback control ownership, and deferred Engine/graph-plan behavior.
- Javadoc review: the package, outer record, four nested declarations, constructor, components,
  accessors, value methods, and generated pages are complete and consistent with tests; no source
  correction was required.
- Glossary impact: added reusable definitions for the request facade, representative-profile
  identity, and strict required-result versus safe-heuristic fallback boundary.
- Unresolved issues: None within task 0006A scope.
- Follow-up required: None. Config 0004–0006 and 0007–0008 and tuning 0002–0003 remain Draft and
  require separate future planning; no next task was selected here.

Status: Complete
