# Task 0006A: [Model-Autotuning Request](../../../../glossary.md#model-autotuning-request--preparation-result) Configuration

## Status

Complete

## Goal

Add the smallest public [immutable](../../../../glossary.md#term-immutable) Config [facade](../../../../glossary.md#term-facade) for the user-owned policy and [identity](../../../../glossary.md#term-identity-meanings) inputs
that are stable after completed tools/tuning task 0001:

```text
objective + bounded sampling budget + representative-profile identity
          + fallback policy + explicit workload-cache path
  -> ModelAutotuningConfig
  -> later composition translates Config values into tool-local request values
```

The facade describes a request. It does not enumerate candidates, [run](../../../../glossary.md#term-run) measurements, interpret
[backend](../../../../glossary.md#backend) values, read or write a [cache](../../../../glossary.md#term-cache), select a decision, [prepare](../../../../glossary.md#prepare) an executable, or orchestrate
[Engine](../../../../glossary.md#term-engine) or [Runtime](../../../../glossary.md#term-runtime) behavior.

This is an explicit staged ordering exception. Config tasks 0004–0006 remain Draft because their
cost-bearing and relaxed-numerics consumers are not stable. Task 0006A may proceed now because
completed tuning 0001 supplies its exact/default consumer evidence, this task uses the separate
`config.tuning` package and exact paths below, and it neither depends on nor overlaps those Draft
tasks. The exception changes [planning](../../../../glossary.md#term-planning) order only, not architecture [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

## Pre-implementation behavior and evidence

- `modules/config` currently exposes four [immutable](../../../../glossary.md#term-immutable) leaves only under `config.compile`; it has no
  model-autotuning package, facade, objective, budget, representative-profile identity, fallback
  policy, or cache-location value.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [facade glossary](../../../../glossary.md#term-facade); [representative-profile identity glossary](../../../../glossary.md#representative-profile-identity); [cache glossary](../../../../glossary.md#term-cache))</span>
- Completed tools/tuning 0001 exposes `WorkloadTuningRequest` with the sole [objective](../../../../glossary.md#term-objective)
  `MIN_MEDIAN_ELAPSED_NANOS`, a four-field bounded `Budget`, a caller-defined
  `ProfileFingerprint`, a caller-defined `ModelFingerprint`, ordered Prepare handoff occurrences,
  and one explicit workload-cache <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff))</span>`Path`.
- The tuning consumer owns measurement, candidate comparison, decision reuse, evidence, cache
  parsing, and atomic persistence. It intentionally accepts tool-local inputs and has no Config
  dependency.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [persistence glossary](../../../../glossary.md#term-persistence); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Tuning 0001 validation passed 24 tool tests and the recorded repository checkpoint. Its public
  contract is therefore stable enough to justify a Config-owned user request vocabulary without
  making Config depend on the outer tool.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The initial implemented tuning slice admits only exact/default FLOAT32/FLOAT64 candidates.
  Relaxed numerical permission remains Draft Config 0006 work and is not required by this task.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>

## [Scope](../../../../glossary.md#term-scope)

- Add one public final record
  `io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig` with exactly these five
  components, in this order:
  1. `Objective objective`;
  2. `Budget budget`;
  3. `RepresentativeProfileIdentity representativeProfile`;
  4. `FallbackPolicy fallbackPolicy`; and
  5. `Path workloadCache`.
- The [canonical](../../../../glossary.md#term-canonical-form) constructor rejects a null component with `NullPointerException` naming that
  component. It retains the exact immutable nested values and exact `Path` reference. It performs
  no filesystem access, path normalization, directory creation, cache validation, discovery, or
  default selection.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [cache glossary](../../../../glossary.md#term-cache); [validation glossary](../../../../glossary.md#term-validation))</span>
- Nest exactly four public policy/value declarations inside [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig):
  - [`Objective`](../../../../glossary.md#term-objective) is an enum with the sole [constant](../../../../glossary.md#term-constant) `MIN_MEDIAN_ELAPSED_NANOS`.
  - `Budget` is a record with, in order, positive `maximumDistinctCacheMisses`, positive
    `maximumCandidatesPerMiss`, non-negative `warmupCount`, and positive odd
    `timedSampleCount`. Its constructor validates those conditions before any consumer work.
  - `RepresentativeProfileIdentity` is a public final [immutable](../../../../glossary.md#term-immutable) class with a positive
    `schemaVersion` and non-empty [opaque](../../../../glossary.md#term-opaque) `byte[]` [identity](../../../../glossary.md#term-identity-meanings). Construction snapshots the array;
    `bytes()` returns a fresh copy; equality and hashing compare schema version and byte content;
    diagnostic text reports the schema version and byte length without exposing bytes. Its exact
    public surface is constructor <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`RepresentativeProfileIdentity(int schemaVersion, byte[] bytes)`,
    accessors `int schemaVersion()` and `byte[] bytes()`, plus ordinary value
    `equals(Object)`, `hashCode()`, and `toString()` overrides; it exposes no other method.
  - `FallbackPolicy` is an enum with exactly `REQUIRE_TUNED_RESULT` and
    `ALLOW_SAFE_HEURISTIC`.
- Define fallback semantics narrowly:
  - `REQUIRE_TUNED_RESULT` tells later composition to report failure if the requested tuning
    transaction cannot produce a complete selected result.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>
  - `ALLOW_SAFE_HEURISTIC` permits later composition to abandon an unavailable or failed tuning
    transaction and continue through the ordinary safe heuristic preparation path. It does not
    turn corrupt/incompatible cache data into a hit, grant candidate eligibility, accept partial
    selections, suppress unrelated preparation failures, or require Config to catch exceptions.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Add no enabled/disabled flag or implicit default. Possessing a `ModelAutotuningConfig` means
  tuning was requested; absence and aggregation belong to a later composition owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Use one [representative-profile identity](../../../../glossary.md#representative-profile-identity) because the implemented tuning request accepts one
  `ProfileFingerprint`. Do not store actual Tensor values, shapes, input maps, suppliers,
  callbacks, resources, or descriptive strings. The later Engine/composition boundary owns
  representative inputs and proves equivalent complete candidate execution.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [shapes glossary](../../../../glossary.md#shape); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Keep the model fingerprint out of Config. Later composition derives or supplies the tool-local
  model evidence identity from the actual model/lifecycle request; Config has no stable canonical
  Model identity contract to duplicate.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Keep numerical/determinism constraints out of this type. The current tuning candidate set is
  fixed to exact/default eligibility, while Draft Config 0006 remains the sole future owner of
  explicit relaxed numerical permission.<span data-glossary-navigation="true"> ([tuning candidate glossary](../../../../glossary.md#candidate-batch))</span>
- Add package documentation that distinguishes immutable request data from tuning, cache,
  backend, preparation, Engine, and Runtime behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add one focused test class covering exact public shape, constructor validation, immutable byte
  snapshots, equality/hash/text behavior, ordinary outer-record behavior, exact enum values,
  exact component order/types, and absence of backend/tool/measurement/persistence/lifecycle
  surfaces.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Finalize all new Javadocs and the focused public API, tuning guide, glossary, and planning
  status in a separate clean documentation-focused context in the same overall change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

### Explicit translation to the stable consumer

Later composition may depend on both Config and tools/tuning and must translate explicitly. No
translation code belongs in this task:

| Config value | tools/tuning 0001 input | Resolution |
|---|---|---|
| `Objective.MIN_MEDIAN_ELAPSED_NANOS` | `WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS` | One exhaustive enum switch; never `valueOf`, string [dispatch](../../../../glossary.md#term-dispatch), or a Config-to-tool [dependency](../../../../glossary.md#term-dependency). |
| Four `Budget` components | Same four `WorkloadTuningRequest.Budget` components | Copy one-for-one after Config [validation](../../../../glossary.md#term-validation); the tool repeats its own boundary validation. |
| `RepresentativeProfileIdentity(schemaVersion, bytes)` | `WorkloadTuningRequest.ProfileFingerprint(schemaVersion, bytes)` | Construct a tool-local [snapshot](../../../../glossary.md#term-snapshot) from the Config accessor copy. |
| `workloadCache` | `WorkloadTuningRequest.workloadCache` | Pass the exact requested path; tuning remains responsible for [cache](../../../../glossary.md#term-cache) I/O and validation. |
| `FallbackPolicy` | No `WorkloadTuningRequest` field | Composition decides whether failure ends the request or enters ordinary safe [heuristic](../../../../glossary.md#term-heuristic) preparation; tuning 0001 remains unchanged. |
| Actual [model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings) | `WorkloadTuningRequest.ModelFingerprint` | Deferred to composition because it owns the actual model request and evidence identity. |
| Tunable occurrences, [weights](../../../../glossary.md#term-weight), and contexts | `WorkloadTuningRequest.Occurrence` list | Deferred to [Prepare](../../../../glossary.md#prepare)/[backend](../../../../glossary.md#backend)-aware composition; Config owns no [handoff](../../../../glossary.md#term-handoff) or backend candidate type. |
| Exact/default numerical eligibility | Candidate [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) supplied to tuning | Already enforced by current candidate [producers](../../../../glossary.md#term-producer); future relaxed permission remains Config 0006 and requires a later consumer update. |
| Representative input values/resources | `ColdCandidateMeasurement` behavior | Deferred to [Engine](../../../../glossary.md#term-engine)/composition; no live input or execution resource enters Config. |

The table is a boundary map, not a second [API](../../../../glossary.md#term-application-programming-interface-api). It shows that a later outer composition layer can
translate stable values without adding the forbidden `modules/config -> tools/tuning` dependency.

## Out of [scope](../../../../glossary.md#term-scope)

- a Config dependency on tools/tuning, Prepare, Runtime, Engine, Model, Planning, Compiler, a
  concrete backend, or a provider<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- `WorkloadTuningRequest` changes, a translation adapter, Engine facade, compile/prepare/run
  aggregate, dependency injection, registration, discovery, or orchestration<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [dependency injection glossary](../../../../glossary.md#term-dependency-injection))</span>
- model fingerprints, model objects, Tensor values, Shapes, input maps, input suppliers,
  callbacks, execution actions, resources, or representative-input validation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shapes glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Prepare handoffs, occurrences, occurrence contexts or weights, candidate batches, candidates,
  decisions, compatibility identities, candidate identities, backend routes, thread counts,
  vector species, layouts, materialization choices, kernels, or executable artifacts<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [weights glossary](../../../../glossary.md#term-weight); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [backend glossary](../../../../glossary.md#backend); [vector species glossary](../../../../glossary.md#term-vector-species); [layouts glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- candidate generation, eligibility, timing, warmup execution, measurement, comparison, median
  calculation, selection, correctness checking, evidence, profiling, tracing, or Runtime behavior<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [median glossary](../../../../glossary.md#term-median))</span>
- cache loading, parsing, compatibility, corruption handling, mutation, persistence, atomic move,
  locking, eviction, migration, inspection, file-format/schema ownership, model-plan cache, or
  prepared-executable serialization<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [atomic move glossary](../../../../glossary.md#term-atomic-move); [schema glossary](../../../../glossary.md#term-schema); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- alternative objectives, wall-clock deadlines, adaptive sampling, confidence thresholds,
  energy/memory objectives, multi-objective ranking, or arbitrary constraint/parameter maps<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [objective glossary](../../../../glossary.md#term-objective); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- relaxed or fast-math permission, determinism configuration, backend-specific numerical policy,
  Config 0004–0006 implementation, or changing their Draft status
- Config 0007–0008, tuning 0002–0003, another task specification, or any code outside
  `modules/config`
- architecture contract, ADR, dependency rule, Gradle/build, architecture-test,
  backend-conformance, integration-test, or unrelated documentation changes<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/config`, Concrete backend modules, Performance evidence and optimization tooling,
  Runtime service locator, and Dependency rules<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [Runtime glossary](../../../../glossary.md#term-runtime); [service locator glossary](../../../../glossary.md#term-service-locator); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [ADR 0008: Separate performance evidence and model-autotuning concerns](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)
- [Completed tuning 0001 consumer](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Public API](../../../../api/public-api.md)
- [Benchmarking and tuning guide](../../../../developer-guide/benchmarking.md)<span data-glossary-navigation="true"> ([Benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Config owns immutable declarative request inputs only. It owns no runner, search, cache
  mutation, measurement algorithm, live discovery, mutable evidence, service, executable, or
  Runtime state.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [cache glossary](../../../../glossary.md#term-cache); [mutable glossary](../../../../glossary.md#term-mutable); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Model autotuning completes before Runtime hot-path execution. Runtime profiling remains passive
  and cannot select settings.<span data-glossary-navigation="true"> ([Model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Runtime profiling glossary](../../../../glossary.md#runtime-profiling))</span>
- Backend authors retain candidate vocabulary, generation, compatibility, decisions, and route
  semantics. No backend field or generic parameter bag enters Config.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Tuning retains measurement, comparison, selection, evidence, cache parsing, and persistence.
  Similar field names do not transfer those responsibilities to Config.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- The Config [API](../../../../glossary.md#term-application-programming-interface-api) uses [JDK](../../../../glossary.md#term-java-development-kit-jdk) and Config-owned values only. `modules/config` must not depend on
  tools/tuning or add another module/Gradle edge.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Later outer composition may depend on both modules and use exhaustive typed translation. Config
  must not call outward or mirror tool implementation mechanics beyond the stable user-owned
  values identified in the translation table.
- Exact/default eligibility remains a producer/backend fact for the current slice. An objective
  favoring elapsed time and <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [slice glossary](../../../../glossary.md#slice); [objective glossary](../../../../glossary.md#term-objective))</span>`ALLOW_SAFE_HEURISTIC` never grant relaxed mathematics or candidate
  eligibility.
- No hidden global cache, default cache location, filesystem side effect, automatic discovery,
  service locator, reflection, string dispatch, or Java serialization is authorized.<span data-glossary-navigation="true"> ([side effect glossary](../../../../glossary.md#term-side-effect); [service locator glossary](../../../../glossary.md#term-service-locator); [dispatch glossary](../../../../glossary.md#term-dispatch); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- If implementation requires another top-level public type, component, factory, dependency,
  package, cache behavior, lifecycle object, or architecture change, stop and return the task to
  planning.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.config.compile` — remains unchanged; its conventions establish
  explicit immutable leaves, constructor validation, meaningful Javadoc, and shape-locking tests.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape))</span>

Packages added or changed:

- `io.github.pho001.synaptik.config.tuning` — new public package containing only the immutable
  model-autotuning request facade and its package documentation.<span data-glossary-navigation="true"> ([model-autotuning request facade glossary](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig))</span>

Type placement:

- `io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig` — Config owns this one
  declarative request aggregate; its nested values are meaningful only as parts of that request
  and therefore do not widen the top-level package surface.
- `ModelAutotuningConfig.Objective` — sole stable selection [objective](../../../../glossary.md#term-objective) consumed by tuning 0001.
- `ModelAutotuningConfig.Budget` — bounded workload/candidate and sampling policy consumed by
  tuning 0001.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- `ModelAutotuningConfig.RepresentativeProfileIdentity` — user-owned opaque identity for the one
  representative profile attached to tuning evidence.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `ModelAutotuningConfig.FallbackPolicy` — outer-composition policy for required tuning versus
  ordinary safe heuristic fallback.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>

Tests mirror the production package. No `api`, [`cache`](../../../../glossary.md#term-cache), `profile`, `input`, `internal`, `util`,
`common`, [backend](../../../../glossary.md#backend), service, or adapter subpackage is added.

## Affected files

Expected production and test paths:

- add `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfig.java`
- add `modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/package-info.java`
- add `modules/config/src/test/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfigTest.java`

Expected documentation paths:

- `docs/api/public-api.md` — add the current Config facade and keep tuning execution/composition
  status explicit<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>
- `docs/developer-guide/benchmarking.md` — distinguish the new declarative request from the
  existing operational tool-local workflow and later composition
- `docs/glossary.md` — define the request facade, representative-profile identity, and fallback
  distinction only where reusable terminology requires it<span data-glossary-navigation="true"> ([representative-profile identity glossary](../../../../glossary.md#representative-profile-identity))</span>

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/tools/tuning/master-plan.md` — synchronize the consumer plan's current Config
  status and preserve the deferred composition boundary
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `AGENTS.md`,
`ARCHITECTURE.md`, focused architecture explanations, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0008, other Config source/tests/build,
completed Config tasks, tools/tuning source/tests/build and completed task 0001, [Prepare](../../../../glossary.md#prepare) and [CPU](../../../../glossary.md#term-cpu)
tuning contracts, other APIs/guides/glossary entries, Config 0004–0006 and 0007–0008 rows, tuning
0002–0003, all [Gradle](../../../../glossary.md#term-gradle) files, architecture tests, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) tests, integration tests,
and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact ten paths above:

- 2 Config production/[Javadoc](../../../../glossary.md#term-javadoc) paths;
- 1 focused Config test path;
- 3 explanatory documentation paths; and
- 4 [planning](../../../../glossary.md#term-planning) paths.

The single top-level record and its four nested declarations are one cohesive request vocabulary.
Splitting them into independent top-level types would widen the public package without another
consumer, while omitting any of them would fail the stable requested policy boundary. If another
path, top-level production type, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module) behavior, or public component is
needed, stop and return the task to planning.

## Acceptance criteria

1. [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig) is the sole new public top-level production type and is a public final
   record with exactly the five components, order, types, and names in Scope.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
2. The record has only its canonical constructor, explicitly documented component accessors,
   ordinary record object methods, and exactly the four specified nested public declarations; it
   adds no builder, static factory, overload, adapter, service, callback, or serialization type.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [callback glossary](../../../../glossary.md#term-callback); [serialization glossary](../../../../glossary.md#term-serialization))</span>
3. Null outer components fail immediately with component-named `NullPointerException` messages;
   valid construction retains exact nested values and the exact `Path` reference without I/O or
   normalization.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
4. [`Objective`](../../../../glossary.md#term-objective) has exactly `MIN_MEDIAN_ELAPSED_NANOS`; no tool enum, ordinal, name conversion,
   scoring callback, or alternative objective is exposed.
5. `Budget` has exactly the four ordered primitive components and rejects every non-positive
   maximum, negative warmup, and non-positive or even sample count. Boundary-valid values retain
   their exact primitives and preserve ordinary record equality/hash/text behavior.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [hash glossary](../../../../glossary.md#term-hash))</span>
6. `RepresentativeProfileIdentity` rejects a non-positive schema version, null bytes, and empty
   bytes; snapshots input and output arrays; compares by schema version and byte content; and does
   not expose or retain a mutable array through an accessor, equality, hashing, or diagnostic text.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [mutable glossary](../../../../glossary.md#term-mutable))</span>
7. `FallbackPolicy` has exactly `REQUIRE_TUNED_RESULT` and `ALLOW_SAFE_HEURISTIC`, with Javadoc
   that preserves the strict semantics in Scope and grants no eligibility or cache-hit meaning.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [cache glossary](../../../../glossary.md#term-cache))</span>
8. Public [signatures](../../../../glossary.md#term-signature) use only `java.nio.file.Path`, primitives, and the enclosing Config-owned
   declarations. Apart from the ordinary value `equals(Object)` override, there is no application
   payload typed as <span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`Object`; no import, signature, dependency, reflection, string dispatch,
   unchecked generic, backend vocabulary, or arbitrary map connects Config to tuning mechanics.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [dispatch glossary](../../../../glossary.md#term-dispatch); [backend glossary](../../../../glossary.md#backend))</span>
9. No model fingerprint, actual representative input, Tensor/Shape, Prepare handoff, occurrence,
   candidate, decision, compatibility value, evidence, cache schema, or execution resource enters
   the facade.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [occurrence glossary](../../../../glossary.md#term-occurrence); [facade glossary](../../../../glossary.md#term-facade))</span>
10. The translation table remains exact: objective/budget/profile/cache map as specified;
    fallback, model fingerprint, occurrences, eligibility, and representative inputs remain with
    later composition or their named owners. No translation implementation is added.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
11. The test locks the complete public shape and validates immutability/value behavior and
    forbidden-surface absence so future widening fails automatically.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>
12. Package and type Javadocs meaningfully document purpose, units, constraints, nullability,
    ownership/snapshot behavior, return semantics, expected failures, fallback boundaries, and
    current-versus-planned behavior for every public declaration and member.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
13. Public API, benchmarking/tuning guide, and glossary describe only the implemented Config
    request data and clearly leave execution, mapping, cache operation, Engine integration, and
    later graph/plan tuning outside it.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine); [graph glossary](../../../../glossary.md#graph))</span>
14. Config 0004–0006 and 0007–0008 remain Draft without detailed specifications. Tuning 0001
    remains Complete; tuning 0002–0003 remain Draft without specifications. The recorded
    out-of-order exception and non-overlapping path/package rationale remain visible.
15. Architecture contracts and explanations, ADRs, Gradle dependencies, existing Config and
    tuning Java/tests, architecture tests, backend-conformance tests, integration tests, and
    other modules remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
16. A separate clean documentation-focused context independently reviews the final source/tests,
    finalizes every affected Javadoc and the three explanatory documents, records glossary and
    no-change conclusions, and does not rerun successful Java tests unless executable behavior
    changes or a concrete stale-evidence risk is recorded.
17. Focused/final Config tests, Config Javadoc and generated-page inspection, all seven changed-
    Markdown files' links/anchors/headings/fences/newlines/whitespace, exact ten-path/package/
    public-shape/status/order checks, and `git diff --check` pass before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while implementing. After executable Java stabilizes, run one final [module](../../../../glossary.md#term-module)
command:

```bash
./gradlew :modules:config:test
```

The focused test must perform the public-[shape](../../../../glossary.md#shape) checks rather than relying on a repeated manual
reflection audit. It must cover defensive byte copying, content equality/[hash](../../../../glossary.md#term-hash) behavior, valid
boundaries, every invalid primitive/null/empty condition, exact enum members, exact outer/nested
shape, and the absence of extra public or forbidden [lifecycle](../../../../glossary.md#lifecycle) surfaces.

Hand the exact diff and final Java-test evidence to a distinct clean documentation-focused
context in the same overall change. That context applies General style with [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) as the
primary profile, [Planning](../../../../glossary.md#term-planning) style for status [synchronization](../../../../glossary.md#term-synchronization), and the relevant developer-guide
rules. It independently inspects source and tests, finalizes Javadocs and the three documentation
paths, records glossary impact and reasoned no-change conclusions, then runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short -uall
```

Render and inspect the generated [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig) and package pages, including every
nested declaration and accessor. Validate all changed Markdown [targets](../../../../glossary.md#term-target) and heading anchors,
unique headings, balanced backtick/tilde fences, LF/final newlines, trailing whitespace,
terminology, exact [scope](../../../../glossary.md#term-scope), package placement, public shape, [dependency](../../../../glossary.md#term-dependency) absence, and synchronized
statuses.

Repository-wide validation is deferred to Config task 0008's configuration-contract closure or
[continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one module, adds no [Gradle](../../../../glossary.md#term-gradle) edge or architecture rule,
and changes no [backend](../../../../glossary.md#backend) or end-to-end behavior. No architecture-test, backend-[conformance](../../../../glossary.md#term-conformance),
integration-test, tuning-test, real [OpenBLAS](../../../../glossary.md#term-openblas), performance, [cache](../../../../glossary.md#term-cache)-I/O, or [Runtime](../../../../glossary.md#term-runtime) command is
required.

## Dependencies

- Config tasks 0001–0003 — Complete; establish current immutable Config conventions and package
  discipline.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- tools/tuning task 0001 — Complete; supplies the stable exact/default local-workload consumer
  and its validated objective, budget, profile-fingerprint, and explicit-cache inputs.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [cache glossary](../../../../glossary.md#term-cache))</span>
- CPU 0010E and Prepare 0004 — Complete indirect producer/handoff evidence used by tuning 0001;
  they are reviewed but not imported or modified by this Config task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [producer glossary](../../../../glossary.md#term-producer); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Explicit user authorization for the recorded staged ordering exception around Draft Config
  0004–0006.

Config 0004–0006 are not dependencies for this exact/default request [facade](../../../../glossary.md#term-facade). Config 0006 becomes
a [dependency](../../../../glossary.md#term-dependency) only for a later consumer that admits relaxed or fast-math candidates. [Engine](../../../../glossary.md#term-engine),
[Model](../../../../glossary.md#term-model), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), concrete [backends](../../../../glossary.md#backend), and tuning 0002–0003 are not task dependencies.

## Follow-up tasks

- Config 0004 remains Draft pending a stable [backend](../../../../glossary.md#backend)-neutral cost-bearing [Planning](../../../../glossary.md#term-planning) consumer.
- Config 0005 remains Draft and later composes the [compile](../../../../glossary.md#compile) leaves after Config 0004.
- Config 0006 remains Draft and later owns the smallest backend-neutral numerical/determinism
  permission before any relaxed candidate consumer exists.
- Config 0007 remains Draft for [run](../../../../glossary.md#term-run)/[publication](../../../../glossary.md#term-publication) inputs; Config 0008 remains the closure [checkpoint](../../../../glossary.md#term-checkpoint).
- A later Engine/composition task may translate this facade into tool-local values and supply the
  model fingerprint, occurrences, representative execution, and fallback control flow. Do not
  create that specification here.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [model glossary](../../../../glossary.md#term-model); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Tuning 0002–0003 remain Draft for bounded graph/plan tuning and artifact inspection. Their
  future needs do not widen this task's one-workload-cache surface.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>

## Architecture impact

Expected impact: None.

This task realizes Config's already-authorized [ownership](../../../../glossary.md#term-ownership-meanings) of [immutable](../../../../glossary.md#term-immutable) declarative
[model](../../../../glossary.md#term-model)-[autotuning](../../../../glossary.md#autotuning--model-autotuning) inputs after the consumer stabilized. It adds no [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) or [lifecycle](../../../../glossary.md#lifecycle)
behavior. If implementation requires a Config-to-tool/[backend](../../../../glossary.md#backend)/execution edge, a new ownership
rule, a second [cache](../../../../glossary.md#term-cache) role, or any architecture change, stop and report the conflict without
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

- Use one top-level [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig) record with four nested declarations. They form one
  request vocabulary, share one consumer, and do not justify five independent package-level APIs.
- Use Config-owned types even where names and fields correspond to tuning 0001. Config cannot
  import the outer tool; later composition performs an exhaustive typed translation.
- Keep the sole objective name identical to the stable tool objective because its unit and
  selection meaning are user-visible and exact. Translation still uses an explicit switch rather
  than enum names or ordinals.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective))</span>
- Match the stable four budget fields exactly. Do not add a wall-clock or total-execution budget
  that the current complete-batch consumer cannot honor deterministically.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Store one opaque representative-profile identity, not representative inputs or an informal
  description. The stable consumer needs identity for evidence, while actual input ownership,
  shape binding, resources, equivalence, and execution remain unresolved outer-lifecycle work.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [representative-profile identity glossary](../../../../glossary.md#representative-profile-identity); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Exclude model identity because Config has no canonical Model fingerprint contract and the
  existing tool value is a caller/composition evidence assertion, not tuning policy.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [canonical glossary](../../../../glossary.md#term-canonical-form); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Require an explicit workload-cache path because tuning 0001 requires one and architecture
  forbids hidden global caches. Defer the model-plan cache because tuning 0002 has not stabilized
  that artifact role.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Use `REQUIRE_TUNED_RESULT` and `ALLOW_SAFE_HEURISTIC` as the complete fallback vocabulary.
  Cache-only operation is not implemented by tuning 0001, and candidate or error-specific
  fallback maps would move orchestration into Config.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Do not add a generic constraints object. Current exact/default eligibility is fixed outside the
  request, and Draft Config 0006 remains the future owner of relaxed numerical permission.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Record the staged ordering exception explicitly: the stable tuning consumer now justifies
  0006A, while its separate package and paths do not overlap Draft Config 0004–0006.

## Open questions

None for implementation. If later composition cannot derive a stable [model](../../../../glossary.md#term-model) evidence [identity](../../../../glossary.md#term-identity-meanings) or
cannot honor `ALLOW_SAFE_HEURISTIC` without weakening [cache](../../../../glossary.md#term-cache)/candidate failure semantics, plan that
composition boundary separately; do not widen this Config task.

## Known limitations

- The facade supports one representative-profile identity, one workload-cache path, one elapsed-
  time objective, and fixed-count sampling only.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [representative-profile identity glossary](../../../../glossary.md#representative-profile-identity); [cache glossary](../../../../glossary.md#term-cache); [objective glossary](../../../../glossary.md#term-objective); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- It carries no representative values, shapes, model identity, target identity, occurrence data,
  numerical relaxation, model-plan cache, or executable artifact.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [target glossary](../../../../glossary.md#term-target); [occurrence glossary](../../../../glossary.md#term-occurrence); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `ALLOW_SAFE_HEURISTIC` is declarative until an outer composition consumer exists. It does not
  change current tuning 0001 behavior or catch its failures inside Config.
- Possessing the facade records that tuning was requested, but the value does not execute tuning;
  no current Engine aggregate or supported CPU adapter consumes it.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- This final targeted planning/documentation correction ran in a distinct clean context that was
  not given a visible context identifier. It authorized and audited the tuning master as the
  necessary tenth path because that consumer plan still called completed Config 0006A the next
  Draft task and still deferred its now-current public vocabulary. No Java, test, Javadoc,
  explanatory API/guide, or glossary file changed in this correction context.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- This mandatory distinct clean documentation-focused review context was not given a visible
  context identifier. It applied General style with API/Javadoc as the primary profile,
  Developer-guide style for the benchmarking explanation, Planning style for status records, and
  the repository example format.<span data-glossary-navigation="true"> ([benchmarking glossary](../../../../glossary.md#term-benchmarking))</span>
- Reused the implementation context's final Java evidence because this pass changed no executable
  Java or tests and found no concrete stale-evidence risk: the focused
  `ModelAutotuningConfigTest` [run](../../../../glossary.md#term-run) passed 11 tests, and final
  `./gradlew :modules:config:test` passed 5 suites and 28 tests with 0 failures, 0 errors, and 0
  skipped. The retained XML reports independently show suite counts of 5, 2, 5, 5, and 11 with
  zero failures, errors, or skips. The initial failed private-synthetic-enum-method inspection is
  historical development evidence; the corrected public-surface test is the retained passing
  gate.
- `./gradlew :modules:config:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) review with `BUILD SUCCESSFUL`,
  3 actionable tasks, and no Javadoc warnings.
- Pandoc rendered the generated package summary, [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig), [`Objective`](../../../../glossary.md#term-objective), `Budget`,
  `RepresentativeProfileIdentity`, and `FallbackPolicy` pages to text. Inspection confirmed the
  package/type purpose, all five component and constructor contracts, all eleven explicit accessors,
  units and numeric constraints, non-null and exact-reference semantics, byte snapshot/copy and
  content-value behavior, failure conditions, fallback exclusions, and current-versus-planned
  composition boundary. No graphical browser was available, so generated HTML rendered to text
  is the recorded visual-inspection limitation.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed for the three explanatory documents and four
  planning files: 7 Markdown files validated for local targets and heading anchors, unique
  headings, balanced backtick/tilde fences, LF endings, final newlines, and trailing whitespace.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `javap -public` over the outer type and all four nested declarations confirmed one public final
  record with the exact five-component constructor/accessor surface, one one-value objective enum,
  one four-primitive Budget record, one five-method final profile-identity class, and one two-value
  fallback enum. Source/test inspection confirmed exactly one new public top-level type and no
  factory, builder, adapter, service, callback, serialization contract, generic parameter bag,
  backend vocabulary, or application payload typed as <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [callback glossary](../../../../glossary.md#term-callback); [serialization glossary](../../../../glossary.md#term-serialization); [parameter glossary](../../../../glossary.md#term-parameter); [backend glossary](../../../../glossary.md#backend); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`Object` beyond `equals(Object)`.
- Source, `modules/config/build.gradle.kts`, and path inspection confirmed the new package uses only
  Config-owned declarations and JDK <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Path`/primitive/array values. Config retains only its existing
  public Backend Contract dependency and adds no tools/tuning, Model, Planning, Compiler, Prepare,
  Runtime, Engine, provider, or concrete-backend dependency.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Exact-scope inspection confirmed precisely the ten authorized worktree paths: two new
  production/Javadoc files, one new reviewed-only test, three explanatory documents, and four
  planning files. The test and both production/Javadoc files remained unchanged during this
  corrective documentation pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Status/specification inspection confirmed Config 0006A is Complete; Config 0004–0006 and
  0007–0008 remain Draft without detailed specifications; tuning 0001 remains Complete; tuning
  0002–0003 remain Draft without specifications; and the historical non-overlapping staged
  ordering rationale remains recorded. Closing 0006A selected no next task and made no task Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- `git diff --check` passed on the finalized ten-path change.
- Repository-wide tests were not run: this is a single-module capability with no Gradle,
  dependency, architecture, backend, or end-to-end behavior change, so Config 0008 or continuous
  integration remains the recorded repository-wide checkpoint.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Implementation notes

- Added exactly one public final record and its four nested declarations in the planned
  `config.tuning` package. Constructor checks and immutable snapshots realize the planned value
  contracts without adding an aggregate default, enabled flag, filesystem action, or consumer.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- The focused test locks outer/nested declaration shape, validation, exact reference retention,
  defensive copying, content equality/hash/text behavior, enum vocabularies, and forbidden
  lifecycle/dependency surfaces.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [defensive copying glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [hash glossary](../../../../glossary.md#term-hash); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The documentation pass retained the implementation's already complete Javadocs after an
  independent source/test/render audit; no Javadoc-only edit was needed. It added the current
  request mental model and five inputs to Public API and the benchmarking guide and defined the
  facade, representative-profile identity, and strict-versus-safe fallback distinction in the
  glossary.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [facade glossary](../../../../glossary.md#term-facade); [representative-profile identity glossary](../../../../glossary.md#representative-profile-identity))</span>
- Architecture, focused architecture explanations, ADR 0008, Gradle, architecture tests,
  backend-conformance tests, integration tests, tuning implementation, existing Config APIs/tests,
  and other modules required no change. The facade realizes already-authorized declarative Config
  ownership and changes no dependency direction, backend behavior, or end-to-end lifecycle.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- The final corrective documentation context added the tuning master as the authorized tenth path
  and synchronized its current status, open questions, and decisions. This was necessary because
  leaving that consumer plan outside the task ceiling preserved a direct current-status
  contradiction after 0006A became Complete.

## Completion summary

- Completed changes: added the exact [immutable](../../../../glossary.md#term-immutable) five-input [`ModelAutotuningConfig`](../../../../glossary.md#model-autotuning-request-facade--modelautotuningconfig) request facade,
  its four nested values, package documentation, focused contract test, explanatory documentation,
  glossary definitions, and synchronized planning status, including the corrective tuning-master
  update, without executable orchestration.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the ten paths listed under Affected files.
- Tests and validation: reused the final focused 11-test and Config 28-test implementation
  evidence and clean Javadoc/render evidence; seven-file Markdown checks,
  public-shape/dependency/status/scope checks, and final whitespace validation passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation-agent review: the mandatory distinct clean documentation-focused pass and this
  final targeted corrective clean context independently audited the combined source/test/diff and
  completed every documentation gate without changing executable Java or tests.
- Documentation impact: Public API and benchmarking now describe only the current five-input
  request data, its relationship to operational tuning 0001, explicit later composition mapping,
  fallback control ownership, and deferred Engine/graph-plan behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine); [graph glossary](../../../../glossary.md#graph))</span>
- Javadoc review: the package, outer record, four nested declarations, constructor, components,
  accessors, value methods, and generated pages are complete and consistent with tests; no source
  correction was required.
- Glossary impact: added reusable definitions for the request facade, representative-profile
  identity, and strict required-result versus safe-heuristic fallback boundary.<span data-glossary-navigation="true"> ([representative-profile identity glossary](../../../../glossary.md#representative-profile-identity); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Unresolved issues: None within task 0006A [scope](../../../../glossary.md#term-scope).
- Follow-up required: None. Config 0004–0006 and 0007–0008 and tuning 0002–0003 remain Draft and
  require separate future planning; no next task was selected here.

Status: Complete
