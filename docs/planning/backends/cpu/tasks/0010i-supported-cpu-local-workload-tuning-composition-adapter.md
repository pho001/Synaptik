# Task 0010I: Supported CPU Local-Workload Tuning Composition Adapter

## Status

Complete

## Goal

Expose the smallest supported CPU-owned integration collaboration that lets an outer composition
owner use the completed CPU 0010E candidate and selected-decision semantics with completed
tools/tuning 0001 without importing any `.internal` type:

```text
one supported CompileArtifacts value
  -> fresh CPU analysis
  -> optional exact-partition typed candidate handoff
  -> CPU-owned enumeration, compatibility, identity, decision, and codec operations
  -> one freshly prepared complete candidate recipe per cold trial
  -> one freshly prepared final selected recipe
```

The collaboration remains inside the lifetime of one `CpuBackendIntegration`. It exposes opaque,
typed CPU values in the existing supported root package, while the internal OpenBLAS workload,
candidate, route, representation, provider, workspace, and coordination types remain hidden.
Ordinary `CpuBackendIntegration.prepare(CompileArtifacts)` remains the deterministic safe-
heuristic path and is unchanged when tuning is absent or abandoned.

## Motivation and current seam

Before this task, CPU 0010E already owned the complete exact/default FLOAT32/FLOAT64 bare-MATMUL
candidate batch,
canonical workload compatibility, portable-first candidate order, stable candidate identity,
selected-decision matching, safe heuristic, representation/resource consequences, and fresh-
analysis decision consumption. CPU 0010F already owns the supported one-partition lifecycle,
automatic provider discovery and qualification, finalization, schedule assembly, concurrency
budget, thread coordination, and close. CPU 0010H added source-only published constants to the same
complete preparation path. None of those earlier supported APIs transported a tuning handoff,
enumerated a candidate, encoded a decision, or prepared a recipe under one explicit candidate.

Prepare 0004 supplies only method-free opaque marker roles and an exact-partition typed handoff.
Tools/tuning 0001 deliberately requires a caller-supplied typed collaboration and complete cold
candidate execution action. It must not import CPU internals, and CPU must not depend on
tools/tuning. This task publishes that route-specific supported seam; Engine 0007 remains blocked
on its separate representative-execution, input-binding, cleanup, and fallback contract.

The existing contracts are sufficient. `CpuPartitionAnalysisInputs` already accepts an optional
CPU decision, the selector validates it against a fresh complete batch, the preparation plan
retains the selected identity and exact resources, and finalization validates the live provider
qualification and installs the selected bounded thread count. No shared-contract or architecture
decision is required.

## Scope

- Add one cohesive supported root-package collaboration,
  `io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning`, with the exact public shape in
  Package and API design.
- Add `CpuBackendIntegration.localWorkloadTuning()`, returning the same retained collaboration for
  the integration lifetime without transferring ownership.
- Produce an optional
  `BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>` from a fresh analysis of one
  supported `CompileArtifacts` value. The handoff retains the exact sole `PlannedPartition`, an
  empty selected decision, and a public opaque batch associated with that exact integration,
  artifacts value, partition, and fresh internal batch.
- Return an empty optional, not a synthetic portable-only batch, when the artifacts are otherwise
  supported but current analysis has no tunable 0010E batch. Continue to reject zero, empty,
  non-CPU, mixed-owner, or multi-partition artifacts exactly as the supported integration does.
- Expose typed public operations for complete ordered candidate enumeration, compatibility,
  candidate identity, selected-decision construction, decision encoding, and compatible decoding.
- Expose one preparation operation for a selected trial candidate and one for a selected decision.
  Each operation repeats authoritative CPU analysis and complete Graph preparation; neither reuses
  the enumeration analysis as authority.
- Keep candidate, batch, and decision values immutable and opaque. They may retain package-private
  delegates and association tokens but expose no CPU-internal field or route vocabulary.
- Use bounded, deterministic, versioned CPU-owned binary encodings for compatibility, candidate
  identity, and decisions. Use fixed numeric tags and length prefixes, never Java serialization,
  enum names, class names, reflection, or string dispatch.
- Preserve session-only versus persistent reuse. A persistent compatibility/decision encoding is
  derived only from the 0010E persistent workload projection. A session-only compatibility value
  includes one immutable per-integration nonce and is usable only with that live collaboration;
  it is never promoted to persistent reuse. Cross-session decoding of a session-only decision
  returns empty.
- Preserve CPU provider, coordinator, concurrency-budget, and close ownership. The collaboration
  owns no independent provider and has no independent `close()` operation.
- Add focused native-free tests and the minimum existing real/native-coordination coverage needed
  to prove the supported surface, association checks, fresh analysis, selected preparation,
  fallback preservation, codec behavior, cleanup, and closure.
- Finalize affected Javadocs, supported CPU package documentation, focused CPU backend guide text,
  glossary impact, and planning evidence in the mandatory separate clean documentation context.

## Out of scope

- measurement, warmup counts, timed samples, clocks, objectives, median calculation, comparison,
  winner selection, occurrence weighting, deduplication, budgets, or tuning evidence
- cache paths, file parsing, cache lookup, mutation, persistence, atomic publication, corruption
  recovery, model/profile fingerprints, or tools/tuning orchestration
- representative Tensor values, binding, trial run invocation, publication reads, result cleanup,
  model execution, strict-versus-fallback policy, or any Engine 0007 behavior
- a CPU dependency on tools/tuning or Engine, or an implementation of
  `BackendWorkloadTuning` inside the CPU module
- Engine representative-execution or fallback contracts, an Engine 0007 specification or
  implementation, Engine 0008, or tools/tuning 0002
- cross-route generalization from CPU 0016; the supported seam wraps only the current CPU 0010E
  exact/default FLOAT32/FLOAT64 bare-MATMUL candidate contract
- new route eligibility, candidate pruning or synthesis, reordered candidates, relaxed numerics,
  BFLOAT16 OpenBLAS, vendor-peer routes 0011–0015, or Config 0004–0006
- changing the method-free Prepare marker roles or handoff, Runtime APIs, `PreparedExecution`,
  Compiler artifacts, Planning ownership, Backend Contract, or provider APIs
- exposing internal workload signatures, candidates, decisions, plans, representations, provider
  handles, qualifications, workspaces, thread coordinators, native addresses, or storage objects
- generated-kernel changes, new generated forms, schema changes unrelated to the CPU adapter
  codec, structural performance gates, or generated-versus-clean-Java performance claims
- a registry, service locator, generic parameter bag, `Object` payload, raw type, unchecked cast,
  callback discovery, plugin mechanism, or broad cross-backend tuning facade
- architecture-contract, ADR, module-dependency, Gradle/build, backend-conformance, or integration-
  test changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/prepare`, Concrete backend modules, Performance evidence and optimization tooling,
  Prepare lifecycle, Runtime service locator, and Dependency rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md)
- [CPU master plan](../master-plan.md)
- [CPU 0010E candidate and decision contract](0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)
- [CPU 0010F supported lifecycle adapter](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [CPU 0010H source-only constant materialization](0010h-source-only-published-constant-cpu-materialization.md)
- [Prepare 0004 opaque tuning handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [Tools/tuning 0001 typed collaboration](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Config 0006A request facade](../../../modules/config/tasks/0006a-model-autotuning-request-configuration.md)
- [Engine master plan](../../../modules/engine/master-plan.md)

## Architecture constraints

- CPU owns route-specific candidates, eligibility, compatibility meaning, canonical codec
  semantics, selected-decision construction and validation, lowering, resource declarations,
  finalization, provider/thread lifetime, and the safe heuristic.
- Prepare transports the typed opaque batch and decision only. Its marker roles remain method-free
  and its handoff retains the exact partition without interpreting CPU content.
- Tools/tuning owns invocation counts, timing, objective comparison, winner selection, cache load,
  mutation, persistence, and evidence. CPU exposes operations an outer owner can adapt to that
  tool; CPU neither implements the tool interface nor duplicates tool behavior.
- Engine later owns representative input binding, complete trial run and cleanup, Config mapping,
  fallback policy, model/profile facts, occurrence context and weights, and final lifecycle
  composition. This task neither designs nor implements those contracts.
- Runtime receives only a fully prepared recipe. It never receives a candidate, batch, decision,
  cache policy, or tuning request and never selects or measures one.
- A candidate or selected decision cannot create eligibility, relax semantics, or change numerical
  and determinism requirements. Trial and final preparation repeat fresh authoritative analysis
  and require an exact compatible selection.
- Ordinary `prepare(CompileArtifacts)` remains correct, deterministic, and safe-heuristic-driven.
  Adapter absence, an empty handoff, stale bytes, or rejected selected preparation cannot change
  or poison that path.
- The collaboration and all values are scoped to one `CpuBackendIntegration`. Public operations
  are safe for concurrent independent calls while it remains open. Close prevents new adapter
  operations, coordinates with the existing provider owner, and does not acquire ownership of
  caller artifacts or returned recipes. Recipes and runs must not outlive the integration.
- Current production composition has capacity one and supplies only the qualified single-thread
  OpenBLAS configuration. Fresh finalization therefore preserves the existing coordinator's
  verified count-one contract. This task must not add multiple provider counts or cross-route
  coordination; either would return the task to planning or belong to CPU 0016.
- Prepared trial recipes are immutable and reusable under the same lifetime rules as ordinary
  recipes. The outer owner performs one complete cold run for each requested warmup or sample and
  owns every Runtime run/result cleanup. CPU does not count or time runs.
- Generated CPU code continues to obey the clean-Java equivalence rule. This adapter changes no
  generated algorithm or kernel, so it adds no new performance proof.

## Package and API design

Existing supported package:

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider
  CpuBackendIntegration
  CpuLocalWorkloadTuning
  package-info.java
```

`CpuLocalWorkloadTuning` is one public final collaboration with nested public immutable value
types so the supported package gains one cohesive concept rather than a family of unrelated
top-level types. Its exact intended surface is:

```java
public final class CpuLocalWorkloadTuning {
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(CompileArtifacts artifacts);
    public List<Candidate> candidates(CandidateBatch batch);
    public Compatibility compatibility(CandidateBatch batch);
    public CandidateIdentity candidateIdentity(Candidate candidate);
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate);
    public byte[] encodeDecision(SelectedDecision decision);
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision);
    public PreparedExecution prepareTrial(CandidateBatch batch, Candidate candidate);
    public PreparedExecution prepareSelected(
            CandidateBatch batch, SelectedDecision decision);

    public static final class CandidateBatch implements BackendTuningCandidateBatch { ... }
    public static final class SelectedDecision implements BackendTuningDecision { ... }
    public static final class Candidate { ... }
    public static final class Compatibility { ... }
    public static final class CandidateIdentity { ... }
    public enum ReuseScope { SESSION, PERSISTENT }
}
```

The collaboration constructor is not public. `CpuBackendIntegration.localWorkloadTuning()` is the
sole supported acquisition path and returns the same retained instance. Nested value constructors
are not public. Public value methods are limited to immutable value semantics and these exact
accessors:

- `Compatibility.schemaVersion()`, `Compatibility.bytes()`, and
  `Compatibility.reuseScope()`;
- `CandidateIdentity.bytes()`;
- ordinary `equals(Object)`, `hashCode()`, and redacted `toString()` where required for the
  tools/tuning adapter and round-trip verification.

Every returned byte array is a defensive copy. Diagnostic text reports only schema, reuse scope,
and byte lengths. Batch, candidate, and decision expose no accessor beyond ordinary association-
safe diagnostic/value behavior needed by tests; in particular they expose no route, thread,
representation, workload, plan, provider, or artifacts accessor. Batch and candidate equality is
exact owner/association identity. Decision equality combines that exact owner/current-batch
association with the internal decision's structural value so a compatible decode against the same
batch satisfies tools/tuning's mandatory round-trip equality check without making decisions from
different integrations equal.

An outer Engine composition can implement its own private
`BackendWorkloadTuning<CandidateBatch, SelectedDecision, Candidate>` by delegating the six matching
operations and translating `Compatibility` and `CandidateIdentity` into the corresponding
tools-owned immutable byte values. That outward adapter belongs to Engine 0007, not this task.

## Exact lifecycle and dataflow

1. `CpuBackendIntegration.open()` creates one composition, one retained
   `CpuLocalWorkloadTuning`, and one immutable session-compatibility nonce. No host query, cache
   read, or measurement is added.
2. `candidateHandoff(artifacts)` checks that the integration is open, rejects a null argument,
   and applies the same exact sole non-empty CPU-partition validation as ordinary preparation.
3. It derives fresh authoritative analysis inputs with no selected decision and runs CPU analysis.
   If analysis produces no 0010E batch, it returns empty. It does not manufacture a portable-only
   tuning workload from an ineligible plan.
4. For an eligible workload, it creates one association token retaining the exact integration,
   `CompileArtifacts`, sole `PlannedPartition`, internal batch, and ordered internal candidates.
   The public batch and each public candidate retain that token. The handoff carries the exact
   partition and an empty decision.
5. Enumeration returns the complete immutable public candidate list in CPU 0010E encounter order.
   Compatibility and identity operations canonically encode the batch workload projection and
   candidate identity. No operation consults a clock, cache, or filesystem.
6. `selectedDecision(batch, candidate)` verifies the collaboration owner and exact batch
   association, then creates one opaque public decision around the existing CPU 0010E decision.
   A candidate from another batch or integration is rejected before preparation.
7. `encodeDecision` emits a bounded versioned representation. Persistent scope contains only the
   stable 0010E persistent workload projection and selected candidate identity. Session scope also
   binds the current collaboration nonce. The encoding contains no live object, provider handle,
   path, class name, native address, or measurement.
8. `decodeCompatibleDecision(batch, bytes)` snapshots and bounds the input before parsing. Invalid,
   truncated, trailing, unsupported-schema, wrong-session, stale-workload, unknown-candidate, or
   otherwise incompatible data returns empty without changing backend state. Programmer errors
   such as a null input still fail normally. Successful decoding reconstructs a decision
   associated with the supplied current batch.
9. `prepareTrial(batch, candidate)` constructs the exact selected decision, derives new analysis
   inputs from the retained artifacts, repeats lowering and candidate generation, and requires the
   fresh batch to accept the exact decision and retain the exact selected identity. It then runs
   ordinary Graph preparation, source-only constant contribution, assignment, finalization, and
   schedule assembly once and returns the resulting `PreparedExecution`.
10. `prepareSelected(batch, decision)` follows the same fresh path. It rejects a decision from a
    different batch/integration and rejects an ineligible, stale, or changed fresh batch rather
    than silently selecting the heuristic. Compatible cache bytes must first be decoded against
    the current batch, which produces a current-associated decision.
11. The outer owner binds equivalent representative inputs, invokes Runtime once for each complete
    cold trial requested by tools/tuning, reads or discards publications as its later contract
    requires, and closes every run/result resource. None of that behavior enters CPU.
12. The final selected decision is prepared through step 10. If later Engine policy permits
    fallback after tuning failure, Engine separately invokes ordinary
    `CpuBackendIntegration.prepare(artifacts)`; CPU never chooses that policy through this adapter.
13. Closing `CpuBackendIntegration` atomically prevents new handoff, codec, and preparation work,
    then uses the existing coordinator close/quiescence/restoration behavior. Racing operations
    have the same admitted-before-close contract as existing CPU preparation. Returned recipes
    must not outlive the integration.

Any failure during fresh analysis, assignment, artifact realization, provider configuration, or
schedule assembly propagates unchanged after existing local cleanup. No partial handoff, decision,
or recipe is published. Codec rejection has no side effect. CPU never deletes or mutates cache
state because it never receives a cache path.

## Affected files

Expected production and test paths:

- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- add at most one package-private CPU-internal codec/helper under the existing OpenBLAS package if
  keeping canonical serialization out of the supported wrapper materially improves cohesion
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatch.java` only if a package-local canonical projection helper is required
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningDecision.java` only if a package-local canonical codec helper is required
- add `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuLocalWorkloadTuningPublicTest.java`
- add at most one focused internal codec/coordination test under the existing mirrored OpenBLAS
  package
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md` only if the supported CPU collaboration creates a reusable term not already
  covered by candidate batch, selected tuning decision, or tuning orchestration
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, focused
architecture explanations, ADRs, Prepare 0004 source/tests/docs, tools/tuning 0001 source/tests/docs,
Config 0006A, Engine source/tests, Compiler/Planning/Runtime/Backend Contract, Gradle files,
architecture tests, backend-conformance tests, integration tests, generated-kernel source/tests,
and optional CPU 0011–0017 plans.

## Maximum scope

At most 17 paths:

- 7 CPU production/Javadoc paths: the three required supported/composition paths plus no more than
  four narrowly justified internal codec/delegate paths;
- 4 focused CPU test paths;
- 2 explanatory documentation paths; and
- 3 planning paths in addition to this task.

The preferred implementation uses one supported top-level type and one package-private internal
codec/helper at most. Do not add a module dependency, build path, shared-contract path,
architecture/ADR path, conformance/integration path, generated-kernel path, Engine Java path, or
tools/tuning Java path. If fresh selected preparation cannot be expressed within this ceiling and
the existing composition, analysis-input, preparation, and finalization contracts, stop and
return the task to planning.

## Acceptance criteria

1. `CpuBackendIntegration.localWorkloadTuning()` returns one retained non-null supported
   collaboration while open; no public constructor, registry, discovery hook, or independent
   close surface exists.
2. The supported CPU package adds exactly one top-level tuning type. Its nested batch and decision
   implement the Prepare marker roles, and all collaboration signatures are fully typed without
   raw types, unchecked casts, arbitrary `Object` payloads, reflection, string dispatch, or maps.
3. A supported one-partition artifact receives a handoff only when fresh CPU analysis produces the
   complete current 0010E candidate batch. The handoff retains the exact partition and has no
   selected decision. A valid non-tunable artifact returns empty; zero/empty/non-CPU/mixed/multi-
   partition artifacts fail before candidate exposure.
4. Candidate enumeration is immutable, complete, stable, portable-first, and exactly ordered like
   the internal batch. No candidate is invented, pruned, reordered, or exposed structurally.
5. Compatibility and candidate identities are non-empty, canonical, defensively copied, bounded,
   and typed. Persistent scope exists only with a stable qualified binary projection; otherwise
   compatibility is session-only and bound to the current integration.
6. Selected-decision construction accepts only a candidate belonging to the exact supplied batch
   and collaboration. Cross-batch and cross-integration combinations fail before any analysis,
   finalization, provider mutation, or artifact realization.
7. Decision encoding is deterministic and versioned, contains no live/internal object, and uses
   fixed binary tags. Compatible decode rejects corruption, unsupported versions, trailing data,
   stale workloads, unknown candidates, and wrong session by returning empty. A successful
   persistent or same-session round trip returns a decision equal to the original public value.
8. Trial preparation selects exactly the requested candidate, repeats authoritative analysis,
   validates fresh compatibility and membership, contributes current source-only constants, and
   returns one complete `PreparedExecution`. It never runs, times, compares, or caches it.
9. Final selected preparation repeats the same authoritative path. An eligibility change,
   incompatible/stale decision, or changed candidate set is rejected and never silently converted
   to heuristic success.
10. Ordinary `prepare(CompileArtifacts)` retains its existing safe heuristic and exact observable
    behavior when tuning is absent or disabled. Failure or rejection through the tuning adapter
    does not mutate later ordinary preparation.
11. Provider qualification, count-one configuration, shared budget admission, prepared execution,
    source-only initializer, and coordinator close/restoration behavior remain owned by the
    existing composition. Portable-only operation requires no provider; native selected
    preparation uses only the exact live qualified coordinator.
12. Independent adapter calls are thread-safe while open. Close prevents new work, racing admitted
    work follows the existing lifecycle contract, repeated close is inert, and all public methods
    reject use after close. Artifacts, trial recipes, Runtime runs, results, and input storage are
    never closed by the collaboration.
13. CPU performs no measurement, cache I/O, model/profile orchestration, representative-input
    binding, Runtime selection, fallback-policy choice, or Engine work. CPU has no tools/tuning or
    Engine dependency.
14. CPU 0016 remains Draft and unchanged as later cross-route generalization; CPU 0011 remains
    Blocked and CPU 0012–0015/0017 retain their current optional statuses and ordering.
15. No generated kernel, generated-artifact identity, lowering semantic, numerical rule,
    performance threshold, architecture contract, ADR, Gradle file, conformance test, or
    integration test changes.
16. A separate clean documentation-focused context independently reviews the implementation and
    tests, finalizes every affected Javadoc and supported-package explanation, updates the focused
    CPU guide/glossary only where needed, and records reasoned no-change conclusions for shared
    APIs, architecture, Config, Engine API, and Runtime documentation. It does not repeat a
    successful Java suite unless it changes executable Java or identifies stale evidence.
17. Focused and full CPU tests, CPU Javadoc, public-shape/import/javap checks, codec malformed-input
    checks, changed-Markdown links/anchors/headings/fences/newlines, exact scope, and
    `git diff --check` pass with recorded evidence.

## Validation

Validation tier: focused CPU module validation plus full CPU module validation. Repository-wide
validation is not required because the task adds no dependency, build, shared-contract,
architecture, or multi-module executable change.

Implementation context:

```bash
./gradlew :backends:cpu:test --tests '*CpuLocalWorkloadTuningPublicTest' \
  --tests '*CpuBackendCompositionTest' --tests '*CpuOpenBlas*Tuning*Test'
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

Also inspect the compiled public surface with `javap`; verify imports and bytecode/source contain
no tools/tuning, Engine, `.internal` type in a public signature, raw/unchecked bridge, reflection,
Java serialization, cache/path/I/O, timing, or string-dispatch mechanism. Exercise malformed and
bounded codec inputs, persistent and session-only round trips, wrong-session rejection, exact
association rejection, fresh-analysis stale rejection, ordinary heuristic preservation,
source-only constants, provider/no-provider cases, concurrency, close races, and cleanup.

Documentation context:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Inspect rendered Javadoc for every new or changed public/nested type and method. Check all changed
Markdown links and anchors, unique headings, balanced fences, LF endings, final newlines, exact
path count, and synchronized statuses. Reuse the implementation context's successful CPU tests
unless documentation changes executable Java.

## Dependencies

- CPU 0010E: Complete
- CPU 0010F: Complete
- CPU 0010H: Complete
- Prepare 0004: Complete
- tools/tuning 0001: Complete consumer contract; CPU does not depend on its module
- Config 0006A: Complete boundary evidence only; not an implementation dependency
- Compiler 0006B3 and Prepare 0003/0005: Complete lifecycle and source-only handoff
- Runtime 0010 and closure hardening: Complete recipe/run lifecycle
- Engine 0007: remains Blocked and is not a dependency

## Documentation-focused pass

After implementation and focused/full CPU validation, a distinct clean documentation context
must read the final diff, this task, CPU 0010E/0010F/0010H, Prepare 0004, tuning 0001, Config
0006A, the documentation rules and General/API-Javadoc profiles, the supported CPU package, and
the focused CPU guide/glossary entries. It must:

- finalize meaningful Javadocs for lifetime, association, byte snapshots, nullability, return
  semantics, rejection, failure, ownership, concurrency, and close behavior;
- explain that the supported CPU collaboration is an outer-composition building block, not a
  tuning runner or application facade;
- preserve the CPU/tools/Engine/Runtime ownership split and safe-heuristic path;
- update the glossary only if existing terms cannot describe the new supported seam;
- record explicit no-change conclusions for architecture/ADRs, Prepare and tuning APIs, Config,
  Engine API, Runtime, builds, conformance/integration docs, and generated-code guidance; and
- synchronize this task, CPU master plan, Engine master plan, and roadmap completion evidence only
  after implementation is actually complete.

## Follow-up

- Reassess Engine 0007 in a new clean planning context only after CPU 0010I is Complete and the
  separate Engine representative-execution/fallback contract is explicitly selected. That Engine
  task may privately adapt this collaboration to tools/tuning 0001 and map Config 0006A.
- Do not create the representative-execution/fallback task in this change.
- Keep tools/tuning 0002 as later graph/plan tuning.
- Keep CPU 0016 as later cross-route generalization after concrete vendor peer routes exist; it
  must reuse rather than duplicate this supported composition shape where applicable.

## Architecture impact

No architecture-contract, ADR, module-boundary, or dependency change is intended. The task adds a
narrow supported CPU integration SPI in the already supported root package and delegates to
existing CPU-owned analysis/finalization behavior. If implementation needs a shared marker method,
a tools/tuning or Engine dependency, a Runtime candidate API, multi-provider coordination, or a
new ownership rule, stop and return the task to planning.

## Local decisions and known limits

- The public collaboration is CPU-owned and tools-shaped, but it does not implement a tools-owned
  interface because that would reverse the dependency boundary.
- Valid CPU artifacts with no 0010E batch return empty. A portable plan is not automatically a
  tunable batch; CPU 0010E intentionally creates a batch only when at least one eligible OpenBLAS
  peer exists.
- Public batch/candidate/decision values are exact association handles, not portable domain DTOs.
  Only compatibility, identity, and encoded decision bytes cross invocation or persistence
  boundaries.
- Persistent reuse remains available only for the existing qualified persistent-binary scope.
  Session-only values may deduplicate inside the current tuning invocation/integration but cannot
  become a cross-session cache hit.
- Trial and final preparation use the current capacity-one/single-thread production composition.
  Multiple provider thread counts, overlapping differently configured prepared recipes, and
  vendor-peer coordination are deliberately not generalized here.
- `PreparedExecution` ownership and Runtime cleanup do not change. The collaboration creates a
  recipe; it does not own or close a later run/result.
- Representative inputs and failure fallback remain the separate unresolved Engine prerequisite.
  Their absence does not prevent CPU from exposing this bounded preparation seam.

## Implementation prompt

Work in `/Users/phujka/IdeaProjects/Synaptik`. Read `AGENTS.md`, `ARCHITECTURE.md`, the current
architecture plan, planning guide/roadmap, documentation rules and General/API-Javadoc profiles,
CPU master plan and tasks 0010E/0010F/0010H plus relevant 0010C/0010D, Prepare 0004 and its source
and tests, tools/tuning 0001 public contracts/tests, Config 0006A source, Engine master plan and
current composition sources/tests, and every CPU source/test touched by this specification.

Implement task 0010I exactly within its allowlist and maximum scope. Add the single supported
`CpuLocalWorkloadTuning` top-level type and the retained acquisition method on
`CpuBackendIntegration`. Keep all nested values opaque and association-safe. Generate handoffs
only from fresh eligible 0010E analysis, expose the exact ordered candidates and canonical typed
compatibility/identity/decision operations, implement bounded versioned encoding/compatible
decoding, and prepare trial/final selected recipes only after fresh authoritative analysis accepts
the exact choice. Preserve ordinary heuristic preparation, source-only constants, current
provider/thread coordination, closure, and every ownership boundary. Add no measurement, cache
I/O, Engine orchestration, Runtime selection, dependency, generated-kernel behavior, or cross-
route generalization.

Run focused and full CPU tests, CPU Javadoc, public-surface/import/bytecode/manual codec checks,
Markdown/scope/status validation, and `git diff --check`. Then use the mandatory separate clean
documentation-focused context to finalize affected Javadocs and focused documentation without
repeating successful Java tests unless executable Java changes. Do not commit or push. If the
existing contracts cannot support the exact fresh-analysis and lifetime rules inside the scope
ceiling, stop and report the missing prerequisite instead of widening the architecture.

## Completion summary

Implementation context `01a0b149-462c-7cc2-a326-1b34f5b518f3` completed the bounded supported
adapter. `CpuBackendIntegration` now retains one `CpuLocalWorkloadTuning`; the new collaboration
exposes exact eligible handoffs, portable-first opaque candidates, typed compatibility and
candidate identities, association-safe selected decisions, bounded versioned persistent/session
codecs, and fresh authoritative trial/final selected preparation. Ordinary
`prepare(CompileArtifacts)` retains its safe heuristic. No measurement, representative binding,
execution, winner selection, cache/filesystem behavior, Engine/tools-tuning dependency,
reflection/serialization, generated-code change, or architecture/dependency change was added.

Actual implementation-owned paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuLocalWorkloadTuningPublicTest.java`

Actual documentation-owned paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

Documentation context `01a0b164-e6e6-78f2-9e15-c3c4fef144bf` finalized those paths. The glossary
gained no new term because its existing candidate generator, selected tuning decision, opaque
backend tuning handoff, and model-autotuning definitions already cover the supported seam; only
their stale pre-0010I status text was corrected.

Implementation validation passed 20/20 focused tests and the full CPU suite passed 1,016 tests
with 28 existing conditional skips and zero failures or errors. CPU Javadoc passed with 94
pre-existing warnings in unchanged lowering/preparation-plan declarations and no warning from
0010I. Public-source, `javap`, and bytecode checks found no `.internal` type in a supported
signature and no forbidden dependency or behavior. The documentation pass preserved every dirty
Java/test path byte-for-byte, inspected the rendered supported page to confirm that the hidden
package-private construction seam does not expose `CpuBackendComposition`, and passed the focused
Markdown, status, exact-scope, and whitespace checks recorded for this task.

`ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, Prepare and
tools/tuning public contracts, Config, Engine and Runtime APIs, other guides, backend conformance,
integration tests, generated-code guidance, and Gradle files require no change: ownership,
dependency direction, executable semantics, and cross-module contracts are unchanged. CPU 0010I
closes only the CPU prerequisite. Engine 0007 remains Blocked until a separate Engine-owned
representative-execution/input-binding/cleanup and fallback contract is specified and completed;
Engine 0008 remains Draft without a task specification.

```text
Status: Complete
```
