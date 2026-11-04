# Task 0004: Typed [Metal](../../../../glossary.md#term-metal) Route Candidate Generators and [Cache](../../../../glossary.md#term-cache) Compatibility

## Reader context

The cold-path objects here describe choices, not tensor values. A workload signature describes
the validated partition; a candidate describes a complete implementation alternative; a decision
identifies the selected candidate. Metal regenerates those facts before trusting a supplied
decision, then fixes the route before shared Prepare assigns resources.

Structural compatibility ignores incidental graph identifiers, but the context's session nonce
deliberately prevents another integration from reusing the decision. These encoded bytes contain
neither native handles nor a compiled executable. This task supplied that backend-private
foundation, not measurement or cache-file orchestration.
The [current session decision and tuning boundary](../../../../backend-guide/metal-backend.md#session-decision-codec-and-limitations)
explains the later public collaborations and why session reuse still does not mean persistent
Metal executable caching.

## Status

Complete

## Change class

Class C — candidate choice changes [Metal](../../../../glossary.md#term-metal) native-route preparation, exact resource declarations,
and the [opaque](../../../../glossary.md#term-opaque) [Prepare](../../../../glossary.md#prepare)/tuning boundary; decision reuse also carries compatibility, [lifecycle](../../../../glossary.md#lifecycle), and
cold-path [authentication](../../../../glossary.md#term-authentication) risk even though no shared [API](../../../../glossary.md#term-application-programming-interface-api), native [ABI](../../../../glossary.md#term-application-binary-interface-abi), or [Runtime](../../../../glossary.md#term-runtime) behavior changes.

## Goal

Add a [Metal](../../../../glossary.md#term-metal)-local, typed, complete candidate-generation and session-compatible decision-[codec](../../../../glossary.md#term-codec)
foundation for the two implemented [`FLOAT32`](../../../../glossary.md#term-float32) [unary](../../../../glossary.md#term-unary) `NEG` routes, while preserving the current
[heuristic](../../../../glossary.md#term-heuristic) when no compatible decision is supplied and keeping all Metal fields [backend](../../../../glossary.md#backend)-private.

## [Scope](../../../../glossary.md#term-scope)

Size justification: this [atomic](../../../../glossary.md#resource-transaction-and-rollback) Class C brief exceeds 15 KB because it keeps candidate
compatibility, decision [authentication](../../../../glossary.md#term-authentication), route-specific declarations, exact isolation, [validation](../../../../glossary.md#term-validation),
and independent-review evidence together at one [backend](../../../../glossary.md#backend)/[Prepare](../../../../glossary.md#prepare) boundary.

- Derive a canonical version-1 Metal NEG workload signature from the already validated exact
  semantics, attributes, ordered topology, descriptors, layouts, constants, boundary facts,
  exact/default policy, and live Metal target context.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Metal glossary](../../../../glossary.md#term-metal); [signature glossary](../../../../glossary.md#term-signature); [topology glossary](../../../../glossary.md#term-topology); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout); [constants glossary](../../../../glossary.md#term-constant); [target glossary](../../../../glossary.md#term-target))</span>
- Generate a stable budget-bounded candidate list: the current safe heuristic first, followed by
  the other valid route. A custom singleton candidate exists only for one node, one feed, one
  target, and checked element count <span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [singleton glossary](../../../../glossary.md#term-singleton); [node glossary](../../../../glossary.md#node); [element glossary](../../../../glossary.md#term-element))</span>`1..UINT32_MAX`; MPSGraph remains valid for every currently
  supported partition.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [partition glossary](../../../../glossary.md#partition))</span>
- Add version-1 candidate, compatibility, and decision-codec schemas. Compatibility is
  conservatively session-scoped to the exact live <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [schemas glossary](../../../../glossary.md#term-schema))</span>`MetalDeviceContext`; [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `3` alone is
  not a stable cross-session device fingerprint.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Construct absent- and present-decision `BackendPartitionTuningHandoff` values inside Metal, then
  regenerate and authenticate current facts during Metal analysis before applying a selection.
  This is a package-private backend foundation only: no `tools/tuning` adapter, outer
  `WorkloadCompatibility`, [cache](../../../../glossary.md#term-cache) [lookup](../../../../glossary.md#term-lookup), measurement, or [Engine](../../../../glossary.md#term-engine) consumer is added in this task.
- Keep the package-private NEG preparer as the sole route selector and exact declaration [owner](../../../../glossary.md#term-owner).

## Non-goals

- New operations, data types, layouts, routes, fusion, capability, partitioning, performance
  claims, or changes to the current two-route native implementations.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Persistent or cross-session Metal cache reuse, outer workload-cache participation, cache-file
  I/O, measurement, winner selection, public Metal/Engine composition, or a dependency on<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [cache glossary](../../../../glossary.md#term-cache); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency))</span>
  `tools/tuning`.
- Native ABI changes, executable serialization, pipeline caches, hidden global state, Runtime
  cache/selection work, or shared Prepare/Config/Tuning API changes.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [serialization glossary](../../../../glossary.md#term-serialization); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership, backend prepare selects the route before Runtime, and tuning completes
  before the hot path.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [`ARCHITECTURE.md` — Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> — Metal
  keeps its current inward production edges and neither Runtime nor Prepare imports Metal.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span> — Metal owns typed complete candidates, route selection,
  compatible lookup semantics, safe heuristic fallback, and exact declarations.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- [Backend execution — Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling) — tooling owns cache-file
  orchestration while treating Metal candidates, identities, and decisions opaquely.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — [lowering](../../../../glossary.md#lowering), custom [kernels](../../../../glossary.md#kernel), [MPSGraph](../../../../glossary.md#term-mpsgraph), storage, and [trace](../../../../glossary.md#trace) remain [Metal](../../../../glossary.md#term-metal)-owned.
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span> and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) — analysis selects one route and declares all shared resources before
  assignment; finalization cannot reselect or add requirements.
- [Runtime/Prepare/Engine — `modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime) — [Runtime](../../../../glossary.md#term-runtime) receives prepared typed work and performs no selection or [cache](../../../../glossary.md#term-cache) access.
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>, [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md), [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md), [ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md), and the [Metal strategy](../../../../design/notes/metal-backend-strategy.md) explain the accepted [ownership](../../../../glossary.md#term-ownership-meanings) and [lifecycle](../../../../glossary.md#lifecycle).

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegRouteCandidateGenerator.java` (new) — package-private generator, [canonical](../../../../glossary.md#term-canonical-form) `WorkloadSignature`, stable pruning, and [opaque](../../../../glossary.md#term-opaque) [handoff](../../../../glossary.md#term-handoff) construction.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningBatch.java` (new) — [immutable](../../../../glossary.md#term-immutable) marker-role [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), typed route candidates, [schema](../../../../glossary.md#term-schema) [constants](../../../../glossary.md#term-constant), [target](../../../../glossary.md#term-target)/session compatibility, and stable [identities](../../../../glossary.md#term-identity-meanings).
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningDecision.java` (new) — immutable marker-role selection and fresh-batch matching.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningCodec.java` (new) — bounded canonical [Metal](../../../../glossary.md#term-metal) compatibility/candidate bytes and [fail-closed](../../../../glossary.md#term-fail-closed) decision encode/decode; no `tools/tuning` type or file I/O.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java` — one fresh private session nonce per context; no native probing or [ABI](../../../../glossary.md#term-application-binary-interface-abi) change.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegAnalysisInputs.java` — optional selected Metal handoff carried opaquely into fresh analysis, while preserving the current no-decision construction form.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionPreparer.java` and `MetalNegPreparationPlan.java` — apply an authenticated decision or the unchanged safe [heuristic](../../../../glossary.md#term-heuristic), then declare exactly the selected route's resources.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java` — package boundary and [cache](../../../../glossary.md#term-cache)/[Runtime](../../../../glossary.md#term-runtime) exclusions.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegRouteCandidateGeneratorTest.java` (new) — schema, completeness, pruning, [identity](../../../../glossary.md#term-identity-meanings), compatibility, [codec](../../../../glossary.md#term-codec), corruption, and fresh-[authentication](../../../../glossary.md#term-authentication) coverage.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutionTest.java` — selected-route declarations/finalization plus heuristic and execution regressions.
- `docs/backend-guide/metal-backend.md` and `docs/glossary.md` — explain typed candidates, session compatibility, opaque handoff, fallback, and current limitations.
- This task, `docs/planning/backends/metal/master-plan.md`, and `docs/planning/roadmap.md` — result and synchronized frontier only.

Maximum implementation [scope](../../../../glossary.md#term-scope) is exactly these 16 paths: nine Metal production paths, two Metal
test paths, two explanatory documents, and three [planning](../../../../glossary.md#term-planning) paths counting this task while noting
that `MetalNegPartitionPreparer.java` and `MetalNegPreparationPlan.java` share one bullet. No
other file, package, [module](../../../../glossary.md#term-module), build edge, or generated [artifact](../../../../glossary.md#term-artifact) may change.

## Acceptance criteria

- Capability remains exactly fully static positive [rank](../../../../glossary.md#term-rank)-`1..16`, resolved dense-contiguous,
  non-view, zero-offset <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings))</span>[`FLOAT32`](../../../../glossary.md#term-float32) NEG with equal input/output [Shape](../../../../glossary.md#shape) and `requiresGrad`; the only
  candidates are typed `CUSTOM_SINGLE_NEG` and [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) configurations.
- Generation is deterministic and exposes every valid complete candidate within the positive
  budget: current heuristic first; custom is pruned outside its exact singleton/index domain;
  budget one therefore preserves the safe current choice without tuning.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Version-1 workload compatibility covers all mutable semantic, topology, type, Shape, layout,
  constant, boundary, exact-policy, candidate-schema, ABI-schema, and live-target/session facts.
  Canonical topology uses structural positions rather than <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [topology glossary](../../../../glossary.md#term-topology); [layout glossary](../../../../glossary.md#layout); [constant glossary](../../../../glossary.md#term-constant); [schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [target glossary](../../../../glossary.md#term-target); [Canonical glossary](../../../../glossary.md#term-canonical-form))</span>[`NodeId`](../../../../glossary.md#nodeid), [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), or partition object
  identity, so equal valid workloads in one context compare equally across occurrences. Every
  independently variable valid fact produces a distinct compatibility; an invalid semantic change
  fails generation, and operation family alone is never a key.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [operation glossary](../../../../glossary.md#operation))</span>
- The Metal decision encoding is bounded, canonical, contains no executable/native handle, and
  returns an empty incompatible result for wrong magic/version/scope,
  malformed/truncated/trailing/corrupt bytes, changed workload or session target, and unknown
  candidates. The task does not adapt these bytes to the outer workload cache; its artifact
  version, checksum, atomic replacement, objective, and sampling remain owned by <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [scope glossary](../../../../glossary.md#term-scope); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact); [checksum glossary](../../../../glossary.md#term-checksum); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [objective glossary](../../../../glossary.md#term-objective); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>`tools/tuning`
  and unchanged.
- A present decision is never trusted directly: fresh analysis regenerates the batch and accepts
  it only when schema, workload, session target, and candidate identity match. A stale/foreign
  present decision fails closed; absence uses the existing heuristic without cache or measurement.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- The authenticated route is fixed before declarations. Custom declares feed and target buffers
  only; MPSGraph declares the same buffers plus exactly one address workspace. Finalization still
  cannot reselect, retry another route, or introduce a requirement.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- The existing Prepare marker roles and handoff are sufficient opaque transport types, but no
  shared Prepare or tuning consumer is added. No private Metal field, generic map, reflection,
  string dispatch, registry, new public type, shared source change, or production dependency leaks
  outward.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [handoff glossary](../../../../glossary.md#term-handoff); [opaque glossary](../../../../glossary.md#term-opaque); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Candidate generation, encoding, and authentication are cold, immutable, thread-safe, and free
  of native allocation. Runtime hot execution, persistent-resource ownership, concurrent-run
  isolation, close behavior, and one typed native call per prepared invocation remain unchanged.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [immutable glossary](../../../../glossary.md#term-immutable); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership); [run glossary](../../../../glossary.md#term-run))</span>
- Existing fake-native execution and capability tests pass; Javadocs explain every changed input,
  result, constraint, ownership rule, and failure mode.
- Stop and return to planning before editing if truthful compatibility needs a shared contract,
  public API, <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`tools/tuning` adapter or outer compatibility type, cross-session target promise,
  native ABI change, new capability/route, cache-file implementation, shared/module/build change,
  or more than the exact allowlist.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## [Validation](../../../../glossary.md#term-validation)

```bash
./gradlew :backends:metal:test --tests '*MetalNegRouteCandidateGeneratorTest' --tests '*MetalNegPreparedExecutionTest'
./gradlew :backends:metal:test
./gradlew :testing:backend-conformance:test --tests '*MetalNegCapabilityPartitionConformanceTest'
./gradlew :backends:metal:javadoc
rg -n -i 'candidate|tuning|cache|compatib|fingerprint|Metal NEG prepared executable' docs/glossary.md docs/backend-guide/metal-backend.md
test -f ARCHITECTURE.md \
  -a -f docs/architecture/contracts/backend-execution.md \
  -a -f docs/architecture/contracts/runtime-prepare-engine.md \
  -a -f docs/design/decisions/0002-backend-owned-lowering.md \
  -a -f docs/design/decisions/0008-performance-evidence-and-tuning-boundaries.md \
  -a -f docs/design/decisions/0010-staged-backend-preparation.md \
  -a -f docs/design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md \
  -a -f docs/design/notes/metal-backend-strategy.md
awk '/^```/{n++} END{exit n%2}' docs/backend-guide/metal-backend.md docs/glossary.md docs/planning/backends/metal/tasks/0004-typed-metal-route-candidate-generators-and-cache-compatibility.md
git diff --check
git status --short
```

The focused tests must cover both candidate domains, budget prefixes, independent [signature](../../../../glossary.md#term-signature)
changes, structurally equal workloads with different [graph](../../../../glossary.md#graph) [identities](../../../../glossary.md#term-identity-meanings), defensive bytes, [codec](../../../../glossary.md#term-codec)
corruption/version/session rejection, fresh decision [authentication](../../../../glossary.md#term-authentication), exact declarations, no
native allocation during generation, and concurrent cold use. The ordinary [module](../../../../glossary.md#term-module) suite covers
existing native seams and hot-path/[lifecycle](../../../../glossary.md#lifecycle) regressions.
The current [conformance](../../../../glossary.md#term-conformance) harness is relevant only to prove unchanged capability/partitioning.
No [architecture test](../../../../glossary.md#term-architecture-test) applies because production dependencies, module boundaries, and shared
source remain unchanged; no integration harness supports public [Metal](../../../../glossary.md#term-metal) composition.

Repository-wide validation is deferred to [continuous integration](../../../../glossary.md#term-continuous-integration-ci) or a later Metal capability
checkpoint: this is one [backend](../../../../glossary.md#backend)-local cold-preparation change with no shared/build/[dependency](../../../../glossary.md#term-dependency) or
public-composition change. The independent review reuses successful executable evidence unless it
changes Java behavior or identifies a concrete stale-evidence risk.

## Dependencies and follow-up

- Metal 0002–0003, Prepare 0004/0006, Runtime 0016, and tuning 0001 are Complete and provide the
  two routes, opaque handoff types, prepared-resource transaction, Runtime owner, and separate
  outer-cache rules. The outer tuning workflow does not consume Metal 0004 output.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff); [resource transaction glossary](../../../../glossary.md#resource-transaction-and-rollback); [owner glossary](../../../../glossary.md#term-owner); [cache glossary](../../../../glossary.md#term-cache))</span>
- Cross-session reuse requires a separately authorized stable Metal device/library fingerprint;
  public Metal/Engine and <span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Engine glossary](../../../../glossary.md#term-engine))</span>`tools/tuning` composition remains a later task and cannot widen 0004.
  The current generic workload tuner loads its explicit cache file before applying per-workload
  `SESSION` reuse, so later Metal composition must separately review that outer behavior rather
  than treating 0004's session codec as a no-I/O end-to-end workflow.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>

## Documentation and review impact

- Update the Metal backend guide, affected implementation Javadocs, package Javadoc, and the
  existing glossary term after a targeted search; do not present session compatibility as a
  persistent cache, current tuning integration, or performance claim.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [Javadoc glossary](../../../../glossary.md#term-javadoc); [cache glossary](../../../../glossary.md#term-cache))</span>
- A clean implementation context and an independent targeted Class C documentation/review context
  are mandatory because route decisions, native-resource declarations, tuning compatibility, and
  a durable backend/Prepare boundary change.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>

## Result

Implemented the package-private [Metal](../../../../glossary.md#term-metal) NEG candidate and session-[authentication](../../../../glossary.md#term-authentication) foundation. Metal
now generates stable budget prefixes over typed `CUSTOM_SINGLE_NEG` and [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) candidates,
uses structural version-one workload fingerprints plus one fresh context nonce, constructs [opaque](../../../../glossary.md#term-opaque)
absent/present [Prepare](../../../../glossary.md#prepare) handoffs, and freshly authenticates every supplied selection before fixing
the route and exact declarations. The bounded checksummed Metal [codec](../../../../glossary.md#term-codec) rejects malformed, corrupt,
stale, foreign-session, and unknown-candidate decisions. It performs no native allocation, file
I/O, measurement, outer workload-[cache](../../../../glossary.md#term-cache) adaptation, or [Runtime](../../../../glossary.md#term-runtime) work.

Exact changed files (16):

- `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`,
  `MetalNegTuningDecision.java`, and `MetalNegTuningCodec.java` were added under the Metal internal
  production package.
- `MetalDeviceContext.java`, `MetalNegAnalysisInputs.java`,
  `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java`, and `package-info.java` were
  updated under that package.
- `MetalNegRouteCandidateGeneratorTest.java` was added and
  `MetalNegPreparedExecutionTest.java` was updated.
- `docs/backend-guide/metal-backend.md`, `docs/glossary.md`, this task, the Metal master plan, and
  the repository roadmap were updated.

[Validation](../../../../glossary.md#term-validation) completed successfully:

- `./gradlew :backends:metal:test --tests '*MetalNegRouteCandidateGeneratorTest' --tests '*MetalNegPreparedExecutionTest'`
  passed 34 tests: 32 passed and two native opt-in tests skipped.
- `./gradlew :backends:metal:test`
  passed 51 tests: 48 passed and three native opt-in tests skipped.
- `./gradlew :testing:backend-conformance:test --tests '*MetalNegCapabilityPartitionConformanceTest'`
  passed both tests without skips.
- `./gradlew :backends:metal:javadoc`
  completed without warnings or errors.
- the task's targeted terminology search, required-file check, Markdown-fence check, exact
  16-path audit, and `git diff --check` passed.

The focused tests cover both domains, budget prefixes, structural [identity](../../../../glossary.md#term-identity-meanings) independence,
independent compatibility changes, invalid semantics, defensive codec failures, session/workload
rejection, fresh authentication, exact declarations/finalization, no native allocation, and
concurrent cold use. The ordinary [module](../../../../glossary.md#term-module) suite retains fake-native execution and [lifecycle](../../../../glossary.md#lifecycle)
coverage; the [conformance](../../../../glossary.md#term-conformance) test confirms unchanged capability and partitioning.

Documentation/[Javadoc](../../../../glossary.md#term-javadoc) impact: implementation and package Javadocs describe [schemas](../../../../glossary.md#term-schema), [ownership](../../../../glossary.md#term-ownership-meanings),
session [scope](../../../../glossary.md#term-scope), failure behavior, and cache/Runtime exclusions. The [backend](../../../../glossary.md#backend) guide and glossary now
explain typed candidates, opaque [handoff](../../../../glossary.md#term-handoff), fresh authentication, safe fallback, codec boundaries,
and the absence of current persistent-cache or tools integration. No architecture, public [API](../../../../glossary.md#term-application-programming-interface-api),
shared source, native [ABI](../../../../glossary.md#term-application-binary-interface-abi), build/[dependency](../../../../glossary.md#term-dependency), capability, or Runtime documentation changed.

The mandatory independent targeted Class C documentation/code review inspected the exact
16-path diff, affected implementation and tests, generated test reports, and the named
architecture headings and decision records. It found and repaired one candidate-completeness
defect: generation had treated an already selected plan route as structural eligibility, so an
eligible [singleton](../../../../glossary.md#term-singleton) plan carrying an authenticated MPSGraph selection could omit the still-valid
custom candidate if regenerated. Generation now derives custom validity only from structural
workload facts, and a focused [regression](../../../../glossary.md#term-regression) covers regeneration from that selected plan. The review
also repaired stale post-completion wording in this task, the Metal master plan, and the roadmap.
After the repair,
`./gradlew :backends:metal:test --tests '*MetalNegRouteCandidateGeneratorTest' --tests '*MetalNegPreparedExecutionTest' :backends:metal:javadoc`
passed 34 focused tests (32 passed and two native opt-in tests skipped) and regenerated Javadoc
without warnings or errors. The earlier successful full Metal and conformance results remain
relevant because the repair changes only cold candidate regeneration and the focused regression
directly exercises that behavior.

Limitations and follow-up: cross-session reuse still requires a separately authorized stable
Metal [device](../../../../glossary.md#term-device)/library [fingerprint](../../../../glossary.md#term-fingerprint). Public Metal/[Engine](../../../../glossary.md#term-engine) and `tools/tuning` composition, persistent
cache I/O, measurement, and winner selection remain outside this task. No unresolved
implementation issue remains.

Status: Complete
