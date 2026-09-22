# Task 0004: Typed Metal Route Candidate Generators and Cache Compatibility

## Status

Complete

## Change class

Class C — candidate choice changes Metal native-route preparation, exact resource declarations,
and the opaque Prepare/tuning boundary; decision reuse also carries compatibility, lifecycle, and
cold-path authentication risk even though no shared API, native ABI, or Runtime behavior changes.

## Goal

Add a Metal-local, typed, complete candidate-generation and session-compatible decision-codec
foundation for the two implemented `FLOAT32` unary `NEG` routes, while preserving the current
heuristic when no compatible decision is supplied and keeping all Metal fields backend-private.

## Scope

Size justification: this atomic Class C brief exceeds 15 KB because it keeps candidate
compatibility, decision authentication, route-specific declarations, exact isolation, validation,
and independent-review evidence together at one backend/Prepare boundary.

- Derive a canonical version-1 Metal NEG workload signature from the already validated exact
  semantics, attributes, ordered topology, descriptors, layouts, constants, boundary facts,
  exact/default policy, and live Metal target context.
- Generate a stable budget-bounded candidate list: the current safe heuristic first, followed by
  the other valid route. A custom singleton candidate exists only for one node, one feed, one
  target, and checked element count `1..UINT32_MAX`; MPSGraph remains valid for every currently
  supported partition.
- Add version-1 candidate, compatibility, and decision-codec schemas. Compatibility is
  conservatively session-scoped to the exact live `MetalDeviceContext`; ABI version `3` alone is
  not a stable cross-session device fingerprint.
- Construct absent- and present-decision `BackendPartitionTuningHandoff` values inside Metal, then
  regenerate and authenticate current facts during Metal analysis before applying a selection.
  This is a package-private backend foundation only: no `tools/tuning` adapter, outer
  `WorkloadCompatibility`, cache lookup, measurement, or Engine consumer is added in this task.
- Keep the package-private NEG preparer as the sole route selector and exact declaration owner.

## Non-goals

- New operations, data types, layouts, routes, fusion, capability, partitioning, performance
  claims, or changes to the current two-route native implementations.
- Persistent or cross-session Metal cache reuse, outer workload-cache participation, cache-file
  I/O, measurement, winner selection, public Metal/Engine composition, or a dependency on
  `tools/tuning`.
- Native ABI changes, executable serialization, pipeline caches, hidden global state, Runtime
  cache/selection work, or shared Prepare/Config/Tuning API changes.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership, backend prepare selects the route before Runtime, and tuning completes
  before the hot path.
- [`ARCHITECTURE.md` — Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) — Metal
  keeps its current inward production edges and neither Runtime nor Prepare imports Metal.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules) — Metal owns typed complete candidates, route selection,
  compatible lookup semantics, safe heuristic fallback, and exact declarations.
- [Backend execution — Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling) — tooling owns cache-file
  orchestration while treating Metal candidates, identities, and decisions opaquely.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — lowering, custom kernels, MPSGraph, storage, and trace remain Metal-owned.
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare) and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) — analysis selects one route and declares all shared resources before
  assignment; finalization cannot reselect or add requirements.
- [Runtime/Prepare/Engine — `modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime) — Runtime receives prepared typed work and performs no selection or cache access.
- [ADR 0002](../../../../design/decisions/0002-backend-owned-lowering.md), [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md), [ADR 0010](../../../../design/decisions/0010-staged-backend-preparation.md), [ADR 0013](../../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md), and the [Metal strategy](../../../../design/notes/metal-backend-strategy.md) explain the accepted ownership and lifecycle.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegRouteCandidateGenerator.java` (new) — package-private generator, canonical `WorkloadSignature`, stable pruning, and opaque handoff construction.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningBatch.java` (new) — immutable marker-role batch, typed route candidates, schema constants, target/session compatibility, and stable identities.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningDecision.java` (new) — immutable marker-role selection and fresh-batch matching.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegTuningCodec.java` (new) — bounded canonical Metal compatibility/candidate bytes and fail-closed decision encode/decode; no `tools/tuning` type or file I/O.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalDeviceContext.java` — one fresh private session nonce per context; no native probing or ABI change.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegAnalysisInputs.java` — optional selected Metal handoff carried opaquely into fresh analysis, while preserving the current no-decision construction form.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPartitionPreparer.java` and `MetalNegPreparationPlan.java` — apply an authenticated decision or the unchanged safe heuristic, then declare exactly the selected route's resources.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/internal/package-info.java` — package boundary and cache/Runtime exclusions.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegRouteCandidateGeneratorTest.java` (new) — schema, completeness, pruning, identity, compatibility, codec, corruption, and fresh-authentication coverage.
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/internal/MetalNegPreparedExecutionTest.java` — selected-route declarations/finalization plus heuristic and execution regressions.
- `docs/backend-guide/metal-backend.md` and `docs/glossary.md` — explain typed candidates, session compatibility, opaque handoff, fallback, and current limitations.
- This task, `docs/planning/backends/metal/master-plan.md`, and `docs/planning/roadmap.md` — result and synchronized frontier only.

Maximum implementation scope is exactly these 16 paths: nine Metal production paths, two Metal
test paths, two explanatory documents, and three planning paths counting this task while noting
that `MetalNegPartitionPreparer.java` and `MetalNegPreparationPlan.java` share one bullet. No
other file, package, module, build edge, or generated artifact may change.

## Acceptance criteria

- Capability remains exactly fully static positive rank-`1..16`, resolved dense-contiguous,
  non-view, zero-offset `FLOAT32` NEG with equal input/output Shape and `requiresGrad`; the only
  candidates are typed `CUSTOM_SINGLE_NEG` and `MPSGRAPH` configurations.
- Generation is deterministic and exposes every valid complete candidate within the positive
  budget: current heuristic first; custom is pruned outside its exact singleton/index domain;
  budget one therefore preserves the safe current choice without tuning.
- Version-1 workload compatibility covers all mutable semantic, topology, type, Shape, layout,
  constant, boundary, exact-policy, candidate-schema, ABI-schema, and live-target/session facts.
  Canonical topology uses structural positions rather than `NodeId`, `ValueId`, or partition object
  identity, so equal valid workloads in one context compare equally across occurrences. Every
  independently variable valid fact produces a distinct compatibility; an invalid semantic change
  fails generation, and operation family alone is never a key.
- The Metal decision encoding is bounded, canonical, contains no executable/native handle, and
  returns an empty incompatible result for wrong magic/version/scope,
  malformed/truncated/trailing/corrupt bytes, changed workload or session target, and unknown
  candidates. The task does not adapt these bytes to the outer workload cache; its artifact
  version, checksum, atomic replacement, objective, and sampling remain owned by `tools/tuning`
  and unchanged.
- A present decision is never trusted directly: fresh analysis regenerates the batch and accepts
  it only when schema, workload, session target, and candidate identity match. A stale/foreign
  present decision fails closed; absence uses the existing heuristic without cache or measurement.
- The authenticated route is fixed before declarations. Custom declares feed and target buffers
  only; MPSGraph declares the same buffers plus exactly one address workspace. Finalization still
  cannot reselect, retry another route, or introduce a requirement.
- The existing Prepare marker roles and handoff are sufficient opaque transport types, but no
  shared Prepare or tuning consumer is added. No private Metal field, generic map, reflection,
  string dispatch, registry, new public type, shared source change, or production dependency leaks
  outward.
- Candidate generation, encoding, and authentication are cold, immutable, thread-safe, and free
  of native allocation. Runtime hot execution, persistent-resource ownership, concurrent-run
  isolation, close behavior, and one typed native call per prepared invocation remain unchanged.
- Existing fake-native execution and capability tests pass; Javadocs explain every changed input,
  result, constraint, ownership rule, and failure mode.
- Stop and return to planning before editing if truthful compatibility needs a shared contract,
  public API, `tools/tuning` adapter or outer compatibility type, cross-session target promise,
  native ABI change, new capability/route, cache-file implementation, shared/module/build change,
  or more than the exact allowlist.

## Validation

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

The focused tests must cover both candidate domains, budget prefixes, independent signature
changes, structurally equal workloads with different graph identities, defensive bytes, codec
corruption/version/session rejection, fresh decision authentication, exact declarations, no
native allocation during generation, and concurrent cold use. The ordinary module suite covers
existing native seams and hot-path/lifecycle regressions.
The current conformance harness is relevant only to prove unchanged capability/partitioning.
No architecture test applies because production dependencies, module boundaries, and shared
source remain unchanged; no integration harness supports public Metal composition.

Repository-wide validation is deferred to continuous integration or a later Metal capability
checkpoint: this is one backend-local cold-preparation change with no shared/build/dependency or
public-composition change. The independent review reuses successful executable evidence unless it
changes Java behavior or identifies a concrete stale-evidence risk.

## Dependencies and follow-up

- Metal 0002–0003, Prepare 0004/0006, Runtime 0016, and tuning 0001 are Complete and provide the
  two routes, opaque handoff types, prepared-resource transaction, Runtime owner, and separate
  outer-cache rules. The outer tuning workflow does not consume Metal 0004 output.
- Cross-session reuse requires a separately authorized stable Metal device/library fingerprint;
  public Metal/Engine and `tools/tuning` composition remains a later task and cannot widen 0004.
  The current generic workload tuner loads its explicit cache file before applying per-workload
  `SESSION` reuse, so later Metal composition must separately review that outer behavior rather
  than treating 0004's session codec as a no-I/O end-to-end workflow.

## Documentation and review impact

- Update the Metal backend guide, affected implementation Javadocs, package Javadoc, and the
  existing glossary term after a targeted search; do not present session compatibility as a
  persistent cache, current tuning integration, or performance claim.
- A clean implementation context and an independent targeted Class C documentation/review context
  are mandatory because route decisions, native-resource declarations, tuning compatibility, and
  a durable backend/Prepare boundary change.

## Result

Implemented the package-private Metal NEG candidate and session-authentication foundation. Metal
now generates stable budget prefixes over typed `CUSTOM_SINGLE_NEG` and `MPSGRAPH` candidates,
uses structural version-one workload fingerprints plus one fresh context nonce, constructs opaque
absent/present Prepare handoffs, and freshly authenticates every supplied selection before fixing
the route and exact declarations. The bounded checksummed Metal codec rejects malformed, corrupt,
stale, foreign-session, and unknown-candidate decisions. It performs no native allocation, file
I/O, measurement, outer workload-cache adaptation, or Runtime work.

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

Validation completed successfully:

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

The focused tests cover both domains, budget prefixes, structural identity independence,
independent compatibility changes, invalid semantics, defensive codec failures, session/workload
rejection, fresh authentication, exact declarations/finalization, no native allocation, and
concurrent cold use. The ordinary module suite retains fake-native execution and lifecycle
coverage; the conformance test confirms unchanged capability and partitioning.

Documentation/Javadoc impact: implementation and package Javadocs describe schemas, ownership,
session scope, failure behavior, and cache/Runtime exclusions. The backend guide and glossary now
explain typed candidates, opaque handoff, fresh authentication, safe fallback, codec boundaries,
and the absence of current persistent-cache or tools integration. No architecture, public API,
shared source, native ABI, build/dependency, capability, or Runtime documentation changed.

The mandatory independent targeted Class C documentation/code review inspected the exact
16-path diff, affected implementation and tests, generated test reports, and the named
architecture headings and decision records. It found and repaired one candidate-completeness
defect: generation had treated an already selected plan route as structural eligibility, so an
eligible singleton plan carrying an authenticated MPSGraph selection could omit the still-valid
custom candidate if regenerated. Generation now derives custom validity only from structural
workload facts, and a focused regression covers regeneration from that selected plan. The review
also repaired stale post-completion wording in this task, the Metal master plan, and the roadmap.
After the repair,
`./gradlew :backends:metal:test --tests '*MetalNegRouteCandidateGeneratorTest' --tests '*MetalNegPreparedExecutionTest' :backends:metal:javadoc`
passed 34 focused tests (32 passed and two native opt-in tests skipped) and regenerated Javadoc
without warnings or errors. The earlier successful full Metal and conformance results remain
relevant because the repair changes only cold candidate regeneration and the focused regression
directly exercises that behavior.

Limitations and follow-up: cross-session reuse still requires a separately authorized stable
Metal device/library fingerprint. Public Metal/Engine and `tools/tuning` composition, persistent
cache I/O, measurement, and winner selection remain outside this task. No unresolved
implementation issue remains.

Status: Complete
