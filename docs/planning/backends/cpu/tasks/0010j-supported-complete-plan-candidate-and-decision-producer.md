# Task 0010J: Supported Complete-Plan Candidate and Decision Producer

## Status

Ready

## Goal

Add one bounded CPU-owned producer/consumer collaboration for the already retained CPU 0008D and
0008E complete alternatives:

```text
one supported CompileArtifacts value with exactly one non-empty maximal CPU partition
  + the exact selected Phase-1 local-route decision, when the current workload has one
  -> fresh CPU legality and candidate analysis
  -> one opaque complete-plan candidate batch
     (legal fusion/split topology x retained direct/single/disjoint-pair representation)
  -> one authenticated CPU-owned selected complete-plan decision
  -> fresh shared assignment, CPU finalization, source-resource contribution, and schedule
  -> one complete immutable PreparedExecution recipe
```

This task is the smallest truthful owner prerequisite for later tools/tuning 0002. It produces and
freshly prepares complete CPU alternatives; it does not execute them, measure them, compare elapsed
time, apply user fallback policy, read or write a cache, or compose an Engine workflow.

## Current-state audit

- The supported lifecycle accepts exactly one non-empty maximal CPU partition. For that domain,
  `CompileArtifacts` contains the immutable compiled graph, the sole `PlannedPartition`, logical
  memory, publication plan, constants, and diagnostics. `CpuBackendIntegration` supplies CPU
  analysis/finalization, source-only constant resources, and the sole complete schedule assembler.
- CPU 0008D already enumerates at most 64 complete legal topology candidates from the canonical
  split using existing contraction legality. Its `CpuFusionDecision.LegalCandidate` values carry
  graph-identity-free topology, structural, resource, score, rank, canonical-split, and 0008B-
  baseline facts. Legality rejections and incomplete enumeration are already explicit.
- CPU 0008E already retains, per legal topology, the direct variant, eligible one-source copies,
  and eligible disjoint-two-source copies. It rejects a pair as `CO_CONSUMED_PAIR` when one
  represented instruction consumes both sources. Its immutable variant identities carry exact
  copy/reuse/layout/workspace/generated-specialization facts, and its existing finalizer and
  executable paths can realize those candidates. Ordinary preparation intentionally selects the
  0008D-selected direct form.
- CPU 0010E and 0010I already expose and authenticate the current Phase-1 local route decision for
  the exact/default FLOAT32/FLOAT64 bare-MATMUL portable/OpenBLAS batch. Candidate order,
  compatibility, decision codecs, fresh selected preparation, provider/thread ownership, and
  session-versus-persistent reuse are CPU-owned. Current public Engine composition can select at
  most one such occurrence, index 0 of partition 0 with weight 1.
- Prepare 0004 already provides sufficient method-free `BackendTuningCandidateBatch` and
  `BackendTuningDecision` roles plus `BackendPartitionTuningHandoff<C,D>`. Shared Prepare neither
  needs nor is allowed to enumerate or interpret CPU plan fields.
- The current preparer always runs 0008D/0008E selection internally and exposes only the selected
  analysis. No supported collaboration can enumerate the retained topology/representation
  variants, bind them to a Phase-1 result, or freshly prepare a specifically selected complete
  variant. Treating CPU 0010I local-route candidates as Phase-2 plans would repeat Phase-1 search.
- Tools/tuning 0002 is therefore correctly `Blocked` and has no task specification. CPU 0016 is a
  later cross-route local workload-cache generalization and is not this producer.

No architecture conflict was found. This task implements the already authorized concrete-backend
candidate ownership and opaque Prepare handoff without changing `ARCHITECTURE.md`.

## Scope

### Supported collaboration

- Add one retained supported root-package collaboration named
  `io.github.pho001.synaptik.backend.cpu.CpuCompletePlanTuning` and expose it through
  `CpuBackendIntegration.completePlanTuning()`.
- Mirror the narrow shape proven by `CpuLocalWorkloadTuning`: one public final collaboration with
  nested immutable opaque `CandidateBatch`, `Candidate`, `SelectedDecision`, `Compatibility`, and
  `CandidateIdentity` values plus a `ReuseScope` enum. Constructors are not public.
- Make the nested batch implement `BackendTuningCandidateBatch` and the nested decision implement
  `BackendTuningDecision`. Use the existing Prepare handoff unchanged.
- Retain exactly one collaboration for the owning `CpuBackendIntegration` lifetime. It owns no
  provider, worker, cache, file, measurement session, Runtime state, or independent close method.

### Complete-plan candidate set

- Generate candidates only from the existing complete 0008D `LegalCandidate` values and complete
  0008E `Variant` values. Reuse their identities, ranks, topology, legality, source eligibility,
  copy reuse, resource geometry, and hard ceilings; do not independently rediscover graph
  patterns, duplicate legality formulas, or reinterpret a rejection.
- A candidate is complete only when it fixes all CPU-owned choices for the current supported
  lifecycle: the exact legal computation-unit topology, zero/one/eligible-disjoint-two 0008E
  external-read materializations, the already selected Phase-1 route/configuration when one is
  eligible, resulting CPU preparation plan and exact declarations, deterministic assignment and
  schedule shape, CPU finalization, source-only constant contribution, representation creation,
  and execution/publication order. Preparing that candidate freshly creates the plan-local slots,
  finalized executable recipe, schedule, and immutable `PreparedExecution`.
- The batch order is stable: 0008D legal topology rank first, then direct, retained single-copy
  candidates in stable source order, then retained disjoint-pair candidates in lexicographic
  source order. Deduplicate only by the exact existing typed variant identity.
- Eligible zero-copy candidates include canonical split, the exact 0008B baseline when distinct,
  the 0008D ordinary selected topology, and every other retained legal fused/split topology.
  Eligible copied candidates are exactly the one-copy and disjoint-two-copy variants retained by
  0008E for those topologies.
- Exclude every 0008D legality rejection, every 0008E rejection including `CO_CONSUMED_PAIR`, an
  unsupported source/type/access/carrier/layout, more than two copies, a partial topology, a
  topology or representation not already retained, and any candidate that cannot preserve the
  supplied Phase-1 selection exactly.
- Return an empty handoff for a valid artifact that has fewer than two distinct complete eligible
  Phase-2 alternatives. Return an empty handoff, rather than a partial batch, when 0008D or 0008E
  reports incomplete/uncertain enumeration such that the producer cannot prove it exposed the
  complete bounded set. Invalid supported-lifecycle artifacts still fail as today.
- Candidate generation uses one versioned CPU-owned enumeration profile that enables the already
  implemented 0008E candidate path without authorizing ordinary promotion. The profile may use
  neutral diagnostic costs and the existing hard candidate/resource ceilings; those values are
  included in compatibility but are never treated as measurement or a ranking result.

### Phase-1 decision reuse

- `candidateHandoff` accepts the exact `CompileArtifacts` reference and an
  `Optional<CpuLocalWorkloadTuning.SelectedDecision>`.
- It freshly determines whether the current artifact has an eligible CPU 0010I Phase-1 handoff.
  If it does, an absent Phase-1 decision is rejected before complete-plan candidates escape. The
  supplied decision must belong to this exact integration, its batch must retain the same exact
  artifacts and partition association, and fresh CPU 0010E matching must accept the decision.
- If fresh analysis has no eligible Phase-1 handoff, absence is the authenticated no-local-choice
  state and complete-plan enumeration proceeds. A present Phase-1 decision in that state is
  rejected. CPU must not manufacture a portable-only Phase-1 batch.
- Every complete-plan candidate retains the same authenticated Phase-1 state. Preparation injects
  that decision into fresh analysis and never re-enumerates, reranks, substitutes, or measures
  local route parameters. A candidate for which the selected local decision is no longer eligible
  is rejected; CPU never silently falls back to the local heuristic inside selected preparation.
- Fresh CPU analysis may regenerate the owning 0010E batch solely to perform its existing exact
  selected-decision compatibility check. That validation batch is not exposed as Phase-2
  alternatives, is not timed or ranked again, and cannot change the supplied Phase-1 winner.
- Cross-integration, cross-artifact, cross-partition, cross-batch, stale, or mismatched Phase-1
  values fail before slot assignment, finalization, provider mutation, or recipe publication.
  Association authentication prevents accidental value mixing; it is not a cryptographic
  security claim.

### Explicit complete-plan selection

- Add one CPU-private immutable selected-plan input that contains the exact existing 0008E
  `VariantIdentity`, the complete-plan schema, and the Phase-1 selection fingerprint. It contains
  no timing, objective, cache path, physical resource, slot, executable, or Runtime state.
- Narrowly refactor the existing 0008D/0008E selectors so ordinary analysis remains byte-for-byte
  equivalent while explicit selected analysis finds one exact already-retained legal variant and
  records an exact explicit-selection reason. It must not rerun profitability ranking or create a
  new topology/representation.
- Explicit selection recomputes internally consistent final `CpuFusionDecision.Selection` and
  `CpuRepresentationDecision.Selection` facts for the chosen retained variant. Do not relabel a
  timing-selected plan as `PROFITABLE_FUSION`, `COPIED_PROFITABLE`, or another heuristic result.
- `prepareTrial(batch, candidate)` and `prepareSelected(batch, decision)` each repeat fresh
  authoritative analysis, validate the exact Phase-1 and complete-plan identities, derive exact
  declarations, invoke ordinary shared assignment, finalize all selected CPU units, contribute
  current source-only constants, assemble and validate the full schedule, and return a fresh
  `PreparedExecution`.
- Selected preparation never falls back. Any changed legality, candidate set, resource geometry,
  Phase-1 compatibility, provider qualification, schema, or target fact rejects the selection.

## Out of scope

- tools/tuning 0002 implementation or task specification; plan timing, warmups, samples, clocks,
  objectives, constraints, budgets, ranking, winner selection, evidence, or correctness oracles
- workload-cache or model-plan-cache lookup, parsing, mutation, persistence, atomic publication,
  eviction, migration, or inspection; prepared-executable serialization
- representative inputs, Runtime invocation, output comparison/materialization, result cleanup,
  fallback policy, Config translation, Engine orchestration, or a public user-facing tuning API
- Compiler graph alternatives, Planning ownership/partition alternatives, multiple partitions,
  mixed backends, zero-node/pass-through artifacts, or more than one Phase-1 occurrence
- any new fusion contraction, recognition, legality rule, pointwise source eligibility, copy form,
  generated algorithm, emitter branch, route, provider, numerical mode, relaxed mathematics,
  BFLOAT16 OpenBLAS, or vendor-peer route
- changing ordinary `CpuBackendIntegration.prepare`, `preparations`, or
  `CpuLocalWorkloadTuning`; ordinary preparation remains direct by default for 0008E and keeps the
  existing safe local-route heuristic
- widening Prepare marker roles or handoff, changing Runtime contracts, adding a generic backend
  facade, exposing CPU internals, or using raw types, `Object` payloads, maps, reflection, string
  dispatch, a registry, service locator, or Java serialization
- repurposing CPU 0016, changing module dependencies or Gradle, changing architecture/ADRs, or
  adding backend-conformance/integration/architecture behavior

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0008B decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
- [CPU 0008C recognition](0008c-typed-specialized-subgraph-and-epilogue-recognition.md)
- [CPU 0008D complete topology facts](0008d-bounded-fusion-profitability-and-typed-decision-facts.md)
- [CPU 0008E representation alternatives](0008e-bounded-multi-input-materialization-and-representation-reuse.md)
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008F MATMUL execution](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
- [CPU 0010E local candidates](0010e-float32-float64-openblas-tuning-candidates-and-compatible-decisions.md)
- [CPU 0010F lifecycle integration](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [CPU 0010I local tuning collaboration](0010i-supported-cpu-local-workload-tuning-composition-adapter.md)
- [Prepare 0004 opaque handoff](../../../modules/prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [Tuning 0001 local workflow](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Engine 0006A representative execution](../../../modules/engine/tasks/0006a-representative-tuning-execution-and-safe-fallback.md)
- [Engine 0007 optional composition](../../../modules/engine/tasks/0007-optional-model-autotuning-composition.md)
- [Config 0006A request facade](../../../modules/config/tasks/0006a-model-autotuning-request-configuration.md)

## Architecture constraints

- Planning continues to choose only CPU ownership. CPU alone owns complete CPU topology,
  representation, route, exact-resource, finalization, and candidate compatibility semantics.
- Shared Prepare transports the opaque typed values and performs ordinary assignment/validation;
  it does not inspect candidate fields. The existing method-free roles are sufficient. If
  implementation proves otherwise, stop and report the exact shared-contract gap instead of
  widening Prepare in this task.
- Phase 1 remains authoritative for local route/configuration choice. Phase 2 fixes surrounding
  complete-plan alternatives around that choice and must not repeat local search.
- Analysis remains deterministic from explicit immutable inputs and performs no measurement,
  cache I/O, execution, or policy fallback. Finalization cannot change the selected route or add
  an undeclared shared requirement.
- Runtime sees only a complete prepared recipe. No candidate, decision, graph, cache, tuning, or
  selection vocabulary enters Runtime or its hot path.
- Engine remains the composition root and later owns representative execution, cleanup, public
  request translation, fallback, and final publication. CPU must not depend on Engine or tuning.
- A selected decision can choose only an exact candidate regenerated in the current complete
  batch. It cannot grant legality, weaken semantics/determinism, or invent a resource.

## Exact semantics, dataflow, lifecycle, failure, and identity contracts

### Candidate and lifecycle boundary

The term **complete plan candidate** in this task is deliberately narrower than the architecture's
future multi-owner model-plan space. It is one complete executable alternative inside the current
supported CPU-only lifecycle:

```text
same CompileArtifacts
  -> same sole PlannedPartition owned by cpu
  -> one existing legal 0008D unit topology
  -> one existing 0008E zero/one/disjoint-two-copy representation
  -> same supplied Phase-1 local route/configuration decision, or proved no eligible Phase 1
  -> exact selected CpuPartitionPreparationPlan and complete resource requirements
  -> one fresh PreparedMemoryPlan and stable assignments
  -> freshly finalized CPU executable(s) and representation recipes
  -> source-only constant recipes plus complete execution/publication schedule
  -> one fresh immutable PreparedExecution
```

It is not one local kernel, one route candidate, a partial partition, a bare analysis plan, or a
serialized executable. Compile graph, Planning owner, partition boundary, logical memory, and
publication semantics are fixed across this first candidate batch.

### Canonical compatibility and identity

- Use a bounded, versioned CPU-owned canonical binary projection with fixed numeric tags and
  explicit length/count prefixes. Do not use enum names, class names, Java serialization,
  `toString`, native addresses, object identity, or map iteration order.
- Batch compatibility includes at minimum:
  - complete-plan compatibility schema and CPU candidate-schema versions;
  - CPU backend/device identity; portable generator/Class-File/Vector compatibility facts and
    exact selected species/strategy facts represented by the candidate set; applicable qualified
    provider/target/binary identity and reuse scope from Phase 1; and a per-integration nonce when
    no stable persistent target projection exists;
  - the sole-partition role and stable topological node count/connectivity using relative node,
    input-port, and output-port coordinates rather than `NodeId` or `ValueId` magnitudes;
  - exact boundary and output descriptors: data type, Shape, resolved layout offset/strides,
    stable external-input/source/publication roles, publication order and aliases, logical-memory
    graph-output/constant roles, and canonical scalar constant bits where applicable;
  - the ordered complete 0008D legal identities, ordered retained 0008E variant identities,
    exact declarations/workspace geometry, source-only constant-resource facts, and the fixed
    candidate-enumeration profile; and
  - the authenticated Phase-1 state: either the exact canonical selected local decision and its
    compatibility/scope, or the explicit freshly proved no-eligible-handoff tag.
- Reuse existing CPU IR structural keys, relative member positions, boundary positions, access
  plans, specializations, and resource facts. Do not create a second operation-attribute encoder
  when those typed CPU facts already carry the semantic distinction.
- A candidate identity is the complete batch-compatibility digest plus the exact existing 0008E
  `VariantIdentity` projection and Phase-1 fingerprint. It excludes graph-local numeric IDs,
  requirement IDs except normalized relative resource roles, assigned slots, generated class
  objects, loaders, provider handles, workers, physical storage, timings, and runs.
- Persistent reuse is allowed only when every included target and Phase-1 fact has a stable
  persistent projection. Otherwise compatibility and decisions are `SESSION` scoped and include
  the owning integration nonce. A stable candidate decision may be decoded only against a freshly
  generated batch with byte-identical compatibility and an exact candidate identity.
- Decision encoding contains magic, schema, scope, compatibility digest, candidate identity,
  Phase-1 fingerprint, bounded lengths, and an integrity checksum. It contains no executable or
  live object. Malformed, truncated, trailing, unsupported, corrupt, wrong-session, stale-target,
  stale-graph, stale-resource, stale-Phase-1, or unknown-candidate bytes return empty without side
  effects. Nulls remain programmer errors. This checksum detects accidental corruption; it is not
  a hostile-input cryptographic authorization boundary.

### Preparation and ownership

- Candidate batches, candidates, compatibility values, identities, and decisions are immutable,
  own no closeable resource, and borrow the exact integration/artifact association.
- Each trial/final preparation constructs a new analysis, new shared assignments and memory plan,
  newly finalized immutable executable recipe, new schedule, and new `PreparedExecution`. It
  reuses no earlier `PreparedExecution`, slot plan, mutable run state, or physical representation.
- Preparation executes no representative input. The later outer owner creates a fresh `RunState`
  per warmup/sample, validates publications, and closes each result and borrowed wrapper under the
  Engine 0006A contract.
- Current preparation acquires no run-owned physical resource. Generated-class reuse may use the
  existing CPU-owned compatible in-memory artifact cache, but the selected candidate and bytes do
  not change. Existing preparation failure cleanup applies; no partial recipe escapes.
- Prepared recipes borrow the owning integration's provider/coordinator lifetime and must not
  outlive it. Closing the integration prevents new candidate/codec/preparation operations;
  admitted-call and close behavior remains the existing CPU 0010F/0010I contract.

### Failure behavior

- Invalid lifecycle artifacts fail before exposure, as in ordinary CPU preparation.
- Valid but non-tunable complete-plan work returns an empty handoff.
- Missing required Phase-1 selection, an unexpected Phase-1 selection, or any association mismatch
  throws `IllegalArgumentException` before backend state changes.
- A stale/incompatible encoded complete-plan decision decodes to empty. A live decision associated
  with another batch/integration is rejected.
- Fresh selected preparation rejects any incompatibility and never substitutes ordinary heuristic
  preparation. Whether a later caller falls back is exclusively an Engine policy decision.

## Package impact

### Exact public API

Existing supported package:

```text
io.github.pho001.synaptik.backend.cpu/
  CpuCapabilityProvider
  CpuBackendIntegration
  CpuLocalWorkloadTuning
  CpuCompletePlanTuning
  package-info.java
```

Exact intended supported surface:

```java
public final class CpuCompletePlanTuning {
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(
                    CompileArtifacts artifacts,
                    Optional<CpuLocalWorkloadTuning.SelectedDecision> phaseOneDecision);
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

`CpuBackendIntegration` adds exactly:

```java
public CpuCompletePlanTuning completePlanTuning();
```

It returns the same retained instance while open. Public opaque values expose only defensive
compatibility/identity bytes, schema/scope accessors, association-safe equality needed for the
future tool adapter, and redacted diagnostic text. They expose no CPU internal plan, route,
topology, materialization, resource, provider, graph ID, slot, or executable accessor.

Existing internal packages changed:

- `internal.prepare` — explicit selected complete-plan input and fresh authoritative application;
- `internal.lowering` / `internal.ir` — reuse retained variants and record truthful explicit
  selection without changing ordinary selection;
- `internal.route.nativeblas.openblas` — retained composition delegates the supported producer,
  Phase-1 authentication, and complete fresh preparation.

No package, module dependency, JPMS export, Prepare method, Runtime API, or Engine API is added.

## Affected files

Expected CPU production/Javadoc paths:

- new `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCompletePlanTuning.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuLocalWorkloadTuning.java`,
  limited to package-private exact association/decision authentication for the new collaboration;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuFusionProfitabilitySelector.java` only if needed to apply an exact retained identity without reranking;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuFusionDecision.java`;
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuRepresentationDecision.java`.

The private complete-plan input/codec must be nested in an affected cohesive owner. No second new
top-level production type is allowed.

Expected focused CPU test paths:

- new `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuCompletePlanTuningPublicTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`;
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java` only if a new internal type is added.

Expected documentation and planning paths in the implementation change:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only if existing complete-plan candidate, candidate batch, and selected tuning
  decision entries cannot describe the result accurately;
- this task;
- `docs/planning/backends/cpu/master-plan.md`;
- `docs/planning/tools/tuning/master-plan.md`;
- `docs/planning/roadmap.md`.

Review without modification unless a concrete contradiction is found: `ARCHITECTURE.md`, focused
architecture pages and ADRs, Prepare source/tests/docs, tools/tuning 0001 source/tests, Engine,
Config, Runtime, Compiler, Planning, Backend Contract, build/Gradle, architecture tests, backend-
conformance tests, integration tests, generated emitters, and CPU 0011–0017 task rows.

## Maximum scope

- At most 11 CPU production/Javadoc paths, including exactly one new supported top-level type and
  no new internal top-level type.
- At most 5 focused CPU test paths, including exactly one new supported-surface test.
- At most 2 explanatory documentation paths and exactly 4 planning paths.
- At most 22 total paths. Fewer are preferred; no headroom may be used for another module,
  generated emitter, algorithm, route, shared contract, build file, or unrelated refactor.
- Preserve existing ceilings: 64 legal topologies, 256 topology attempts, 8 units, 64 topology
  boundary positions, 8 topology workspaces, 8 eligible representation sources, 36 non-direct
  copy choices per topology, 37 variants per topology, 2,368 variants, 2 copies per variant,
  16 consumer-use facts, 10 units/workspaces/artifacts including copies, 74 requirements, and
  2,753 combined decision facts.

If complete selection cannot be implemented by reusing the existing facts/finalizer inside this
ceiling, stop and propose the smallest CPU-owned follow-up. Do not create multiple detailed tasks
or weaken this task's completeness.

## Acceptance criteria

1. `CpuCompletePlanTuning` is the sole new supported top-level CPU type, is retained by
   `CpuBackendIntegration`, and has exactly the public collaboration shape specified above.
2. Its nested batch/decision adopt the existing method-free Prepare roles; no shared Prepare
   contract changes and no CPU internal type leaks through public signatures or Javadocs.
3. A complete batch contains every and only retained eligible 0008D/0008E variant in exact stable
   order, or no batch is exposed when completeness cannot be proved or fewer than two alternatives
   exist.
4. Direct, fused/split, one-copy, and eligible disjoint-two-copy cases are covered. Typed 0008D
   legality and 0008E representation rejections remain excluded and are never reinterpreted.
5. An eligible Phase-1 workload requires one exact authenticated selected local decision; a
   noneligible workload requires absence. Every plan candidate preserves that state without local
   candidate enumeration, ranking, timing, or heuristic substitution during Phase-2 preparation.
6. Cross-owner/batch/artifact/partition and stale Phase-1 or complete-plan decisions fail before
   shared assignment, finalization, provider mutation, or publication of a recipe.
7. Compatibility, candidate identity, and codec bytes are canonical, bounded, defensive,
   versioned, graph-ID-independent, target/graph/partition/layout/resource/Phase-1 complete, and
   session-scoped whenever persistent target compatibility is unavailable.
8. Codec corruption/incompatibility returns empty without side effects; exact compatible
   round-trip reconstructs a decision associated with the supplied fresh batch.
9. Trial and final selection each produce a fresh full `PreparedExecution` through ordinary
   assignment/finalization/schedule validation and select exactly the requested retained variant.
   No selected preparation falls back.
10. Ordinary `prepare`, ordinary 0008D profitability choice, 0008E direct-by-default selection,
    and CPU 0010I local tuning behavior remain unchanged.
11. No representative execution, timing, cache I/O, persistence, fallback policy, Engine/Runtime
    selection, new generated bytes, algorithm, route, dependency, or architecture change occurs.
12. Focused tests cover semantic set completeness, identity stability, Phase-1 presence/absence
    and mismatch, exact resources, fresh recipe identity, concurrency/close, codec failures, and
    unchanged ordinary preparation.
13. A separate clean documentation-focused context finalizes affected Javadocs, the CPU guide,
    glossary impact, and all four planning paths before the task becomes Complete.

## Tests / validation

Implementation-focused tests:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuCompletePlanTuningPublicTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRepresentationPlannerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuBackendCompositionTest
./gradlew :backends:cpu:test
```

The focused matrix must include real supported fixtures for one direct-only non-tunable case, a
vertical fused/direct pair, canonical/0008B split roles, one-copy, eligible disjoint-two-copy,
`CO_CONSUMED_PAIR`, no eligible Phase 1, required eligible Phase 1, portable and OpenBLAS Phase-1
selection, stale/mismatched decisions, source-only constants, and fresh selected preparation.

Manual/structural checks must prove the supported public signatures contain no `.internal` type,
candidate and compatibility bytes contain no graph-local ID magnitude or live-object text, and
ordinary and explicitly selected existing generated specialization/class identities are unchanged.
No generated-code performance rerun, direct-Java oracle rerun, or new benchmark is required: this
task changes cold candidate exposure/selection only and must not change an algorithm, emitter,
entry descriptor, structural key, generated schema, or generated class bytes. If any such byte
changes, stop and replan with the generated-code evidence required by `ARCHITECTURE.md`.

The clean documentation-focused context reuses the successful Java evidence unless it changes
executable behavior, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates changed Markdown links and same-file anchors, unique/canonical headings,
balanced fences, LF/final newlines, no trailing whitespace, exact path ceiling, empty staging,
Ready/Blocked ordering, and rendered Javadocs for ownership, nullability, lifetime, failures, and
defensive bytes.

Backend-conformance and integration tests are unnecessary because no backend semantic/execution
contract or public Engine behavior changes. Architecture tests are unnecessary because no module
edge or dependency rule changes. Repository-wide tests remain deferred to the later Phase-2
capability checkpoint or CI.

## Dependencies

- Complete CPU 0008B, 0008C, 0008D, 0008E, 0008E1, and 0008F.
- Complete CPU 0010E, 0010F, and 0010I.
- Complete Prepare 0004 and tools/tuning 0001.
- Complete Engine 0006A and 0007 establish the outer Phase-1 lifecycle evidence but are not CPU
  dependencies and are not changed here.
- Complete Config 0006A establishes current public Phase-1 request vocabulary but is not consumed
  or changed here.

This task is an explicit prerequisite interleave after Complete CPU 0010I and before unrelated or
externally blocked vendor-peer rows 0011–0015. It does not reorder or replace them, and it does not
repurpose broad future CPU 0016.

## Follow-up tasks

- After CPU 0010J is Complete, re-audit tools/tuning 0002 against the implemented batch,
  compatibility, codec, and fresh-preparation evidence. Keep 0002 `Blocked` and do not create its
  detailed specification until that re-audit resolves the Phase-2 budget, correctness oracle,
  model-plan record/cache, and tools-only scope.
- A later separate Engine task must compose representative complete-plan execution, result
  correctness checking, cleanup, Config Phase-2 translation, selected final preparation, and user
  fallback policy. None of that follows automatically from this CPU producer.
- CPU 0016 remains the distinct later cross-route local workload-cache integration row for
  implemented vendor peers. CPU 0010J must not absorb it.

## Architecture impact

Expected architecture impact: None. The root contract already permits concrete backends to
generate complete valid backend-specific candidates, Prepare to transport them opaquely, tuning
to measure them before Runtime, and CPU to finalize selected work. No dependency direction,
module boundary, lifecycle order, ADR, or architecture test changes.

### Documentation, Javadoc, and glossary impact

Documentation impact: update `docs/backend-guide/cpu-backend.md` to explain the supported
complete-plan producer, exact current one-partition meaning of complete, Phase-1 reuse, candidate-
only materializations, fresh selected preparation, and the no-measurement/no-cache/no-Engine/no-
Runtime-selection boundaries.

Javadoc impact: every added or changed public/internal type and method must document semantics,
inputs, nullability, association/lifetime, immutable ownership, defensive copies, return meaning,
and expected failures. Javadoc must distinguish association authentication from security and
trial preparation from trial execution.

Glossary impact: independently review the existing complete-plan candidate, candidate batch,
selected tuning decision, CPU portable preparation plan, and materialization entries. Update only
entries made incomplete by the implemented supported producer; otherwise record a reasoned
no-change conclusion. Do not invent a synonym for model-plan cache or claim that it exists.

No-change review is required for architecture pages/ADRs, public Tensor/Compile/Training APIs,
Prepare/Runtime APIs, Config, Engine user docs, provider docs, generated-code docs, other backend
guides, and conformance/integration documentation unless implementation reveals a concrete
contradiction.

## Local decisions

- Identifier `0010J` is selected because it is the next coherent CPU integration increment after
  Complete 0010I and before unrelated vendor-peer 0011–0015 work.
- One supported `CpuCompletePlanTuning` collaboration parallels `CpuLocalWorkloadTuning`; a broad
  facade or shared Prepare API would expose the wrong owner.
- Phase-1 absence is proved by fresh analysis, not represented by a caller-created token.
- A batch with fewer than two eligible complete alternatives is absent because there is nothing
  for Phase 2 to compare.
- Complete candidate identity uses stable relative CPU facts and a canonical compile projection,
  never raw graph-local ID magnitudes.
- Session scope is mandatory whenever every target fact cannot be projected persistently. This is
  safer than treating a caller/model label as target compatibility.
- Ordinary heuristic preparation remains unchanged and direct by default. Only an explicit exact
  complete-plan decision may select a candidate-only copied form.

## Known limitations

- The first producer covers only the current one-nonempty-CPU-partition lifecycle and at most one
  current Phase-1 local workload. It does not produce Compiler graph or Planning ownership choices.
- A portable-only target may remain session-scoped until a later owner establishes a stable
  cross-session CPU target fingerprint. This limits persistent model-plan reuse but not fresh
  Phase-2 measurement or correctness.
- Model-plan persistence, Phase-2 budgets, numerical equivalence checking, and Engine composition
  remain unresolved follow-ups and keep tools/tuning 0002 Blocked after this specification.
- Existing 0008D/0008E hard ceilings and unsupported materialization types/forms remain unchanged.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0010J. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md,
and every architecture, predecessor, API, source, test, and planning file it identifies as
required. Inspect the existing uncommitted planning diff before editing.

Implement exactly the Ready specification. Preserve ordinary heuristic preparation and the
existing Phase-1 collaboration. Stop and report any architecture/shared-contract conflict or need
to exceed the maximum scope; do not widen Prepare, implement tuning/Engine behavior, or invent a
new generated form.

After executable CPU validation succeeds, hand the same worktree and recorded evidence to a
separate clean documentation-focused agent/thread. That pass must follow
docs/developer-guide/documentation-rules.md, finalize Javadocs, the CPU guide, glossary impact,
and planning evidence, and run the specified documentation validation without repeating successful
Java tests unless executable behavior changes.

Update this task's evidence, notes, completion summary, and status only after every acceptance
criterion and the documentation pass succeed.
```

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
