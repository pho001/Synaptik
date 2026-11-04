# Task 0008A: Representative Complete-Plan Correctness [Oracle](../../../../glossary.md#term-test-oracle)

## Status

Complete

## Goal

Add the smallest [Engine](../../../../glossary.md#term-engine)-owned internal correctness [oracle](../../../../glossary.md#term-test-oracle) required before generic Phase-2
complete-plan tuning can be specified truthfully. The oracle executes freshly prepared complete
recipes against one already-bound representative input set, [captures](../../../../glossary.md#term-capture-meanings) every ordered [publication](../../../../glossary.md#term-publication)
as [detached](../../../../glossary.md#detached-model-results) [canonical](../../../../glossary.md#term-canonical-form) [host](../../../../glossary.md#term-host) bytes, owns the first candidate's reference output, and reports exact
match or mismatch for each later candidate without exposing [Tensor](../../../../glossary.md#tensor) semantics, [backend](../../../../glossary.md#backend)
representations, or output bytes to `tools/tuning`.

This is an upstream prerequisite, not Phase-2 tuning itself. It adds no timing, [ranking](../../../../glossary.md#term-ranking), [cache](../../../../glossary.md#term-cache),
candidate interpretation, public request, selected preparation, or fallback policy. After this
task is Complete, a fresh tools/tuning 0002 specification may define the generic [opaque](../../../../glossary.md#term-opaque)
candidate, budget, correctness-[callback](../../../../glossary.md#term-callback), evidence, and compact [model](../../../../glossary.md#term-model)-plan-record workflow.

```text
exact compiled graph + one admitted representative-input session
  -> preflight every publication descriptor and one aggregate canonical-byte limit
  -> execute the first freshly prepared complete recipe in fresh Runtime state
  -> copy every publication occurrence in order and close the result
  -> retain detached bytes as the session-owned reference
  -> execute each later freshly prepared complete recipe in fresh Runtime state
  -> copy all occurrences, close the result, and compare exact canonical bytes
  -> report MATCH or the first publication/byte mismatch without exporting payloads
```

## Current-state audit

- Complete [Engine](../../../../glossary.md#term-engine) 0006A added `RepresentativeExecutionSession`. It validates and borrows the
  representative inputs once, executes each supplied [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) in a fresh Runtime
  state, validates the publication count, and closes the result, but its contract deliberately
  forbids inspecting or materializing publication payloads.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Complete Engine 0007 composes only Phase-1 local-workload tuning. Its measurement action freshly
  prepares one CPU candidate, calls the representative session's completion-only execution, and
  leaves numerical comparison out of scope.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope))</span>
- `CompiledGraph` already retains the authoritative ordered `PublicationSpec` values and their
  exact final `TensorDescriptor` values. [Runtime](../../../../glossary.md#term-runtime) [`RunResult`](../../../../glossary.md#run-result--runresult) already lends each ordered
  publication representation by dense result index while its lease is open.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease))</span>
- `EngineBackendComposition.copyToCanonicalHostBytes(...)` already supplies the backend-neutral
  Engine seam used by host materialization. The CPU implementation preserves represented bits in
  canonical row-major big-endian bytes, including floating-point NaN payloads and signed zero,
  while validating descriptors, geometry, capacity, carrier, and canonical BOOL values.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [represented bits glossary](../../../../glossary.md#term-represented-bits); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [floating-point glossary](../../../../glossary.md#term-floating-point); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [descriptors glossary](../../../../glossary.md#term-descriptor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Current ordinary Engine materialization already preflights fully static Shape, resolved layout,
  checked element/byte arithmetic, the JVM array ceiling, and an aggregate caller byte limit.
  That logic is private to <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>`AdvancedEngine`; the representative session must reuse or factor the
  same Engine-owned validation rather than create different byte-accounting semantics.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Complete CPU 0010J supplies a session-scoped opaque batch of at least two retained complete
  one-partition CPU topology/representation alternatives around the exact authenticated Phase-1
  selection. It freshly prepares a requested trial or selected recipe, but does not execute,
  compare, time, cache, or apply fallback policy.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [cache glossary](../../../../glossary.md#term-cache))</span>
- Prepare's public tuning roles are intentionally method-free opaque transport. Runtime exposes
  result-indexed representations and cleanup only; neither layer owns Tensor-value comparison.
  Therefore the missing executable correctness contract belongs to Engine, which already owns
  logical publications, representative values, materialization, and result cleanup.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Tensor glossary](../../../../glossary.md#tensor))</span>

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private [immutable](../../../../glossary.md#term-immutable) correctness [model](../../../../glossary.md#term-model) in `io.github.pho001.synaptik.engine` with:
  - an [opaque](../../../../glossary.md#term-opaque) reference value associated by [identity](../../../../glossary.md#term-identity-meanings) with exactly one live
    `RepresentativeExecutionSession`;
  - an exact two-valued comparison result, `MATCH` or `MISMATCH`; and
  - no public constructor, byte accessor, [Tensor](../../../../glossary.md#tensor) [metadata](../../../../glossary.md#term-metadata), [backend](../../../../glossary.md#backend) value, or diagnostic [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication).
- Extend `RepresentativeExecutionSession` with two package-private operations:
  - `captureCorrectnessReference(PreparedExecution, long)` captures the first complete recipe's
    reference output under the supplied aggregate canonical-byte limit; and<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
  - `compareCorrectness(Reference, PreparedExecution)` executes and compares one later complete
    recipe with that exact reference and its already validated byte-count contract.
- Snapshot the compiled graph's complete ordered publication specifications during session
  construction. Do not accept publication descriptors, output meanings, or expected bytes from<span data-glossary-navigation="true"> ([Snapshot glossary](../../../../glossary.md#term-snapshot); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [publication glossary](../../../../glossary.md#term-publication); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
  `tools/tuning`, a backend, or a new [callback](../../../../glossary.md#term-callback).
- Before the first correctness execution, validate all publication descriptors and calculate all
  canonical byte counts with checked arithmetic. Require at least one publication, fully static
  Shapes, resolved layouts, a non-negative aggregate byte limit, each payload within the JVM<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
  `byte[]` ceiling, and the checked aggregate not greater than the supplied limit.
- Execute each correctness [run](../../../../glossary.md#term-run) through the existing [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) and the session's
  already ordered borrowed inputs. Every call creates a fresh isolated Runtime <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`RunState`](../../../../glossary.md#run-state--runstate).
- Validate the Runtime result count before any payload access. Copy every publication occurrence
  independently in dense compiled order through<span data-glossary-navigation="true"> ([publication occurrence glossary](../../../../glossary.md#publication-occurrence))</span>
  `EngineBackendComposition.copyToCanonicalHostBytes(...)`; intentional aliases remain separate
  observable occurrences.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Complete every copy while the Runtime result lease is open, then close the result before
  returning a reference or comparison. Reference bytes are detached Engine-owned snapshots and
  contain no Runtime or backend resource.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [detached glossary](../../../../glossary.md#detached-model-results); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Compare publication count, canonical byte length, and byte content in encounter order. The
  initial policy is exact represented-bit equality only. This deliberately distinguishes signed
  zero and NaN payloads and supplies no tolerance, normalization, data-type-specific rule, or
  relaxed-math permission.<span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [tolerance glossary](../../../../glossary.md#term-tolerance); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Make the first successfully captured candidate the sole reference for the session. Reject a
  reference from another session, a second reference capture, comparison before reference capture,
  comparison after cleanup, and any action after poisoning.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Treat `MISMATCH` as a clean typed outcome only after execution, all copies, and result cleanup
  succeed. It does not poison the session by itself. A later Phase-2 caller must stop the tuning
  transaction, publish no partial evidence or cache state, clean the session, and route the
  failure through Engine's public safe-fallback policy.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction); [cache glossary](../../../../glossary.md#term-cache))</span>
- Preserve the existing failure protocol. Execution, count validation, materialization, or result
  cleanup failure keeps its exact unchecked primary where applicable, suppresses distinct cleanup
  failures deterministically, poisons the session, cleans borrowed inputs once, releases Engine
  admission, and forbids selected preparation and fallback from that session.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [admission glossary](../../../../glossary.md#term-admission))</span>
- Add focused native-free tests for preflight, first-reference ownership, exact match, content and
  length mismatch, multiple publications, aliased occurrences, zero-length
  payloads, signed zero and NaN-payload distinction, detached reference lifetime, wrong-session
  rejection, result/input cleanup, failure identity/suppression, poisoning, and close races.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Finalize affected Javadocs and focused explanatory documentation in the mandatory separate clean
  documentation context during implementation.

## Out of [scope](../../../../glossary.md#term-scope)

- any public [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api) or change to `ModelAutotuningRequest`, `ModelAutotuningPreparation`,
  `Engine`, `CompiledGraph`, public [`RunResult`](../../../../glossary.md#run-result--runresult), or [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue)
- tools/tuning 0002 types, algorithms, dependencies, timing, ranking, deterministic winner
  selection, budgets, evidence, cache parsing/publication, or model-plan records<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [publication glossary](../../../../glossary.md#term-publication); [model glossary](../../../../glossary.md#term-model))</span>
- CPU candidate enumeration, compatibility/identity/decision codecs, selected preparation, Phase-1
  ranking, or interpretation of topology, representation, route, provider, or resource fields<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology))</span>
- changing [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), Runtime result/publication contracts, Prepare opaque handoff roles,
  or serializing an executable, result, representation, Tensor, or prepared recipe<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff); [Tensor glossary](../../../../glossary.md#tensor))</span>
- tolerance, relative/absolute error, ULP comparison, NaN normalization, stochastic comparison,
  relaxed mathematics, or caller-defined numerical callbacks<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [absolute error glossary](../../../../glossary.md#term-absolute-error); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp); [NaN glossary](../../../../glossary.md#term-nan); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- making a [backend](../../../../glossary.md#backend) [target](../../../../glossary.md#term-target) [fingerprint](../../../../glossary.md#term-fingerprint) persistent; [CPU](../../../../glossary.md#term-cpu) 0010J remains honestly `SESSION`
- public safe-fallback policy, fallback classification, final selected preparation, or promotion of
  a correctness or timed trial recipe into production
- timing a correctness run, combining correctness with a warmup/sample, or changing Phase-1
  measurement behavior<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Compiler graph alternatives, Planning ownership alternatives, generated-code tuning,
  multi-partition or mixed-backend candidates, or a public Phase-2 request<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Engine, Runtime, concrete
  backends, performance evidence and optimization tooling, and dependency rules<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)
- [Engine](../../../../glossary.md#term-engine) [0004](0004-explicit-host-materialization-boundary.md),
  [0006A](0006a-representative-tuning-execution-and-safe-fallback.md), and
  [0007](0007-optional-model-autotuning-composition.md)
- [CPU 0010J](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [tools/tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)

## Architecture constraints

- Engine owns representative logical input binding, publication meaning, result materialization,
  cleanup coordination, and public fallback policy. This task stays package-private in Engine.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- Runtime remains unaware of tuning and correctness policy. It executes one immutable recipe at a
  time and lends ordered representations only through the existing result lease.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [lease glossary](../../../../glossary.md#term-lease))</span>
- Concrete backends continue to own physical copy/encoding and complete-plan candidates. Engine
  calls the existing composition copy seam without importing CPU internals.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `tools/tuning` must later see only an opaque reference collaboration and a typed match/mismatch
  result supplied by its caller. It must not receive Tensor descriptors, publication bytes,
  Runtime representations, or an Engine type in its own API.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Correctness executions are separate from warmups and timed samples. The later tools task owns
  invocation ordering and arithmetic; Engine owns the semantics and cleanup of each requested
  correctness action.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Exact byte equality is a conservative current exact/default oracle for the first producer. Any
  tolerance policy requires a separate stable numerical contract and is not inferred from data
  type, objective, provider, target, or observed results.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [producer glossary](../../../../glossary.md#term-producer); [tolerance glossary](../../../../glossary.md#term-tolerance); [data type glossary](../../../../glossary.md#data-type--datatype); [objective glossary](../../../../glossary.md#term-objective); [target glossary](../../../../glossary.md#term-target))</span>
- A mismatch never becomes a selected result. It is a recoverable tuning failure only after
  successful result and representative-input cleanup; the existing Engine fallback owner makes
  the later public policy decision.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- No [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), executable state, Runtime result, representation, or captured output may
  enter a persistent artifact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- No [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) or authoritative architecture rule changes in this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.engine` — owns the representative session, compiled publication
  descriptors, composition copy seam, cleanup, and the new package-private correctness values.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.runtime.execution` and
  `io.github.pho001.synaptik.runtime.run` — existing immutable recipe, runner, and leased result
  contracts only; no source changes.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>

Packages added or changed:

- No package is added.
- `io.github.pho001.synaptik.engine` gains one package-private helper and focused tests in the
  mirrored test package.

Type placement:

- `io.github.pho001.synaptik.engine.RepresentativePlanCorrectness` — package-private immutable
  opaque reference and two-valued comparison model colocated with the sole Engine-owned consumer;
  it is not a public generic tuning contract.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- `io.github.pho001.synaptik.engine.RepresentativeExecutionSession` — remains the owner of
  representative bindings, Runtime execution, result materialization/cleanup, poisoning, and
  reference association.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [bindings glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime); [materialization glossary](../../../../glossary.md#materialization))</span>

## Affected files

Expected production and test paths:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativePlanCorrectness.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java` only to factor
  the existing canonical-byte preflight into one package-private Engine-owned implementation if
  reuse cannot remain private without duplication<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [preflight glossary](../../../../glossary.md#term-preflight); [Engine glossary](../../../../glossary.md#term-engine))</span>
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths during implementation:

- `docs/architecture/performance-evidence-and-tuning.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- this task specification
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

No build file, [dependency](../../../../glossary.md#term-dependency), [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [CPU](../../../../glossary.md#term-cpu), Config, [Compiler](../../../../glossary.md#term-compiler),
[Model](../../../../glossary.md#term-model), or tools/tuning source path is expected to change.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- 3 [Engine](../../../../glossary.md#term-engine) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, including exactly one new package-private top-level type;
- 1 focused Engine test path;
- 2 focused explanatory architecture-documentation paths; and
- exactly 4 planning paths: this task, the Engine master plan, the tuning master plan, and the
  roadmap.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

The total maximum is 10 paths. If another source package, public type, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), numeric
policy, or more than 10 paths are required, stop and propose a follow-up task.

## Exact semantics, [lifecycle](../../../../glossary.md#lifecycle), failure, and [identity](../../../../glossary.md#term-identity-meanings)

### Reference and comparison semantics

- The first successful reference capture executes exactly once and owns the baseline. There is no
  separately prepared heuristic baseline and no backend-provided expected output.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [heuristic glossary](../../../../glossary.md#term-heuristic); [backend glossary](../../../../glossary.md#backend))</span>
- The reference contains one detached canonical byte sequence per compiled publication occurrence
  in exact dense order. It retains no Tensor, storage, representation, <span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [canonical glossary](../../../../glossary.md#term-canonical-form); [publication occurrence glossary](../../../../glossary.md#publication-occurrence); [Tensor glossary](../../../../glossary.md#tensor))</span>[`RunResult`](../../../../glossary.md#run-result--runresult),
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), or executable object.
- A comparison executes the supplied recipe exactly once and evaluates every occurrence. It
  returns <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`MATCH` only when counts, lengths, and every byte match the reference.
- `MISMATCH` carries no payload. Engine may use internal local indexes while comparing, but neither
  the reference nor result exports publication metadata or bytes. The later generic tuning caller
  needs only the two-valued result to stop fail-closed.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication); [metadata glossary](../../../../glossary.md#term-metadata); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Zero-element publication payloads are valid and compare as empty byte sequences. A compiled
  graph with no publications is rejected before execution because it provides no observable
  correctness reference.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- Exact represented-byte equality is policy version 1. Floating signed zeros and NaN payloads are
  intentionally distinct. No value decoding occurs.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>

### [Preflight](../../../../glossary.md#term-preflight) and bounds

- The caller supplies one non-negative maximum aggregate canonical byte count per correctness
  execution. The reference stores the validated publication byte counts, and comparisons require
  the same session/reference contract rather than recalculating an alternative policy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [publication glossary](../../../../glossary.md#term-publication))</span>
- Shape, layout, element count, data-type width, per-array ceiling, aggregate addition, and limit
  validation complete before the Runtime runner is invoked. Overflow fails closed.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element); [validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime); [Overflow glossary](../../../../glossary.md#term-overflow))</span>
- A preflight failure executes no recipe, captures no reference, produces no comparison, changes
  no cache/evidence state, and leaves the session eligible for orderly cleanup.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>

### [Lifecycle](../../../../glossary.md#lifecycle) and cleanup

- The reference is associated with the exact session by identity and is usable only while that
  session remains open and unpoisoned. Detached bytes survive result cleanup internally but do not
  grant use after representative-session cleanup.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Detached glossary](../../../../glossary.md#detached-model-results))</span>
- A successful capture/comparison closes the complete Runtime result before returning. No trial
  result or run-owned resource survives the call.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run))</span>
- A clean mismatch does not poison resources; the later caller may only clean the session and
  resolve the mismatch through its tuning/fallback boundary. It must not continue timing.
- Execution, result-count, physical copy, canonical validation, allocation, or result-cleanup
  failure poisons the session exactly like the existing representative execution failure path.
  Input cleanup is attempted once in reverse order and Engine admission is released.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission))</span>
- If copying fails and result cleanup also fails, copying remains primary and cleanup is
  suppressed once. If result cleanup alone fails, that cleanup failure is primary. Existing
  distinct-suppression and fatal-error behavior remains unchanged.

### Later Phase-2 budget contract

This task does not implement the budget, but its one-execution-per-correctness-action contract
fixes the arithmetic that tools/tuning 0002 must use. For `N` [cache](../../../../glossary.md#term-cache)-miss complete candidates,
`W` warmups, and `S` timed [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters):

```text
correctness executions = N
timed/warmup executions = N * (W + S)
maximum total executions required = N * (1 + W + S)
```

The first correctness execution creates the reference and each of the remaining `N - 1` compares
against it. All `N` correctness executions must match and clean up successfully before the first
[warmup](../../../../glossary.md#term-warmup) or timed execution begins. The later tool must validate a positive candidate ceiling,
non-negative `W`, positive odd `S`, a positive maximum-total-executions ceiling, and all checked
additions/multiplications before the first correctness execution. A compatible [cache hit](../../../../glossary.md#cache-and-memoization) performs
zero correctness, warmup, or timed executions and proceeds only through [backend](../../../../glossary.md#backend)-compatible
decision decoding and fresh selected preparation. Any [preflight](../../../../glossary.md#term-preflight) failure publishes no partial
evidence or cache state.

## Acceptance criteria

1. `RepresentativePlanCorrectness` is package-private, immutable, exposes no output bytes or
   Tensor/backend/Runtime value, and has exactly one opaque reference value plus the two-valued<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [opaque glossary](../../../../glossary.md#term-opaque))</span>
   `MATCH`/`MISMATCH` result required by this task.
2. `RepresentativeExecutionSession` snapshots the exact compiled publication specifications and
   supports one reference capture followed by zero or more same-session comparisons while open.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
3. Complete static/resolved/JVM-array/aggregate-byte preflight finishes before execution, rejects
   an empty publication boundary, uses checked arithmetic, and shares the ordinary Engine host-
   snapshot byte-accounting semantics.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [preflight glossary](../../../../glossary.md#term-preflight); [Engine glossary](../../../../glossary.md#term-engine); [host glossary](../../../../glossary.md#term-host); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
4. Each capture/comparison uses a fresh Runtime run, validates result count, copies every ordered
   publication occurrence independently through the existing composition seam, and closes the
   result before return.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [publication occurrence glossary](../../../../glossary.md#publication-occurrence))</span>
5. Exact [canonical](../../../../glossary.md#term-canonical-form) bytes determine `MATCH`; signed zero, NaN payload, length, and occurrence-order
   differences are tested without decoding Tensor values.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
6. A reference cannot be reused across sessions, recaptured, used before capture, or used after
   cleanup/poisoning. No output byte array escapes the Engine package-private model.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
7. A clean mismatch returns typed metadata only after successful cleanup, does not poison the
   session, and cannot itself authorize continued measurement, selection, or fallback.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
8. Execution, copying, validation-after-run, allocation, and cleanup failures preserve the
   existing exact-primary and deterministic distinct-suppression rules, poison the session, clean
   owned wrappers once, release admission, and forbid final preparation/fallback.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [release glossary](../../../../glossary.md#term-release); [admission glossary](../../../../glossary.md#term-admission))</span>
9. Existing completion-only representative execution and Phase-1 model-autotuning behavior remain
   compatible and their focused tests continue to pass.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
10. Source and reflection checks prove there is no public API addition, CPU-internal import,
    tools/tuning API addition, output serialization, tolerance policy, timing, cache access, or
    fallback-policy change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [CPU glossary](../../../../glossary.md#term-cpu); [serialization glossary](../../../../glossary.md#term-serialization); [tolerance glossary](../../../../glossary.md#term-tolerance); [cache glossary](../../../../glossary.md#term-cache))</span>
11. A separate clean documentation-focused context finalizes Javadoc and the two focused
    explanatory documents, reviews glossary impact, and records a reasoned no-change conclusion
    if no new reusable public term is introduced.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
12. Task/master/roadmap statuses and links are synchronized, changed-path ceilings hold, staging
    remains empty, and all validation below passes.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing:

```bash
./gradlew :modules:engine:test --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest
```

Run the final affected-[module](../../../../glossary.md#term-module) validation once executable code stabilizes:

```bash
./gradlew :modules:engine:test
```

Documentation pass:

```bash
./gradlew :modules:engine:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate all changed Markdown local links and anchors, [canonical](../../../../glossary.md#term-canonical-form) headings, balanced fences,
LF/final newlines, absence of trailing whitespace, exact changed paths, [Ready](../../../../glossary.md#term-ready)/Blocked ordering,
and empty staging. Inspect public [signatures](../../../../glossary.md#term-signature)/imports to confirm the package-private-only boundary.

Repository-wide validation is deferred to the later [Engine](../../../../glossary.md#term-engine) Phase-2 composition [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).
This task changes one module's package-private behavior and no [dependency](../../../../glossary.md#term-dependency) or shared [architecture
contract](../../../../glossary.md#architecture-contract).

## Dependencies

- Complete [Engine](../../../../glossary.md#term-engine) 0004, 0006A, and 0007
- Complete [Runtime](../../../../glossary.md#term-runtime) 0015 result-indexed [publication](../../../../glossary.md#term-publication) [lease](../../../../glossary.md#term-lease)
- Complete [CPU](../../../../glossary.md#term-cpu) 0010G [canonical](../../../../glossary.md#term-canonical-form) [host](../../../../glossary.md#term-host) copying
- Complete CPU 0010J [complete-plan candidate](../../../../glossary.md#complete-plan-candidate)/decision [producer](../../../../glossary.md#term-producer)
- Current [Prepare](../../../../glossary.md#prepare) 0004 [opaque](../../../../glossary.md#term-opaque) [handoff](../../../../glossary.md#term-handoff) roles

## Follow-up tasks

- **Required next:** after this task is Complete, perform a fresh audit and create tools/tuning
  0002 only if the implemented oracle supports the specified generic callback/result boundary.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [callback glossary](../../../../glossary.md#term-callback))</span>
- **After tools/tuning 0002 stabilizes:** add one Config-owned Phase-2 request extension with
  `maximumPlanCandidates`, `maximumTotalPlanExecutions`, `maximumCorrectnessBytes`, and an explicit
  model-plan-cache path. Preserve all current Phase-1 <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache))</span>`Budget` fields and meanings; do not add a
  tolerance field for the exact-only first slice. This remains unnumbered and has no detailed
  specification until the consumer exists.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [slice glossary](../../../../glossary.md#slice))</span>
- **After both consumer contracts stabilize:** add one Engine-owned CPU Phase-2 composition task.
  It must pass the exact Phase-1 selected decision into CPU 0010J, adapt the Engine oracle and
  representative execution to the generic tool, stop on mismatch, freshly prepare only the
  authenticated winner, translate rich evidence, and apply the existing public safe-fallback
  policy. It must not retime or rerank Phase-1 routes. This remains unnumbered and has no detailed
  specification.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [prepare glossary](../../../../glossary.md#prepare))</span>
- Tolerance policies, persistent CPU complete-plan compatibility, Compiler graph alternatives,
  Planning ownership alternatives, multiple partitions, mixed backends, generated-code tuning,
  and a public general Phase-2 API remain separate future work.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backends glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Architecture impact

Expected impact: None. This task realizes existing [Engine](../../../../glossary.md#term-engine) [ownership](../../../../glossary.md#term-ownership-meanings) using current public inward
contracts and adds no [dependency](../../../../glossary.md#term-dependency) or authoritative rule.

If implementation requires a public [oracle](../../../../glossary.md#term-test-oracle), a tools-to-Engine dependency, [CPU](../../../../glossary.md#term-cpu)-internal access,
[Runtime](../../../../glossary.md#term-runtime) result semantic changes, or a relaxed numerical policy, stop and report the architecture
conflict instead of editing `ARCHITECTURE.md` or an [ADR](../../../../glossary.md#term-architecture-decision-record-adr).

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/modules/engine/tasks/0008a-representative-complete-plan-correctness-oracle.md

Implement this task exactly as specified. Do not implement out-of-scope items. Stop and report if
the package-private Engine oracle cannot be implemented through current contracts or if the scope
or architecture ceiling would be exceeded.

After code implementation and module validation, hand the resulting diff and recorded test
evidence to a separate documentation-focused agent or thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md and finalize affected Javadoc, the two focused
architecture documents, glossary impact, and documentation validation in the same overall change.
It must not repeat successful Java tests unless it changes executable behavior or records a
concrete reason.

At the end, update this task file with implementation notes, validation evidence including the
documentation-agent pass, completion summary, and final status. Do not mark the task Complete
before that pass finishes. Do not commit or push unless explicitly instructed.
```

## Local decisions

- The correctness model is one package-private final top-level type colocated with the
  representative session. Its reference owns transferred detached arrays and exposes no accessor;
  its comparison result has exactly <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [detached glossary](../../../../glossary.md#detached-model-results))</span>`MATCH` and `MISMATCH`.
- Ordinary host snapshots and correctness capture share one Engine-owned descriptor preflight so
  fully static Shape, resolved layout, checked byte arithmetic, per-array ceiling, occurrence
  order, and aggregate-limit behavior cannot drift.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine); [descriptor glossary](../../../../glossary.md#term-descriptor); [preflight glossary](../../../../glossary.md#term-preflight); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Correctness capture stores the validated per-occurrence byte counts for later comparisons.
  Candidate comparison therefore cannot introduce a different byte policy after the reference is
  established.
- Completion-only execution remains the Phase-1 measurement action. Correctness capture and
  comparison are distinct package-private session actions and are not wired into current public
  tuning composition.

## Known limitations

- The [oracle](../../../../glossary.md#term-test-oracle) is package-private [Engine](../../../../glossary.md#term-engine) machinery and has no public request or result surface.
- Exact canonical represented-byte equality is the only policy. There is no tolerance, NaN
  normalization, relaxed-math permission, or decoded numeric comparison.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [tolerance glossary](../../../../glossary.md#term-tolerance); [NaN glossary](../../../../glossary.md#term-nan); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- No tools/tuning Phase-2 orchestration consumes the oracle yet. Task 0002 remains Blocked pending
  a fresh post-0008A dependency/readiness audit and still has no detailed specification.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Model-plan persistence, Phase-2 budgets and timing order, candidate interpretation, selected
  production preparation, evidence translation, and public fallback remain follow-up work.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0b91e-76ac-7ee3-a697-015e98495a1c` ran
  `./gradlew :modules:engine:test --tests
  io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest`: 34 tests, zero failures,
  errors, or skips.
- The same implementation context ran `./gradlew :modules:engine:test`: 77 tests, zero failures,
  errors, or skips. Documentation context `01a0b937-c31b-7591-8c79-f8577d6264b2` changed no
  executable Java token, so it did not repeat either successful Java suite.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- The implementation context passed `./gradlew :modules:engine:javadoc` and the required
  `javap`, reflection, import, public-surface, and source scans. Those checks confirmed one new
  package-private top-level final type, no public/protected oracle surface, exactly two comparison
  values, and no CPU-internal or new tools/tuning API dependency.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [CPU glossary](../../../../glossary.md#term-cpu); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The implementation context also passed its diff, changed-path, and staging checks. It reported
  no repository-wide checkpoint, architecture test, integration test, backend-conformance run, or
  performance benchmark; none is claimed by this task.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [architecture test glossary](../../../../glossary.md#term-architecture-test); [integration test glossary](../../../../glossary.md#term-integration-test); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [run glossary](../../../../glossary.md#term-run))</span>
- Mandatory clean documentation context `01a0b937-c31b-7591-8c79-f8577d6264b2` independently
  reviewed the complete source/test diff, finalized the three affected production Javadoc/comment
  paths, the two focused architecture explanations, and the four planning/status paths, and
  reviewed the central glossary. It used General plus API/Javadoc style for Java contracts,
  General plus Architecture style for the focused explanations, and General plus Planning style
  for planning files.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- The documentation context ran `./gradlew :modules:engine:javadoc` after final comment edits and
  passed. Its repository-relative Markdown validation over the six documentation/planning paths
  passed local targets and anchors, unique headings, balanced fences, LF/final newlines, and
  trailing-whitespace checks.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- A documentation-pass source-delta audit reviewed all three permitted production Java paths.
  `AdvancedEngine` and `RepresentativePlanCorrectness` required no documentation-pass edit;
  every hunk added by this pass to `RepresentativeExecutionSession` was inside a Javadoc comment.
  Therefore this pass changed no executable token. Final `git diff --check` and
  `git diff --cached --check` passed; staging was empty.
- Final inventory validation found exactly ten changed or new paths: three Engine production
  paths, one existing Engine test, two architecture explanations, this task, the Engine master
  plan, the tuning master plan, and the roadmap. Exactly one new top-level Java type exists.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- `ARCHITECTURE.md`, ADRs, module dependencies, architecture tests, integration tests,
  backend-conformance tests, Runtime/Prepare/CPU/Config/Compiler/Model contracts, and public Engine
  API need no change because this task realizes the existing Engine-owned package-private
  publication/copy/cleanup boundary without changing architecture or cross-module behavior.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [publication glossary](../../../../glossary.md#term-publication))</span>
- Glossary no-change conclusion: the implementation adds no reusable public domain term. Its
  opaque reference and two-valued result are private session mechanics, while the existing
  complete-plan candidate and canonical host-snapshot terminology remains accurate.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [complete-plan candidate glossary](../../../../glossary.md#complete-plan-candidate); [canonical glossary](../../../../glossary.md#term-canonical-form); [host glossary](../../../../glossary.md#term-host); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>

## Implementation notes

- Added `RepresentativePlanCorrectness`, an immutable package-private result model whose opaque
  same-session reference owns detached canonical bytes and whose result exposes only exact<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [model glossary](../../../../glossary.md#term-model); [opaque glossary](../../../../glossary.md#term-opaque); [detached glossary](../../../../glossary.md#detached-model-results); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
  `MATCH` or `MISMATCH`.
- Extended `RepresentativeExecutionSession` to snapshot ordered publication specifications,
  capture one reference after complete preflight and fresh execution, and compare later fresh
  executions only after all occurrence copies and result cleanup succeed. Aliases, empty values,
  order, signed zero, and NaN payload bits remain observable.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [publication glossary](../../../../glossary.md#term-publication); [capture glossary](../../../../glossary.md#term-capture-meanings); [preflight glossary](../../../../glossary.md#term-preflight); [occurrence glossary](../../../../glossary.md#term-occurrence); [Aliases glossary](../../../../glossary.md#aliasing-and-arena); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Factored AdvancedEngine's canonical descriptor byte preflight for shared ordinary-host-snapshot
  and correctness use. The existing Phase-1 completion-only action and public tuning composition
  are unchanged.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [host glossary](../../../../glossary.md#term-host))</span>
- Reused the existing execution-failure cleanup path so copy/count/length/result-cleanup failure
  preserves primary and suppressed failures, poisons once, cleans borrowed wrappers once, and
  releases admission. A clean mismatch does not poison the session.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [admission glossary](../../../../glossary.md#term-admission))</span>

## Completion summary

- Completed changes: implemented and documented the bounded Engine-owned representative complete-
  plan exact correctness primitive without public API, timing, cache, candidate, or fallback
  integration.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [cache glossary](../../../../glossary.md#term-cache))</span>
- Files changed or created: the exact ten-path inventory recorded in [Validation](../../../../glossary.md#term-validation) evidence.
- Tests and validation: reused the implementation context's passing 34 focused and 77 full Engine
  tests; final Engine Javadoc, Markdown, no-executable-token source-delta, diff, staging, and scope
  checks passed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [token glossary](../../../../glossary.md#term-token); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `01a0b937-c31b-7591-8c79-f8577d6264b2` finalized all
  affected Javadocs/comments, focused explanations, planning evidence, and no-change reviews.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: current completion-only versus correctness behavior, exact byte semantics,
  lifecycle/failure handling, task status, and next frontier are synchronized.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: all three affected production paths were reviewed; their ownership, lifecycle,
  inputs, results, exact represented-byte semantics, preflight order, cleanup, and failures are
  accurate.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Glossary impact: no change; the new concepts are package-private mechanics rather than reusable
  public terminology.
- Unresolved issues: Phase-2 composition and its readiness remain unaudited after this change.
- Follow-up required: perform the required fresh tools/tuning 0002 dependency/readiness audit;
  keep task 0002 Blocked until that audit supports a complete detailed specification.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

Status: Complete
