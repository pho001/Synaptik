# Task 0008A: Representative Complete-Plan Correctness Oracle

## Status

Ready

## Goal

Add the smallest Engine-owned internal correctness oracle required before generic Phase-2
complete-plan tuning can be specified truthfully. The oracle executes freshly prepared complete
recipes against one already-bound representative input set, captures every ordered publication
as detached canonical host bytes, owns the first candidate's reference output, and reports exact
match or mismatch for each later candidate without exposing Tensor semantics, backend
representations, or output bytes to `tools/tuning`.

This is an upstream prerequisite, not Phase-2 tuning itself. It adds no timing, ranking, cache,
candidate interpretation, public request, selected preparation, or fallback policy. After this
task is Complete, a fresh tools/tuning 0002 specification may define the generic opaque
candidate, budget, correctness-callback, evidence, and compact model-plan-record workflow.

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

- Complete Engine 0006A added `RepresentativeExecutionSession`. It validates and borrows the
  representative inputs once, executes each supplied `PreparedExecution` in a fresh Runtime
  state, validates the publication count, and closes the result, but its contract deliberately
  forbids inspecting or materializing publication payloads.
- Complete Engine 0007 composes only Phase-1 local-workload tuning. Its measurement action freshly
  prepares one CPU candidate, calls the representative session's completion-only execution, and
  leaves numerical comparison out of scope.
- `CompiledGraph` already retains the authoritative ordered `PublicationSpec` values and their
  exact final `TensorDescriptor` values. Runtime `RunResult` already lends each ordered
  publication representation by dense result index while its lease is open.
- `EngineBackendComposition.copyToCanonicalHostBytes(...)` already supplies the backend-neutral
  Engine seam used by host materialization. The CPU implementation preserves represented bits in
  canonical row-major big-endian bytes, including floating-point NaN payloads and signed zero,
  while validating descriptors, geometry, capacity, carrier, and canonical BOOL values.
- Current ordinary Engine materialization already preflights fully static Shape, resolved layout,
  checked element/byte arithmetic, the JVM array ceiling, and an aggregate caller byte limit.
  That logic is private to `AdvancedEngine`; the representative session must reuse or factor the
  same Engine-owned validation rather than create different byte-accounting semantics.
- Complete CPU 0010J supplies a session-scoped opaque batch of at least two retained complete
  one-partition CPU topology/representation alternatives around the exact authenticated Phase-1
  selection. It freshly prepares a requested trial or selected recipe, but does not execute,
  compare, time, cache, or apply fallback policy.
- Prepare's public tuning roles are intentionally method-free opaque transport. Runtime exposes
  result-indexed representations and cleanup only; neither layer owns Tensor-value comparison.
  Therefore the missing executable correctness contract belongs to Engine, which already owns
  logical publications, representative values, materialization, and result cleanup.

## Scope

- Add one package-private immutable correctness model in `io.github.pho001.synaptik.engine` with:
  - an opaque reference value associated by identity with exactly one live
    `RepresentativeExecutionSession`;
  - an exact two-valued comparison result, `MATCH` or `MISMATCH`; and
  - no public constructor, byte accessor, Tensor metadata, backend value, or diagnostic payload.
- Extend `RepresentativeExecutionSession` with two package-private operations:
  - `captureCorrectnessReference(PreparedExecution, long)` captures the first complete recipe's
    reference output under the supplied aggregate canonical-byte limit; and
  - `compareCorrectness(Reference, PreparedExecution)` executes and compares one later complete
    recipe with that exact reference and its already validated byte-count contract.
- Snapshot the compiled graph's complete ordered publication specifications during session
  construction. Do not accept publication descriptors, output meanings, or expected bytes from
  `tools/tuning`, a backend, or a new callback.
- Before the first correctness execution, validate all publication descriptors and calculate all
  canonical byte counts with checked arithmetic. Require at least one publication, fully static
  Shapes, resolved layouts, a non-negative aggregate byte limit, each payload within the JVM
  `byte[]` ceiling, and the checked aggregate not greater than the supplied limit.
- Execute each correctness run through the existing `PreparedExecutionRunner` and the session's
  already ordered borrowed inputs. Every call creates a fresh isolated Runtime `RunState`.
- Validate the Runtime result count before any payload access. Copy every publication occurrence
  independently in dense compiled order through
  `EngineBackendComposition.copyToCanonicalHostBytes(...)`; intentional aliases remain separate
  observable occurrences.
- Complete every copy while the Runtime result lease is open, then close the result before
  returning a reference or comparison. Reference bytes are detached Engine-owned snapshots and
  contain no Runtime or backend resource.
- Compare publication count, canonical byte length, and byte content in encounter order. The
  initial policy is exact represented-bit equality only. This deliberately distinguishes signed
  zero and NaN payloads and supplies no tolerance, normalization, data-type-specific rule, or
  relaxed-math permission.
- Make the first successfully captured candidate the sole reference for the session. Reject a
  reference from another session, a second reference capture, comparison before reference capture,
  comparison after cleanup, and any action after poisoning.
- Treat `MISMATCH` as a clean typed outcome only after execution, all copies, and result cleanup
  succeed. It does not poison the session by itself. A later Phase-2 caller must stop the tuning
  transaction, publish no partial evidence or cache state, clean the session, and route the
  failure through Engine's public safe-fallback policy.
- Preserve the existing failure protocol. Execution, count validation, materialization, or result
  cleanup failure keeps its exact unchecked primary where applicable, suppresses distinct cleanup
  failures deterministically, poisons the session, cleans borrowed inputs once, releases Engine
  admission, and forbids selected preparation and fallback from that session.
- Add focused native-free tests for preflight, first-reference ownership, exact match, content and
  length mismatch, multiple publications, aliased occurrences, zero-length
  payloads, signed zero and NaN-payload distinction, detached reference lifetime, wrong-session
  rejection, result/input cleanup, failure identity/suppression, poisoning, and close races.
- Finalize affected Javadocs and focused explanatory documentation in the mandatory separate clean
  documentation context during implementation.

## Out of scope

- any public Engine API or change to `ModelAutotuningRequest`, `ModelAutotuningPreparation`,
  `Engine`, `CompiledGraph`, public `RunResult`, or `HostTensorValue`
- tools/tuning 0002 types, algorithms, dependencies, timing, ranking, deterministic winner
  selection, budgets, evidence, cache parsing/publication, or model-plan records
- CPU candidate enumeration, compatibility/identity/decision codecs, selected preparation, Phase-1
  ranking, or interpretation of topology, representation, route, provider, or resource fields
- changing `PreparedExecution`, Runtime result/publication contracts, Prepare opaque handoff roles,
  or serializing an executable, result, representation, Tensor, or prepared recipe
- tolerance, relative/absolute error, ULP comparison, NaN normalization, stochastic comparison,
  relaxed mathematics, or caller-defined numerical callbacks
- making a backend target fingerprint persistent; CPU 0010J remains honestly `SESSION`
- public safe-fallback policy, fallback classification, final selected preparation, or promotion of
  a correctness or timed trial recipe into production
- timing a correctness run, combining correctness with a warmup/sample, or changing Phase-1
  measurement behavior
- Compiler graph alternatives, Planning ownership alternatives, generated-code tuning,
  multi-partition or mixed-backend candidates, or a public Phase-2 request

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Engine, Runtime, concrete
  backends, performance evidence and optimization tooling, and dependency rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)
- Engine [0004](0004-explicit-host-materialization-boundary.md),
  [0006A](0006a-representative-tuning-execution-and-safe-fallback.md), and
  [0007](0007-optional-model-autotuning-composition.md)
- [CPU 0010J](../../../backends/cpu/tasks/0010j-supported-complete-plan-candidate-and-decision-producer.md)
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)
- [tools/tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)

## Architecture constraints

- Engine owns representative logical input binding, publication meaning, result materialization,
  cleanup coordination, and public fallback policy. This task stays package-private in Engine.
- Runtime remains unaware of tuning and correctness policy. It executes one immutable recipe at a
  time and lends ordered representations only through the existing result lease.
- Concrete backends continue to own physical copy/encoding and complete-plan candidates. Engine
  calls the existing composition copy seam without importing CPU internals.
- `tools/tuning` must later see only an opaque reference collaboration and a typed match/mismatch
  result supplied by its caller. It must not receive Tensor descriptors, publication bytes,
  Runtime representations, or an Engine type in its own API.
- Correctness executions are separate from warmups and timed samples. The later tools task owns
  invocation ordering and arithmetic; Engine owns the semantics and cleanup of each requested
  correctness action.
- Exact byte equality is a conservative current exact/default oracle for the first producer. Any
  tolerance policy requires a separate stable numerical contract and is not inferred from data
  type, objective, provider, target, or observed results.
- A mismatch never becomes a selected result. It is a recoverable tuning failure only after
  successful result and representative-input cleanup; the existing Engine fallback owner makes
  the later public policy decision.
- No `PreparedExecution`, executable state, Runtime result, representation, or captured output may
  enter a persistent artifact.
- No module dependency or authoritative architecture rule changes in this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.engine` — owns the representative session, compiled publication
  descriptors, composition copy seam, cleanup, and the new package-private correctness values.
- `io.github.pho001.synaptik.runtime.execution` and
  `io.github.pho001.synaptik.runtime.run` — existing immutable recipe, runner, and leased result
  contracts only; no source changes.

Packages added or changed:

- No package is added.
- `io.github.pho001.synaptik.engine` gains one package-private helper and focused tests in the
  mirrored test package.

Type placement:

- `io.github.pho001.synaptik.engine.RepresentativePlanCorrectness` — package-private immutable
  opaque reference and two-valued comparison model colocated with the sole Engine-owned consumer;
  it is not a public generic tuning contract.
- `io.github.pho001.synaptik.engine.RepresentativeExecutionSession` — remains the owner of
  representative bindings, Runtime execution, result materialization/cleanup, poisoning, and
  reference association.

## Affected files

Expected production and test paths:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativePlanCorrectness.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java` only to factor
  the existing canonical-byte preflight into one package-private Engine-owned implementation if
  reuse cannot remain private without duplication
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`

Expected documentation and planning paths during implementation:

- `docs/architecture/performance-evidence-and-tuning.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- this task specification
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/tools/tuning/master-plan.md`
- `docs/planning/roadmap.md`

No build file, dependency, architecture contract, ADR, Runtime, Prepare, CPU, Config, Compiler,
Model, or tools/tuning source path is expected to change.

## Maximum scope

This task may create or modify at most:

- 3 Engine production/Javadoc paths, including exactly one new package-private top-level type;
- 1 focused Engine test path;
- 2 focused explanatory architecture-documentation paths; and
- exactly 4 planning paths: this task, the Engine master plan, the tuning master plan, and the
  roadmap.

The total maximum is 10 paths. If another source package, public type, module dependency, numeric
policy, or more than 10 paths are required, stop and propose a follow-up task.

## Exact semantics, lifecycle, failure, and identity

### Reference and comparison semantics

- The first successful reference capture executes exactly once and owns the baseline. There is no
  separately prepared heuristic baseline and no backend-provided expected output.
- The reference contains one detached canonical byte sequence per compiled publication occurrence
  in exact dense order. It retains no Tensor, storage, representation, `RunResult`,
  `PreparedExecution`, or executable object.
- A comparison executes the supplied recipe exactly once and evaluates every occurrence. It
  returns `MATCH` only when counts, lengths, and every byte match the reference.
- `MISMATCH` carries no payload. Engine may use internal local indexes while comparing, but neither
  the reference nor result exports publication metadata or bytes. The later generic tuning caller
  needs only the two-valued result to stop fail-closed.
- Zero-element publication payloads are valid and compare as empty byte sequences. A compiled
  graph with no publications is rejected before execution because it provides no observable
  correctness reference.
- Exact represented-byte equality is policy version 1. Floating signed zeros and NaN payloads are
  intentionally distinct. No value decoding occurs.

### Preflight and bounds

- The caller supplies one non-negative maximum aggregate canonical byte count per correctness
  execution. The reference stores the validated publication byte counts, and comparisons require
  the same session/reference contract rather than recalculating an alternative policy.
- Shape, layout, element count, data-type width, per-array ceiling, aggregate addition, and limit
  validation complete before the Runtime runner is invoked. Overflow fails closed.
- A preflight failure executes no recipe, captures no reference, produces no comparison, changes
  no cache/evidence state, and leaves the session eligible for orderly cleanup.

### Lifecycle and cleanup

- The reference is associated with the exact session by identity and is usable only while that
  session remains open and unpoisoned. Detached bytes survive result cleanup internally but do not
  grant use after representative-session cleanup.
- A successful capture/comparison closes the complete Runtime result before returning. No trial
  result or run-owned resource survives the call.
- A clean mismatch does not poison resources; the later caller may only clean the session and
  resolve the mismatch through its tuning/fallback boundary. It must not continue timing.
- Execution, result-count, physical copy, canonical validation, allocation, or result-cleanup
  failure poisons the session exactly like the existing representative execution failure path.
  Input cleanup is attempted once in reverse order and Engine admission is released.
- If copying fails and result cleanup also fails, copying remains primary and cleanup is
  suppressed once. If result cleanup alone fails, that cleanup failure is primary. Existing
  distinct-suppression and fatal-error behavior remains unchanged.

### Later Phase-2 budget contract

This task does not implement the budget, but its one-execution-per-correctness-action contract
fixes the arithmetic that tools/tuning 0002 must use. For `N` cache-miss complete candidates,
`W` warmups, and `S` timed samples:

```text
correctness executions = N
timed/warmup executions = N * (W + S)
maximum total executions required = N * (1 + W + S)
```

The first correctness execution creates the reference and each of the remaining `N - 1` compares
against it. All `N` correctness executions must match and clean up successfully before the first
warmup or timed execution begins. The later tool must validate a positive candidate ceiling,
non-negative `W`, positive odd `S`, a positive maximum-total-executions ceiling, and all checked
additions/multiplications before the first correctness execution. A compatible cache hit performs
zero correctness, warmup, or timed executions and proceeds only through backend-compatible
decision decoding and fresh selected preparation. Any preflight failure publishes no partial
evidence or cache state.

## Acceptance criteria

1. `RepresentativePlanCorrectness` is package-private, immutable, exposes no output bytes or
   Tensor/backend/Runtime value, and has exactly one opaque reference value plus the two-valued
   `MATCH`/`MISMATCH` result required by this task.
2. `RepresentativeExecutionSession` snapshots the exact compiled publication specifications and
   supports one reference capture followed by zero or more same-session comparisons while open.
3. Complete static/resolved/JVM-array/aggregate-byte preflight finishes before execution, rejects
   an empty publication boundary, uses checked arithmetic, and shares the ordinary Engine host-
   snapshot byte-accounting semantics.
4. Each capture/comparison uses a fresh Runtime run, validates result count, copies every ordered
   publication occurrence independently through the existing composition seam, and closes the
   result before return.
5. Exact canonical bytes determine `MATCH`; signed zero, NaN payload, length, and occurrence-order
   differences are tested without decoding Tensor values.
6. A reference cannot be reused across sessions, recaptured, used before capture, or used after
   cleanup/poisoning. No output byte array escapes the Engine package-private model.
7. A clean mismatch returns typed metadata only after successful cleanup, does not poison the
   session, and cannot itself authorize continued measurement, selection, or fallback.
8. Execution, copying, validation-after-run, allocation, and cleanup failures preserve the
   existing exact-primary and deterministic distinct-suppression rules, poison the session, clean
   owned wrappers once, release admission, and forbid final preparation/fallback.
9. Existing completion-only representative execution and Phase-1 model-autotuning behavior remain
   compatible and their focused tests continue to pass.
10. Source and reflection checks prove there is no public API addition, CPU-internal import,
    tools/tuning API addition, output serialization, tolerance policy, timing, cache access, or
    fallback-policy change.
11. A separate clean documentation-focused context finalizes Javadoc and the two focused
    explanatory documents, reviews glossary impact, and records a reasoned no-change conclusion
    if no new reusable public term is introduced.
12. Task/master/roadmap statuses and links are synchronized, changed-path ceilings hold, staging
    remains empty, and all validation below passes.

## Tests / validation

Run focused tests while developing:

```bash
./gradlew :modules:engine:test --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest
```

Run the final affected-module validation once executable code stabilizes:

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

Also validate all changed Markdown local links and anchors, canonical headings, balanced fences,
LF/final newlines, absence of trailing whitespace, exact changed paths, Ready/Blocked ordering,
and empty staging. Inspect public signatures/imports to confirm the package-private-only boundary.

Repository-wide validation is deferred to the later Engine Phase-2 composition checkpoint or CI.
This task changes one module's package-private behavior and no dependency or shared architecture
contract.

## Dependencies

- Complete Engine 0004, 0006A, and 0007
- Complete Runtime 0015 result-indexed publication lease
- Complete CPU 0010G canonical host copying
- Complete CPU 0010J complete-plan candidate/decision producer
- Current Prepare 0004 opaque handoff roles

## Follow-up tasks

- **Required next:** after this task is Complete, perform a fresh audit and create tools/tuning
  0002 only if the implemented oracle supports the specified generic callback/result boundary.
- **After tools/tuning 0002 stabilizes:** add one Config-owned Phase-2 request extension with
  `maximumPlanCandidates`, `maximumTotalPlanExecutions`, `maximumCorrectnessBytes`, and an explicit
  model-plan-cache path. Preserve all current Phase-1 `Budget` fields and meanings; do not add a
  tolerance field for the exact-only first slice. This remains unnumbered and has no detailed
  specification until the consumer exists.
- **After both consumer contracts stabilize:** add one Engine-owned CPU Phase-2 composition task.
  It must pass the exact Phase-1 selected decision into CPU 0010J, adapt the Engine oracle and
  representative execution to the generic tool, stop on mismatch, freshly prepare only the
  authenticated winner, translate rich evidence, and apply the existing public safe-fallback
  policy. It must not retime or rerank Phase-1 routes. This remains unnumbered and has no detailed
  specification.
- Tolerance policies, persistent CPU complete-plan compatibility, Compiler graph alternatives,
  Planning ownership alternatives, multiple partitions, mixed backends, generated-code tuning,
  and a public general Phase-2 API remain separate future work.

## Architecture impact

Expected impact: None. This task realizes existing Engine ownership using current public inward
contracts and adds no dependency or authoritative rule.

If implementation requires a public oracle, a tools-to-Engine dependency, CPU-internal access,
Runtime result semantic changes, or a relaxed numerical policy, stop and report the architecture
conflict instead of editing `ARCHITECTURE.md` or an ADR.

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

Empty until implemented.

## Known limitations

Empty until implemented.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
