# Task 0015: Leased Publication Representation Access

## Status

Complete

## Goal

Expose the exact physical representation selected for one successfully published Runtime result
occurrence to an inward lifecycle consumer while the complete `RunResult` lease remains open.
This closes the smallest demonstrated prerequisite for Engine host materialization: Runtime
already snapshots the ordered direct references, including aliases, but currently exposes only
their count and lifecycle.

The boundary remains representation-level Runtime service-provider interface (SPI), not ordinary
Engine value access. It performs no host copy, conversion, transfer, allocation, backend lookup,
or ownership transfer.

## Scope

- Add exactly one public method to `io.github.pho001.synaptik.runtime.run.RunResult`:

  ```java
  public BufferRepresentation publicationRepresentation(int resultIndex);
  ```

- Require the result lease to be open, then return the exact retained representation reference at
  the dense result index. Repeated access to one occurrence returns the same reference; aliased
  occurrences return the same exact reference.
- Treat the returned representation as borrowed from the result. The caller must not close it,
  transfer it, retain it beyond result closure, or infer ownership from alias identity.
- Preserve `RunResult`'s existing whole-`RunState` lease, result count, idempotent cleanup, and
  borrowed-input/run-owned-resource rules unchanged.
- Add focused Runtime behavior and public-shape coverage.
- Finalize affected Runtime/Public API explanations and glossary wording through a separate clean
  documentation-focused pass after implementation.

## Out of scope

- Host values, byte encodings, logical-layout traversal, copying, conversion, allocation, arena or
  `MemorySegment` access, caller-owned output storage, or a backend materializer.
- A representation list, mutable view, `RunState` accessor, buffer or representation coordinate,
  `ValueId`, Tensor identity, publication role, descriptor, slot, backend type, or ownership flag.
- Returning a new wrapper, transferring an individual representation out of the run state, or
  changing which resources result closure releases.
- Making `RunResult` thread-safe, racing representation access with closure, asynchronous access,
  callbacks, pinning, reference counting, or extending a representation's lifetime.
- Changes to Prepare, Engine, CPU or another backend, Config, Trace, module dependencies,
  architecture rules, ADRs, schedule construction, publication execution, or Runtime hot-path
  behavior.
- Creating a CPU 0010G or Engine 0004 implementation or a later detailed Engine task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Runtime ownership, run
  lifecycle, Engine composition, dependency rules, and ADR 0011's backend-owned representation
  boundary.
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0009](0009-publication-and-result-schedule-steps.md)
- [Runtime 0010](0010-prepared-runner-and-dynamic-execution.md)
- [Engine master plan](../../engine/master-plan.md)

## Architecture constraints

- Runtime owns the result lease and already retains the selected representations in dense
  publication-occurrence order. Exposing that retained reference from the inward Runtime SPI does
  not move physical access mechanics into Runtime.
- The returned type is only the existing nominal `BufferRepresentation`; Runtime must not inspect
  or name CPU, Metal, CUDA, `MemorySegment`, backend storage, or logical Tensor facts.
- Concrete backends continue to own physical access and copying. A later CPU task may consume the
  borrowed reference through its supported integration SPI; ordinary Engine signatures must not.
- The reference remains governed by the whole result lease. This task must not transfer, split,
  pin, or extend ownership and must not let callers bypass result cleanup.
- The accessor is cold result handling after successful schedule execution, not a Runtime hot-path
  action. It performs constant-time indexed access only.
- If implementation requires a new dependency, another public type, a callback abstraction,
  per-representation ownership, concurrency semantics, or an architecture change, stop and
  report the conflict.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.runtime.run` remains the owner of result publication order and lease
  lifetime.

Type placement:

- `io.github.pho001.synaptik.runtime.run.RunResult` gains the sole accessor because it already
  privately snapshots the exact representation array and owns the lease that bounds access.

No package or top-level type is added or moved.

## Affected files

Expected production and test paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunResult.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunResultTest.java`

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`

Expected planning paths:

- `docs/planning/modules/runtime/tasks/0015-leased-publication-representation-access.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `ARCHITECTURE.md`, ADR 0011, focused architecture pages, Runtime
package Javadocs, Prepare/Engine/CPU source and Javadocs, build files, architecture tests,
conformance tests, and integration tests. Record reasoned no-change conclusions.

## Maximum scope

This task may modify exactly the eight paths listed above: one production file, one focused test,
three explanatory documents, and three planning documents. If another source/test owner,
dependency, architecture document, or ninth path is required, stop and replan.

## Exact behavior and failures

- `publicationRepresentation(resultIndex)` first requires the leased run state to be open, using
  the existing stable `IllegalStateException("run state is closed")`. Closed-state failure wins
  over index validation.
- For an open result, a negative index or an index greater than or equal to `resultCount()` throws
  `IndexOutOfBoundsException` with the rejected index in its message.
- On success, return `representations[resultIndex]` exactly. Do not copy, wrap, cast, close, or
  mutate it and do not query Runtime validity again; successful result construction already proved
  that every occurrence published a valid selected representation.
- The returned reference is usable only under the existing non-thread-safe result contract. The
  caller must externally prevent access from racing any result/state activity or closure.
- Repeated calls before closure return the identical reference. Two result indices that aliased at
  publication return the identical reference; distinct occurrence count and order remain intact.
- After closure begins, every access fails even when the index would otherwise be valid. Metadata
  methods `resultCount()` and `isClosed()` keep their current post-close behavior.

## Acceptance criteria

- `RunResult` has exactly the one specified new method and no other public/protected surface
  change.
- Open-result access, both index bounds, repeated identity, alias identity, distinct identity,
  closed precedence, and no-close/no-copy behavior are covered by focused tests.
- Construction, publication, result count, failure cleanup, and whole-state cleanup tests remain
  unchanged in meaning and pass.
- Runtime imports no Model, Engine, Prepare, or concrete backend type and gains no dependency.
- No `RunState`, coordinate, storage, backend type, transfer, or host-value access is exposed.
- Javadocs precisely describe borrowing, lifetime, aliasing, external synchronization, failure
  precedence, and the prohibition on closing or retaining the reference beyond the result.
- Runtime/Public API and glossary prose stop claiming that the inward Runtime result has no
  representation access, while clearly stating that ordinary Engine still has metadata only.
- A separate clean documentation-focused pass finalizes Javadoc, explanatory documentation,
  glossary impact, and planning evidence before completion.

## Tests / validation

Implementation-focused final validation after executable Java stabilizes:

```bash
./gradlew :modules:runtime:test
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest
```

Documentation-focused pass:

```bash
./gradlew :modules:runtime:javadoc
git diff --check
```

Also automate or inspect the exact `RunResult` public shape, Runtime dependency inventory, source
imports, generated Javadoc, changed Markdown links/anchors/headings/fences, trailing whitespace,
final newlines, exact eight-path scope, and task/master/roadmap status agreement. Confirm no CPU
0010G or Engine 0004 detailed task exists.

Repository-wide tests are deferred to Engine 0008/CI. This task changes one narrow Runtime SPI
accessor without changing dependencies, schedule execution, concrete backend behavior, or the
hot path. Backend conformance and integration tests are unchanged.

## Dependencies

- Runtime 0009 publication and result schedule steps — Complete.
- Runtime 0010 prepared runner and dynamic execution — Complete.
- Runtime 0012 shared-throwable cleanup correction — Complete.
- Engine 0003 — Complete and demonstrates the first real outer consumer, while importing no new
  Runtime contract into its ordinary surface.

## Follow-up tasks

- CPU 0010G: supported canonical caller-owned host snapshot export through
  `CpuBackendIntegration`, consuming the leased representation without exposing CPU internals.
- Engine 0004: lazy per-publication host materialization into a detached immutable bounded
  `HostTensorValue`, after Runtime 0015 and CPU 0010G are Complete.

Do not create either detailed follow-up specification during Runtime 0015 implementation.

## Architecture impact

Expected impact: None.

This realizes the authoritative result-lease and backend-owned physical-access split. Runtime
exposes only its existing nominal representation reference while the lease is open; concrete
backends still perform all physical access. No dependency direction or ownership rule changes.
If implementation evidence contradicts that conclusion, stop rather than editing architecture
from this task.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Runtime task 0015. Do not use GSD,
commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the Runtime master plan,
docs/planning/modules/runtime/tasks/0015-leased-publication-representation-access.md, Runtime
0009–0010, ADR 0011, the focused Runtime boundary documentation, and the actual Runtime result,
publication, state, tests, APIs, glossary, and architecture enforcement in full.

Implement exactly the one result-indexed borrowed publication-representation accessor and its
focused tests. Preserve the whole-state lease, aliases, failure precedence, non-thread-safe
contract, dependencies, and eight-path ceiling. Add no host copy, backend knowledge, coordinates,
ownership transfer, callback abstraction, Engine/CPU change, or later task. Stop on architecture
or scope conflict.

Run the task-tier validation once after Java stabilizes, then hand the exact diff and evidence to
a distinct clean documentation-focused context. That pass must finalize Javadoc, Runtime/Public
API wording, glossary impact, and planning evidence without repeating successful Java tests unless
executable behavior changes. Mark Complete only after all acceptance and documentation gates pass.
```

## Local decisions

- Direct result-indexed access is smaller and safer than exposing `RunState`, prepared
  coordinates, a representation list, or a generic callback whose returned value could disguise
  lifetime escape.
- The exact representation is borrowed under the existing whole-state lease. Individual output
  ownership is not transferred, so cleanup and aliases remain unchanged.
- Existing non-thread-safe result semantics remain explicit. Engine 0004 must add its own outward
  materialization/close coordination instead of making this inward carrier concurrently mutable.

## Known limitations

- The returned representation is not a host value and has no shared physical-access methods.
- Callers must use a concrete backend's supported integration operation while the result remains
  open; Runtime itself cannot copy or interpret it.
- Ordinary Engine callers still receive metadata-only results until CPU 0010G and Engine 0004 are
  implemented.

## Validation evidence

- Implementation context: `01a0a4be-e136-72b1-93a1-7361f64a24cd`.
- The implementation context's focused `RunResultTest` run passed 9 tests. Its final
  `./gradlew :modules:runtime:test` passed 17 suites and 148 tests with zero failures, errors, or
  skips. Its focused
  `./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest`
  passed 1 suite and 3 tests with zero failures, errors, or skips. Documentation context
  `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` reused this stable evidence because it changed no
  executable Java token and found no stale-evidence reason; it did not rerun Java tests.
- The implementation context's `javap` inspection confirmed exactly one new public method,
  `publicationRepresentation(int)`. Its Runtime dependency/import inventory remained unchanged,
  its exact two-Java-path implementation scope passed, and `git diff --check` passed.
- Documentation context `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` independently read the authority,
  lifecycle and boundary explanations, ADR 0011, planning guide/roadmap, Runtime master and tasks
  0009/0010/0012/0014/0015, documentation rules and General/API-Javadoc/Planning/Example profiles,
  final Runtime source/tests and representation/state/publication/runner contracts, Engine
  ordinary/advanced result boundaries, supported CPU integration boundary, Runtime/Public APIs,
  glossary, and the actual diff.
- `./gradlew :modules:runtime:javadoc` passed with no warnings after final documentation review.
  Generated `RunResult.html` was inspected and contains the borrowed exact-reference lifetime,
  identity/alias, external-synchronization, no-copy/no-validity-query, closed-precedence, bounds,
  parameter, return, and failure contracts.
- Final `javap -classpath modules/runtime/build/classes/java/main -public
  io.github.pho001.synaptik.runtime.run.RunResult` reports the existing public constructor,
  `resultCount()`, `isClosed()`, and `close()`, plus exactly the one new
  `BufferRepresentation publicationRepresentation(int)` method. The comment-stripped final
  production source hash remained
  `56e03e5fb775375ebf0e1c640651fcae7d4b39950338abf9fb8fcfed97995e4f` throughout the
  documentation finalization, confirming no executable Java token changed after implementation.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md docs/api/public-api.md
  docs/glossary.md docs/planning/modules/runtime/tasks/0015-leased-publication-representation-access.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md` passed all six changed
  Markdown files for local targets, generated effective anchors, balanced fences, LF/final
  newlines, and trailing whitespace. Repeated example-subheadings in Runtime API use the
  renderer's deterministic suffixed effective anchors rather than being treated as collisions.
- The final tracked/untracked union contains Runtime 0015's exact eight permitted paths plus only
  the two preserved prerequisite-planning paths,
  `docs/planning/backends/cpu/master-plan.md` and
  `docs/planning/modules/engine/master-plan.md`. Their SHA-256 values remained
  `f78bbb430f5eaa47fbd3565f7ab52e904a166b9ecfffec916215c0f48c4e2e34` and
  `6d9a3d2e129543b75fe085ff1b066072f87cddd5075152631e7c6dd600cb1633`, respectively,
  throughout this documentation pass.
- Final status/specification inspection confirms Runtime 0015 Complete, CPU 0010G Draft, Engine
  0004 Blocked, Engine 0005–0008 Draft, and no detailed CPU 0010G or Engine 0004–0008 task
  specification. `git diff --check` passed.
- Repository-wide, backend-conformance, and integration suites were not run. The task changes one
  narrow Runtime accessor with no module edge, schedule execution, concrete-backend behavior, or
  ordinary Engine execution change; existing implementation and focused architecture evidence is
  the applicable task tier, and repository-wide coverage remains deferred to Engine 0008 or CI.

## Implementation notes

- Added the sole result-indexed accessor to the existing Runtime `RunResult`. It checks the exact
  leased state before bounds and returns the retained array element without copy, wrapper, cast,
  validity query, transfer, mutation, or ownership change.
- Extended the existing focused test with exact public shape, open/closed bounds, repeated
  identity, alias identity, distinct identity, no-early-close, and unchanged cleanup evidence.
- The documentation pass retained the implementation-authored `RunResult` Javadoc after
  independent review and finalized Runtime/Public API, glossary, task, Runtime master, and
  roadmap wording. It preserved the two prerequisite-planning diffs without editing them.

## Completion summary

- Completed changes: inward Runtime result-indexed borrowed representation access with exact
  open-lease, order, identity, alias, failure-precedence, and non-thread-safe lifetime semantics.
- Files changed or created: exactly the eight Runtime-0015 paths listed under Affected files. The
  shared tree additionally preserves the two pre-existing CPU/Engine prerequisite master-plan
  paths, for ten total uncommitted paths.
- Tests and validation: reused the implementation context's passing 9-test focused result run,
  17-suite/148-test Runtime run, and 1-suite/3-test Runtime architecture run; documentation
  context passed Runtime Javadoc without warnings, generated-page and `javap` inspection,
  executable-token preservation, six-file Markdown validation, exact scope/status/spec absence,
  prerequisite-file hash preservation, and final whitespace checks.
- Documentation-agent review: clean context `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` completed the
  mandatory independent API/Javadoc/planning/example review without rerunning stable Java tests.
- Documentation impact: Runtime API, Public API, and glossary now distinguish the inward borrowed
  representation SPI from ordinary Engine metadata access and state the complete occurrence,
  identity, alias, lifetime, synchronization, failure, ownership, and no-host-semantics boundary.
- Javadoc review: the final `RunResult` type and accessor precisely document borrowing, the exact
  lease lifetime, no close/transfer/retention, alias identity, external synchronization,
  closed-state precedence, bounds, return identity, and absent physical work; generated output
  passed inspection.
- Glossary impact: the Runtime implementation-status and `RunResult` entries now describe the sole
  inward representation accessor while retaining the separate metadata-only ordinary Engine
  definition.
- No-change conclusions: `ARCHITECTURE.md`, focused architecture pages, and ADR 0011 already
  assign Runtime result leasing and backend-owned physical access correctly, so no authority or
  architecture-test rule changed. `BufferRepresentation`, `RunState`, `BoundPublication`, the
  runner, schedule, Prepare, CPU integration, and Engine Java/Javadocs remain accurate because
  representation selection, validity, execution, cleanup, and outward facades did not change.
  Ordinary and advanced Engine results remain metadata/lifecycle-only. Compile, Tensor, and
  Training APIs remain unchanged because the accessor exposes neither graph/Tensor identity nor a
  value or gradient. Config, Trace, Backend Contract, other backends/modules, Gradle/dependencies,
  architecture tests, backend conformance, and integration tests need no edit because no policy,
  trace, module edge, backend behavior, or end-to-end value access changed.
- Unresolved issues: None for Runtime 0015.
- Follow-up required: CPU 0010G remains the next Draft prerequisite; Engine 0004 remains Blocked
  until that CPU boundary is Complete. Engine 0005–0008 remain Draft. No detailed follow-up
  specification was created.

Status: Complete
