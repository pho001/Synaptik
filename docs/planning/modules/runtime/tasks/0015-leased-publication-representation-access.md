# Task 0015: Leased [Publication](../../../../glossary.md#term-publication) Representation Access

> **Editorial historical reading key.**
>
> This task opens one inward SPI boundary: a lifecycle consumer calls Runtime
> `RunResult.publicationRepresentation(resultIndex)` and receives the exact representation already
> selected for that publication occurrence. An index is not a Tensor identity or buffer coordinate.
> Aliased occurrences retain separate positions even when their references match. Runtime authorizes
> access through the open whole-state result lease; it neither interprets physical storage nor copies
> a value. The consumer must not close the returned representation or use it beyond that lease.
>
> The current [publication
> contract](../../../../reference/runtime/publication.md#current-prepared-publication-and-result-lease)
> retains this distinction. The current [ordinary Engine
> boundary](../../../../reference/runtime/engine-boundary.md#current-ordinary-engine-boundary)
> separately authenticates outward publication occurrences and orchestrates explicit backend
> materialization into detached host values. Thus the task's metadata-only Engine limitation and
> blocked CPU/Engine follow-ups describe its historical frontier, not today's ordinary API.

## Status

Complete

## Goal

Expose the exact physical representation selected for one successfully published [Runtime](../../../../glossary.md#term-runtime) result
[occurrence](../../../../glossary.md#term-occurrence) to an inward [lifecycle](../../../../glossary.md#lifecycle) consumer while the complete [`RunResult`](../../../../glossary.md#run-result--runresult) [lease](../../../../glossary.md#term-lease) remains open.
This closes the smallest demonstrated prerequisite for [Engine](../../../../glossary.md#term-engine) [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization): Runtime
already snapshots the ordered direct references, including [aliases](../../../../glossary.md#aliasing-and-arena), but currently exposes only
their count and lifecycle.

The boundary remains representation-level Runtime service-provider interface ([SPI](../../../../glossary.md#term-service-provider-interface-spi)), not ordinary
Engine value access. It performs no host copy, conversion, transfer, allocation, [backend](../../../../glossary.md#backend) [lookup](../../../../glossary.md#term-lookup),
or [ownership](../../../../glossary.md#term-ownership-meanings) transfer.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public method to `io.github.pho001.synaptik.runtime.run.RunResult`:

  ```java
  public BufferRepresentation publicationRepresentation(int resultIndex);
  ```

- Require the result lease to be open, then return the exact retained representation reference at
  the dense result index. Repeated access to one occurrence returns the same reference; aliased
  occurrences return the same exact reference.<span data-glossary-navigation="true"> ([lease glossary](../../../../glossary.md#term-lease); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Treat the returned representation as borrowed from the result. The caller must not close it,
  transfer it, retain it beyond result closure, or infer ownership from alias identity.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Preserve [`RunResult`](../../../../glossary.md#run-result--runresult)'s existing whole-[`RunState`](../../../../glossary.md#run-state--runstate) lease, result count, idempotent cleanup, and
  borrowed-input/run-owned-resource rules unchanged.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent); [run glossary](../../../../glossary.md#term-run))</span>
- Add focused [Runtime](../../../../glossary.md#term-runtime) behavior and public-[shape](../../../../glossary.md#shape) coverage.
- Finalize affected Runtime/Public API explanations and glossary wording through a separate clean
  documentation-focused pass after implementation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [Host](../../../../glossary.md#term-host) values, byte encodings, logical-[layout](../../../../glossary.md#layout) traversal, copying, conversion, allocation, [arena](../../../../glossary.md#term-arena) or
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) access, caller-owned output storage, or a [backend](../../../../glossary.md#backend) materializer.
- A representation list, [mutable](../../../../glossary.md#term-mutable) [view](../../../../glossary.md#term-view-meanings), [`RunState`](../../../../glossary.md#run-state--runstate) accessor, [buffer](../../../../glossary.md#term-buffer) or representation coordinate,
  [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [publication](../../../../glossary.md#term-publication) role, [descriptor](../../../../glossary.md#term-descriptor), slot, backend type, or [ownership](../../../../glossary.md#term-ownership-meanings) flag.
- Returning a new wrapper, transferring an individual representation out of the run state, or
  changing which resources result closure releases.<span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Making [`RunResult`](../../../../glossary.md#run-result--runresult) thread-safe, racing representation access with closure, asynchronous access,
  callbacks, pinning, reference counting, or extending a representation's lifetime.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Changes to Prepare, Engine, CPU or another backend, Config, Trace, module dependencies,
  architecture rules, ADRs, schedule construction, publication execution, or Runtime hot-path
  behavior.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [module glossary](../../../../glossary.md#term-module); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Creating a [CPU](../../../../glossary.md#term-cpu) 0010G or [Engine](../../../../glossary.md#term-engine) 0004 implementation or a later detailed Engine task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Runtime ownership, run
  lifecycle, Engine composition, dependency rules, and ADR 0011's backend-owned representation
  boundary.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0009](0009-publication-and-result-schedule-steps.md)
- [Runtime 0010](0010-prepared-runner-and-dynamic-execution.md)
- [Engine master plan](../../engine/master-plan.md)

## Architecture constraints

- Runtime owns the result lease and already retains the selected representations in dense
  publication-occurrence order. Exposing that retained reference from the inward Runtime SPI does
  not move physical access mechanics into Runtime.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lease glossary](../../../../glossary.md#term-lease); [publication glossary](../../../../glossary.md#term-publication); [occurrence glossary](../../../../glossary.md#term-occurrence); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- The returned type is only the existing [nominal](../../../../glossary.md#term-nominal) `BufferRepresentation`; Runtime must not inspect
  or name CPU, Metal, CUDA, <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), [backend](../../../../glossary.md#backend) storage, or logical [Tensor](../../../../glossary.md#tensor) facts.
- Concrete backends continue to own physical access and copying. A later CPU task may consume the
  borrowed reference through its supported integration SPI; ordinary Engine signatures must not.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Engine glossary](../../../../glossary.md#term-engine); [signatures glossary](../../../../glossary.md#term-signature))</span>
- The reference remains governed by the whole result lease. This task must not transfer, split,
  pin, or extend ownership and must not let callers bypass result cleanup.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The accessor is cold result handling after successful schedule execution, not a Runtime hot-path
  action. It performs constant-time indexed access only.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- If implementation requires a new dependency, another public type, a callback abstraction,
  per-representation ownership, concurrency semantics, or an architecture change, stop and
  report the conflict.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [callback glossary](../../../../glossary.md#term-callback); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.runtime.run` remains the owner of result publication order and lease
  lifetime.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [publication glossary](../../../../glossary.md#term-publication); [lease glossary](../../../../glossary.md#term-lease); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>

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

Expected [planning](../../../../glossary.md#term-planning) paths:

- `docs/planning/modules/runtime/tasks/0015-leased-publication-representation-access.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011, focused architecture pages, [Runtime](../../../../glossary.md#term-runtime)
package Javadocs, [Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/[CPU](../../../../glossary.md#term-cpu) source and Javadocs, build files, architecture tests,
[conformance](../../../../glossary.md#term-conformance) tests, and integration tests. Record reasoned no-change conclusions.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify exactly the eight paths listed above: one production file, one focused test,
three explanatory documents, and three [planning](../../../../glossary.md#term-planning) documents. If another source/test [owner](../../../../glossary.md#term-owner),
[dependency](../../../../glossary.md#term-dependency), architecture document, or ninth path is required, stop and replan.

## Exact behavior and failures

- `publicationRepresentation(resultIndex)` first requires the leased run state to be open, using
  the existing stable <span data-glossary-navigation="true"> ([run state glossary](../../../../glossary.md#run-state--runstate))</span>`IllegalStateException("run state is closed")`. Closed-state failure wins
  over index validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- For an open result, a negative index or an index greater than or equal to `resultCount()` throws
  `IndexOutOfBoundsException` with the rejected index in its message.
- On success, return `representations[resultIndex]` exactly. Do not copy, wrap, cast, close, or
  mutate it and do not query Runtime validity again; successful result construction already proved
  that every occurrence published a valid selected representation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- The returned reference is usable only under the existing non-thread-safe result contract. The
  caller must externally prevent access from racing any result/state activity or closure.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>
- Repeated calls before closure return the identical reference. Two result indices that aliased at
  publication return the identical reference; distinct occurrence count and order remain intact.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- After closure begins, every access fails even when the index would otherwise be valid. Metadata
  methods <span data-glossary-navigation="true"> ([Metadata glossary](../../../../glossary.md#term-metadata))</span>`resultCount()` and `isClosed()` keep their current post-close behavior.

## Acceptance criteria

- [`RunResult`](../../../../glossary.md#run-result--runresult) has exactly the one specified new method and no other public/protected surface
  change.
- Open-result access, both index bounds, repeated identity, alias identity, distinct identity,
  closed precedence, and no-close/no-copy behavior are covered by focused tests.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Construction, publication, result count, failure cleanup, and whole-state cleanup tests remain
  unchanged in meaning and pass.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [Runtime](../../../../glossary.md#term-runtime) imports no [Model](../../../../glossary.md#term-model), [Engine](../../../../glossary.md#term-engine), [Prepare](../../../../glossary.md#prepare), or [concrete backend](../../../../glossary.md#concrete-backend) type and gains no [dependency](../../../../glossary.md#term-dependency).
- No [`RunState`](../../../../glossary.md#run-state--runstate), coordinate, storage, [backend](../../../../glossary.md#backend) type, transfer, or [host](../../../../glossary.md#term-host)-value access is exposed.
- Javadocs precisely describe borrowing, lifetime, aliasing, external synchronization, failure
  precedence, and the prohibition on closing or retaining the reference beyond the result.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [lifetime glossary](../../../../glossary.md#term-lifetime); [aliasing glossary](../../../../glossary.md#term-aliasing); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Runtime/Public API and glossary prose stop claiming that the inward Runtime result has no
  representation access, while clearly stating that ordinary Engine still has metadata only.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- A separate clean documentation-focused pass finalizes Javadoc, explanatory documentation,
  glossary impact, and planning evidence before completion.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

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

Also automate or inspect the exact [`RunResult`](../../../../glossary.md#run-result--runresult) public [shape](../../../../glossary.md#shape), [Runtime](../../../../glossary.md#term-runtime) [dependency](../../../../glossary.md#term-dependency) inventory, source
imports, generated [Javadoc](../../../../glossary.md#term-javadoc), changed Markdown links/anchors/headings/fences, trailing whitespace,
final newlines, exact eight-path [scope](../../../../glossary.md#term-scope), and task/master/roadmap status agreement. Confirm no [CPU](../../../../glossary.md#term-cpu)
0010G or [Engine](../../../../glossary.md#term-engine) 0004 detailed task exists.

Repository-wide tests are deferred to Engine 0008/[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one narrow Runtime [SPI](../../../../glossary.md#term-service-provider-interface-spi)
accessor without changing dependencies, schedule execution, [concrete backend](../../../../glossary.md#concrete-backend) behavior, or the
[hot path](../../../../glossary.md#term-hot-path). [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration tests are unchanged.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0009 [publication](../../../../glossary.md#term-publication) and result schedule steps — Complete.
- Runtime 0010 prepared runner and dynamic execution — Complete.
- Runtime 0012 shared-throwable cleanup correction — Complete.
- Engine 0003 — Complete and demonstrates the first real outer consumer, while importing no new
  Runtime contract into its ordinary surface.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0010G: supported [canonical](../../../../glossary.md#term-canonical-form) caller-owned [host](../../../../glossary.md#term-host) [snapshot](../../../../glossary.md#term-snapshot) export through
  `CpuBackendIntegration`, consuming the leased representation without exposing CPU internals.
- [Engine](../../../../glossary.md#term-engine) 0004: [lazy](../../../../glossary.md#term-lazy-evaluation) per-[publication](../../../../glossary.md#term-publication) host [materialization](../../../../glossary.md#materialization) into a [detached](../../../../glossary.md#detached-model-results) [immutable](../../../../glossary.md#term-immutable) bounded
  [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue), after [Runtime](../../../../glossary.md#term-runtime) 0015 and CPU 0010G are Complete.

Do not create either detailed follow-up specification during Runtime 0015 implementation.

## Architecture impact

Expected impact: None.

This realizes the authoritative result-[lease](../../../../glossary.md#term-lease) and [backend](../../../../glossary.md#backend)-owned physical-access split. [Runtime](../../../../glossary.md#term-runtime)
exposes only its existing [nominal](../../../../glossary.md#term-nominal) representation reference while the lease is open; concrete
backends still perform all physical access. No [dependency direction](../../../../glossary.md#java-platform-and-build-tools) or [ownership](../../../../glossary.md#term-ownership-meanings) rule changes.
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

- Direct result-indexed access is smaller and safer than exposing [`RunState`](../../../../glossary.md#run-state--runstate), prepared
  coordinates, a representation list, or a generic callback whose returned value could disguise
  lifetime escape.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The exact representation is borrowed under the existing whole-state lease. Individual output
  ownership is not transferred, so cleanup and aliases remain unchanged.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lease glossary](../../../../glossary.md#term-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Existing non-thread-safe result semantics remain explicit. Engine 0004 must add its own outward
  materialization/close coordination instead of making this inward carrier concurrently mutable.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization); [mutable glossary](../../../../glossary.md#term-mutable))</span>

## Known limitations

- The returned representation is not a [host](../../../../glossary.md#term-host) value and has no shared physical-access methods.
- Callers must use a concrete backend's supported integration operation while the result remains
  open; Runtime itself cannot copy or interpret it.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [operation glossary](../../../../glossary.md#operation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Ordinary Engine callers still receive metadata-only results until CPU 0010G and Engine 0004 are
  implemented.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [metadata glossary](../../../../glossary.md#term-metadata); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context: `01a0a4be-e136-72b1-93a1-7361f64a24cd`.
- The implementation context's focused `RunResultTest` [run](../../../../glossary.md#term-run) passed 9 tests. Its final
  `./gradlew :modules:runtime:test` passed 17 suites and 148 tests with zero failures, errors, or
  skips. Its focused
  `./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest`
  passed 1 suite and 3 tests with zero failures, errors, or skips. Documentation context
  `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` reused this stable evidence because it changed no
  executable Java token and found no stale-evidence reason; it did not rerun Java tests.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- The implementation context's `javap` inspection confirmed exactly one new public method,
  `publicationRepresentation(int)`. Its Runtime dependency/import inventory remained unchanged,
  its exact two-Java-path implementation scope passed, and <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation context `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` independently read the authority,
  lifecycle and boundary explanations, ADR 0011, planning guide/roadmap, Runtime master and tasks
  0009/0010/0012/0014/0015, documentation rules and General/API-Javadoc/Planning/Example profiles,
  final Runtime source/tests and representation/state/publication/runner contracts, Engine
  ordinary/advanced result boundaries, supported CPU integration boundary, Runtime/Public APIs,
  glossary, and the actual diff.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [publication glossary](../../../../glossary.md#term-publication); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `./gradlew :modules:runtime:javadoc` passed with no warnings after final documentation review.
  Generated `RunResult.html` was inspected and contains the borrowed exact-reference lifetime,
  identity/alias, external-synchronization, no-copy/no-validity-query, closed-precedence, bounds,
  parameter, return, and failure contracts.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../../../../glossary.md#term-lifetime); [identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [synchronization glossary](../../../../glossary.md#term-synchronization); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Final `javap -classpath modules/runtime/build/classes/java/main -public
  io.github.pho001.synaptik.runtime.run.RunResult` reports the existing public constructor,
  `resultCount()`, `isClosed()`, and `close()`, plus exactly the one new
  `BufferRepresentation publicationRepresentation(int)` method. The comment-stripped final
  production source hash remained<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
  `56e03e5fb775375ebf0e1c640651fcae7d4b39950338abf9fb8fcfed97995e4f` throughout the
  documentation finalization, confirming no executable Java token changed after implementation.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md docs/api/public-api.md
  docs/glossary.md docs/planning/modules/runtime/tasks/0015-leased-publication-representation-access.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md` passed all six changed
  Markdown files for local targets, generated effective anchors, balanced fences, LF/final
  newlines, and trailing whitespace. Repeated example-subheadings in Runtime API use the
  renderer's deterministic suffixed effective anchors rather than being treated as collisions.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- The final tracked/untracked union contains Runtime 0015's exact eight permitted paths plus only
  the two preserved prerequisite-planning paths,
  `docs/planning/backends/cpu/master-plan.md` and
  `docs/planning/modules/engine/master-plan.md`. Their [SHA-256](../../../../glossary.md#term-sha-256) values remained
  `f78bbb430f5eaa47fbd3565f7ab52e904a166b9ecfffec916215c0f48c4e2e34` and
  `6d9a3d2e129543b75fe085ff1b066072f87cddd5075152631e7c6dd600cb1633`, respectively,
  throughout this documentation pass.
- Final status/specification inspection confirms Runtime 0015 Complete, CPU 0010G Draft, Engine
  0004 Blocked, Engine 0005–0008 Draft, and no detailed CPU 0010G or Engine 0004–0008 task
  specification. `git diff --check` passed.
- Repository-wide, backend-conformance, and integration suites were not run. The task changes one
  narrow Runtime accessor with no module edge, schedule execution, concrete-backend behavior, or
  ordinary Engine execution change; existing implementation and focused architecture evidence is
  the applicable task tier, and repository-wide coverage remains deferred to Engine 0008 or CI.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Added the sole result-indexed accessor to the existing [Runtime](../../../../glossary.md#term-runtime) [`RunResult`](../../../../glossary.md#run-result--runresult). It checks the exact
  leased state before bounds and returns the retained array element without copy, wrapper, cast,
  validity query, transfer, mutation, or ownership change.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Extended the existing focused test with exact public shape, open/closed bounds, repeated
  identity, alias identity, distinct identity, no-early-close, and unchanged cleanup evidence.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The documentation pass retained the implementation-authored `RunResult` Javadoc after
  independent review and finalized Runtime/Public API, glossary, task, Runtime master, and
  roadmap wording. It preserved the two prerequisite-planning diffs without editing them.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: inward Runtime result-indexed borrowed representation access with exact
  open-lease, order, identity, alias, failure-precedence, and non-thread-safe lifetime semantics.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lease glossary](../../../../glossary.md#term-lease); [identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Files changed or created: exactly the eight Runtime-0015 paths listed under Affected files. The
  shared tree additionally preserves the two pre-existing CPU/Engine prerequisite master-plan
  paths, for ten total uncommitted paths.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Tests and validation: reused the implementation context's passing 9-test focused result run,
  17-suite/148-test Runtime run, and 1-suite/3-test Runtime architecture run; documentation
  context passed Runtime Javadoc without warnings, generated-page and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run))</span>`javap` inspection,
  executable-token preservation, six-file Markdown validation, exact scope/status/spec absence,
  prerequisite-file hash preservation, and final whitespace checks.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [scope glossary](../../../../glossary.md#term-scope); [hash glossary](../../../../glossary.md#term-hash))</span>
- Documentation-agent review: clean context `01a0a4c3-6043-7a51-a083-7d3f4ddeec77` completed the
  mandatory independent API/Javadoc/planning/example review without rerunning stable Java tests.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Runtime API, Public API, and glossary now distinguish the inward borrowed
  representation SPI from ordinary Engine metadata access and state the complete occurrence,
  identity, alias, lifetime, synchronization, failure, ownership, and no-host-semantics boundary.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [metadata glossary](../../../../glossary.md#term-metadata); [occurrence glossary](../../../../glossary.md#term-occurrence); [synchronization glossary](../../../../glossary.md#term-synchronization); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [host glossary](../../../../glossary.md#term-host))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: the final [`RunResult`](../../../../glossary.md#run-result--runresult) type and accessor precisely document borrowing, the exact
  lease lifetime, no close/transfer/retention, alias identity, external synchronization,
  closed-state precedence, bounds, return identity, and absent physical work; generated output
  passed inspection.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing))</span>
- Glossary impact: the [Runtime](../../../../glossary.md#term-runtime) implementation-status and `RunResult` entries now describe the sole
  inward representation accessor while retaining the separate metadata-only ordinary Engine
  definition.
- No-change conclusions: `ARCHITECTURE.md`, focused architecture pages, and ADR 0011 already
  assign Runtime result leasing and backend-owned physical access correctly, so no authority or
  architecture-test rule changed. <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>`BufferRepresentation`, [`RunState`](../../../../glossary.md#run-state--runstate), [`BoundPublication`](../../../../glossary.md#bound-publication--boundpublication), the
  runner, schedule, Prepare, CPU integration, and Engine Java/Javadocs remain accurate because
  representation selection, validity, execution, cleanup, and outward facades did not change.
  Ordinary and advanced Engine results remain metadata/lifecycle-only. Compile, Tensor, and
  Training APIs remain unchanged because the accessor exposes neither graph/Tensor identity nor a
  value or gradient. Config, Trace, Backend Contract, other backends/modules, Gradle/dependencies,
  architecture tests, backend conformance, and integration tests need no edit because no policy,
  trace, module edge, backend behavior, or end-to-end value access changed.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [Trace glossary](../../../../glossary.md#trace); [modules glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: None for Runtime 0015.
- Follow-up required: CPU 0010G remains the next Draft prerequisite; Engine 0004 remains Blocked
  until that CPU boundary is Complete. Engine 0005–0008 remain Draft. No detailed follow-up
  specification was created.

Status: Complete
