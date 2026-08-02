# Task 0012: Run-State Shared-Throwable Cleanup

## Status

Complete

## Goal

Resolve audit finding `RUNTIME-CLEANUP-001` with the smallest Runtime-only behavior repair:
`RunState.close()` must preserve its closed-first, idempotent, reverse-order, attempt-all cleanup
contract and rethrow the original primary failure even when two distinct owned resources throw the
same exact `Throwable` object.

The repair closes only the cleanup defect selected after Runtime 0011's `BLOCKING_GAP` verdict. It
does not close the Runtime milestone by itself because the separate documentation-status and
architecture-enforcement findings remain Draft follow-ups.

## Scope

- Change only `RunState.close()` failure aggregation as needed to prevent Java self-suppression
  from aborting cleanup when a later owned resource throws the same exact object as the first
  cleanup failure.
- Preserve the first encountered `RuntimeException` or `Error` by reference identity and rethrow
  that exact object after every owned resource has been attempted.
- Continue attaching later failures with identity different from the primary failure as suppressed
  exceptions in cleanup encounter order. A later occurrence of the primary object is not attached
  to itself because Java forbids self-suppression; it does not replace the primary failure, abort
  traversal, or prevent still-later distinct failures from being suppressed.
- Preserve the existing cleanup traversal exactly: mark closed before callbacks, traverse
  workspaces from last to first, then buffer positions and their representations from last to
  first, skip borrowed buffers, and attempt each owned occurrence once.
- Add one focused regression in `RunStateTest` whose distinct owned resources throw the same exact
  primary failure object and whose remaining resources prove attempt-all traversal. The regression
  must also prove the exact primary identity, suppression of a later distinct failure, reverse
  order, closed state, one attempt per owned resource, borrowed-resource exclusion where included,
  and inert repeated closure.
- Finalize the affected `RunState.close()` Javadoc and the focused Runtime API and glossary cleanup
  wording so repeated primary-failure identity is explicit and does not over-promise an impossible
  self-suppressed entry.
- Synchronize this task, the Runtime master plan, and the roadmap after implementation and the
  required clean documentation pass.

## Out of scope

- `RunStateCreation` rollback behavior, runner cleanup composition, `RunResult` leasing, resource
  ownership, validity, residency, transfer, publication, schedule traversal, or any other Runtime
  behavior
- new public or package-private Java types, methods, overloads, exception types, or packages
- throwable deduplication beyond the required primary-object self-suppression guard; distinct later
  failure objects remain suppressed in encounter order, including repeated occurrences when Java
  permits them
- changing cleanup order, borrowed/run-owned ownership, thread-safety, allocation, storage,
  backend, Prepare, Engine, Config, Trace, or tuning contracts
- `DOCUMENTATION-STATUS-001`, including stale implementation-status text in general architecture
  pages or the architecture-test guide
- `ARCHITECTURE-ENFORCEMENT-001`, including Runtime dependency or hot-path architecture tests
- changes to `ARCHITECTURE.md`, architecture explanations, ADRs, Gradle files, dependency edges,
  architecture tests, backend conformance tests, integration tests, or another module
- stale or unrelated documentation cleanup and any later detailed task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Runtime module responsibilities](../../../../../ARCHITECTURE.md#modulesruntime)
- [Run lifecycle](../../../../../ARCHITECTURE.md#run-lifecycle)
- [Runtime master plan](../master-plan.md)
- [Runtime contract closure audit](../runtime-contract-closure-audit.md)
- [Runtime 0003 run-state foundation](0003-run-state-and-runtime-resource-foundation.md)
- [Runtime 0007 representation creation](0007-representation-creation-and-residency-foundation.md)
- [Runtime 0009 publication and result steps](0009-publication-and-result-schedule-steps.md)
- [Runtime 0010 prepared runner](0010-prepared-runner-and-dynamic-execution.md)
- [Runtime 0011 closure audit](0011-runtime-contract-closure-audit.md)
- [Planning guide](../../../planning-guide.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Public API](../../../../api/public-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Runtime owns each run's resource-lifecycle orchestration and failure cleanup; concrete
  representations continue to own physical release mechanics.
- One active logical run has exactly one mutable `RunState`; concurrent runs retain isolated
  run-owned resources.
- Failure cleanup releases only resources still owned by the run, never borrowed inputs.
- Runtime remains independent of Engine and concrete backend implementations and retains no Model,
  Compiler, Planning, or Prepare object on the hot path.
- The existing architecture contract already authorizes this behavior. This task must not modify,
  reinterpret, or extend authoritative architecture.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.run` — existing owner of `RunState`, cleanup orchestration,
  and its focused tests.

Packages added or changed:

- No package is added or structurally changed.

Type placement:

- `io.github.pho001.synaptik.runtime.run.RunState` — remains the sole production type changed
  because it already owns the affected close lifecycle.
- `io.github.pho001.synaptik.runtime.run.RunStateTest` — remains the focused same-package contract
  test for cleanup order, ownership, failure aggregation, and idempotence.

## Affected files

Expected exact changed paths for the completed task:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/run/RunStateTest.java`
- `docs/api/runtime-api.md`
- `docs/glossary.md`
- `docs/planning/modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review-only documentation whose current high-level wording must remain unchanged unless this task
is formally replanned:

- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/architecture/current-architecture-plan.md`
- `docs/architecture/lifecycle.md`
- `docs/architecture/module-boundaries.md`
- `docs/developer-guide/architecture-tests.md`

## Maximum scope

This task may modify exactly the seven paths listed under expected changed paths: one Runtime
production file, one Runtime test file, two explanatory documentation files, and three planning
files. It creates no Java type and no additional documentation or task file.

If another path, package, public declaration, behavior owner, dependency, architecture rule, or
test project is required, stop and propose a separately authorized replan. Do not absorb Runtime
0013 or 0014 and do not create either detailed specification.

## Exact regression contract

The focused regression must arrange cleanup so that:

1. the first attempted owned resource throws one `RuntimeException` or `Error` instance;
2. a distinct later owned resource throws that same exact instance;
3. at least one owned resource is still attempted after the repeated-identity failure;
4. a later distinct failure is available to prove ordinary suppression still works; and
5. at least one final non-throwing owned resource is attempted to prove traversal completes.

The test must assert all of the following:

- the thrown object is the exact first failure by `assertSame`;
- the repeated identical failure does not appear as a self-suppressed entry;
- the later distinct failure appears once in the primary's suppressed array in encounter order;
- close callbacks occur in the existing workspace-reverse then owned-buffer-reverse order;
- every owned occurrence is attempted exactly once despite both earlier failures;
- the state is closed before callbacks and remains closed after the failure;
- a second `close()` performs no callback and throws no remembered failure; and
- borrowed buffers, if present in the fixture, remain unclosed.

The existing distinct-failure and first-`Error` tests remain unchanged and passing. Do not replace
their coverage with the new regression.

## Acceptance criteria

- `RunState.close()` never invokes `Throwable.addSuppressed` with the primary failure as both
  receiver and argument.
- Distinct resources throwing the same exact primary object cannot interrupt remaining cleanup.
- Closed-first state, exact reverse traversal, attempt-all behavior, borrowed exclusion,
  idempotence, original primary identity, and distinct later suppression all satisfy the exact
  regression contract.
- Runtime exception and error behavior remains symmetric and the existing first-error coverage
  passes.
- No list, set, map, reflection, synchronization, new allocation, or new abstraction is introduced
  solely to aggregate cleanup failures; the repair remains a constant-space identity guard within
  the existing cleanup mechanism.
- The public Java surface, packages, dependencies, build, and all non-cleanup Runtime semantics are
  unchanged.
- `RunState.close()` Javadoc, Runtime API, and glossary explicitly distinguish attempt-all cleanup
  from the impossible self-suppression of a repeated primary object.
- The clean documentation pass records reasoned no-change conclusions for Public API, focused
  Runtime architecture text, stale general architecture status, architecture-test enforcement,
  RunState creation, runner/result behavior, backend guide, other modules, Gradle, conformance,
  and integration.
- Runtime 0001-0011 remain Complete and their completed history is preserved. Runtime 0012 becomes
  Complete only after all evidence lands; Runtime 0013 and 0014 remain Draft rows without detailed
  specifications, and the Runtime milestone remains open.
- Runtime 0012 owns exactly the seven authorized paths. The final shared working tree contains
  those seven paths plus the preserved Runtime-0011-only task and durable audit, for exactly nine
  legitimate paths. Markdown links, anchors, fences, final newlines, status ordering, exact scope,
  and whitespace checks pass.

## Tests / validation

Validation tier: task validation. This is one focused Runtime behavior repair with no dependency,
build, architecture, or multi-module change.

Focused development and regression validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.run.RunStateTest
```

Exactly one final affected-module command after executable Java stabilizes:

```bash
./gradlew :modules:runtime:test
```

The separate documentation-focused pass reuses that successful Runtime test evidence unless it
changes executable Java behavior or records a concrete reason to repeat it. After final Javadoc
and documentation edits, run:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md docs/glossary.md \
  docs/planning/modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md \
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md
git diff --check
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check local links and anchors, unique effective anchors, fences, final
newlines, and trailing whitespace for all changed Markdown.

Final scope and status validation must also run:

```bash
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
rg -n '^Ready$|^Complete$|^Draft$|\| (Ready|Complete|Draft) \|' \
  docs/planning/modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md \
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md
rg --files docs/planning/modules/runtime/tasks | sort
```

Inspect the final diff to confirm the seven 0012-owned paths plus the two preserved 0011-only
paths, the required same-identity guard and regression assertions, preserved Runtime 0001-0011
history, one synchronized 0012 status, Draft 0013/0014 rows, and absence of later task files. Also
inspect generated `RunState` Javadoc for the final close wording.

Repository-wide and architecture validation is deferred to continuous integration or the later
Runtime milestone re-closure after the remaining 0013/0014 findings are resolved. Architecture
tests are deliberately not run or changed here because 0014 owns their missing enforcement.
Backend conformance and integration tests are inapplicable because this task changes no concrete
backend and no public Engine execution path.

## Mandatory clean documentation handoff

After the final Runtime test, hand this task, the actual diff, the exact regression evidence, the
seven-path ceiling, and the architecture constraints to a separate documentation-focused agent or
thread with clean context. That pass must read `AGENTS.md`, `ARCHITECTURE.md`, the documentation
rules, General/API-Javadoc/Planning profiles, final `RunState` source/test/Javadoc, Runtime API,
focused Public API and Runtime architecture text, glossary, task/master/roadmap, and the actual
diff.

The documentation pass must finalize `RunState.close()` Javadoc, Runtime API, glossary, and
planning evidence; inspect the generated `RunState` page; validate Markdown/scope/status; and
record exact commands and results. It must not repeat successful Java tests unless executable Java
changes after the recorded result or a concrete risk is documented.

## Dependencies

- Runtime 0003 — Complete; defines the cleanup ordering, ownership, idempotence, and failure
  aggregation contract.
- Runtime 0007, 0009, and 0010 — Complete; preserve creation/rollback, result leasing, and runner
  cleanup composition without modification.
- Runtime 0011 — Complete audit with `RUNTIME-CLEANUP-001` selected as the next bounded remediation.
- Existing Runtime dependencies and the Java 26 build contract — unchanged.

Runtime 0013, Runtime 0014, Prepare 0003, concrete backends, Engine, Trace, Config, conformance,
integration, and tuning are not dependencies for this repair.

## Follow-up tasks

- Runtime 0013 — Draft row only: correct `DOCUMENTATION-STATUS-001` in a separately authorized
  documentation scope without changing authoritative architecture.
- Runtime 0014 — Draft row only: add focused Runtime dependency/hot-path enforcement for
  `ARCHITECTURE-ENFORCEMENT-001` without changing architecture rules.
- Reassess Runtime milestone closure only after 0012-0014 are complete and their evidence is
  synchronized.

Do not create a detailed specification for either follow-up in this task.

## Architecture impact

Expected impact: None.

This task makes current implementation satisfy the already documented Runtime cleanup lifecycle.
It changes no ownership, dependency, module boundary, public surface, or authoritative rule. If
implementation reveals that an architecture change or broader ownership decision is required,
stop and report the conflict without editing architecture.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository. Do not commit or
push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md. Read the directly
referenced Runtime source, tests, audit finding, and documentation required by that specification.

Implement Runtime 0012 exactly as specified. Preserve all existing uncommitted work and do not
implement out-of-scope documentation-status or architecture-enforcement follow-ups. Stop and
report any architecture or maximum-scope conflict instead of expanding the task.

After Java implementation and Runtime validation, hand the resulting diff and exact test evidence
to a separate documentation-focused agent or thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize the affected Javadoc,
explanatory documentation, glossary impact, planning evidence, and documentation validation in the
same overall change, and avoid repeating successful Java tests unless executable behavior changes.

At the end, update the task's local decisions, known limitations, validation evidence,
implementation notes, completion summary, and final status. Do not mark the task Complete before
the documentation pass and every exact-scope/status gate pass.
```

## Local decisions

- Use one reference-identity comparison in each existing cleanup loop. This is the smallest
  constant-space repair and changes neither traversal nor ordinary distinct-failure suppression.
- Treat only recurrence of the exact primary object specially. Equal messages, types, causes, or
  stack traces do not deduplicate distinct failure objects.
- Retain the implementation draft of `RunState.close()` Javadoc unchanged after independent
  review because it already documents order, ownership, attempt-all behavior, primary identity,
  distinct suppression, repeated-primary exclusion, idempotence, threading, and both unchecked
  failure categories precisely.
- Account for task scope separately from shared-tree scope: 0012 owns seven paths, while the two
  untracked 0011 audit paths are legitimate preserved pre-existing work.

## Known limitations

- Runtime 0013 and 0014 remain accepted Draft follow-ups without detailed specifications. Their
  stale general architecture status and missing Runtime architecture-test enforcement findings
  keep the Runtime milestone open but are not limitations of this bounded cleanup repair.
- This repair deliberately does not deduplicate later distinct failure objects; Java permits the
  same distinct object to be suppressed according to each later occurrence.

## Validation evidence

- Implementation context `019fbef9-a9f8-7392-a4a9-8d8609d8035c` changed only `RunState.java` and
  `RunStateTest.java`. Its focused command
  `./gradlew :modules:runtime:test --tests io.github.pho001.synaptik.runtime.run.RunStateTest`
  passed 16 tests. Its exactly one final `./gradlew :modules:runtime:test` passed 17 suites and
  144 tests with zero failures, errors, or skips. Clean documentation context
  `019fbefd-f12e-7450-b554-81a816c3e6b8` inspected
  the final XML evidence and confirmed the same counts. No executable Java changed afterward, so
  this pass correctly did not repeat either Java test command.
- Clean documentation context `019fbefd-f12e-7450-b554-81a816c3e6b8` independently reviewed the
  architecture contract,
  documentation rules and General/API-Javadoc/Planning profiles, planning guide, Runtime 0011
  task and durable audit, final `RunState` source and 16-test contract suite, current/generated
  Javadoc, Runtime/Public APIs, focused Runtime architecture text, glossary, Runtime master plan,
  roadmap, and actual diff.
- `./gradlew :modules:runtime:javadoc` passed with `BUILD SUCCESSFUL`; generated
  `modules/runtime/build/docs/javadoc/io/github/pho001/synaptik/runtime/run/RunState.html` states
  that later distinct failures are suppressed, the same exact primary object is not
  self-suppressed or substituted, remaining cleanup continues, and repeated close is inert.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md docs/glossary.md
  docs/planning/modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md
  docs/planning/modules/runtime/master-plan.md docs/planning/roadmap.md` passed explicit five-file
  local-link, anchor, unique-effective-anchor, fence, final-newline, and trailing-whitespace
  validation.
- `git diff --check` passed with no output. The tracked/untracked union and `git status --short`
  contain exactly nine paths: the seven 0012-owned paths plus preserved 0011-only
  `docs/planning/modules/runtime/tasks/0011-runtime-contract-closure-audit.md` and
  `docs/planning/modules/runtime/runtime-contract-closure-audit.md`.
- Final source/diff inspection confirmed the identity guard in both cleanup loops, the exact
  same-`Throwable` regression and all required assertions, no public-surface/package/dependency/
  build change, unchanged completed Runtime 0001-0011 history, synchronized Complete status for
  0012, Draft 0013/0014 rows, no 0013/0014 specification, and no later Runtime task file.
- Reasoned no-change conclusions: Public API and focused Runtime architecture already state the
  correct high-level reverse attempt-all lifecycle and need no edge-case duplication; the stale
  general architecture status and missing architecture-test enforcement remain exclusively owned
  by Draft 0013/0014; `RunStateCreation` already guards rollback self-suppression and is unaffected;
  runner/result behavior merely delegates cleanup and retains its existing ownership contract;
  backend guidance and other modules own physical mechanics rather than failure aggregation;
  Gradle and dependency edges are unchanged; backend conformance is inapplicable without a
  concrete backend behavior change; integration is inapplicable without a public Engine path.

## Implementation notes

- Added `firstFailure != failure` before `addSuppressed` in both existing `RunState.close()`
  cleanup loops. No other executable statement changed.
- Added one regression using distinct owned resources that throw the same primary object, a later
  distinct failure, remaining non-throwing resources, and one borrowed resource. It proves exact
  identity, suppression, order, closed-first state, attempt counts, borrowed exclusion, and inert
  repeated close.
- Finalized Runtime API and glossary wording and synchronized this task, the Runtime master plan,
  and roadmap. Preserved both 0011-only files without modification and created no later task spec.

## Completion summary

- Completed changes: repaired shared-primary throwable cleanup without changing traversal,
  ownership, public surface, or any other Runtime behavior; added the exact regression; finalized
  Javadoc review, Runtime API, glossary, and planning status/evidence.
- Files changed or created: exactly seven Runtime-0012-owned paths—`RunState.java`,
  `RunStateTest.java`, Runtime API, glossary, this task, Runtime master plan, and roadmap. The final
  shared tree also preserves the two pre-existing Runtime-0011-only audit paths, for nine total.
- Tests and validation: reused the implementation context's successful 16-test focused run and
  one final 17-suite/144-test Runtime run; documentation context
  `019fbefd-f12e-7450-b554-81a816c3e6b8` passed Runtime Javadoc,
  generated close-page inspection, explicit five-file Markdown validation, exact scope/status/
  history/later-spec checks, and `git diff --check` without repeating Java tests.
- Documentation-agent review: clean documentation context
  `019fbefd-f12e-7450-b554-81a816c3e6b8` completed the mandatory independent review and
  finalization.
- Documentation impact: Runtime API, glossary, task, master plan, and roadmap now state the exact
  repeated-primary identity behavior and synchronized completion status; Public API and focused
  architecture remain accurate without modification.
- Javadoc review: the existing implementation draft is accurate and was retained unchanged;
  generated output was refreshed and inspected.
- Glossary impact: the `RunState` entry now distinguishes attempt-all cleanup from impossible
  self-suppression of the repeated primary object.
- Unresolved issues: Runtime 0013 and 0014 remain Draft follow-ups from the 0011 audit; the Runtime
  milestone remains open.
- Follow-up required: complete Runtime 0013 and 0014 in order under separately authorized tasks.

Status: Complete
