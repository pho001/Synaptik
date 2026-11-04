# [Planning](../glossary.md#term-planning) Guide

## Purpose

This guide [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) non-trivial Synaptik work through concise master plans and executable task
briefs. [Planning](../glossary.md#term-planning) keeps order, [scope](../glossary.md#term-scope), contracts, [validation](../glossary.md#term-validation), and results visible without making
historical reading or repeated process work the dominant cost of a change.

## Authority

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the authoritative architecture root and sole
authority index. Only the six scoped contracts that it explicitly incorporates are [normative](../glossary.md#term-normative),
and only within their stated scopes. Other documents under
[`docs/architecture/`](../architecture/) explain that architecture. Documents under
`docs/planning/` [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix) implementation and are not authoritative.

[Planning](../glossary.md#term-planning) documents must not introduce architecture changes silently. If a task conflicts with an
authoritative contract or requires an unapproved [module](../glossary.md#term-module), [ownership](../glossary.md#term-ownership-meanings), or [dependency](../glossary.md#term-dependency) change, stop and
request an explicit architecture decision.

## Roles and reading

Every participant reads `AGENTS.md`, the current task brief or user request, affected source,
focused tests, and the exact authoritative contract headings named by the brief.

A planner or coordinator that creates, reorders, or materially updates work also reads this guide,
the [roadmap](roadmap.md), and the relevant active master-plan section. Before launching an
executor, the planner verifies that the brief is [`Ready`](../glossary.md#term-ready), the task is at one [authorized frontier](../glossary.md#term-authorized-frontier),
its dependencies are satisfied, and its [concurrency](../glossary.md#term-concurrency) and integration [metadata](../glossary.md#term-metadata) are current. Put
that verification in the brief so the executor does not need to reconstruct global history.

An executor starts with the root [scope](../glossary.md#term-scope) index, then uses the brief's exact root and incorporated
scoped-contract headings for each affected [module](../glossary.md#term-module) or boundary. Executors do not read
the full roadmap, full [planning](../glossary.md#term-planning) guide, completed predecessor tasks, task histories, full glossary,
unrelated architecture sections, or prior completion evidence by default. If the applicable
contract is missing or ambiguous, stop rather than assume.

Current source, focused tests, and authoritative current contracts take precedence over historical
task narratives. Read predecessor tasks only for explicitly identified unresolved evidence.

## Directory layout

```text
docs/planning/
  README.md
  planning-guide.md
  roadmap.md
  modules/<module>/
    master-plan.md
    tasks/<task-id>-<short-title>.md
  backends/<backend>/
    master-plan.md
    tasks/
  extensions/<extension>/
    master-plan.md
    tasks/
  tools/<tool>/
    master-plan.md
    tasks/
```

Each project area has one master plan. Detailed briefs live beside it under `tasks/`.

## Change classes

Classify work by actual impact, not source visibility. If uncertain, classify upward.

### Class A — Local

Bounded internal, test, or documentation-only work with no change to observable semantics or
numerics, public or supported [API](../glossary.md#term-application-programming-interface-api), [lifecycle](../glossary.md#lifecycle) or [ownership](../glossary.md#term-ownership-meanings), [concurrency](../glossary.md#term-concurrency), [persistence](../glossary.md#term-persistence) or [application
binary interface](../glossary.md#term-application-binary-interface-abi) ([ABI](../glossary.md#application-binary-interface-and-native-bridge)), [backend capability](../glossary.md#backend-capability), [dependency direction](../glossary.md#java-platform-and-build-tools), cross-[module](../glossary.md#term-module) workflow, or a
hot-path contract.

One context is sufficient, and the main coordination context may implement the task. The
implementer performs targeted documentation and [Javadoc](../glossary.md#term-javadoc) impact review.

### Class B — Capability or public [API](../glossary.md#term-application-programming-interface-api)

A [module](../glossary.md#term-module) capability, public or supported API, user-visible behavior, or lasting module-local
[lifecycle](../glossary.md#lifecycle) contract.

Use a clean implementation context for non-trivial work. An independent documentation or review
context is required when public API, user workflow or behavior, terminology, or a durable contract
changes. Otherwise record a reasoned no-change conclusion.

### Class C — Architecture or high-risk boundary

A [module](../glossary.md#term-module) edge or [ownership](../glossary.md#term-ownership-meanings) rule, [Runtime](../glossary.md#term-runtime)/[Prepare](../glossary.md#prepare) boundary, native [ABI](../glossary.md#term-application-binary-interface-abi) or [resource lifetime](../glossary.md#ownership-borrowing-and-lease),
[concurrency](../glossary.md#term-concurrency), mixed-[backend composition](../glossary.md#backend-composition), generated-code or hot-path [invariant](../glossary.md#term-invariant), or similarly
high-risk cross-module contract.

A clean implementation context and independent targeted review/documentation context are
mandatory. Add [ADR](../glossary.md#term-architecture-decision-record-adr), architecture, [backend](../glossary.md#backend)-[conformance](../glossary.md#term-conformance), integration, structural, or performance
[validation](../glossary.md#term-validation) where the affected contract requires it.

A private method is not automatically Class A. A behavior-changing bug fix is at least Class B
and may be Class C.

## Execution contexts

The main context may implement Class A and coordinates Classes B and C. Every clean [execution
context](../glossary.md#term-execution-context) receives only the compact packet needed to act:

- exact goal;
- relevant files and symbols;
- exact contract links or headings and applicable routing information;
- acceptance criteria;
- [validation](../glossary.md#term-validation) commands;
- documentation and review impact; and
- required completion status format.

The task brief plus a standard short launcher is the complete execution packet. Do not embed a
custom implementation prompt in every task or rely on remembered conversation.

An executor works toward a complete result. On a contract ambiguity, architecture conflict, or
[scope](../glossary.md#term-scope) expansion, stop and report the exact issue. Do not invent architecture or silently add work.

## Progressive [planning](../glossary.md#term-planning) and ordering

Create master plans early enough to make [ownership](../glossary.md#term-ownership-meanings) and sequencing visible. [Model](../glossary.md#term-model) unfinished work
as an explicit [dependency](../glossary.md#term-dependency) [directed acyclic graph](../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../glossary.md#intermediate-representation-dag-and-ssa)). [Prepare](../glossary.md#prepare) detailed briefs for authorized
frontiers and, when their contracts and dependencies are stable, immediate successors. Do not
speculate farther ahead.

Each independent workstream may expose one [authorized frontier](../glossary.md#term-authorized-frontier). Task IDs and table order aid
navigation but do not create dependencies or authorize execution. The planner controls status,
the frontier set, and DAG edges; verifies each task before launch; and records these fields in
every [`Ready`](../glossary.md#term-ready) brief:

- **Depends on** — predecessor task IDs, or `None`;
- **Conflicts with** — task IDs or shared semantic/write scopes that require [serialization](../glossary.md#term-serialization), or
  `None`;
- **Parallel group** — the group identifier, or `None` for serial work;
- **Common base revision** — the stable revision or branch shared by the group, or `N/A`;
- **Integration order** — the required merge order or `Any`;
- **Integration [validation](../glossary.md#term-validation)** — the post-integration command or named [checkpoint](../glossary.md#term-checkpoint); and
- **Shared-document integration [owner](../glossary.md#term-owner)** — the owner, or `N/A`.

Parallel tasks must share a stable contract set and base revision and have both disjoint write
scopes and disjoint semantic ownership. Disjoint files alone are insufficient. Change a shared
contract first, integrate it, and rebase dependent work before parallel implementation.
Serialize tasks that change the same [API](../glossary.md#term-application-programming-interface-api), [architecture contract](../glossary.md#architecture-contract), [lifecycle](../glossary.md#lifecycle) contract, generated
format, or shared authoritative document contract.

Concurrent write tasks [run](../glossary.md#term-run) in isolated Git worktrees. A read-only audit needs no worktree. Assign
one integration owner for shared planning and explanatory documentation; worker tasks do not edit
those shared documents unless the integration plan gives one worker exclusive ownership. The
integration owner applies the recorded order and resolves integration [scope](../glossary.md#term-scope) without silently
changing task contracts.

Do not skip work silently. Record `Blocked`, `Superseded`, or `Cancelled` with a concise reason and
identify the next valid frontier or frontiers.

## Task granularity

A task normally delivers one cohesive capability or independently reviewable foundation, touches
one [module](../glossary.md#term-module), has explicit acceptance and [validation](../glossary.md#term-validation), and fits one focused [execution context](../glossary.md#term-execution-context).

Prefer a complete vertical module capability when semantics, supported [facade](../glossary.md#term-facade), tests, [Javadoc](../glossary.md#term-javadoc),
and focused explanatory documentation form one decision. Split work when it crosses architecture
boundaries, combines unrelated responsibilities, depends on an unresolved decision, or becomes too
large to review safely. File count is a guardrail, not a reason to separate tightly coupled work.

Package placement is explicit only where it matters. A master plan maintains the authorized and
immediately following frontiers' package map. A task names every added or moved production type
and its owning package, but does not repeat boilerplate for unchanged packages or files. Never
change planned package structure silently.

## [Validation](../glossary.md#term-validation) tiers

Validation is proportional to impact and risk. Do not rerun the same successful suite in another
context unless executable code changed, evidence is stale or missing, or a concrete risk requires
an independent rerun.

### Task [validation](../glossary.md#term-validation)

Each worker runs focused tests during development and one final affected-[module](../glossary.md#term-module) [run](../glossary.md#term-run) after
executable code stabilizes. It validates its owned documentation after final edits. Add reflection,
[bytecode](../glossary.md#term-bytecode), [ABI](../glossary.md#term-application-binary-interface-abi), import, manual [API](../glossary.md#term-application-programming-interface-api)-[shape](../glossary.md#shape), or performance checks only for a named risk not covered by
ordinary [compilation](../glossary.md#compiler-engine-and-execution-stage-names) and tests.

### Integration [checkpoint](../glossary.md#term-checkpoint)

After a parallel group is integrated in the recorded order, its integration [owner](../glossary.md#term-owner) runs the named
cross-task, cross-[module](../glossary.md#term-module), architecture, or [conformance](../glossary.md#term-conformance) checks once and validates shared
documentation. `git diff --check` passes on the combined change. Do not repeat successful worker
commands unless integration changed the tested code or a concrete risk requires a rerun.

### Repository and [CI](../glossary.md#term-continuous-integration-ci) [validation](../glossary.md#term-validation)

The integration [owner](../glossary.md#term-owner) runs repository-wide validation once when the integrated change affects
dependencies, architecture boundaries, shared build configuration, multiple [modules](../glossary.md#term-module), or another
repository-wide contract. Otherwise the recorded [checkpoint](../glossary.md#term-checkpoint) or CI may be the final independent
gate. A small task does not [run](../glossary.md#term-run) the full repository suite merely to duplicate focused coverage.

Preserve focused architecture tests for [dependency](../glossary.md#term-dependency) changes, [backend](../glossary.md#backend)-[conformance](../glossary.md#term-conformance) tests for backend
behavior, integration tests for end-to-end behavior, and precise [lifecycle](../glossary.md#lifecycle)/[ABI](../glossary.md#term-application-binary-interface-abi)/[concurrency](../glossary.md#term-concurrency)
acceptance where applicable.

## Status values

- **Draft** — [scope](../glossary.md#term-scope), dependencies, acceptance, contracts, or decisions are incomplete.
- **[Ready](../glossary.md#term-ready)** — actionable, bounded, at an [authorized frontier](../glossary.md#term-authorized-frontier), and [dependency](../glossary.md#term-dependency)-verified.
- **In progress** — execution is active.
- **Blocked** — a recorded dependency, decision, or architecture conflict prevents progress.
- **Review needed** — implementation is ready for a required review but not final acceptance.
- **Complete** — acceptance, required [validation](../glossary.md#term-validation), documentation, review, and result are complete.
- **Superseded** — a linked task replaces the work.
- **Cancelled** — work will not proceed and the reason is recorded.

## Master plan format

A master plan is a concise [dependency](../glossary.md#term-dependency) map, not a duplicate task archive. Keep current [ownership](../glossary.md#term-ownership-meanings),
package direction, [DAG](../glossary.md#term-directed-acyclic-graph-dag) edges, authorized frontiers, integration [metadata](../glossary.md#term-metadata), milestones, and risks
visible. Completed rows [carry](../glossary.md#term-carry) a one-line result summary and link to the task; evidence stays in the
task brief.

~~~markdown
# <Area> Master Plan

## Goal
## Contracts and common base

- Contract set: `<stable authoritative headings>`
- Common base revision: `<revision/branch>`

## Scope and non-goals
## Module invariants and dependencies
## Package map
## Dependency DAG and authorized frontiers

<T1 --> T3; T2 --> T3>

Authorized frontiers: <T1, T2>

## Task list

| ID | Task | Status | Depends on | Conflicts with | Parallel group | Integration order | Integration validation | Intent/result |
|---|---|---|---|---|---|---|---|---|

## Integration ownership and shared documents

- Integration owner: `<owner>`
- Shared documents: `<paths and exclusive owner>`

## Milestones
## Open decisions and risks
~~~

Existing completed master plans and tasks remain valid, need no retrospective rewrite, and are not
current authority or default executor inputs. For an existing active master plan, preserve
completed rows and add DAG, frontier, and integration metadata only to unfinished work before it
becomes [`Ready`](../glossary.md#term-ready) or runs concurrently. Until that transition is recorded, its single currently
listed frontier remains authorized and parallel writes are not authorized.

## Task brief size guardrails

A task brief [targets](../glossary.md#term-target) no more than 200 lines and 15 KB. This is a readability guardrail, not an
excuse for dense prose. If it exceeds 15 KB, record why. If it exceeds 25 KB, split it or record
why one [atomic](../glossary.md#resource-transaction-and-rollback) [scope](../glossary.md#term-scope) is safer than the split.

Remove package/file boilerplate when irrelevant, embedded implementation prompts,
pre-implementation chronicles, context IDs, duplicated predecessor evidence, and verbose
completion [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings).

## Task brief format

Every [`Ready`](../glossary.md#term-ready) brief fills the [dependency](../glossary.md#term-dependency) and integration [metadata](../glossary.md#term-metadata); use `None`, `Any`, or `N/A`
explicitly where applicable. Omit `Follow-up` when there is none. Keep the completed `Result` to
about 20 lines.

~~~markdown
# Task <ID>: <Title>

## Status

Draft

## Change class

Class <A, B, or C> — <impact-based rationale>.

## Goal

## Scope

## Non-goals

## Contracts

- `ARCHITECTURE.md` heading `<exact root heading>` — <applicable global constraint and relative link>
- `docs/architecture/contracts/<owner>.md` heading `<exact scoped heading>` — <applicable boundary>

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: `<task IDs or None>`
- Conflicts with: `<task IDs or shared scopes, or None>`
- Parallel group: `<group ID or None>`
- Common base revision: `<revision/branch for the group, or N/A>`
- Integration order: `<ordering constraint or Any>`
- Integration validation: `<exact command or named checkpoint>`
- Shared-document integration owner: `<owner or N/A>`

## Files and symbols

- `<path>` — <symbols or responsibility>

## Acceptance criteria

- ...

## Validation

Worker validation:

```bash
<exact focused and affected-module commands>
```

Integration/repository validation: <integration-owner command/checkpoint/CI, or reasoned
deferral>.

## Follow-up

- ...

## Documentation and review impact

- <documents/Javadocs/terminology affected or `Documentation impact: none; <reason>`>
- <implementer review or required independent targeted context and why>

## Result

Empty until execution. On completion record:

- completed changes;
- changed files;
- exact validation commands and outcomes, with key counts/skips when meaningful;
- documentation, Javadoc, and targeted glossary impact;
- limitations or unresolved issues;
- follow-up, if any; and
- `Status: Complete`, or `Status: Incomplete` plus a specific follow-up.
~~~

## Documentation and review

Follow the [documentation rules](../developer-guide/documentation-rules.md). Separate clean
documentation/review is mandatory only for public or supported [API](../glossary.md#term-application-programming-interface-api), user-visible behavior or
workflow, architecture or [module](../glossary.md#term-module) boundaries, terminology, cross-module contracts, a lasting
module-local [lifecycle](../glossary.md#lifecycle) or other durable contract, or Class C. Ordinary internal work is reviewed
by the implementer, including affected Javadocs. Record `Documentation impact: none; <reason>`
when appropriate.

Review the glossary with targeted `rg` or search for relevant terms and anchors; never require a
full-glossary read by default. Documentation review does not own executable tests and reuses
current successful evidence unless executable behavior changes or a specific risk justifies a
rerun.

## Evidence and results

The task brief is the single home for compact result evidence. Record the exact command and
outcome, important test counts or skips when meaningful, and limitations. Detailed [logs](../glossary.md#commit-release-distribution-log-and-seed-meanings) belong in
[CI](../glossary.md#term-continuous-integration-ci) or tool [artifacts](../glossary.md#term-artifact). Do not paste full build logs.

Master plans and the roadmap [carry](../glossary.md#term-carry) status, a one-line result, and links. They do not duplicate
task narratives, context IDs, command transcripts, or completion evidence.

## Follow-up and architecture impact

Do not expand a task silently. Add a follow-up only when the work is real and outside current
acceptance; create its detailed brief only when it reaches an [authorized frontier](../glossary.md#term-authorized-frontier) or its immediate
successor. A follow-up must not hide incomplete acceptance.

The default architecture impact is none. If implementation requires an architecture change, stop
and follow the coordinated architecture process in `AGENTS.md`. Preserve architecture authority,
resource [ownership](../glossary.md#term-ownership), [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare)/[run](../glossary.md#term-run) separation, [dependency direction](../glossary.md#java-platform-and-build-tools), [fail-closed](../glossary.md#term-fail-closed) capability,
performance and hot-path rules, and required [conformance](../glossary.md#term-conformance) [validation](../glossary.md#term-validation).

## Advancing frontiers

Before launching a task, the planner confirms:

1. every declared predecessor has a final result and correct status;
2. required predecessor [validation](../glossary.md#term-validation) and documentation/review are complete;
3. master-plan and roadmap status/link updates are concise and synchronized;
4. the task is at an authorized frontier, declared conflicts are inactive, and the recorded common
   base and contracts are still stable; and<span data-glossary-navigation="true"> ([authorized frontier glossary](../glossary.md#term-authorized-frontier))</span>
5. its compact brief is [`Ready`](../glossary.md#term-ready) with complete [dependency](../glossary.md#term-dependency) and integration [metadata](../glossary.md#term-metadata).

Before launching concurrent writes, also confirm isolated worktrees and the parallel group's
integration [owner](../glossary.md#term-owner). Multiple ready frontiers do not broaden worker reading: executors still receive
only their scoped packet, and change-class rules still determine clean-context and independent
review requirements.
