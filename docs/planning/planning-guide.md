# Planning Guide

## Purpose

This guide coordinates non-trivial Synaptik work through concise master plans and executable task
briefs. Planning keeps order, scope, contracts, validation, and results visible without making
historical reading or repeated process work the dominant cost of a change.

## Authority

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the authoritative architecture root and sole
authority index. Only the six scoped contracts that it explicitly incorporates are normative,
and only within their stated scopes. Other documents under
[`docs/architecture/`](../architecture/) explain that architecture. Documents under
`docs/planning/` coordinate implementation and are not authoritative.

Planning documents must not introduce architecture changes silently. If a task conflicts with an
authoritative contract or requires an unapproved module, ownership, or dependency change, stop and
request an explicit architecture decision.

## Roles and reading

Every participant reads `AGENTS.md`, the current task brief or user request, affected source,
focused tests, and the exact authoritative contract headings named by the brief.

A planner or coordinator that creates, reorders, or materially updates work also reads this guide,
the [roadmap](roadmap.md), and the relevant active master-plan section. Before launching an
executor, the planner verifies that the brief is `Ready`, the task is at an authorized frontier,
and dependencies are satisfied. Put that verification in the brief so the executor does not need
to reconstruct global history.

An executor starts with the root scope index, then uses the brief's exact root and incorporated
scoped-contract headings for each affected module or boundary. Executors do not read
the full roadmap, full planning guide, completed predecessor tasks, task histories, full glossary,
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
numerics, public or supported API, lifecycle or ownership, concurrency, persistence or application
binary interface (ABI), backend capability, dependency direction, cross-module workflow, or a
hot-path contract.

One context is sufficient, and the main coordination context may implement the task. The
implementer performs targeted documentation and Javadoc impact review.

### Class B — Capability or public API

A module capability, public or supported API, user-visible behavior, or lasting module-local
lifecycle contract.

Use a clean implementation context for non-trivial work. An independent documentation or review
context is required when public API, user workflow or behavior, terminology, or a durable contract
changes. Otherwise record a reasoned no-change conclusion.

### Class C — Architecture or high-risk boundary

A module edge or ownership rule, Runtime/Prepare boundary, native ABI or resource lifetime,
concurrency, mixed-backend composition, generated-code or hot-path invariant, or similarly
high-risk cross-module contract.

A clean implementation context and independent targeted review/documentation context are
mandatory. Add ADR, architecture, backend-conformance, integration, structural, or performance
validation where the affected contract requires it.

A private method is not automatically Class A. A behavior-changing bug fix is at least Class B
and may be Class C.

## Execution contexts

The main context may implement Class A and coordinates Classes B and C. Every clean execution
context receives only the compact packet needed to act:

- exact goal;
- relevant files and symbols;
- exact contract links or headings and applicable routing information;
- acceptance criteria;
- validation commands;
- documentation and review impact; and
- required completion status format.

The task brief plus a standard short launcher is the complete execution packet. Do not embed a
custom implementation prompt in every task or rely on remembered conversation.

An executor works toward a complete result. On a contract ambiguity, architecture conflict, or
scope expansion, stop and report the exact issue. Do not invent architecture or silently add work.

## Progressive planning and ordering

Create master plans early enough to make ownership and sequencing visible. Create a detailed task
brief only for the current frontier or immediately following frontier.

Master-plan task tables are ordered queues. Unless an explicit exception records non-overlapping
dependencies and files plus an integration plan, execute tasks in ascending ID and table order.
The roadmap applies the same rule across project areas.

The planner controls status and ordering. Only the planner creates or materially replans `Ready`
work, verifies frontier and dependencies before launch, and records any parallel or out-of-order
exception. Executors receive that verification in the brief and do not re-audit completed global
history.

Do not skip work silently. Record `Blocked`, `Superseded`, or `Cancelled` with a concise reason and
identify the next valid frontier.

## Task granularity

A task normally delivers one cohesive capability or independently reviewable foundation, touches
one module, has explicit acceptance and validation, and fits one focused execution context.

Prefer a complete vertical module capability when semantics, supported facade, tests, Javadoc,
and focused explanatory documentation form one decision. Split work when it crosses architecture
boundaries, combines unrelated responsibilities, depends on an unresolved decision, or becomes too
large to review safely. File count is a guardrail, not a reason to separate tightly coupled work.

Package placement is explicit only where it matters. A master plan maintains the current and next
frontier's package map. A task names every added or moved production type and its owning package,
but does not repeat boilerplate for unchanged packages or files. Never change planned package
structure silently.

## Validation tiers

Validation is proportional to impact and risk. Do not rerun the same successful suite in another
context unless executable code changed, evidence is stale or missing, or a concrete risk requires
an independent rerun.

### Task validation

Run focused tests during development and one final affected-module run after executable code
stabilizes. Validate documentation after final edits. `git diff --check` passes on the combined
change. Add reflection, bytecode, ABI, import, manual API-shape, or performance checks only for a
named risk not covered by ordinary compilation and tests.

### Capability checkpoint

At a master-plan checkpoint, run the identified cross-task tests, affected architecture or
conformance suites, and final documentation checks. Use checkpoints for a completed capability
family, shared foundation, release, or merge where CI is not the only final gate.

### Repository and CI validation

Run repository-wide validation for changes to dependencies, architecture boundaries, shared build
configuration, multiple modules, or another repository-wide contract. CI is the final independent
repository-wide gate. A small task does not run the full repository suite merely to duplicate
focused coverage.

Preserve focused architecture tests for dependency changes, backend-conformance tests for backend
behavior, integration tests for end-to-end behavior, and precise lifecycle/ABI/concurrency
acceptance where applicable.

## Status values

- **Draft** — scope, dependencies, acceptance, contracts, or decisions are incomplete.
- **Ready** — actionable, bounded, at an authorized frontier, and dependency-verified.
- **In progress** — execution is active.
- **Blocked** — a recorded dependency, decision, or architecture conflict prevents progress.
- **Review needed** — implementation is ready for a required review but not final acceptance.
- **Complete** — acceptance, required validation, documentation, review, and result are complete.
- **Superseded** — a linked task replaces the work.
- **Cancelled** — work will not proceed and the reason is recorded.

## Master plan format

A master plan is a concise map, not a duplicate task archive. Keep current ownership, package
direction, ordered work, dependencies, milestones, risks, and frontier visible. Completed rows
carry a one-line result summary and link to the task; evidence stays in the task brief.

~~~markdown
# <Area> Master Plan

## Goal
## Contracts
## Scope and non-goals
## Module invariants and dependencies
## Package map
## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|

## Milestones and current frontier
## Open decisions and risks
~~~

Existing completed master plans and tasks remain valid and need no retrospective rewrite.
Historical/completed plans, evidence, and tasks are not default executor inputs.

## Task brief size guardrails

A task brief targets no more than 200 lines and 15 KB. This is a readability guardrail, not an
excuse for dense prose. If it exceeds 15 KB, record why. If it exceeds 25 KB, split it or record
why one atomic scope is safer than the split.

Remove package/file boilerplate when irrelevant, embedded implementation prompts,
pre-implementation chronicles, context IDs, duplicated predecessor evidence, and verbose
completion logs.

## Task brief format

Omit `Dependencies and follow-up` when there are none. Keep the completed `Result` to about 20
lines.

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

## Files and symbols

- `<path>` — <symbols or responsibility>

## Acceptance criteria

- ...

## Validation

```bash
<exact focused commands>
```

Repository-wide validation: <required, named checkpoint/CI, or reasoned deferral>.

## Dependencies and follow-up

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
documentation/review is mandatory only for public or supported API, user-visible behavior or
workflow, architecture or module boundaries, terminology, cross-module contracts, a lasting
module-local lifecycle or other durable contract, or Class C. Ordinary internal work is reviewed
by the implementer, including affected Javadocs. Record `Documentation impact: none; <reason>`
when appropriate.

Review the glossary with targeted `rg` or search for relevant terms and anchors; never require a
full-glossary read by default. Documentation review does not own executable tests and reuses
current successful evidence unless executable behavior changes or a specific risk justifies a
rerun.

## Evidence and results

The task brief is the single home for compact result evidence. Record the exact command and
outcome, important test counts or skips when meaningful, and limitations. Detailed logs belong in
CI or tool artifacts. Do not paste full build logs.

Master plans and the roadmap carry status, a one-line result, and links. They do not duplicate
task narratives, context IDs, command transcripts, or completion evidence.

## Follow-up and architecture impact

Do not expand a task silently. Add a follow-up only when the work is real and outside current
acceptance; create its detailed brief only when it reaches the current or next frontier. A
follow-up must not hide incomplete acceptance.

The default architecture impact is none. If implementation requires an architecture change, stop
and follow the coordinated architecture process in `AGENTS.md`. Preserve architecture authority,
resource ownership, compile/prepare/run separation, dependency direction, fail-closed capability,
performance and hot-path rules, and required conformance validation.

## Advancing the frontier

Before launching the next task, the planner confirms:

1. the current task has a final result and correct status;
2. required validation and documentation/review are complete;
3. master-plan and roadmap status/link updates are concise and synchronized;
4. the next task is the authorized frontier with satisfied dependencies; and
5. its compact brief is `Ready`.
