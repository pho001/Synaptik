# Planning Guide

## Purpose

This guide defines how Synaptik coordinates non-trivial implementation work through module master plans and cohesive executable task specifications. The planning system keeps scope, dependencies, validation, decisions, and handoff evidence visible across isolated agentic implementation sessions without making repeated process work the primary cost of a change.

## Authority

[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the authoritative architecture contract. Documents under [`docs/architecture/`](../architecture/) explain that architecture. Documents under `docs/planning/` coordinate implementation and are not authoritative.

Planning documents must not introduce architecture changes silently. If a task cannot be completed without changing architecture, module boundaries, or dependency rules, the implementation agent must stop, record the conflict, and report it for an explicit architecture decision.

## Directory layout

```text
docs/planning/
  README.md
  planning-guide.md
  roadmap.md
  modules/<module>/
    master-plan.md
    tasks/
      <task-id>-<short-title>.md
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

Each project area has one master plan. Detailed executable task specifications live beside it under `tasks/`.

## Progressive planning policy

Create master plans early for every module, backend, extension, and tool so ownership and sequencing remain visible. Create detailed task specifications only for the current implementation frontier or the immediately following frontier.

Do not create detailed task specs far ahead of implementation. Upstream public APIs, module contracts, and validation evidence may change the correct design of downstream work. Future work should remain a concise row in a master plan until its dependencies and constraints are sufficiently stable.

## Package structure planning

Package structure is part of implementation planning, not an incidental implementation decision. Do not default every new type to a module's root package merely because no package plan exists.

Use two levels of package planning:

- the module master plan defines the target package map, the responsibility of each package, and which packages form the intended public surface; and
- each detailed task specification maps every expected production and test type to an existing or proposed package and explains any package it adds or changes.

Plan the module-level map progressively. It should be detailed enough for the current and next implementation frontier without attempting to predict every future internal package. Group types by cohesive responsibility and useful visibility boundaries rather than by file count. A module root package may hold a small, deliberate public facade or closely related foundational contracts, but it must not become the automatic destination for unrelated APIs, implementation helpers, and internal models.

Package-private helpers should live with the contracts they support. Tests should normally mirror the production package when they need package-private access; black-box API tests may use a distinct test package when the task explains why. Avoid generic packages such as `util`, `common`, or `misc` unless the master plan gives them a narrow, stable responsibility.

A task must not become `Ready` until its package impact is explicit. Implementation must not create a different package structure silently. If evidence shows that the planned package placement is unsuitable, update the task and master-plan package map before continuing. Moving an already published or completed contract to another package requires an explicit refactoring or migration task with compatibility impact and validation; it must not be hidden inside unrelated work.

## Task ordering

The task table in each master plan is an ordered implementation queue. Unless the master plan explicitly says otherwise, execute tasks in ascending ID and table order.

At any time, create a detailed task specification only for the first task that is not `Complete`, `Superseded`, or `Cancelled`. Finish and validate that task, update its status and master-plan row, and only then create the specification for the next task.

Do not skip a task silently. Record why it is `Blocked`, `Superseded`, or `Cancelled`, and identify the next valid task. Parallel or out-of-order execution is allowed only when the master plan explicitly records the reason, confirms that dependencies and files do not overlap, and identifies how results will be integrated.

The global [implementation roadmap](roadmap.md) applies the same rule to project areas: finish the active module frontier before advancing to the next module or backend, except for an explicitly recorded parallel exception.

## Task granularity

A task should usually:

- deliver one cohesive capability or one independently reviewable foundation contract;
- touch one module;
- affect approximately 3–12 source and test files;
- have explicit acceptance criteria and validation commands; and
- fit in one isolated agentic implementation session.

Prefer a complete vertical model capability when its semantic contracts, public facade, package-private construction helper, tests, Javadoc, and explanatory API documentation form one coherent decision. Do not split those artifacts into consecutive tasks merely because they are separate files. Split semantic definition from public API only when the semantic contract is an independently uncertain decision, is reused by multiple later capabilities, or can be validated meaningfully on its own.

Avoid tasks named `implement module X`. Prefer focused names such as `implement data type model`, `implement shape model`, `define typed trace envelope`, or `implement binary comparison expressions`.

If a task needs more than 12–18 files, split it unless a documented technical reason makes an atomic larger change safer. File count is a guardrail, not a reason to separate tightly coupled pieces of one capability. Split work that combines model API design with compiler, runtime, preparation, or backend behavior. If a task crosses architecture boundaries, split it along those boundaries or stop and request clarification.

## Validation tiers

Validation effort must be proportional to the scope and risk of the change. Do not run the same successful suite again in another agent unless executable code changed after the recorded run or a concrete failure risk justifies repetition.

### Task validation

Every implementation task runs focused tests while developing and records one final test run for each affected module after executable code stabilizes. For a task limited to `modules/model`, the normal final Java validation is:

```bash
./gradlew :modules:model:test
```

The documentation-focused pass runs final Javadoc generation after it has finalized Javadoc and runs the applicable documentation checks. `git diff --check` must pass on the final combined change. The coordinator may reuse this evidence and normally needs only to inspect the final diff and status.

Reflection, `javap`, bytecode inspection, import scans, and manual API-shape checks belong in a task only when they address a concrete risk that ordinary compilation and tests do not cover. A recurring invariant must be moved into an automated test or reusable test assertion instead of repeated manually in every task.

### Capability checkpoint

A capability checkpoint closes a coherent family of tasks or a foundation-hardening sequence. The relevant master plan must identify the checkpoint. At that point run repository-wide tests, affected architecture tests, final documentation checks, and any cross-module validation that was intentionally deferred from individual tasks.

Use a checkpoint before moving to a new operation family or architecture layer, after a sequence that changes a shared foundational contract, and before a release or merge when CI is not the final gate.

### Repository and CI validation

Run the full repository suite for changes to module dependencies, architecture boundaries, shared Gradle configuration, multiple modules, or another repository-wide contract. CI remains the final independent repository-wide validation gate. A small single-module task does not run the full repository suite merely to duplicate coverage already supplied by its module tests and the next checkpoint.

## Status values

- **Draft** — scope, dependencies, acceptance criteria, or decisions are incomplete.
- **Ready** — the task or plan is actionable, bounded, and has sufficient validation criteria.
- **In progress** — implementation is active; local decisions and evidence must be kept current.
- **Blocked** — a recorded external dependency, missing decision, or architecture conflict prevents progress.
- **Review needed** — implementation is complete enough for review but has not passed final acceptance.
- **Complete** — all acceptance criteria and required validation have passed and the completion summary is final.
- **Superseded** — another linked plan or task replaces this document.
- **Cancelled** — the work will not proceed; the reason is recorded.

Only mark a task `Complete` when its implementation, tests, documentation, validation evidence, and completion summary are all complete. When code or behavior changes, a documentation-focused agent or thread with clean context, separate from the implementation context, must independently review and finalize the affected explanatory documentation, Javadoc, and glossary impact. The separate context works on the same overall branch and change; it does not defer documentation to a later commit or task. It reuses successful implementation-test evidence and does not rerun Java tests unless it changes executable Java behavior or records a concrete reason.

## Master plan format

A master plan defines the stable implementation outline for one project area. It should remain concise and should link to detailed tasks instead of duplicating them.

~~~markdown
# <Module Name> Master Plan

## Goal

## Architecture references

## Scope

## Out of scope

## Module invariants

## Allowed dependencies

## Forbidden dependencies

## Package structure

```text
<base.package>/
  <subpackage>/  <responsibility and intended visibility>
```

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|

## Milestones

## Current status

## Open questions

## Decisions made

## Risks

## Notes
~~~

## Task specification format

A task specification is the executable contract for one isolated implementation session. Replace every placeholder before changing its status to `Ready`.

~~~markdown
# Task <ID>: <Title>

## Status

Draft

## Goal

## Scope

## Out of scope

## Architecture references

## Architecture constraints

## Package impact

Existing packages used:

- ...

Packages added or changed:

- ...

Type placement:

- `<fully.qualified.Type>` — <reason this package owns the type>

## Affected files

Expected:

- ...

## Maximum scope

This task may create or modify at most:

- ...

If more files are needed, stop and propose a follow-up task.

## Acceptance criteria

- ...
- A separate documentation-focused agent pass has finalized affected documentation, Javadoc, and glossary impact in this same overall change.

## Tests / validation

Run:

```bash
./gradlew <module-path>:test
```

Documentation pass:

```bash
./gradlew <module-path>:javadoc
git diff --check
```

Repository-wide validation: deferred to <named capability checkpoint or CI>, unless this task changes a repository-wide contract.

## Dependencies

- ...

## Follow-up tasks

- ...

## Architecture impact

Expected impact: None.

If this task requires architecture changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- <this task file>

Implement this task exactly as specified.
Do not implement out-of-scope items.

After code implementation and module validation, hand the resulting diff and recorded test evidence to a separate documentation-focused agent or thread with clean context. That targeted pass must follow docs/developer-guide/documentation-rules.md and finalize affected documentation, Javadoc, glossary impact, and documentation validation in the same overall change. It must not repeat successful Java tests unless it changes executable behavior or records a concrete reason.

At the end, update this task file with implementation notes, validation evidence including the documentation-agent pass, completion summary, and final status. Do not mark the task Complete before that pass finishes.
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
~~~

## Implementation prompt format

Every implementation prompt must create a separate agentic task or thread with a clean context. It must identify the exact task file and require the agent to read `AGENTS.md`, `ARCHITECTURE.md`, this guide, and the task specification.

Keep the implementation prompt short. The task specification is the single detailed execution contract, so the prompt should not reproduce its complete scope, exclusions, validation matrix, or file list. It must state only the task file, required authoritative reading, the instruction to implement exactly that specification, the stop condition for architecture or scope conflicts, the targeted documentation handoff, and whether commit or push is allowed. It must not rely on remembered conversation context.

The prompt must require a second, documentation-focused agent or thread with clean context after implementation whenever code or behavior changes. That pass reviews the resulting diff and independently finalizes affected explanatory documentation, Javadoc, glossary impact, and documentation validation. Both contexts work in the same overall branch and change, and the task remains incomplete until the documentation pass and its evidence are present. The documentation agent receives the existing module-test evidence and does not reproduce it unless executable code changes after that evidence or a concrete risk requires a rerun.

The documentation-agent handoff must identify the task specification, affected APIs or behavior, implementation diff, architecture constraints, expected documentation, and validation to perform. See the [documentation rules](../developer-guide/documentation-rules.md) for the complete handoff and review workflow.

## Completion summary format

Every completed or blocked implementation session must add a concise summary to its task specification:

~~~markdown
## Completion summary

- Completed changes: ...
- Files changed or created: ...
- Tests and validation: ...
- Documentation-agent review: ...
- Documentation impact: ...
- Javadoc review: ...
- Glossary impact: ...
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
~~~

For incomplete work, use:

```text
Status: Incomplete
Follow-up required: <specific follow-up>
```

Do not mark a task complete merely because its maximum scope or session time was reached.

## Validation evidence

Evidence must record:

- every command executed;
- whether it passed, failed, or was not run;
- the relevant result, including test counts or task outcomes when available;
- any environmental limitation or skipped validation; and
- manual checks required by the acceptance criteria;
- the separate documentation-focused agent or thread used, the files and topics it reviewed, and its result;
- documentation, Javadoc, and glossary impact, including an explicit no-change conclusion with rationale where applicable;
- documentation validation commands and link checks, including their results; and
- confirmation that created and moved types match the package map and task-level type placement.

Evidence may reference a successful command recorded by the implementation pass instead of rerunning it in the documentation or coordination pass. Identify which context ran the command and confirm whether executable code changed afterward. Do not claim independent execution when evidence was reused.

Claims such as `tests pass` without commands and results are insufficient. Keep evidence concise; do not paste entire build logs when a result summary identifies the outcome.

## Follow-up tasks

Do not silently expand a task when implementation uncovers additional work. If the new work is not required to satisfy the current acceptance criteria, add it to the master plan and create a `Draft` follow-up task only when it is at the current or next frontier.

Link follow-up tasks to their origin, record dependencies in both task specs, and identify whether they are required or optional. Do not use a follow-up task to hide incomplete acceptance criteria in the current task.

## Local decisions

Record implementation-level choices that resolve ambiguity without changing architecture. Include the decision, rationale, alternatives considered when material, and affected files or APIs.

Local decisions must remain within the task's architecture constraints. A local decision cannot redefine module ownership or dependency direction.

## Known limitations

Record limitations that remain after implementation, including unsupported cases, deferred validation, portability constraints, or temporary compatibility boundaries. Every limitation must either be accepted by the task specification or linked to a follow-up task.

## Architecture impact

The default expected impact is `None`. Implementation must preserve the existing contract.

If work reveals a required architecture change, stop implementation. Report the conflicting rule, affected modules, and the decision needed. Architecture changes require the coordinated updates defined in `AGENTS.md`; they cannot be approved by editing a planning document.

## When to create a new task

Create a new task when work:

- introduces a separate concept or public contract;
- introduces or restructures a package boundary beyond the current task's approved package impact;
- belongs to another module or architecture layer;
- exceeds the current file or scope limit;
- needs independent validation;
- depends on an unresolved decision; or
- can be completed and reviewed independently.

Keep a change in the current task when it is necessary to deliver one cohesive capability and remains inside its module, package plan, architecture constraints, and documented scope. Separate files or the presence of both a semantic contract and its public model facade are not by themselves reasons to split a task.

## When to update architecture docs

Update architecture documentation when implementation changes or clarifies architecture-related behavior. If an architectural decision or dependency rule changes, follow the coordinated update rules in `AGENTS.md`, including `ARCHITECTURE.md`, focused architecture documentation, an ADR when significant, and architecture tests when dependency rules change.

Do not update architecture documentation merely to make a conflicting implementation appear compliant. Stop and resolve the architecture decision first.

## Templates

Use the [master plan format](#master-plan-format) for each project area and the [task specification format](#task-specification-format) for executable tasks. Copy templates into the relevant directory, replace all placeholders, and keep the master task table synchronized with task status.

Existing master plans may adopt the package-structure section progressively, but the active project's master plan must contain it before the next detailed task becomes `Ready`. Existing task specifications do not need retrospective package sections after completion; every new or materially replanned task must use the current template.
