# Planning documentation style

## Purpose

Planning documentation coordinates executable implementation work. It defines scope, order, constraints, acceptance, validation, and handoff evidence. It is not a tutorial and is not an architecture contract.

The authoritative planning format and workflow are in the [Planning Guide](../../planning/planning-guide.md). Apply [General style](general-style.md) only where it improves clarity without turning a task brief into teaching material.

## Required content

Master plans are concise dependency maps. They keep ownership, package direction, an explicit DAG,
authorized frontiers by independent workstream, integration metadata, status, and risks visible.
Completed rows use a one-line summary and link rather than duplicating task evidence; completed
plans and tasks are historical records, not current authority or default executor reading.

Compact task briefs follow the planning guide and include only:

- title or ID, status, and change class with rationale;
- exact goal and bounded scope;
- explicit exclusions;
- exact contract links or headings;
- `Depends on`, `Conflicts with`, `Parallel group`, `Common base revision`, `Integration order`,
  `Integration validation`, and `Shared-document integration owner` metadata;
- affected files and symbols;
- falsifiable acceptance criteria;
- proportional worker and integration validation commands;
- follow-up only when real;
- documentation, shared-document integration ownership, and review impact; and
- a concise result after completion.

Use concrete names, paths, commands, and outcomes. Target at most 200 lines and 15 KB. If a brief
exceeds 15 KB, justify it; above 25 KB, split it or record why atomic scope is safer. The brief plus
a standard short launcher is the execution packet. Do not embed a per-task implementation prompt
or broaden scoped reading. Change-class risk still determines clean-context and independent-review
isolation.

## Avoid

- tutorial chapters, broad background essays, or speculative implementation detail;
- redefining module ownership or dependency rules;
- unverifiable acceptance language such as “works well”;
- hidden scope in implementation notes;
- placeholder sections in a `Ready` task;
- package/file boilerplate when irrelevant;
- pre-implementation chronicles, context IDs, duplicated predecessor evidence, or verbose logs;
- embedded per-task implementation prompts;
- repeated full-repository suites for a worker task when the integration owner will run the
  recorded gate;
- duplicate Java-test execution by implementation and review contexts without executable changes;
- manual reflection, bytecode, or import checks that should be stable automated tests;
- parallel work based only on disjoint files, without disjoint semantic ownership, a stable common
  contract/base, isolated worktrees for concurrent writes, and recorded integration ownership;
- parallel edits to the same API, architecture or lifecycle contract, generated format, or shared
  authoritative document contract; and
- marking work complete without evidence and synchronized status.

## Validation

- Validate the task against the current [Planning Guide](../../planning/planning-guide.md).
- Confirm every architecture constraint traces to the contract rather than the plan itself.
- Check the explicit DAG, authorized frontiers, conflict/write scopes, semantic ownership, common
  base, integration order, commands, and status synchronization.
- For parallel groups, confirm contract-first sequencing, isolated write worktrees, one integration
  owner for shared documents, worker-focused/module validation, and one post-integration
  cross-module or full validation.
- Verify that acceptance criteria can be observed or tested.
- Confirm validation evidence records exact commands and outcomes, key counts or skips when
  meaningful, reused evidence, limitations, and documentation/review impact.

## Task-brief template

Use the complete canonical template in the [Planning Guide](../../planning/planning-guide.md#task-brief-format). Its outline is:

```markdown
# Task <ID>: <Title>

## Status
## Change class
## Goal
## Scope
## Non-goals
## Contracts
## Dependencies and integration

- Depends on: `<task IDs or None>`
- Conflicts with: `<task IDs or shared scopes, or None>`
- Parallel group: `<group ID or None>`
- Common base revision: `<revision/branch or N/A>`
- Integration order: `<constraint or Any>`
- Integration validation: `<command/checkpoint>`
- Shared-document integration owner: `<owner or N/A>`

## Files and symbols
## Acceptance criteria
## Validation
## Follow-up
## Documentation and review impact
## Result
```

Keep dependency and integration metadata even when its values are `None`, `Any`, or `N/A`; omit
only `Follow-up` when empty. Replace every placeholder before setting `Ready`.
