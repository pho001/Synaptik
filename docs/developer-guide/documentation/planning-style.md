# [Planning](../../glossary.md#term-planning) documentation style

## Purpose

[Planning](../../glossary.md#term-planning) documentation coordinates executable implementation work. It defines [scope](../../glossary.md#term-scope), order, constraints, acceptance, [validation](../../glossary.md#term-validation), and [handoff](../../glossary.md#term-handoff) evidence. It is not a tutorial and is not an [architecture contract](../../glossary.md#architecture-contract).

The authoritative planning format and workflow are in the [Planning Guide](../../planning/planning-guide.md). Apply [General style](general-style.md) only where it improves clarity without turning a task brief into teaching material.

A **[directed acyclic graph](../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../glossary.md#intermediate-representation-dag-and-ssa))** is a set of items joined by one-way relationships with no loop
back to an item. In a plan, “A depends on B” means B must be satisfied before A can start.
An **[authorized frontier](../../glossary.md#term-authorized-frontier)** is the work currently allowed to start after those prerequisites and
scope checks, not every task whose predecessors look finished. **Acceptance criteria** state
observable outcomes; **validation** is the activity that checks them and records evidence.

## Required content

Master plans are concise [dependency](../../glossary.md#term-dependency) maps. They keep [ownership](../../glossary.md#term-ownership-meanings), package direction, an explicit [DAG](../../glossary.md#term-directed-acyclic-graph-dag),
authorized frontiers by independent workstream, integration [metadata](../../glossary.md#term-metadata), status, and risks visible.
Completed rows use a one-line summary and link rather than duplicating task evidence; completed
plans and tasks are historical records, not current authority or default executor reading.

Compact task briefs follow the [planning](../../glossary.md#term-planning) guide and include only:

- title or ID, status, and change class with rationale;
- exact goal and bounded [scope](../../glossary.md#term-scope);
- explicit exclusions;
- exact contract links or headings;
- `Depends on`, `Conflicts with`, `Parallel group`, `Common base revision`, `Integration order`,
  `Integration validation`, and `Shared-document integration owner` metadata;
- affected files and symbols;
- falsifiable acceptance criteria;
- proportional worker and integration [validation](../../glossary.md#term-validation) commands;
- follow-up only when real;
- documentation, shared-document integration ownership, and review impact; and
- a concise result after completion.

Use concrete names, paths, commands, and outcomes. [Target](../../glossary.md#term-target) at most 200 lines and 15 KB. If a brief
exceeds 15 KB, justify it; above 25 KB, split it or record why [atomic](../../glossary.md#resource-transaction-and-rollback) scope is safer. The brief plus
a standard short launcher is the execution packet. Do not embed a per-task implementation prompt
or broaden scoped reading. Change-class risk still determines clean-context and independent-review
isolation.

## Distinguish work assignment from runtime ownership

The coordinator authorizes a bounded task, the implementer changes its assigned
files, and the integration owner joins the results and records the agreed checks.
Those roles explain who may edit or accept work; they do not imply that an agent
must close a Tensor or release a native resource. A handoff here carries the diff,
contract links and evidence, not application data ownership.

For example, two isolated edits may still conflict if both change the same format
or lifecycle promise. Record that semantic conflict even when their files differ.
Follow the [Planning Guide](../../planning/planning-guide.md) for authorization and
[documentation rules](../documentation-rules.md#risk-based-documentation-workflow)
for the targeted independent review boundary.

## Avoid

- tutorial chapters, broad background essays, or speculative implementation detail;
- redefining [module](../../glossary.md#term-module) [ownership](../../glossary.md#term-ownership-meanings) or [dependency](../../glossary.md#term-dependency) rules;
- unverifiable acceptance language such as “works well”;
- hidden [scope](../../glossary.md#term-scope) in implementation notes;
- placeholder sections in a [`Ready`](../../glossary.md#term-ready) task;
- package/file boilerplate when irrelevant;
- pre-implementation chronicles, context IDs, duplicated predecessor evidence, or verbose [logs](../../glossary.md#commit-release-distribution-log-and-seed-meanings);
- embedded per-task implementation prompts;
- repeated full-repository suites for a worker task when the integration owner will run the
  recorded gate;
- duplicate Java-test execution by implementation and review contexts without executable changes;
- manual reflection, [bytecode](../../glossary.md#term-bytecode), or import checks that should be stable automated tests;
- parallel work based only on disjoint files, without disjoint semantic ownership, a stable common
  contract/base, isolated worktrees for concurrent writes, and recorded integration ownership;
- parallel edits to the same API, architecture or lifecycle contract, generated format, or shared
  authoritative document contract; and
- marking work complete without evidence and synchronized status.

## [Validation](../../glossary.md#term-validation)

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

Use the complete [canonical](../../glossary.md#term-canonical-form) template in the [Planning Guide](../../planning/planning-guide.md#task-brief-format). Its outline is:

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

Keep [dependency](../../glossary.md#term-dependency) and integration [metadata](../../glossary.md#term-metadata) even when its values are `None`, `Any`, or `N/A`; omit
only `Follow-up` when empty. Replace every placeholder before setting [`Ready`](../../glossary.md#term-ready).
