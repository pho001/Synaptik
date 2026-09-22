# Planning documentation style

## Purpose

Planning documentation coordinates executable implementation work. It defines scope, order, constraints, acceptance, validation, and handoff evidence. It is not a tutorial and is not an architecture contract.

The authoritative planning format and workflow are in the [Planning Guide](../../planning/planning-guide.md). Apply [General style](general-style.md) only where it improves clarity without turning a task brief into teaching material.

## Required content

Master plans are concise maps. They keep ownership, package direction, ordered tasks, real
dependencies, status, risks, and the current frontier visible. Completed rows use a one-line
summary and link rather than duplicating task evidence. Historical plans and completed tasks are
not default executor reading.

Compact task briefs follow the planning guide and include only:

- title or ID, status, and change class with rationale;
- exact goal and bounded scope;
- explicit exclusions;
- exact contract links or headings;
- affected files and symbols;
- falsifiable acceptance criteria;
- proportional validation commands;
- dependencies and follow-up only when real;
- documentation and review impact; and
- a concise result after completion.

Use concrete names, paths, commands, and outcomes. Target at most 200 lines and 15 KB. If a brief
exceeds 15 KB, justify it; above 25 KB, split it or record why atomic scope is safer. The brief plus
a standard short launcher is the execution packet. Do not embed a per-task implementation prompt.

## Avoid

- tutorial chapters, broad background essays, or speculative implementation detail;
- redefining module ownership or dependency rules;
- unverifiable acceptance language such as “works well”;
- hidden scope in implementation notes;
- placeholder sections in a `Ready` task;
- package/file boilerplate when irrelevant;
- pre-implementation chronicles, context IDs, duplicated predecessor evidence, or verbose logs;
- embedded per-task implementation prompts;
- repeated full-repository suites for a small single-module change without a recorded risk;
- duplicate Java-test execution by implementation and review contexts without executable changes;
- manual reflection, bytecode, or import checks that should be stable automated tests;
- out-of-order work without recorded justification; and
- marking work complete without evidence and synchronized status.

## Validation

- Validate the task against the current [Planning Guide](../../planning/planning-guide.md).
- Confirm every architecture constraint traces to the contract rather than the plan itself.
- Check package impact, file limits, dependencies, commands, and status synchronization.
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
## Files and symbols
## Acceptance criteria
## Validation
## Dependencies and follow-up
## Documentation and review impact
## Result
```

Omit the dependencies/follow-up section when there are none. Use the canonical planning template
and replace every placeholder before setting `Ready`.
