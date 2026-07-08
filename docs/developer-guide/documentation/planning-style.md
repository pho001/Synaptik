# Planning documentation style

## Purpose

Planning documentation coordinates executable implementation work. It defines scope, order, constraints, acceptance, validation, and handoff evidence. It is not a tutorial and is not an architecture contract.

The authoritative planning format and workflow are in the [Planning Guide](../../planning/planning-guide.md). Apply [General style](general-style.md) only where it improves clarity without turning a task specification into teaching material.

## Required content

Master plans must keep ownership, package direction, ordered tasks, dependencies, status, risks, and current frontier visible. Detailed task specifications must follow the planning guide and include:

- exact goal and bounded scope;
- explicit exclusions;
- architecture references and constraints;
- package and type placement;
- expected affected files and maximum scope;
- falsifiable acceptance criteria;
- exact validation commands at the appropriate task, capability-checkpoint, or repository tier;
- dependencies and follow-up work;
- architecture impact;
- a self-contained implementation prompt;
- local decisions, known limitations, evidence, implementation notes, and completion summary; and
- the separate documentation-agent pass when code or behavior changes.

Use concrete names, paths, commands, and outcomes. A `Ready` task must be executable by a clean-context agent without relying on remembered conversation. Keep its implementation prompt concise and put detailed execution rules in the task specification once.

## Avoid

- tutorial chapters, broad background essays, or speculative implementation detail;
- redefining module ownership or dependency rules;
- unverifiable acceptance language such as “works well”;
- hidden scope in implementation notes;
- placeholder sections in a `Ready` task;
- repeated full-repository suites for a small single-module change without a recorded risk;
- duplicate Java-test execution by implementation and documentation agents without executable changes;
- manual reflection, bytecode, or import checks that should be stable automated tests;
- out-of-order work without recorded justification; and
- marking work complete without evidence and synchronized status.

## Validation

- Validate the task against the current [Planning Guide](../../planning/planning-guide.md).
- Confirm every architecture constraint traces to the contract rather than the plan itself.
- Check package impact, file limits, dependencies, commands, and status synchronization.
- Verify that acceptance criteria can be observed or tested.
- Confirm validation evidence records commands, results, justified manual checks, reused evidence, documentation review, checkpoint deferrals, and limitations.

## Task-specification template

Use the complete canonical template in the [Planning Guide](../../planning/planning-guide.md#task-specification-format). Its working outline is:

```markdown
# Task <ID>: <Title>

## Status
## Goal
## Scope
## Out of scope
## Architecture references and constraints
## Package impact
## Affected files and maximum scope
## Acceptance criteria
## Tests / validation
## Dependencies and follow-up tasks
## Architecture impact
## Implementation prompt
## Local decisions
## Known limitations
## Validation evidence
## Implementation notes
## Completion summary
```

Do not copy this abbreviated outline when creating a task; use the canonical planning template and replace every placeholder before setting `Ready`.
