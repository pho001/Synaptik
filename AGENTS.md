# AGENTS.md

This file defines mandatory working instructions for AI agents and automated contributors.

These instructions apply to the entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## Required reading

Before making changes related to architecture, module boundaries, dependencies, compiler behavior, runtime behavior, backend behavior, tracing, tensor model, training, documentation, or build structure, read:

- `ARCHITECTURE.md`
- `docs/architecture/current-architecture-plan.md`, if present
- `docs/planning/planning-guide.md` and `docs/planning/roadmap.md`, before creating, updating, or executing implementation plans
- the applicable master plan and task specification under `docs/planning/`, if they exist

`ARCHITECTURE.md` is the authoritative architecture contract.

Files under `docs/` are explanatory unless they are explicitly referenced by `ARCHITECTURE.md`.

## Agent task isolation and completion

Do not perform coding or documentation implementation work directly in the main planning agent context.

The main agent context is for planning, architecture discussion, task decomposition, review, and coordination.

Each concrete coding, refactoring, testing, or documentation task must be executed in a separate agentic task/thread with a clean context.

The separate task/thread must receive only the relevant instructions, files, constraints, and acceptance criteria needed for that task.

Do not rely on the implementation agent remembering prior conversation context unless that context is explicitly included in the task prompt.

When creating a separate task context, include:

- the exact task goal
- relevant files or modules
- applicable constraints from `ARCHITECTURE.md`
- expected output
- tests or validation to run
- whether documentation or architecture tests must be updated

The implementation agent must work toward a complete result and should not intentionally leave the task half-finished.

If the task cannot be fully completed, the implementation agent must clearly state:

- what was completed
- what was not completed
- why it could not be completed
- what follow-up is required
- which files or areas are affected

At the end of every separate task/thread, the implementation agent must provide a completion summary.

The completion summary must include:

- completed changes
- files changed or created
- tests or validation performed
- unresolved issues, if any
- required follow-up, if any
- whether the task is complete or incomplete

Use this completion status format:

```text
Status: Complete
```

or:

```text
Status: Incomplete
Follow-up required: <specific follow-up>
```

A task should only be marked complete when the requested work has been implemented, documented when needed, and validated as far as possible within the task context.

If architectural uncertainty appears during implementation, the implementation agent must stop, explain the uncertainty, and request clarification instead of inventing new architecture.

## Architecture contract

All changes must preserve the architecture contract in `ARCHITECTURE.md`.

If a requested change conflicts with `ARCHITECTURE.md`, stop and explain the conflict before editing code.

Do not duplicate architecture rules in this file. Architecture rules belong in `ARCHITECTURE.md`.

When an architectural decision changes, update all relevant files in the same change:

1. `ARCHITECTURE.md`
2. the relevant document under `docs/architecture/`
3. an ADR under `docs/design/decisions/`, when the decision is significant
4. architecture tests under `testing/architecture-tests/`, when dependency rules change

## Documentation discipline

Documentation changes should keep the following distinction clear:

```text
ARCHITECTURE.md
  authoritative architecture contract

docs/
  explanations, guides, design notes, examples, ADRs

docs/planning/
  non-authoritative implementation plans

AGENTS.md
  agent working instructions
```

When adding or changing architecture-related behavior, update documentation in the same change.

After every code change, review the affected documentation. Update or extend it in the same change when public APIs, behavior, configuration, architecture, module boundaries, workflows, or examples have changed. If no documentation update is needed, state that explicitly in the completion summary.

Every Java code change must be reflected in the Javadoc for the affected API or implementation contract. In the same change, add or update Javadoc whenever behavior, invariants, ownership, lifecycle, side effects, threading, nullability, parameters, return values, or failure modes change. Do not add Javadoc that merely restates the implementation. If the affected Javadoc remains accurate without modification, review it and state that explicitly in the completion summary.

Javadoc must always provide a meaningful, detailed description of the documented type, constructor, or method. Constructor and method Javadoc must document every input with `@param`, including relevant constraints, units, nullability, ownership, or mutation behavior. Every non-`void` method must document its result with `@return`, including result semantics and nullability. Document expected failure conditions with `@throws`. Constructors and `void` methods do not use `@return`.

## Planning discipline

Implementation plans and task specifications live under `docs/planning/`. Before creating, updating, or executing planning tasks, read `docs/planning/planning-guide.md` and `docs/planning/roadmap.md`.

Planning documents are not authoritative architecture contracts. If a planning document conflicts with `ARCHITECTURE.md`, the architecture contract wins and implementation must stop until the conflict is resolved.

Represent non-trivial implementation work as a small task specification under the relevant `tasks/` directory. Task specifications must follow the planning guide and define the goal, scope, exclusions, architecture constraints, affected files, acceptance criteria, validation, dependencies, follow-up tasks, implementation prompt, and completion summary.

Execute tasks in the order listed by the relevant master plan. Create a detailed task specification for the next unfinished task only. Parallel or out-of-order execution is an explicit exception that must be justified and recorded in the master plan.

## Legacy implementation reference

The `legacy/pre-rewrite` branch is a read-only reference for capabilities, observable behavior, tests, and historical context. Implement the new architecture from scratch. Do not copy or import legacy source files, internal package structure, dependency direction, runtime coupling, or implementation shortcuts into the new project. Reproduce only explicitly selected capabilities, expressed through new designs that comply with `ARCHITECTURE.md` and verified by new or adapted tests.

## Testing expectations

When changing module boundaries or dependencies, add or update architecture tests under:

```text
testing/architecture-tests/
```

When changing backend behavior, add or update backend conformance tests under:

```text
testing/backend-conformance/
```

When changing end-to-end behavior, add or update integration tests under:

```text
testing/integration-tests/
```

## Code discipline

Prefer small, focused, readable classes, and split classes that own multiple concepts. Do not create god classes, catch-all managers, broad facades, or vague utilities; place each responsibility in its owning module and layer. Keep public APIs minimal, prefer explicit domain names over generic `Manager`, `Helper`, `Util`, `Processor`, or `Service` names, and avoid unrelated refactors.

Add abstractions and interfaces only for a concrete current need, a real boundary, multiple implementations, a test seam, or an architecture contract. Do not conceal architecture violations behind facades, adapters, registries, or service locators.

## Performance discipline

Treat performance as a design priority, especially on runtime hot paths. Keep code readable without adding avoidable overhead per tensor element, operation, graph node, or execution step, and move expensive decisions to compile or prepare time when possible.

Avoid unnecessary allocation, boxing, reflection, string dispatch, map lookup, synchronization, and virtual indirection in hot paths. Do not obscure code for speculative optimization; document necessary optimizations that reduce readability and add tests or benchmarks when appropriate.

## Change discipline

Prefer small, focused changes.

Do not perform unrelated refactors in the same change.

Do not silently change architecture rules.

Do not leave work intentionally half-finished.

When unsure whether a change violates the architecture, stop and explain the uncertainty before editing.
