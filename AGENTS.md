# AGENTS.md

This file defines mandatory working instructions for AI agents and automated contributors.

These instructions apply to the entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## Required reading

Before making changes related to architecture, module boundaries, dependencies, compiler behavior, runtime behavior, backend behavior, tracing, tensor model, training, documentation, or build structure, read:

- `ARCHITECTURE.md`
- `docs/architecture/current-architecture-plan.md`, if present

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

AGENTS.md
  agent working instructions
```

When adding or changing architecture-related behavior, update documentation in the same change.

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

## Change discipline

Prefer small, focused changes.

Do not perform unrelated refactors in the same change.

Do not silently change architecture rules.

Do not leave work intentionally half-finished.

When unsure whether a change violates the architecture, stop and explain the uncertainty before editing.
