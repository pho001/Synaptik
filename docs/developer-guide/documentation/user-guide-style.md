# User guide style

## Purpose

A user guide helps a user complete a concrete Synaptik task. Organize it around the user's goal and observable result rather than the repository's internal module structure.

Apply [General style](general-style.md) and the [example format](example-format.md) alongside this profile.

## Required content

- State the task and the result the user will obtain.
- List prerequisites, supported environment, and required setup.
- Define task-specific terms at first use.
- Provide a runnable example or exact command sequence when the required API or tool exists.
- Show complete inputs, configuration, and starting state.
- Explain steps in execution order and call out irreversible or expensive actions before they occur.
- Show the expected output or observable state, then explain how to interpret it.
- Include common errors with symptoms, likely causes, and corrective actions.
- State limitations, unsupported variants, and links to related tasks or API details.

If the public workflow is planned but not implemented, do not invent a runnable API. Mark the example as conceptual and direct the reader to implementation status.

## Avoid

- beginning with architecture history instead of the user's goal;
- examples that cannot run in the documented environment;
- placeholders presented as literal commands;
- unexplained output;
- internal implementation detail that does not help complete or troubleshoot the task; and
- success-only instructions that omit likely errors.

## Validation

- Execute the documented commands or example in the stated environment when possible.
- Compare actual output with documented output.
- Test at least the most likely documented error or validate it against tests.
- Confirm prerequisites are sufficient for a new user.
- Check links, terminology, and version-sensitive statements.

## Template

```markdown
# <Task-oriented title>

## Outcome

What the user will have or observe.

## Prerequisites

## Inputs and setup

## Steps

1. ...

## Expected result

Exact output or state and its interpretation.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|

## Limitations

## Related tasks and API reference
```
