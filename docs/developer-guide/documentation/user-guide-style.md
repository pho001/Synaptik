# User guide style

## Purpose

A user guide helps a user complete a concrete Synaptik task. Organize it around the user's goal and observable result rather than the repository's internal [module](../../glossary.md#term-module) structure.

Apply [General style](general-style.md) and the [example format](example-format.md) alongside this profile.

A **prerequisite** is something the reader must already have or know. **Configuration** is the
explicit set of choices supplied to the program. An **observable result** is something the reader
can inspect, such as returned values, a saved file or a specific failure. State what each result
demonstrates without treating one successful example as proof of every supported case.

## Required content

- State the task and the result the user will obtain.
- List prerequisites, supported environment, and required setup.
- Define task-specific terms at first use.
- Provide a runnable example or exact command sequence when the required [API](../../glossary.md#term-application-programming-interface-api) or tool exists.
- Show complete inputs, configuration, and starting state.
- Explain steps in execution order and call out irreversible or expensive actions before they occur.
- Show the expected output or observable state, then explain how to interpret it.
- Include common errors with symptoms, likely causes, and corrective actions.
- State limitations, unsupported variants, and links to related tasks or API details.

If the public workflow is planned but not implemented, do not invent a runnable API. Mark the example as conceptual and direct the reader to implementation status.

## Put the user's action before the framework vocabulary

For a saved-model workflow, start with the application loading a model, binding
input values and reading a prediction. Identify which method reads bytes, which
constructs a description and which actually executes it. Explain why any Engine,
session or Arena must stay alive, without implying that a detached host value needs
closing merely because an internal execution result did.

Compare [running models](../../user-guide/running-models.md) for the execution
boundary and [model checkpoints](../../api/model-checkpoint-api.md) for load failure
and format limits. Use those focused destinations rather than asking the user to
infer setup from an index of unrelated APIs.

## Avoid

- beginning with architecture history instead of the user's goal;
- examples that cannot run in the documented environment;
- placeholders presented as literal commands;
- unexplained output;
- internal implementation detail that does not help complete or troubleshoot the task; and
- success-only instructions that omit likely errors.

## [Validation](../../glossary.md#term-validation)

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
