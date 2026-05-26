# Agent Guidance

## Project Context

This repository is a Java tensor and compiled computation graph framework with CPU, Metal, CUDA, and shared accelerator/runtime code. Before planning or implementation work, read the relevant source files, tests, and docs for the area being changed.

Use local planning documents only when they are directly relevant to the task. Do not require a specific planning framework or external workflow tool to understand or modify the codebase.

## Engineering Rules

- Keep public `Tensor` API logical; backend residency belongs in compile/prepare/execute runtime state.
- Prefer backend-neutral accelerator abstractions over Metal-only or CUDA-only shortcuts.
- Preserve CPU hot-path performance while improving accelerator execution.
- Make fallback visible in traces and benchmark reports.
- Do not commit local benchmark/calibration artifacts unless intentionally updating canonical profiles or fixtures.
- Do not commit temporary verification scratch files.

## Verification Expectations

Before completing accelerator/runtime changes, run focused tests for the touched area. Use targeted Gradle filters when full `./gradlew test` is too slow because debug benchmark tests may run as part of the default suite.

Common commands:

```bash
./gradlew classes
./gradlew test --tests <TestClassOrPattern>
./gradlew metalTest
```

For Metal native work, build or point to the native shim as required by the relevant tests and docs.

## Workflow

Keep planning notes in the project’s normal task/planning location when needed, and implementation changes in source/docs/tests. Commit by topic.

## Coding task execution rules

Every coding task must be executed in a dedicated, isolated agent context.

For each coding task, the agent must:

1. Clearly understand the requested change before editing code.
2. Implement only what is necessary to satisfy the task.
3. Avoid introducing unnecessary technical debt.
4. Avoid adding transitional compatibility layers, temporary facades, adapter layers, wrappers, or extra abstractions unless they are strictly necessary for correctness, readability, or long-term maintainability.
5. Prefer direct, simple, explicit code over generalized abstractions that are not yet justified by real use cases.
6. Keep the code readable, maintainable, and easy to reason about.
7. Preserve existing architecture and conventions unless the task explicitly requires changing them.
8. Avoid broad refactors unrelated to the task.
9. Avoid hidden behavior, implicit coupling, duplicated logic, and unnecessary indirection.
10. Remove obsolete code when it is made unnecessary by the change, instead of leaving compatibility paths behind.

After every coding task, the agent must review the implementation against the original assignment.

The final response for every coding task must include:

### What changed

A precise list of files, components, functions, or behaviors that were changed.

### Why it changed

A short explanation of why each change was necessary to satisfy the assignment.

### Validation against the assignment

A checklist or concise explanation confirming how the implementation satisfies the original request.

### Remaining debt or follow-up work

A clear statement of any remaining technical debt, limitations, trade-offs, or follow-up work.

If there is no remaining debt, say explicitly:

> No known remaining technical debt was introduced by this change.

## Technical debt policy

Coding tasks must not create avoidable technical debt.

Do not introduce:

- unnecessary compatibility layers
- transitional facades
- unused abstractions
- speculative extension points
- duplicated logic
- dead code
- temporary code paths without a removal plan
- broad architectural changes unrelated to the task

Compatibility or migration layers are allowed only when they are required by the assignment, required for safe rollout, or clearly necessary to preserve existing behavior. When such a layer is introduced, the agent must document:

- why it is necessary
- what risk it mitigates
- when or how it should be removed

## Code quality expectations

Code must be written for long-term readability and maintainability.

Prefer:

- simple control flow
- clear naming
- explicit behavior
- small focused functions
- minimal necessary abstractions
- consistency with existing project style
- tests or validation appropriate to the change

Do not optimize for cleverness. Optimize for clarity.
