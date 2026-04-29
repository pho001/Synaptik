# Agent Guidance

This repository uses GSD planning artifacts under `.planning/`.

## Project Context

Read these first for planning or implementation work:

- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- relevant files in `.planning/codebase/`

Current focus is Phase 1: Accelerator Buffer Layout ABI.

## Engineering Rules

- Keep public `Tensor` API logical; backend residency belongs in compile/prepare/execute runtime state.
- Prefer backend-neutral accelerator abstractions over Metal-only or CUDA-only shortcuts.
- Preserve CPU hot-path performance while improving accelerator execution.
- Make fallback visible in traces and benchmark reports.
- Do not commit local benchmark/calibration artifacts unless intentionally updating canonical profiles or fixtures.
- Do not commit `.planning/tmp/` verification scratch files.

## Verification Expectations

Before completing accelerator/runtime changes, run focused tests for the touched area. Use targeted Gradle filters when full `./gradlew test` is too slow because debug benchmark tests may run as part of the default suite.

Common commands:

```bash
./gradlew classes
./gradlew test --tests <TestClassOrPattern>
./gradlew metalTest
```

For Metal native work, build or point to the native shim as required by the relevant tests and docs.

## GSD Workflow

Typical next steps:

```text
$gsd-plan-phase 1
$gsd-execute-phase 1
$gsd-code-review 1
```

Keep planning changes in `.planning/` and implementation changes in source/docs/tests. Commit by topic.
