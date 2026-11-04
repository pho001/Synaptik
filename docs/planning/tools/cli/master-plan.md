# [CLI](../../../glossary.md#term-cli) Master Plan

## Goal

Provide command-line diagnostics for [graph](../../../glossary.md#graph), [partition](../../../glossary.md#partition), scoring, [trace](../../../glossary.md#trace), benchmark, and [model](../../../glossary.md#term-model) workflows.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- command parsing
- [graph](../../../glossary.md#graph) and [partition](../../../glossary.md#partition) dumps
- [trace](../../../glossary.md#trace) and scoring inspection
- benchmark and [model](../../../glossary.md#term-model) commands

## Out of [scope](../../../glossary.md#term-scope)

- [compiler](../../../glossary.md#term-compiler) or [runtime](../../../glossary.md#term-runtime) business logic
- [backend](../../../glossary.md#backend) discovery in the runtime [hot path](../../../glossary.md#term-hot-path)
- [kernel](../../../glossary.md#kernel) implementations

## [Module](../../../glossary.md#term-module) invariants

- [CLI](../../../glossary.md#term-cli) delegates to public [engine](../../../glossary.md#term-engine) and tooling contracts.
- CLI remains an outer adapter.

## Allowed dependencies

- [modules](../../../glossary.md#term-module)/[engine](../../../glossary.md#term-engine) and explicitly required public tool contracts

## Forbidden dependencies

- Production [modules](../../../glossary.md#term-module) depending on [CLI](../../../glossary.md#term-cli) code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Command framework
- Diagnostic commands
- Benchmark and [model](../../../glossary.md#term-model) commands

## Current status

Draft.

This tool is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Reimplementing core logic inside command handlers.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
