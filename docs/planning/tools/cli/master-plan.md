# CLI Master Plan

## Goal

Provide command-line diagnostics for graph, partition, scoring, trace, benchmark, and model workflows.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- command parsing
- graph and partition dumps
- trace and scoring inspection
- benchmark and model commands

## Out of scope

- compiler or runtime business logic
- backend discovery in the runtime hot path
- kernel implementations

## Module invariants

- CLI delegates to public engine and tooling contracts.
- CLI remains an outer adapter.

## Allowed dependencies

- modules/engine and explicitly required public tool contracts

## Forbidden dependencies

- Production modules depending on CLI code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Command framework
- Diagnostic commands
- Benchmark and model commands

## Current status

Draft.

This tool is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Reimplementing core logic inside command handlers.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
