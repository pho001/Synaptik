# [CLI](../../../glossary.md#term-cli) Master Plan

## Editorial reading key: an outer adapter is a proposed caller

A command-line interface (CLI) turns a person's command and arguments into calls to existing
public APIs, then formats the returned data. Here “outer adapter” means the proposed command
handlers would call Engine or tooling; they would not define operation semantics, choose backend
eligibility or implement Runtime execution. A partition dump would describe which backend
executes a group of graph operations, not transfer ownership of application memory.

The [real model/live terminal task](tasks/0001-live-model-training-and-terminal.md) is Ready
after verified Tuning 0006; no executable command is claimed until its actual delivery gate.
For an existing read-only diagnostic API rather than a promised CLI, read
[tuning artifact inspection](../../../developer-guide/benchmarking.md#inspecting-current-tuning-artifacts-and-evidence).


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

- [CLI](../../../glossary.md#term-cli) delegates to public NN/Training/Checkpoint/Engine and tooling contracts; existing inward owners remain unchanged.
- CLI remains an outer adapter.

## Allowed dependencies

- Public `extensions/nn` and its exported Model/Training/Checkpoint/Engine contracts, plus explicitly required public tool contracts.

## Forbidden dependencies

- Production [modules](../../../glossary.md#term-module) depending on [CLI](../../../glossary.md#term-cli) code.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Real model training and live terminal](tasks/0001-live-model-training-and-terminal.md) | Ready | Verified Tuning 0006; current public NN/Training/Engine | Real numeric data/model commands and 500 ms ASCII progress; public-owner composition, no core architecture change. |


## Milestones

- Command framework
- Diagnostic commands
- Benchmark and [model](../../../glossary.md#term-model) commands

## Current status

Ready: [0001](tasks/0001-live-model-training-and-terminal.md). Its public-workflow research and
verified Tuning dependency are complete. The authorized implementation remains an outer caller;
no executable command is claimed until real training, prediction and terminal acceptance runs.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Reimplementing core logic inside command handlers.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
