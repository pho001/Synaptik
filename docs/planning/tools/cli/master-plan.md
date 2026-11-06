# [CLI](../../../glossary.md#term-cli) Master Plan

<a id="editorial-reading-key-an-outer-adapter-is-a-proposed-caller"></a>
## Reader's key: CLI and TUI are outer callers

A command-line interface (CLI) turns a person's command and arguments into calls to existing
public APIs, then formats the returned data. The interactive TUI obtains the same arguments from
forms and calls the same workflows directly. Neither owns operation semantics, backend eligibility
or Runtime execution. The application owns its input buffers until synchronous calls return.

[0001](tasks/0001-live-model-training-and-terminal.md) and
[0002](tasks/0002-interactive-tui-workbench.md) are complete. Start with the
[actual user guide](../../../how-to/train-with-cli.md) and
[behavior/terminal qualification](../../terminal-interface-validation.md).
For read-only tuning diagnostics, see
[artifact inspection](../../../developer-guide/benchmarking.md#inspecting-current-tuning-artifacts-and-evidence).


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
| 0001 | [Real model training and live terminal](tasks/0001-live-model-training-and-terminal.md) | Complete | Verified Tuning 0006; current public NN/Training/Engine | Real commands, external-directory CPU/native consumers, meaningful tests and actual terminal qualification. |
| 0002 | [Interactive TUI workbench and shared branding](tasks/0002-interactive-tui-workbench.md) | Complete | Integrated 0001 workflows; explicit user scope selection | Real menu/forms, shared workflows, live state/events/safe stop, scrollable inspection and common ASCII/ANSI logo. |


## Milestones

- Command framework
- Diagnostic commands
- Benchmark and [model](../../../glossary.md#term-model) commands

## Current status

The combined CLI/TUI delivery has 89 passing affected behavioral tests and 25 freshly forced
architecture tests. Actual keyboard workflows train/save/predict/inspect, resize, recover from
editable errors and stop safely. Raw Ctrl-C/native INT, NO_COLOR and terminal restoration have
separate real-terminal receipts. See [qualification](../../terminal-interface-validation.md)
for exact scope and platform limits. PyTorch comparisons have a separate [measured report](../../native-performance-comparison.md).

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Reimplementing core logic inside command handlers.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
