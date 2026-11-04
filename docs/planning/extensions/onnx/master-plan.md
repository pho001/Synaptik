# [ONNX](../../../glossary.md#term-onnx) Master Plan

## Goal

Provide isolated [ONNX](../../../glossary.md#term-onnx) import, export, and mapping to and from the Synaptik [model](../../../glossary.md#term-model).

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>

## [Scope](../../../glossary.md#term-scope)

- [ONNX](../../../glossary.md#term-onnx) import
- ONNX export
- ONNX-to-[model](../../../glossary.md#term-model) mapping
- model-to-ONNX mapping

## Out of [scope](../../../glossary.md#term-scope)

- [runtime](../../../glossary.md#term-runtime) execution
- [backend](../../../glossary.md#backend) [lowering](../../../glossary.md#lowering)
- [kernel](../../../glossary.md#kernel) selection
- runtime [residency](../../../glossary.md#residency)

## [Module](../../../glossary.md#term-module) invariants

- [ONNX](../../../glossary.md#term-onnx) remains outside the [runtime](../../../glossary.md#term-runtime) [hot path](../../../glossary.md#term-hot-path).
- Mappings [target](../../../glossary.md#term-target) [backend](../../../glossary.md#backend)-independent [model](../../../glossary.md#term-model) contracts.

## Allowed dependencies

- [modules](../../../glossary.md#term-module)/[model](../../../glossary.md#term-model) and explicitly required import/export libraries when approved

## Forbidden dependencies

- [runtime](../../../glossary.md#term-runtime) hot-path internals
- [concrete backend](../../../glossary.md#concrete-backend) [modules](../../../glossary.md#term-module)

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- [Model](../../../glossary.md#term-model) mapping contracts
- Importer
- Exporter and compatibility [validation](../../../glossary.md#term-validation)

## Current status

Draft.

This extension is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current [architecture contract](../../../glossary.md#architecture-contract).
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Coupling interchange mapping to [runtime](../../../glossary.md#term-runtime) or [backend](../../../glossary.md#backend) implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
