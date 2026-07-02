# ONNX Master Plan

## Goal

Provide isolated ONNX import, export, and mapping to and from the Synaptik model.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- ONNX import
- ONNX export
- ONNX-to-model mapping
- model-to-ONNX mapping

## Out of scope

- runtime execution
- backend lowering
- kernel selection
- runtime residency

## Module invariants

- ONNX remains outside the runtime hot path.
- Mappings target backend-independent model contracts.

## Allowed dependencies

- modules/model and explicitly required import/export libraries when approved

## Forbidden dependencies

- runtime hot-path internals
- concrete backend modules

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|


## Milestones

- Model mapping contracts
- Importer
- Exporter and compatibility validation

## Current status

Draft.

This extension is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- No open questions recorded.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.

## Risks

- Coupling interchange mapping to runtime or backend implementation details.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
