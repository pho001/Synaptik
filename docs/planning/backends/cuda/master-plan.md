# CUDA Backend Master Plan

## Goal

Implement CUDA capability, backend-owned lowering, kernels, storage, native integration, and prepared execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CUDA capability provider
- partition lowering, specialization, fusion, and route selection
- CUDA executables, storage, workspace, and transfers
- native bridge and typed tracing
- typed, version-controlled, tested CUDA route candidate generators and compatible workload-cache
  lookup during prepare

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- training-owned CUDA optimizer bridge

## Module invariants

- CUDA prepare owns concrete implementation selection.
- CUDA backend never depends on engine.
- Runtime receives only prepared executable contracts.
- CUDA candidate generators return complete valid route-specific configurations and remain
  opaque to shared tuning orchestration.
- Safe CUDA heuristics remain correct without a compatible tuning result.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- modules/engine
- extensions/training implementation dependencies

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | CUDA capability and native foundation | Draft | Stable shared planning, runtime, prepare, backend-contract, and trace contracts | Establish truthful capability and native lifecycle contracts. |
| 0002 | CUDA prepared execution and routes | Draft | 0001 | Add validated lowering, storage, transfers, kernels, and prepared execution. |
| 0003 | Typed CUDA route candidate generators and cache compatibility | Draft | 0002, opaque prepare/tuning boundary and artifact versioning | Add colocated typed complete-candidate generation and canonical workload compatibility without exposing CUDA knobs to planning or shared parameter bags. |


## Milestones

- Capability and native bridge
- Prepared execution and storage
- Kernel coverage and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented CUDA routes and the shared opaque orchestration consumer.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Operation family selects the appropriate CUDA candidate generator but is not a universal cache
  key. Model tuning may compare complete plans while CUDA retains route and lowering ownership.

## Risks

- Leaking concrete CUDA details into planning or runtime.
- Exposing private CUDA candidate fields through generic maps, reflection, or shared string
  dispatch.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
