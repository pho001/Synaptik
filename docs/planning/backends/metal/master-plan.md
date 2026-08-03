# Metal Backend Master Plan

## Goal

Implement Metal capability, MPSGraph and custom-kernel preparation, storage, native integration, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- Metal capability provider
- MPSGraph and custom-kernel lowering
- Metal executables, storage, workspace, and materialization
- native bridge and typed tracing
- typed, version-controlled, tested MPSGraph/custom-route candidate generators and compatible
  workload-cache lookup during prepare

## Out of scope

- global autograd
- public Tensor ownership
- engine dependency
- training-owned Metal optimizer bridge

## Module invariants

- Metal prepare owns lowering and route selection.
- Planning must select Metal ownership before MPSGraph or a custom Metal kernel can be considered.
- MPSGraph and custom Metal kernels are exclusive to this backend. CPU never calls them as an
  internal Apple optimization route.
- Metal-specific optimizer execution remains in backend prepare or kernels.
- Metal backend never depends on engine.
- Metal candidate generators return complete valid route-specific configurations and remain
  opaque to shared tuning orchestration.
- Safe Metal heuristics remain correct without a compatible tuning result.
- Metal mixed-precision routes prioritize FLOAT16. BFLOAT16 remains a distinct logical type and is
  eligible only through an exact capability-, device-, and operation-gated route.
- Model task 0026 must define FLOAT16 and every affected input, accumulation/intermediate, and
  output contract before Metal advertises FLOAT16. FLOAT32 accumulation is the expected default
  for numerically sensitive 16-bit work unless Model explicitly specifies an exception.
- A logical data type or 16-bit storage representation alone never advertises or selects a Metal
  route. Exact target/operation/numerical filtering precedes route/workload benchmarking, safe
  heuristics, or compatible tuning evidence.

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
| 0001 | Metal capability, storage, and native foundation | Draft | Stable shared planning, runtime, prepare, backend-contract, and trace contracts; Model 0026 before any FLOAT16 claim | Establish truthful Metal capability plus native and resource lifecycle contracts without CPU-owned offload. |
| 0002 | MPSGraph prepared execution route | Draft | 0001 | Add backend-owned MPSGraph lowering, executable creation, storage/materialization integration, and execution only for Metal-owned partitions. |
| 0003 | Custom Metal kernel routes | Draft | 0001–0002 | Add validated custom-kernel lowering and execution for eligible Metal-owned work without making custom kernels a CPU route. |
| 0004 | Typed Metal route candidate generators and cache compatibility | Draft | 0002–0003, opaque prepare/tuning boundary and artifact versioning | Add colocated typed complete-candidate generation and canonical workload compatibility without exposing Metal knobs to planning or shared parameter bags. |


## Milestones

- Capability and bridge
- MPSGraph preparation
- Custom routes, storage, and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented Metal routes and the shared opaque orchestration consumer.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Operation family selects the appropriate Metal candidate generator but is not a universal cache
  key. Model tuning may compare complete plans while Metal retains route and lowering ownership.
- Apple CPU acceleration through Accelerate belongs to the CPU backend. MPSGraph and custom Metal
  kernels remain here and are available only after Planning selects `owner = Metal`; CPU does not
  fall through or offload to this backend internally.
- FLOAT16 is the prioritized Metal mixed-precision type, but only after Model owns its true
  IEEE-754 binary16 semantics. BFLOAT16 remains supported only when exact device and operation
  capabilities admit it; neither route follows from storage width alone.

## Risks

- Moving Metal lowering into shared prepare or training.
- Exposing private Metal candidate fields through generic maps, reflection, or shared string
  dispatch.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
