# CPU Backend Master Plan

## Goal

Implement CPU capability reporting, backend-owned preparation, kernel routes, storage, workspace, and execution.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- CPU capability provider
- partition lowering, specialization, and fusion
- scalar, Vector API, ASM, OpenBLAS, and specialized routes
- CPU executables, storage, workspace, scheduling, and tracing
- typed, version-controlled, tested route candidate generators and compatible workload-cache
  lookup during prepare

## Out of scope

- global compiler logic
- public Tensor ownership
- engine dependency
- separate CPU route backends

## Module invariants

- Planning selects CPU ownership; CPU prepare selects the route.
- All CPU routes remain inside one concrete backend.
- CPU backend never depends on engine.
- CPU candidate generators return complete valid route-specific configurations; shared tuning
  sees them opaquely.
- Safe CPU heuristics remain correct when tuning is disabled or a compatible cache entry is
  absent.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/runtime
- modules/prepare
- modules/backend-contract
- modules/trace
- backends/openblas-provider

## Forbidden dependencies

- modules/engine

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | CPU capability and scalar correctness baseline | Draft | Stable planning, runtime, prepare, backend-contract, and trace contracts | Establish truthful capability and safe scalar preparation before optimized route search. |
| 0002 | CPU prepared execution, storage, and optimized routes | Draft | 0001, OpenBLAS provider where required | Add CPU execution/storage contracts and validated Vector API, OpenBLAS, specialized, and fused routes without splitting backend ownership. |
| 0003 | Typed CPU route candidate generators and cache compatibility | Draft | 0002, opaque prepare/tuning boundary and artifact versioning | Add colocated typed generators for complete valid route configurations, canonical workload compatibility, and safe heuristic/cache-hit selection without generic knob maps. |


## Milestones

- Capability and scalar baseline
- Prepared execution and storage
- Optimized routes and conformance

## Current status

Draft.

This backend is not yet planned in detail. Detailed task specifications will be created when it becomes the current or next implementation frontier.

## Open questions

- Exact route-specific configuration records, target fingerprints, and candidate-schema versions
  wait for implemented CPU routes and the shared opaque orchestration consumer.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Matrix-multiplication candidates may include supported JDK Vector API species and strategy,
  unroll, tile, parallelism, and OpenBLAS thread configurations derived and pruned from target
  capabilities, workload facts, and budget.
- Scalar, vector, and OpenBLAS are typed route configurations, not booleans in
  `Map<String,Object>`. Operation family selects a generator but does not key one universal
  configuration.

## Risks

- Leaking route selection into planning or splitting CPU routes into false backends.
- Exposing private CPU knobs through string dispatch, reflection annotations, a central registry,
  or a generic configuration language.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
