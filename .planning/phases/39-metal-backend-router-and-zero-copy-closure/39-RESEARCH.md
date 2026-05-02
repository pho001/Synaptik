# Phase 39 Research: Metal Backend Router And Zero-Copy Closure

**Phase:** 39 Metal Backend Router And Zero-Copy Closure
**Requirements:** METALROUTER-01, METALROUTER-02, METALROUTER-03
**Status:** Complete

## Current State

- `DefaultBackendSelectionPolicy` chooses a backend-owned partition using `AcceleratorPlanCostModel`, but it does not choose an internal Metal execution route.
- `PreparedMetalExecutable` already computes a prepare-time static transport plan and performs execute-time validation for buffer binding versus tensor-array fallback or CPU fallback.
- The Metal bridge SPI is MPSGraph-centric: `MetalMpsGraphBridge.compile(...)`, `execute(...)`, and `executeBuffers(...)`.
- The native buffer path avoids Java-array copy-back between adjacent Metal regions, but the Objective-C shim still conservatively copies returned MPSGraph result storage into caller output buffers and reports this as `nativeDeviceCopyNs`.
- Coverage reports already expose selected region length, lowered primitive count, buffer binding, tensor-array steps, CPU fallback, CPU materialization, native-device copy time, and target gates.

## Planning Findings

### Router Scope

Phase 39 should add a Metal-internal router, not replace backend selection. Backend selection answers "CPU vs GPU_METAL"; the Metal router answers "inside GPU_METAL, should this region run through MPSGraph, a custom Metal kernel path, tensor-array fallback, or explicit CPU fallback?"

The router should be prepare-time first:

- inputs: lowered manifest, operation family, dtype, layout class, shape/rank, estimated work, native bridge capabilities, buffer ABI support, custom-kernel availability, runtime accelerator policy, historical/calibrated cost factors where available;
- output: stable route decision with selected route, rejected routes, reason codes, estimated costs, and required execute-time validation;
- trace/report: route name and reasons must be visible in `ExecutionStepTrace`, benchmark text/JSON, and coverage summaries.

### Custom Kernel Integration Point

There is no checked-in custom Metal kernel bridge today. Phase 39 should introduce a narrow SPI and route decision surface before adding real kernels:

- `MPS_GRAPH` stays the supported route for existing native coverage.
- `CUSTOM_KERNEL` starts capability-gated unless a scoped kernel is implemented.
- `CPU_FALLBACK`, `TENSOR_ARRAY`, and `BUFFER_BINDING` remain explicit transport/execution paths.

This avoids hardcoding "custom kernel" into MPSGraph classes and gives later phases a safe place to plug in kernels without disturbing existing MPSGraph coverage.

### Zero-Copy / Native Copy Proof

The current result-copy path is intentionally conservative. Phase 39 should not simply set `nativeDeviceCopyNs` to zero. It should either:

1. prove an output-buffer write contract for selected MPSGraph executions and expose `TRUE_OUTPUT_BUFFER_WRITE` evidence, or
2. keep the copy but classify it explicitly as `MPSGRAPH_RESULT_COPY` and gate unexpected regressions.

Safe proof work should use small native and Java tests that seed output buffers with sentinel values, run MPSGraph into provided `MTLBuffer` outputs, and verify whether the returned result storage aliases or writes the provided buffer for supported operation families.

### Cost Evidence

Existing profile-derived accelerator cost factors are backend-selection level. Phase 39 needs route-level cost evidence:

- route dispatch overhead;
- expected MPSGraph native-device result copy cost;
- custom-kernel availability/cost placeholder;
- CPU fallback estimated cost;
- tensor-array copy penalty;
- buffer binding validation/materialization cost.

Local tuning outputs under `profiles/platform/.../tuning/...` must remain unstaged unless intentionally promoting canonical fixtures.

## Proposed Plan Split

1. Router policy and cost evidence model.
2. Custom-kernel SPI and MPSGraph route wiring.
3. Output-buffer/zero-copy proof or explicit copy-classification strategy.
4. Final coverage regression, docs, and milestone audit readiness.

## Validation Strategy

- Unit tests for route decision records and reason codes.
- Prepared execution tests proving MPSGraph route is selected for existing supported Metal regions.
- Native Metal tests for copy classification or output-buffer write proof.
- Coverage/report tests that fail hidden CPU/tensor-array exits and quantify native-device copy.
- Source hygiene and profile artifact hygiene checks.
