---
phase: 12-fused-gpu-region-execution
plan: "04"
status: complete
completed: 2026-04-30
requirements: [GPUFUSE-01, GPUFUSE-02, GPUFUSE-03, GPUFUSE-04]
---

# 12-04 Summary: Reduction Adjacent And Compound Docs Closure

## Outcome

Phase 12 final verification is complete.

- `LINEAR_BIAS_ACTIVATION` remains covered by the focused accelerator/Metal/CUDA lowerer gates.
- `ELEMENTWISE_CHAIN` remains covered by prepared execution trace metadata and buffer-binding residency tests.
- `REDUCTION_ADJACENT` candidates are explicitly recognized and rejected with stable reason detail for the Phase 12 minimal subset.
- `Operation.OpType.FUSED remains CPU-only`; Metal and CUDA planner diagnostics reject synthetic `FUSED` nodes with `CPU_FUSED_OPERATION_UNSUPPORTED`.
- GPU compound lowering remains independent of CPU fused ASM/vector internals; no accelerator, Metal, or CUDA production package imports `backend.cpu.fused`.

## Changes

- Added reduction-adjacent planner diagnostic prefixes for Metal and CUDA unsupported reduction/normalization candidates.
- Added regression tests for `LAYER_NORM`/`RMS_NORM` compound rejection, Metal/CUDA `FUSED` rejection, and reduction-adjacent rejection trace evidence.
- Updated GPU lowering, optimizer, compute-flow, and development docs to describe GPU compound region lowering, backend-specific Metal/CUDA coverage, logical public Tensor semantics, and device residency in `ExecutionState` / `DeviceBufferBinding`.

## Verification

| Command | Result |
|---|---|
| `./gradlew classes` | passed |
| `./gradlew test --tests backend.accelerator.lowering.* --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | passed with quoted shell pattern: `--tests 'backend.accelerator.lowering.*'` |
| `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | passed |
| `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` | passed |
| `./gradlew metalTest` | passed |
| `./gradlew buildCudaGraphShim cudaTest` | capability-skipped / unavailable locally; Gradle task completed successfully with `buildCudaGraphShim SKIPPED` and `cudaTest SKIPPED` |
| `rg -n "import backend.cpu.fused" src/main/java/backend/accelerator src/main/java/backend/metal src/main/java/backend/cuda` | passed: no matches |
| `git status --short` | confirmed local `profiles/platform/.../tuning/abc/*` files remained unstaged |

The first unquoted shell form of the accelerator-lowering test command was rejected by `zsh` glob expansion before Gradle ran. The rerun used the same Gradle filter quoted for the shell and passed.

## Source Hygiene

`git status --short` after verification still showed the pre-existing local tuning artifacts under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*`. Those profile files were not staged for this plan and are not part of the Phase 12 code/docs changes.
