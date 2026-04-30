# Phase 06: CUDA Shim And Capability Probe - Patterns

## Closest Analogs

| New/Changed Area | Closest Existing Analog | Pattern To Reuse |
|------------------|-------------------------|------------------|
| `scripts/build-cuda-graph-shim.sh` | `scripts/build-metal-mps-shim.sh` | Resolve repo root from script path, write native library under `build/native/<backend>/`, echo path and `-D...lib` usage. |
| CUDA Gradle native tasks | `build.gradle` `buildMetalMpsShim`, `nativeBuild`, `metalTest` | Optional native tasks are guarded by platform/tooling checks and do not affect default `test`. |
| CUDA FFM symbol discovery | `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` | Required core symbols gate bridge availability; optional buffer symbols gate `supportsBufferBindings()`. |
| CUDA unavailable bridge | `src/main/java/backend/cuda/bridge/UnavailableCudaGraphBridge.java` | Unavailable bridge/context/executable records carry diagnostic reasons and do not throw until execution is incorrectly invoked. |
| CUDA prepared executable policy | `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` | Build an `AcceleratorBufferRequest`, publish `AcceleratorBufferDecision`, enforce REQUIRED mode before fallback. |
| CUDA buffer policy | `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` | Validate mode, bridge support, dtype, layout, and existing binding compatibility through backend-neutral reason codes. |
| Shared buffer records | `src/main/java/backend/accelerator/buffer/*.java` and `src/main/java/backend/memory/DeviceBufferBinding.java` | Keep shape/stride/storage-offset/dtype/access/native identity metadata backend-neutral. |
| Portable CUDA tests | `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` and `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` | Tests must pass without native CUDA by asserting unavailable diagnostics or using assumptions for explicit native-library scenarios. |

## File Plan

Expected new files:

- `src/main/native/cuda/synaptik_cuda_graph_stub.cu`
- `scripts/build-cuda-graph-shim.sh`
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java`
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilityCode.java`
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java`
- `src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java`

Expected changed files:

- `build.gradle`
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java`
- `src/main/java/backend/cuda/bridge/CudaBridgeExecutable.java`
- `src/main/java/backend/cuda/bridge/UnavailableCudaGraphBridge.java`
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `docs/development.md`
- `docs/configuration.md`
- `docs/architecture.md`
- `docs/metal-backend.md`
- `src/test/java/SourceTreeHygieneTest.java`

## Guardrails

- Do not add CUDA-native fields to `AcceleratorBufferLayout`, `AcceleratorBufferRequest`, `AcceleratorBufferDecision`, or `DeviceBufferBinding`.
- Do not make `./gradlew test`, `./gradlew classes`, or `./gradlew check` require CUDA hardware or `nvcc`.
- Do not stage `profiles/platform/.../tuning/abc/*` local changes.
- Do not commit native build outputs from `build/native/cuda/`.
- Keep Phase 7 items out of Phase 6: real CUDA device-buffer execution, CPU materialization, and adjacent handoff.
