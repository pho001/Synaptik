---
phase: 09-native-layout-abi-v2
status: passed
score: 5/5
requirements_verified: [GPULAYOUT-01, GPULAYOUT-02, GPULAYOUT-03]
human_verification: []
gaps: []
verified: 2026-04-30
---

# Phase 9 Verification: Native Layout ABI v2

## Verdict

Passed. Phase 9 achieved its roadmap goal: Metal and CUDA now have an additive layout ABI v2 contract with shared metadata records, optional native capability/version probes, and stable ABI-specific fallback diagnostics without changing the public logical `Tensor` API or breaking existing dense buffer execution.

## Requirement Coverage

| Requirement | Status | Evidence |
|---|---|---|
| GPULAYOUT-01 | Passed | `AcceleratorLayoutAbiV2Descriptor`, status/support records, and descriptor tests cover rank, shape, strides, storage offset, logical element/byte counts, physical byte span, access mode, backend id, dtype, layout class, and native handle identity. |
| GPULAYOUT-02 | Passed | `MetalMpsBridgeCapabilities`, extended `CudaBridgeCapabilities`, optional Metal/CUDA layout ABI v2 native symbols, and bridge tests report layout ABI v2 support/version separately from dense graph and buffer execution availability. |
| GPULAYOUT-03 | Passed | `AcceleratorBufferReasonCode` now includes stable `NATIVE_LAYOUT_*` values, `AcceleratorLayoutAbiV2ReasonCodes` maps ABI statuses to buffer reasons, and CUDA binder tests prove visible AUTO rejection and REQUIRED-mode diagnostics for non-dense metadata. |

## Success Criteria

| # | Status | Evidence |
|---|---|---|
| 1. Shared Java records represent layout ABI v2 metadata | Passed | `AcceleratorLayoutAbiV2Descriptor` stores backend-neutral metadata and rejects invalid/overflowing physical spans. |
| 2. Metal and CUDA capability probes report layout ABI version/support | Passed | Bridge capability tests cover no-throw capability reporting; native version symbols return ABI version `2` when built and present. |
| 3. Native ABI changes are optional-symbol/version gated | Passed | Missing layout ABI v2 symbols do not change existing bridge availability or dense buffer support. |
| 4. Unsupported layout metadata uses stable reason codes | Passed | Shared `NATIVE_LAYOUT_*` codes exist and CUDA non-dense metadata maps to ABI-v2-specific decisions; Metal keeps the previously supported dense-physical logical-view path instead of introducing a false rejection. |
| 5. Focused portable tests cover metadata, capability, and fallback seams | Passed | Accelerator buffer, Metal bridge, CUDA bridge, CUDA binder, and prepared executable tests passed in the combined phase gate. |

## Plan Completion

| Plan | Status | Summary |
|---|---|---|
| 09-01 Shared ABI v2 Metadata Contract | Complete | Added backend-neutral descriptor/status/support records, checked physical-span calculation, defensive-copy tests, and metadata docs. |
| 09-02 Metal/CUDA Layout ABI v2 Capability Handshake | Complete | Added Metal capability records, CUDA capability fields, optional native version/validation symbols, and bridge capability tests. |
| 09-03 ABI v2 Fallback And Required-Mode Diagnostics | Complete | Added ABI-v2-specific reason codes and CUDA non-dense metadata diagnostics; documented AUTO/REQUIRE semantics. |

## Automated Checks

- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` - passed.
- `./gradlew classes` - passed.
- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest` - passed.
- `./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest` - passed.
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed.
- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed.
- `rg -n "record AcceleratorLayoutAbiV2Descriptor|physicalByteSpan|REQUIRED_VERSION = 2|record MetalMpsBridgeCapabilities|layoutAbiV2Supported|synaptik_apple_mps_layout_abi_version|synaptik_cuda_graph_layout_abi_version|NATIVE_LAYOUT_ABI_UNAVAILABLE|NATIVE_LAYOUT_PHYSICAL_SPAN_OVERFLOW" src/main/java src/main/native src/test/java docs` - passed.
- `./gradlew metalTest` - passed; built and used the local Metal MPS shim.
- `./gradlew buildCudaGraphShim cudaTest` - build successful; `buildCudaGraphShim` and `cudaTest` were skipped by Gradle capability gates, so no native CUDA hardware execution ran here.
- `gsd-sdk query verify.schema-drift 09` - no schema drift.
- `gsd-sdk query verify.codebase-drift` - no action required.

## Accepted Deviation

Plan 09-03 originally expected Metal non-dense logical-view rejection to switch to `NATIVE_LAYOUT_ABI_UNAVAILABLE`. Execution deliberately preserved the existing Metal dense-physical logical-view path from the previous milestone. That avoids regressing already-supported Metal logical-view execution while Phase 9 still adds common ABI v2 capability metadata, native symbols, shared reason codes, and CUDA-specific non-dense ABI rejection diagnostics.

## Review Gate

Code review was not run in this command chain, and no `09-REVIEW.md` exists yet. The phase is verified by focused automated gates; run `$gsd-code-review 09` before a review-driven milestone close if required.

## Git Hygiene

Local profile tuning changes under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*` remain unstaged and are not part of Phase 9. Build outputs from `metalTest` remain under `build/` and were not staged.

## Human Verification

None required. Phase 9 changes are runtime contracts, native bridge probes, tests, and documentation.

## Residual Risk

Native CUDA execution was not exercised on a CUDA-capable host in this run because the CUDA Gradle tasks were capability-skipped. Layout ABI v2 is also only a contract in Phase 9; Phase 10 owns GPU-side layout/view execution that consumes this contract.
