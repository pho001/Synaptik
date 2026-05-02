# Phase 29 Verification: Metal DType ABI And Capability Truth

## Verdict

**PASS.**

Phase 29 completes `METALDTYPE-01`, `METALDTYPE-02`, and `METALDTYPE-03` by adding role-specific Metal dtype decisions, optional dtype ABI v3 native discovery, report-visible dtype reason codes, tests, and docs. It does not widen native execution beyond current FLOAT32 compute/output.

## Requirement Evidence

| Requirement | Evidence | Status |
|---|---|---|
| `METALDTYPE-01` | `MetalMpsCapabilities` now distinguishes storage, external input, external input role, compute, output, and operation dtype decisions for all public dtypes. `MetalMpsCapabilitiesTest` locks current truth, including explicit `FLOAT64` unsupported decisions. | Passed |
| `METALDTYPE-02` | `MetalDTypeAbiV3Support`, `MetalMpsBridgeCapabilities`, `MetalMpsFfmBridge`, and `synaptik_apple_mps_stub.m` expose optional dtype ABI v3 version/validation symbols without changing the `_f32` execution path. | Passed |
| `METALDTYPE-03` | `AcceleratorDTypeResidencyPolicy` now renders Metal dtype decisions with backend/role/dtype/reason-code detail, and docs preserve the distinction between dtype residency and native dtype compute. | Passed |

## Source And Test Evidence

| Area | Evidence |
|---|---|
| DType decision model | `src/main/java/backend/metal/MetalDTypeCapabilityDecision.java`, `MetalDTypeRole.java`, `MetalDTypeReasonCode.java`, `MetalMpsCapabilities.java` |
| Native ABI v3 discovery | `src/main/java/backend/metal/bridge/MetalDTypeAbiV3Support.java`, `MetalMpsBridgeCapabilities.java`, `MetalMpsFfmBridge.java`, `src/main/native/apple/synaptik_apple_mps_stub.m` |
| Report-visible dtype truth | `src/main/java/backend/accelerator/residency/AcceleratorDTypeResidencyPolicy.java`, `docs/gpu-lowering-coverage.md` |
| Public docs | `docs/metal-backend.md`, `docs/gpu-lowering-coverage.md` |
| Tests | `src/test/java/backend/metal/MetalMpsCapabilitiesTest.java`, `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` |

## Commands

| Command | Result |
|---|---|
| `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest` | Passed |
| `./gradlew classes` | Passed |
| `./gradlew metalTest` | Passed; built `build/native/apple/libsynaptik_apple_mps.dylib` |

## Current Metal DType Truth

| Role | Current decision |
|---|---|
| Storage metadata/residency | All public dtypes are representable as metadata/storage decisions. |
| Native compute | `FLOAT32` only. |
| Native output | `FLOAT32` only. |
| External data input | `FLOAT32` only. |
| Predicate external input | `BOOL` only for supported predicate roles such as `WHERE` input 0. |
| Descriptor ABI | dtype ABI v3 can describe all public dtypes, but descriptor support is not compute support. |
| Explicit unsupported native compute/output | `BFLOAT16`, `INT32`, `BOOL`, `FLOAT64`. |

## Residual Risks

- BF16 native compute/output remains Phase 30.
- BOOL-producing native compute remains Phase 31.
- INT32 index execution remains Phase 32.
- Existing local profile artifacts under `profiles/platform/...` remain dirty and intentionally unstaged.
