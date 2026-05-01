---
phase: 09
slug: native-layout-abi-v2
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 09 - Security

Per-phase security contract: threat register, accepted risks, and audit trail for Native Layout ABI v2.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Public tensor/runtime boundary | Public `Tensor` remains logical while backend residency and native handles stay inside runtime binding state. | Layout metadata, backend id, access mode, opaque native handle identity. |
| Java accelerator metadata to native bridges | Shared Java ABI v2 records describe layout without owning backend-native objects. | Rank, shape, strides, storage offset, dtype, physical span, logical byte length. |
| Optional native symbols to bridge capabilities | Metal/CUDA layout ABI v2 support is discovered through optional symbols without changing dense bridge availability. | Native library state, graph ABI state, buffer ABI state, layout ABI v2 version/support. |
| Buffer binding decisions to fallback execution | AUTO fallback and REQUIRE failure must expose stable reason codes before tensor-array or CPU fallback can hide an unsupported layout. | Accelerator buffer decisions, reason codes, trace/report diagnostics. |
| Local workspace to committed artifacts | Local tuning/profile outputs must stay outside security evidence commits. | Machine-local JSON/JSONL files under `profiles/platform/.../tuning/abc/*`. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-09-01 | Information Disclosure | Shared ABI v2 descriptor | mitigate | `AcceleratorLayoutAbiV2Descriptor` stores only `backendId`, `accessMode`, and opaque `nativeHandleIdentity`; tests construct it from `DeviceBufferBinding` and assert no backend-native handle object crosses the shared record boundary. | closed |
| T-09-02 | Tampering / Denial of Service | Physical span calculation | mitigate | Descriptor physical span uses checked arithmetic and rejects negative stride metadata; descriptor tests cover non-zero offsets, permuted views, broadcasts, defensive copies, and overflow failure. | closed |
| T-09-03 | Denial of Service / Regression | Metal/CUDA optional layout ABI v2 symbols | mitigate | Metal and CUDA bridge capability tests prove missing layout ABI v2 symbols report `layoutAbiV2Supported=false` while dense bridge availability and buffer binding support keep their existing meanings. | closed |
| T-09-04 | Repudiation | Bridge capability reporting | mitigate | `MetalMpsBridgeCapabilities`, `MetalMpsCapabilityCode`, `CudaBridgeCapabilities`, and `CudaBridgeCapabilityCode` report native library, runtime/context, graph ABI, buffer ABI, and layout ABI v2 state separately with stable codes and reasons. | closed |
| T-09-05 | Repudiation / Fallback Bypass | Required buffer execution | mitigate | Prepared Metal/CUDA buffer-policy tests and CUDA binder tests assert non-dense layout metadata surfaces ABI-v2-specific decisions and REQUIRE-mode failures before hidden tensor-array or CPU replay. | closed |
| T-09-06 | Repudiation | Fallback reason taxonomy | mitigate | `AcceleratorBufferReasonCode` and `AcceleratorLayoutAbiV2ReasonCodes` define stable `NATIVE_LAYOUT_*` reason codes; docs and tests assert layout ABI v2 absence/mismatch/metadata reasons are visible. | closed |

---

## Evidence

| Threat Ref | Evidence |
|------------|----------|
| T-09-01 | `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java`, `src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java`, `.planning/phases/09-native-layout-abi-v2/09-01-SUMMARY.md` |
| T-09-02 | `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2StatusCode.java`, `src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java` |
| T-09-03 | `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`, `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`, `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`, `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` |
| T-09-04 | `src/main/java/backend/metal/bridge/MetalMpsBridgeCapabilities.java`, `src/main/java/backend/metal/bridge/MetalMpsCapabilityCode.java`, `src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java`, `src/main/java/backend/cuda/bridge/CudaBridgeCapabilityCode.java` |
| T-09-05 | `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java`, `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`, `src/test/java/backend/cuda/buffer/CudaAcceleratorBufferBinderTest.java`, `docs/native-bridges-and-blas.md` |
| T-09-06 | `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java`, `src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2ReasonCodes.java`, `docs/metal-backend.md`, `docs/native-bridges-and-blas.md` |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 6 | 6 | 0 | Codex inline auditor |

### Security Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Threats found | 6 |
| Closed | 6 |
| Open | 0 |

Verification commands:

- `rg -n "nativeHandleIdentity|backendId\\(|accessMode\\(|DeviceBufferBinding|physicalByteSpan|Math\\.|multiplyExact|addExact|PHYSICAL_SPAN_OVERFLOW|negative stride" src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2Descriptor.java src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java src/main/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2StatusCode.java` - PASS
- `rg -n "layoutAbiV2Supported|layoutAbiV2Version|supportsBufferBindings|isAvailable|capability|CapabilityCode|synaptik_apple_mps_layout_abi_version|synaptik_cuda_graph_layout_abi_version|symbols unavailable|version mismatch" src/main/java/backend/metal/bridge src/main/java/backend/cuda/bridge src/main/native/apple/synaptik_apple_mps_stub.m src/main/native/cuda/synaptik_cuda_graph_stub.cu src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java` - PASS
- `rg -n "NATIVE_LAYOUT_ABI_UNAVAILABLE|NATIVE_LAYOUT_ABI_VERSION_MISMATCH|NATIVE_LAYOUT_METADATA_UNSUPPORTED|NATIVE_LAYOUT_RANK_UNSUPPORTED|NATIVE_LAYOUT_DTYPE_UNSUPPORTED|NATIVE_LAYOUT_PHYSICAL_SPAN_OVERFLOW|layout ABI v2|REQUIRE|AUTO|CPU_CONSUMER|tensor-array" src/main/java/backend/accelerator/buffer src/main/java/backend/metal src/main/java/backend/cuda src/test/java/backend/metal src/test/java/backend/cuda docs/metal-backend.md docs/native-bridges-and-blas.md` - PASS
- `./gradlew test --tests 'backend.accelerator.buffer.*' --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS

Native Metal and native CUDA capability gates are recorded in `09-VALIDATION.md`. CUDA native execution remains capability-skipped on this host.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
