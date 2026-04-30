---
phase: 002
slug: metal-layout-aware-device-flow
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 002 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Runtime tensor layout -> Metal preflight | Runtime shape/stride/offset facts decide whether native Metal buffers may be used. | layout class, shape, strides, storage offset, dtype |
| Existing device binding -> native execution | Device-owned bindings can feed later Metal execution without a CPU round trip. | existing binding metadata, access mode, availability, logical count |
| Java logical layout -> dense native buffer | Java metadata maps dense native bytes back into logical CPU tensor storage. | dense buffer bytes, target shape/strides/storage offset |
| Native buffer handle -> CPU materializer | Device-owned bindings become user-visible CPU data at publication boundaries. | Metal binding, materialization reason, destination tensor |
| Java FFM bridge -> Objective-C shim | Optional native buffer execution must be distinguishable from fallback. | buffer handles, native ABI availability, bridge execution stats |
| Native Metal execution -> traced runtime state | Native success, tensor-array fallback, and CPU fallback must be separately observable. | execution path, reason codes, copy timings, storage residency |
| Device-owned output/gradient -> public tensors | Graph outputs and gradients require correct CPU materialization before publication. | `GRAPH_OUTPUT`, `CPU_CONSUMER`, `GRADIENT_PUBLICATION` traces |
| Local verification -> repository state | Native/benchmark/test runs can leave local artifacts that must not be committed. | `.planning/tmp/`, profile tuning artifacts, generated native build outputs |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-002-01-01 | Tampering | `MetalLayoutPolicy` | mitigate | `MetalLayoutPolicy` encodes exact `DIRECT_DENSE_BUFFER`, `DENSE_PHYSICAL_LOGICAL_VIEW`, and `REJECT` actions and emits reason text with layout class, shape, strides, and storage offset. | closed |
| T-002-01-02 | Elevation of Privilege | `MetalAcceleratorBufferBinder.inputDecisions` | mitigate | Existing device bindings are accepted only after policy classification and `incompatibleBindingReason(...)` checks for availability, dtype, shape, strides, storage offset, logical element count, and access. | closed |
| T-002-01-03 | Repudiation | `AcceleratorBufferDecision` reason text | mitigate | Metal fake-bridge tests assert stable `INPUT_LAYOUT_UNSUPPORTED`, `OUTPUT_LAYOUT_UNSUPPORTED`, and `NATIVE_BUFFER_ABI_UNAVAILABLE` reason codes and policy details. | closed |
| T-002-01-04 | Information Disclosure | Trace diagnostics | accept | Accepted risk documented below: layout diagnostics expose tensor shape/stride metadata only inside local execution traces; no network or multi-user boundary exists. | closed |
| T-002-02-01 | Tampering | `MetalBufferAllocator.readToCpu` | mitigate | `readToCpu` scatters dense buffer contents into destination storage by target shape, strides, and storage offset, with tests for permuted and non-zero-offset destinations. | closed |
| T-002-02-02 | Spoofing | `MetalDeviceToCpuMaterializer.supports` | mitigate | Materializer support requires an available Metal binding, matching dtype, shape, strides, storage offset, logical count, accepted layout policy, and non-broadcast/non-unsupported layout class. | closed |
| T-002-02-03 | Repudiation | `PreparedMetalExecutable.markBufferOutputsCurrent` | mitigate | Output bindings are promoted only after `executeBuffers(...)` succeeds without CPU fallback; CPU materialization traces carry explicit reasons. | closed |
| T-002-02-04 | Elevation of Privilege | Native ABI boundary | mitigate | No new native layout ABI symbol was added; Java supplies dense physical buffers and performs logical materialization, with docs/tests covering the contract. | closed |
| T-002-02-05 | Denial of Service | Layout scatter rank | mitigate | Logical-view materialization rejects rank greater than 4 with a stable message matching the Metal native rank limit. | closed |
| T-002-03-01 | Repudiation | `MetalBufferTraceSmokeTest` | mitigate | Trace tests assert execution path, storage residency, CPU materialization reasons, native copy fields, and fallback reason codes. | closed |
| T-002-03-02 | Tampering | `MetalLayoutAwareDeviceFlowTest` CPU parity | mitigate | Layout-aware forward and forward-backward tests compare Metal-visible results and gradients against CPU baselines. | closed |
| T-002-03-03 | Spoofing | Native success diagnostics | mitigate | Native buffer success assertions require `metalExecutionPath=BUFFER_BINDING`, `acceleratorBufferExecutionPath=BUFFER_BINDING`, `metalNativeToJavaCopyNs=0L`, and `storageResidency=DEVICE_OWNED`. | closed |
| T-002-03-04 | Information Disclosure | Local profile artifacts | accept | Accepted risk documented below: no secrets are involved, and phase rules forbid staging local benchmark/calibration/profile outputs. | closed |
| T-002-03-05 | Denial of Service | Verification command selection | mitigate | Phase 2 validation and docs use targeted Gradle filters before `metalTest`, avoiding default debug benchmark suite delays. | closed |

*Status: open / closed*
*Disposition: mitigate (implementation required) / accept (documented risk) / transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-002-01 | T-002-01-04 | Trace diagnostics expose local tensor layout facts needed to diagnose fallback. They stay inside local execution traces and are not exposed through a network or multi-user boundary. | Codex | 2026-04-30 |
| AR-002-02 | T-002-03-04 | Local profile and benchmark artifacts contain no secrets in this workflow, but they remain excluded from Phase 2 commits; summaries and status checks confirm they were not staged. | Codex | 2026-04-30 |

---

## Summary Threat Flags

All Phase 2 summaries reported no additional threat flags beyond the plan threat models.

---

## Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 14 |
| Closed | 14 |
| Open | 0 |

## Evidence

| Threats | Evidence | Result |
|---------|----------|--------|
| T-002-01-01, T-002-01-03 | `MetalLayoutPolicy.java:17-128` defines accepted/rejected policy actions and detailed reason text; `PreparedMetalExecutableBufferBindingTest` asserts policy actions, layout classes, storage offsets, strides, and stable reason codes. | PASS |
| T-002-01-02 | `MetalAcceleratorBufferBinder.java:112-137` applies `MetalLayoutPolicy.existingDeviceInput(...)`; `MetalAcceleratorBufferBinder.java:327-365` checks availability, dtype, shape, strides, storage offset, logical count, and access compatibility. | PASS |
| T-002-01-04 | Accepted risk `AR-002-01`; trace fields are local diagnostics and are documented as execution metadata. | PASS |
| T-002-02-01, T-002-02-05 | `MetalBufferAllocator.java:288-306` scatters dense logical data by shape/strides/storage offset and rejects rank greater than 4; allocator tests cover permuted and non-zero-offset destinations. | PASS |
| T-002-02-02 | `MetalDeviceToCpuMaterializer.java:34-51` validates available Metal binding, target layout equality, accepted policy action, and rejected broadcast/unsupported layout classes; unit tests cover supported and rejected logical views. | PASS |
| T-002-02-03 | `PreparedMetalExecutable.java:141-153` promotes output bindings only after successful `executeBuffers(...)`; `PreparedMetalExecutable.java:362-370` attaches promoted bindings; runtime traces label CPU materialization reasons. | PASS |
| T-002-02-04 | `docs/metal-backend.md` and `docs/compute-flow.md` document dense physical logical-view handling and native buffer ABI boundaries; no new native layout ABI symbol was introduced in Phase 2. | PASS |
| T-002-03-01, T-002-03-03 | `MetalBufferTraceSmokeTest.java:47-90` asserts `BUFFER_BINDING`, stable reason code, prepared input flag, `metalNativeToJavaCopyNs=0L`, `DEVICE_OWNED`, and materialization reasons. | PASS |
| T-002-03-02 | `MetalLayoutAwareDeviceFlowTest.java:54-120` checks CPU parity for forward results and forward-backward gradients with visible materialization/fallback evidence. | PASS |
| T-002-03-04 | Accepted risk `AR-002-02`; `002-03-SUMMARY.md` records that local profile tuning artifacts and `.planning/tmp/` were not staged. | PASS |
| T-002-03-05 | `docs/testing.md` and `002-VALIDATION.md` list focused Phase 2 Gradle filters plus `metalTest`, not the full default test suite. | PASS |
| All mitigate threats | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` | PASS |
| Native gate | `./gradlew metalTest` | PASS |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 14 | 14 | 0 | Codex |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
