---
phase: 17
slug: normalization-reduction-and-loss-adjacent-lowering
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 17 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Shared lowering coverage matrix -> backend legality adapters | Metal and CUDA consume backend-neutral support/rejection rows but retain backend-owned gates. | Operation type, semantic family, support status, rejection reason, target evidence |
| Planner diagnostics -> trace/report output | Unsupported accelerator candidates must stay visible without exposing false native support. | Planner rejection strings, lowered manifest text, coverage summaries, benchmark reports |
| Accelerator-configured execution -> CPU correctness oracle | Numerically sensitive flows compare accelerator-prepared behavior against CPU-owned reference execution. | Tensor values for softmax, normalization, and loss-adjacent flows |
| Local benchmark/profile outputs -> source control | Machine-local profile artifacts must not enter committed phase evidence. | `profiles/platform/.../tuning/abc/*` local calibration data |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-17-01 | Integrity | `GpuLoweringCoverageMatrix` support statuses | mitigate | `phaseSeventeenMatrixCoversNormalizationReductionSoftmaxAndLossRows` and `phaseSeventeenNonSupportedRowsUseStableReasonCodes` prove unsupported reduction, normalization, conv, and loss rows are not marked supported without lowering/parity evidence. | closed |
| T-17-02 | Integrity | Shared target evidence | mitigate | Target evidence stays in shared matrix notes/docs (`target=layer_norm_small`, `target=conv2d_resnet_3x3`, `target=transformer_block_hot_path`) and backend adapters consume `plannerUnsupportedDetail(...)` rather than forked text. | closed |
| T-17-03 | Tampering | Metal/CUDA semantic legality | mitigate | `MetalPartitionSupport` and `CudaGpuRegionLegalityAdapter` both route unsupported semantic rows through `GpuLoweringCoverageMatrix.plannerUnsupportedDetail(...)`; backend tests assert the shared family/status/target detail. | closed |
| T-17-04 | Elevation of privilege | Backend dtype/layout guard ordering | mitigate | Metal dtype/capability gates remain before matrix checks, and CUDA direct non-dense layout rejection remains exact through `cudaPhaseSeventeenKeepsDirectNonDenseLayoutRejectionBeforeExecution`. | closed |
| T-17-05 | Repudiation | Loss-adjacent fallback observability | mitigate | Trace, coverage, benchmark, and prepared-execution tests assert `UNSUPPORTED_DTYPE`, `family=LOSS_ADJACENT`, and `target=transformer_block_hot_path` for index-target loss fallback. | closed |
| T-17-06 | Integrity | Softmax-ish supported lowering | mitigate | `LOG_SOFTMAX` support is verified by parity and manifest tests that require original `LOG_SOFTMAX` plus lowered `SOFTMAX` and `LOG` primitives. | closed |
| T-17-07 | Supply chain / integrity | Phase closure docs and local artifacts | mitigate | Docs state that native reduction and normalization support is not implied by a fallback row, validation records `profiles/platform/.../tuning/abc/* remained unstaged`, and `SourceTreeHygieneTest` passed. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 7 | 7 | 0 | Codex inline security audit |

---

## Audit Evidence

| Evidence | Result |
|----------|--------|
| `rg -n "plannerUnsupportedDetail|target=layer_norm_small|target=conv2d_resnet_3x3|target=transformer_block_hot_path" src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java src/main/java/backend/metal/lowering/MetalPartitionSupport.java src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` | Shared matrix and backend adapter mitigations present |
| `rg -n "phaseSeventeenMatrixCovers|phaseSeventeenNonSupportedRowsUseStableReasonCodes|metalPhaseSeventeen|cudaPhaseSeventeen|family=LOSS_ADJACENT|family=NORMALIZATION" src/test/java` | Matrix, backend legality, trace, coverage, and benchmark tests present |
| `rg -n "native reduction and normalization support is not implied by a fallback row|loss-adjacent fallback remained visible|CPU parity remained the correctness oracle|profiles/platform/.../tuning/abc/\\* remained unstaged" docs .planning/phases/17-normalization-reduction-and-loss-adjacent-lowering` | Docs, validation, summary, and verification evidence present |
| `17-REVIEW.md` | Clean review, 0 findings |
| `17-VERIFICATION.md` | Passed, human verification not required |

## Security Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Threats found | 7 |
| Closed | 7 |
| Open | 0 |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
