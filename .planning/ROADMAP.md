# Roadmap: Synaptik

## Milestones

- ✅ **v1.0 Accelerator Runtime Architecture** - Phases 1-5 shipped 2026-04-30. Full archive: [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 CUDA Native Runtime** - Phases 6-8 shipped 2026-04-30. Full archive: [v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)
- ✅ **v1.2 GPU Region Coverage** - Phases 9-13 shipped 2026-05-01. Full archive: [v1.2-ROADMAP.md](milestones/v1.2-ROADMAP.md)
- ✅ **v1.3 Coverage-Driven GPU Region Expansion** - Phases 14-21 shipped 2026-05-01. Full archive: [v1.3-ROADMAP.md](milestones/v1.3-ROADMAP.md)
- ✅ **v1.4 Native GPU Operation Coverage Closure** - Phases 22-28 shipped 2026-05-02. Full archive: [v1.4-ROADMAP.md](milestones/v1.4-ROADMAP.md)
- ✅ **v1.5 Production-Grade Metal Backend Expansion** - Phases 29-39 shipped 2026-05-02. Full archive: [v1.5-ROADMAP.md](milestones/v1.5-ROADMAP.md)
- 🚧 **v1.6 Accelerator Backend Parity And Native Kernel Closure** - Phases 40-46 active. Scope: CUDA parity for v1.5 Metal families, real custom Metal kernel execution, Metal output-copy closure, and cross-backend route regression gates.

## Summary

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 40 | CUDA Parity Gap Triage And Capability Baseline | Establish the source-of-truth CUDA parity gap matrix and capability gates before widening CUDA support. | CUDAPARITY-01, CUDAPARITY-02, CUDAPARITY-03 | 5 |
| 41 | CUDA DType Layout And Index Residency | Expand CUDA dtype/layout/index residency and forward indexing coverage without overclaiming unsupported compute. | CUDADTYPE-01, CUDADTYPE-02, CUDAINDEX-01 | 5 |
| 42 | CUDA NN Operation Parity | Add native/lowered CUDA coverage or stable rejection for high-value NN forward families. | CUDANN-01, CUDANN-02, CUDANN-03 | 5 |
| 43 | CUDA Training And Index Semantics | Close CUDA training/backward and scatter/index-gradient evidence gaps with explicit blockers where support is not proven. | CUDATRAIN-01, CUDATRAIN-02, CUDATRAIN-03 | 5 |
| 44 | Custom Metal Kernel Execution Route | Turn the custom Metal kernel route from a visible seam into real scoped native kernel execution. | METALKERNEL-01, METALKERNEL-02, METALKERNEL-03 | 5 |
| 45 | Metal Output Buffer Write And Copy Closure | Prove true output-buffer writes or implement a lower-copy strategy without false zero-copy claims. | METALCOPY-01, METALCOPY-02, METALCOPY-03 | 5 |
| 46 | Cross-Backend Router Calibration And Regression Gates | Calibrate and gate backend route decisions across MPSGraph, custom Metal, CUDA, tensor-array fallback, and CPU fallback. | BACKENDROUTE-01, BACKENDROUTE-02, BACKENDROUTE-03 | 5 |

## Milestone Rule

v1.6 is parity-and-proof driven. A backend row can become `SUPPORTED` only when it has semantic contract, lowering, backend legality, native or routed execution, CPU parity evidence, trace/report evidence, and regression gate coverage. Capability skips, route seams, and matrix entries are evidence only when they are labeled as such.

CUDA parity means matching the v1.5 Metal support-or-rejection discipline, not blindly marking every Metal-supported row as CUDA-supported. Custom Metal kernels must execute through a real backend route before they can be counted. Metal output-buffer behavior must be proven or explicitly remain a copy/lower-copy strategy.

## Phase Details

### Phase 40: CUDA Parity Gap Triage And Capability Baseline

**Status:** Complete

**Goal:** Establish the source-of-truth CUDA parity gap matrix and capability gates before widening CUDA support.

**Requirements:** CUDAPARITY-01, CUDAPARITY-02, CUDAPARITY-03

**Depends on:** v1.5 Phase 39 route/copy evidence and current CUDA bridge/lowering contracts.

**Success Criteria:**
1. CUDA parity report compares each v1.5 Metal-supported family against CUDA status, blocker, and required evidence.
2. CUDA capability probes distinguish dtype, layout, DAG primitive, vendor-library, buffer-binding, hardware, and toolchain availability.
3. Capability skip is never treated as support in coverage reports or milestone evidence.
4. Hot-path targets identify blocker CPU exits versus accepted explicit capability gaps.
5. Docs explain which CUDA rows are implementation targets for v1.6 and which remain future scope.

**Planned waves:**
- [x] 40-01 CUDA parity matrix and report vocabulary.
- [x] 40-02 CUDA native capability and ABI probe hardening.
- [x] 40-03 Hot-path target policy and blocker classification.
- [x] 40-04 Docs, tests, and baseline closure.

### Phase 41: CUDA DType Layout And Index Residency

**Status:** Complete

**Goal:** Expand CUDA dtype/layout/index residency and forward indexing coverage without overclaiming unsupported compute.

**Requirements:** CUDADTYPE-01, CUDADTYPE-02, CUDAINDEX-01

**Depends on:** Phase 40.

**Success Criteria:**
1. CUDA can represent BF16, BOOL, and INT32 runtime roles needed by selected regions without claiming unsupported compute/output roles.
2. CUDA layout router decisions cover metadata-only views, dense materialization, unsupported strided compute, and stable reason codes.
3. Forward `GATHER` and `TAKE_ALONG_AXIS` either execute under the scoped INT32 contract or reject with explicit bounds/layout/dtype reasons.
4. Adjacent CUDA GPU producers remain device-owned when an index/layout step is supported or explicitly shortened when not.
5. Coverage gates distinguish dtype residency, native dtype compute, layout repair, and CPU materialization.

**Planned waves:**
- [x] 41-01 CUDA dtype role and residency contract.
- [x] 41-02 CUDA layout router/materialization parity.
- [x] 41-03 CUDA forward gather/take execution or stable rejection.
- [x] 41-04 Coverage gates and docs.

### Phase 42: CUDA NN Operation Parity

**Status:** Not started

**Goal:** Add native/lowered CUDA coverage or stable rejection for high-value NN forward families.

**Requirements:** CUDANN-01, CUDANN-02, CUDANN-03

**Depends on:** Phases 40 and 41.

**Success Criteria:**
1. CUDA masked/causal SDPA forward has native/lowered execution or stable capability-gated rejection with mask polarity evidence.
2. CUDA conv/pool forward rows expose execution or precise blockers for layout, dtype, shape, groups, dilation, and average-pool divisor semantics.
3. CUDA dense NLL/CE loss matches the Metal dense contract or rejects with stable dense-loss primitive blockers.
4. CPU parity tests cover admitted CUDA NN rows and unsupported variants.
5. Reports prove supported CUDA NN rows use native buffer execution instead of hidden tensor-array or CPU replay.

**Planned waves:**
- [ ] 42-01 CUDA SDPA forward parity.
- [ ] 42-02 CUDA conv/pool forward parity.
- [ ] 42-03 CUDA dense loss parity.
- [ ] 42-04 Coverage, docs, and regression closure.

### Phase 43: CUDA Training And Index Semantics

**Status:** Not started

**Goal:** Close CUDA training/backward and scatter/index-gradient evidence gaps with explicit blockers where support is not proven.

**Requirements:** CUDATRAIN-01, CUDATRAIN-02, CUDATRAIN-03

**Depends on:** Phases 40-42.

**Success Criteria:**
1. CUDA backward coverage distinguishes native-executable rows from capability-missing rows and gradient publication boundaries.
2. Scatter/index-gradient operations preserve duplicate-index, bounds, dtype, and layout semantics before any native support claim.
3. Unsupported training rows keep stable blockers instead of shortening regions silently.
4. Representative training hot paths report native execution, CPU fallback, tensor-array replay, and materialization reasons.
5. CUDA training gates fail hidden internal CPU materialization while allowing explicit graph-output or gradient-publication boundaries.

**Planned waves:**
- [ ] 43-01 CUDA backward truth and gate policy.
- [ ] 43-02 Scatter/index-gradient semantics and support-or-rejection.
- [ ] 43-03 Training hot-path report targets.
- [ ] 43-04 Final parity, docs, and validation closure.

### Phase 44: Custom Metal Kernel Execution Route

**Status:** Not started

**Goal:** Turn the custom Metal kernel route from a visible seam into real scoped native kernel execution.

**Requirements:** METALKERNEL-01, METALKERNEL-02, METALKERNEL-03

**Depends on:** v1.5 Phase 39.

**Success Criteria:**
1. Custom Metal kernel bridge can compile/load and execute at least one scoped kernel family with buffer bindings.
2. Route legality selects custom kernels only when dtype, layout, shape, and capability checks pass.
3. CPU parity tests cover custom-kernel outputs and fallback behavior when the route is unavailable.
4. Trace/report fields distinguish selected MPSGraph and custom-kernel routes plus rejected alternatives.
5. Existing MPSGraph-supported rows do not regress when custom-kernel route is disabled or unprofitable.

**Planned waves:**
- [ ] 44-01 Custom Metal kernel bridge and native build contract.
- [ ] 44-02 First scoped custom kernel family.
- [ ] 44-03 Route selection, parity, and fallback evidence.
- [ ] 44-04 Reports, docs, and hard gates.

### Phase 45: Metal Output Buffer Write And Copy Closure

**Status:** Not started

**Goal:** Prove true output-buffer writes or implement a lower-copy strategy without false zero-copy claims.

**Requirements:** METALCOPY-01, METALCOPY-02, METALCOPY-03

**Depends on:** Phase 44 where custom kernels may provide direct output writes; v1.5 Phase 39 for current copy classification.

**Success Criteria:**
1. Sentinel/alias tests prove whether MPSGraph writes into caller output buffers for scoped operations.
2. Current behavior remains classified as `MPSGRAPH_RESULT_COPY` unless true output-buffer writes are proven.
3. A lower-copy alias/materialization strategy reduces avoidable native copies where true writes are unavailable.
4. Reports expose copy strategy, native copy timing, alias/write status, and route context.
5. Regression gates fail false zero-copy claims and unexpected copy reintroduction.

**Planned waves:**
- [ ] 45-01 Output-buffer write proof harness.
- [ ] 45-02 Lower-copy or alias strategy.
- [ ] 45-03 Copy strategy reporting and gates.
- [ ] 45-04 Docs and milestone evidence closure.

### Phase 46: Cross-Backend Router Calibration And Regression Gates

**Status:** Not started

**Goal:** Calibrate and gate backend route decisions across MPSGraph, custom Metal, CUDA, tensor-array fallback, and CPU fallback.

**Requirements:** BACKENDROUTE-01, BACKENDROUTE-02, BACKENDROUTE-03

**Depends on:** Phases 40-45.

**Success Criteria:**
1. Router decisions use calibrated evidence for shape, dtype, layout, route, copy, fallback, and native capability.
2. Representative workloads fail gates on hidden CPU exits, tensor-array replay, unsupported route overclaims, and native-copy regressions.
3. Reports show selected route, rejected alternatives, copy strategy, region length, lowered primitive count, and backend ownership.
4. Docs explain backend parity scope, CUDA capability skips, custom-kernel scope, and copy strategy boundaries.
5. Milestone audit can verify every v1.6 support claim from tests, docs, reports, and trace evidence.

**Planned waves:**
- [ ] 46-01 Router calibration evidence model.
- [ ] 46-02 Representative workload gates.
- [ ] 46-03 Report and docs closure.
- [ ] 46-04 Milestone audit readiness.

## Current Status

v1.6 is active. Phase 40 CUDA Parity Gap Triage And Capability Baseline is complete, security-passed, and Nyquist-validated. Phase 41 CUDA DType Layout And Index Residency is complete, security-passed, and Nyquist-validated. Phase 42 CUDA NN Operation Parity is ready for `$gsd-discuss-phase 42 --auto` or `$gsd-plan-phase 42 --auto`.

---

*Last updated: 2026-05-02 after Phase 41 validation*
