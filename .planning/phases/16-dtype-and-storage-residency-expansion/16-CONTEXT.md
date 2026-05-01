# Phase 16: DType And Storage Residency Expansion - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Source:** Auto context from roadmap, requirements, Phase 14 targets, Phase 15 manifest contract, and codebase inspection.

<domain>
## Phase Boundary

Phase 16 expands accelerator residency plumbing for `BFLOAT16`, `INT32`, and `BOOL` where those dtypes block GPU regions. The phase is about representing and preserving dtype-specific storage in runtime memory binding, device buffer metadata, capability decisions, and trace/report diagnostics.

It is not a promise that every operation can compute natively in those dtypes. Execution legality remains gated by backend, operation, layout, role, and dtype.
</domain>

<decisions>
## Implementation Decisions

### Runtime Storage Binding
- D-01: `RuntimeMemoryBinder` must stop treating `BFLOAT16`, `INT32`, and `BOOL` as no-op dtypes when a region slot is reusable and size-compatible.
- D-02: Runtime slot reuse must allocate dtype-specific Java arrays: `short[]` for `BFLOAT16`, `int[]` for `INT32`, and `byte[]` for `BOOL`, while preserving existing `double[]` and `float[]` behavior.
- D-03: Alias/view nodes and workspace-sensitive nodes must keep the same skip behavior as existing `FLOAT32` binding.

### Accelerator Residency And Capability
- D-04: Device buffer layout metadata must continue to represent dtype byte length for `BFLOAT16`, `INT32`, and `BOOL`; Phase 16 must add tests around residency decisions, not a native ABI rewrite.
- D-05: Metal and CUDA dtype legality must be capability-gated by backend role: external input, internal value, output, and native compute.
- D-06: Unsupported dtype cases must report stable backend-specific detail strings and stable `GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE` attribution instead of silently forcing CPU materialization.
- D-07: Metal's current contract remains conservative: `FLOAT32` compute/output and `BOOL` predicate inputs only unless code evidence proves more.
- D-08: CUDA's current buffer allocation remains conservative until explicit dtype support exists; BOOL or other dtype handling must not be counted as general native compute unless implemented.

### Trace, Reports, And Phase Handoff
- D-09: Lowered-region manifests and benchmark coverage should expose dtype residency assumptions and dtype-related rejection/materialization evidence for `BFLOAT16`, `INT32`, and `BOOL`.
- D-10: Hidden CPU materialization is a bug for supported internal GPU-region values; true graph output, CPU consumer, and gradient publication boundaries can materialize but must be visible.
- D-11: Phase 14 targets `mlp_classifier_small` and `layer_norm_small` are the primary hot-path drivers for `GPUSTORAGE`.

### Guardrails
- D-12: Public `Tensor` remains a logical tensor API; no public GPU tensor or device residency API should be added in this phase.
- D-13: CPU hot paths, `FLOAT32`, and `FLOAT64` behavior must remain stable.
- D-14: Local files under `profiles/platform/.../tuning/abc/*` are calibration artifacts and must remain unstaged unless intentionally promoted.
</decisions>

<canonical_refs>
## Canonical References

### Planning And Requirements
- `.planning/ROADMAP.md` - Phase 16 goal, dependencies, and success criteria.
- `.planning/REQUIREMENTS.md` - `GPUSTORAGE-01`, `GPUSTORAGE-02`, and `GPUSTORAGE-03`.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` - downstream `GPUSTORAGE` target workloads.
- `.planning/phases/15-gpu-region-internal-lowered-dag-contract/15-VERIFICATION.md` - completed lowered-region manifest contract.

### Runtime Binding
- `src/main/java/graph/execution/RuntimeMemoryBinder.java` - runtime slot binding implementation and current non-floating no-op.
- `src/test/java/graph/execution/RuntimeMemoryBinderTest.java` - existing binding regression tests.
- `src/main/java/tensor/Tensor.java` - typed storage getters and `setData(short[]/int[]/byte[])`.
- `src/main/java/tensor/BFloat16Storage.java`, `src/main/java/tensor/Int32Storage.java`, `src/main/java/tensor/BoolStorage.java` - storage array contracts.

### Accelerator Residency
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` - dtype byte length and layout metadata.
- `src/main/java/graph/execution/ExecutionState.java` - storage residency, device binding, materialization tracing, and dtype byte sizes.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` - current Metal dtype capability source.
- `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java` and `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java` - current CUDA dense `FLOAT32` buffer boundary.
- `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` - stable unsupported reason vocabulary.
</canonical_refs>

<specifics>
## Specific Ideas

- Start with runtime typed slot reuse because it is the concrete blocker already called out in `.planning/codebase/CONCERNS.md`.
- Add a backend-neutral dtype residency decision helper only if it reduces duplicated Metal/CUDA diagnostics; keep backend-native execution choices backend-owned.
- Extend trace/report evidence after decisions are explicit, so Phase 20 can later gate "hot path stayed on GPU" by dtype.
</specifics>

<deferred>
## Deferred Ideas

- Universal native `BFLOAT16`, `INT32`, or `BOOL` arithmetic.
- Native Metal/CUDA ABI changes for every dtype.
- Public device tensor API.
- Region-internal fusion and multi-op execution mechanics beyond dtype residency evidence; those belong to Phases 18 and 19.
</deferred>

---

*Phase: 16-dtype-and-storage-residency-expansion*
*Context gathered: 2026-05-01 via auto plan-phase*
