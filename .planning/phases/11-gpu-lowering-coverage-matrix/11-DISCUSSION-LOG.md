# Phase 11: GPU Lowering Coverage Matrix - Discussion Log

**Gathered:** 2026-04-30
**Mode:** `gsd-next` non-interactive default selection

## Auto-Selected Gray Areas

### Coverage Matrix Shape
- **Question:** Should the matrix be a broad claim document or a testable source-aligned contract?
- **Selected:** Testable source-aligned contract.
- **Reason:** Phase 13 depends on trustworthy coverage evidence; unsupported entries need stable reasons instead of vague "not yet" notes.

### Operation Prioritization
- **Question:** Should Phase 11 chase many isolated ops or prioritize patterns that lengthen realistic GPU regions?
- **Selected:** Prioritize patterns that lengthen realistic GPU regions.
- **Reason:** The milestone goal is fewer GPU-to-CPU exits, not a long unsupported checklist.

### Shared Versus Backend-Specific Lowering
- **Question:** Should Metal and CUDA grow separate allowlists or converge around a shared semantic coverage model?
- **Selected:** Shared semantic coverage model with backend-owned capability checks.
- **Reason:** Existing Metal/CUDA allowlists overlap and can drift; native handles, kernels, and capability checks still belong under each backend.

### Layout And Residency Boundaries
- **Question:** Should Phase 11 directly accept arbitrary non-dense CUDA compute now that layout ABI v2 exists?
- **Selected:** No. Keep CUDA non-dense compute conservative unless Phase 10 metadata-only view propagation or dense materialization makes the path legal.
- **Reason:** Phase 10 explicitly avoided overclaiming arbitrary non-dense CUDA compute; broad lowering must preserve visible fallback and CPU parity.

### Verification Scope
- **Question:** What closes Phase 11?
- **Selected:** Portable coverage matrix tests, legality/lowering selected and rejected candidate tests, docs, and capability-gated native checks only where local tooling allows.
- **Reason:** Benchmarks and coverage regression gates are Phase 13; Phase 11 should not commit local profile artifacts.

## Deferred Ideas

- Fused GPU compound execution belongs to Phase 12.
- Coverage ratio benchmark/regression gates belong to Phase 13.
- Universal accelerator support beyond the roadmap matrix belongs after v1.2.

## Next Step

Run `$gsd-plan-phase 11`.
