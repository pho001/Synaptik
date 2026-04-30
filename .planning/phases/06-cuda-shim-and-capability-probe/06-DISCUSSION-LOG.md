# Phase 6: CUDA Shim And Capability Probe - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 6-CUDA Shim And Capability Probe
**Areas discussed:** Native Shim And Build Workflow, Capability Probe And Reason Codes, Shared Accelerator Buffer ABI, Execution And Fallback Policy, Tests Docs And Hygiene

---

## Native Shim And Build Workflow

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal checked-in shim with optional build/probe task | Add native source, script, and Gradle wiring while keeping portable Java gates independent of CUDA hardware. | ✓ |
| External library only | Keep relying on user-provided native libraries and only document lookup. | |
| Full CUDA runtime implementation now | Implement broad native operation coverage immediately. | |

**User's choice:** Auto-selected recommended default: minimal checked-in shim with optional build/probe task.
**Notes:** This satisfies `CUDA-01` without expanding Phase 6 into Phase 7 execution/materialization scope.

---

## Capability Probe And Reason Codes

| Option | Description | Selected |
|--------|-------------|----------|
| Layered capability state with stable reasons | Distinguish missing library, missing symbols, unavailable runtime, context/compile/execute failures, buffer ABI absence, unsupported dtype/layout, and required-unavailable paths. | ✓ |
| Single available boolean | Collapse capability into bridge availability only. | |
| Throw on unavailable CUDA | Let native lookup or execution failures escape instead of publishing unavailable records. | |

**User's choice:** Auto-selected recommended default: layered capability state with stable reasons.
**Notes:** This aligns with existing fallback visibility requirements and avoids silent CPU fallback masking ABI mismatches.

---

## Shared Accelerator Buffer ABI

| Option | Description | Selected |
|--------|-------------|----------|
| Consume shared ABI, keep CUDA handles private | Use `AcceleratorBufferRequest`, `AcceleratorBufferLayout`, `AcceleratorBufferDecision`, and `DeviceBufferBinding` while placing CUDA native handles under CUDA packages. | ✓ |
| Add CUDA fields to common records | Encode CUDA native details in shared accelerator contracts. | |
| Skip buffer ABI seam until Phase 7 | Leave Phase 6 as native loading only. | |

**User's choice:** Auto-selected recommended default: consume shared ABI, keep CUDA handles private.
**Notes:** This satisfies `CUDA-02` while preserving backend-neutral accelerator contracts.

---

## Execution And Fallback Policy

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative fallback with REQUIRED-mode preflight failure | AUTO may fall back visibly; REQUIRED fails before tensor-list execution when CUDA buffer execution is unavailable. | ✓ |
| Always tensor-list fallback | Ignore REQUIRED buffer semantics and always replay through legacy execution when possible. | |
| Force native execution whenever library loads | Treat native library presence as enough to bypass fallback guards. | |

**User's choice:** Auto-selected recommended default: conservative fallback with REQUIRED-mode preflight failure.
**Notes:** Preserves existing CUDA policy tests and matches Metal-era explicit fallback contracts.

---

## Tests Docs And Hygiene

| Option | Description | Selected |
|--------|-------------|----------|
| Portable tests plus optional native tests | Cover unavailable and policy behavior without hardware; use assumptions/targeted task for native CUDA checks. | ✓ |
| Require CUDA in default test | Make standard test gate depend on CUDA hardware/toolkit. | |
| Docs-only verification | Document behavior without expanding tests. | |

**User's choice:** Auto-selected recommended default: portable tests plus optional native tests.
**Notes:** Build outputs stay under ignored build locations; profile/tuning artifacts remain out of scope.

## the agent's Discretion

- Exact CUDA helper class names, native source extension, and Gradle task names may be selected by the implementer if they match repository conventions.
- The implementer may mirror Metal build/test ergonomics where practical, but must preserve CUDA-specific package boundaries.

## Deferred Ideas

- Actual CUDA native buffer execution, materialization, and adjacent handoff belong to Phase 7.
- CUDA trace/report parity and final documentation/hygiene closure belong to Phase 8.
