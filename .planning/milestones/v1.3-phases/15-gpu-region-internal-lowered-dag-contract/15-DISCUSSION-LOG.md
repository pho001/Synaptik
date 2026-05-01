# Phase 15: GPU Region Internal Lowered DAG Contract - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-01T07:12:42.000Z
**Phase:** 15-GPU Region Internal Lowered DAG Contract
**Areas discussed:** DAG metadata contract, Rejection attribution, Trace/debug format, Compatibility boundary

---

## DAG Metadata Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Full region manifest | Original ops, lowered primitives, input/output mapping, dtype/layout assumptions, fused summaries, region length. | yes |
| Minimal trace fields | Basic counts and types only. | |
| Planner-only internal model | Metadata mainly for planning; trace remains brief. | |

**User's choice:** Full region manifest.
**Notes:** The manifest must include bidirectional original-op to lowered-primitive mapping, explicit dtype/layout/storage assumptions per input/primitive/output, and a fusion placeholder using existing `GpuCompoundRegionSummary`.

---

## Rejection Attribution

| Option | Description | Selected |
|--------|-------------|----------|
| Per original op + lowered primitive + region boundary | Most auditable; shows exactly what broke a region. | yes |
| Per original op only | Simpler but weaker for backend primitive debugging. | |
| Per region only | Smallest change but too coarse for coverage-driven work. | |

**User's choice:** Attribute reasons per original op, lowered primitive, and region boundary.
**Notes:** Failing primitive owns the primary reason when an expanded op partially fails. Original op aggregates the reason. Extend `GpuLoweringUnsupportedReason` with DAG-specific codes. Candidate shortening must record original span, accepted span, rejected node/primitive, and reason.

---

## Trace/debug Format

| Option | Description | Selected |
|--------|-------------|----------|
| Structured trace object + compact text summary | Machine-readable for tests/reports plus human-readable compact summary. | yes |
| Execution step trace attributes only | Fits current attributes but becomes flat and hard to evolve. | |
| Debug renderer/report outside runtime trace | Cleaner runtime trace but harder gates/report integration. | |

**User's choice:** Structured trace object plus compact text summary.
**Notes:** Structured data belongs primarily in prepare/backend-selection trace. Run trace references region id and runtime outcome. JSON/text contract should be stable enough for tests and Phase 20 gates.

---

## Compatibility Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| New `GpuLoweredRegionManifest` wrapper | Reuses existing artifacts without overloading native DAG ABI. | yes |
| Extend `AcceleratorDagSpec` directly | Simple lookup but mixes native ABI and debug contract. | |
| Extend only `GpuCompoundRegionSummary` | Too narrow because manifest is not only fusion metadata. | |

**User's choice:** New wrapper manifest with shared core plus backend-specific extensions map.
**Notes:** Phase 15 is Java-side trace/manifest only. No native Metal/CUDA ABI changes. Public `Tensor` and CPU hot path require hard guardrails and focused tests.

## the agent's Discretion

- Exact class/package names and renderer/test decomposition can be chosen during planning.
- Backend-specific extension-map shape can be finalized by research/planning as long as shared Metal/CUDA core stays stable.

## Deferred Ideas

None.
