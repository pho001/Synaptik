# Phase 15: GPU Region Internal Lowered DAG Contract - Context

**Gathered:** 2026-05-01T07:12:42.000Z
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 15 defines the Java-side contract that describes one selected GPU region as an internal lowered DAG. The phase should make region structure, source operation mapping, backend primitive mapping, dtype/layout/storage assumptions, fused subpattern placeholders, and rejection/materialization attribution visible enough for tests, traces, benchmark reports, and later v1.3 phases.

This phase does not implement broad new operation coverage, dtype/storage residency expansion, fused execution, or multi-op GPU execution itself. It defines the manifest and observability contract that later phases consume.

</domain>

<decisions>
## Implementation Decisions

### DAG Metadata Contract
- **D-01:** Use a full region manifest for each GPU region. The manifest must include original ops, lowered primitives, input/output mapping, dtype/layout/storage assumptions, fused summaries, selected backend, and region length.
- **D-02:** The manifest must provide bidirectional mapping between original operations and lowered primitives. Each original op identifies the lowered primitive or primitives it produced, and each lowered primitive references its source op or source span.
- **D-03:** Dtype, layout, and storage assumptions are explicit per region input, lowered primitive, and region output. Do not rely only on backend legality reason codes for this information.
- **D-04:** Include fused subpattern metadata as a Phase 15 placeholder by wiring existing `GpuCompoundRegionSummary` into the manifest. Phase 18 can expand this, but the Phase 15 manifest should already reserve and expose the field.

### Rejection Attribution
- **D-05:** Rejection, fallback, and materialization reasons must be attributable at three levels: original operation, lowered primitive, and region boundary.
- **D-06:** When one original op expands into multiple lowered primitives and only one primitive fails, the failing primitive owns the primary reason. The original op aggregates that reason for user-facing/debug readability.
- **D-07:** Extend the existing `GpuLoweringUnsupportedReason` vocabulary with DAG-specific codes rather than introducing free-form strings or a separate region-only enum.
- **D-08:** Candidate shortening must be explicit metadata. Record the original candidate span, accepted span, rejected original node or primitive, and reason so Phase 20 gates can detect shortened hot paths.

### Trace/debug Format
- **D-09:** Expose the manifest through a structured trace object plus a compact text summary. The structured form is for tests, reports, and gates; the text form is for human debugging.
- **D-10:** The text summary should be a compact one-region block containing region id, backend, length, original ops, lowered primitives, fused summary, and rejection summary.
- **D-11:** Structured manifest data belongs primarily in prepare/backend-selection trace because the manifest is produced by planning/lowering. Run trace should reference the region id and runtime outcome rather than duplicating the whole manifest.
- **D-12:** JSON/text field names and section headings should be stable enough for Phase 15 tests and Phase 20 regression gates. Treat renames as intentional contract changes.

### Compatibility Boundary
- **D-13:** Introduce a new Java-side wrapper, tentatively `GpuLoweredRegionManifest`, around existing artifacts. Reuse `AcceleratorDagSpec`, `GpuCompoundRegionSummary`, `LoweredRegion`, and backend selection data without pushing debug/report concerns directly into the native DAG ABI.
- **D-14:** Manifest shape is shared core plus backend-specific extension map. The core must be common for Metal and CUDA; backend-specific capability/native detail may live in typed or string-keyed extensions so the model does not fork.
- **D-15:** Phase 15 must not change native Metal/CUDA ABI. It is Java-side trace/manifest only.
- **D-16:** Protect public `Tensor` and CPU hot paths with hard guardrails and focused tests. No public GPU tensor API, no CPU fused behavior change, and no CPU execution contract changes should be needed for this phase.

### the agent's Discretion
The agent may choose exact class/package names and renderer/test decomposition as long as the manifest stays backend-neutral, Java-side only, and trace/report stable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` — Phase 15 goal, success criteria, dependencies, and constraints.
- `.planning/REQUIREMENTS.md` — `GPUDAG-01`, `GPUDAG-02`, and `GPUDAG-03`.
- `.planning/PROJECT.md` — project-level accelerator/runtime constraints and public `Tensor` boundary.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` — source-of-truth v1.3 downstream target list; Phase 15 owns `GPUDAG`.

### Codebase Maps
- `.planning/codebase/ARCHITECTURE.md` — compile/prepare/execute layering and accelerator backend scaffolding.
- `.planning/codebase/INTEGRATIONS.md` — native Metal/CUDA bridge and capability boundaries.
- `.planning/codebase/STRUCTURE.md` — package ownership and where accelerator/shared contracts belong.
- `.planning/codebase/CONCERNS.md` — native ABI fragility, CPU hot-path risk, local profile artifact hygiene, and known accelerator concerns.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/main/java/backend/accelerator/dag/AcceleratorDagSpec.java` — existing backend-neutral lowered DAG passed to native graph bridges; should be referenced by the manifest, not overloaded with debug/report-only fields.
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLoweringResult.java` — current lowering result already carries `AcceleratorDagSpec`, estimated work, and `GpuCompoundRegionSummary`; likely integration point for manifest creation.
- `src/main/java/backend/accelerator/lowering/GpuCompoundRegionSummary.java` — existing compound/fusion summary to embed as Phase 15 fused subpattern placeholder.
- `src/main/java/backend/lowering/LoweredRegion.java` and `src/main/java/backend/lowering/LoweredExecutionUnit.java` — existing lowered-region/unit model that can connect region id, lowering family, ordered node ids, and artifacts.
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` and `BackendSelectionTrace.java` — current prepare/backend-selection trace model; primary target for structured region manifest attachment.
- `src/main/java/graph/execution/PreparedExecution.java` — run-step trace currently publishes accelerator buffer and compound summary attributes; Phase 15 should add only region id/runtime outcome references here if needed.

### Established Patterns
- Accelerator execution is region/partition anchored; `ComputeEngine` skips partition interiors when `PartitionExecutionRole.INTERIOR` is set.
- Backend-neutral orchestration lives under `backend/accelerator`, `backend/lowering`, `backend/prepare`, `backend/select`, and `graph/execution/trace`; backend-specific implementation stays under `backend/metal` and `backend/cuda`.
- Existing trace/report contracts are immutable Java records plus focused JUnit tests and stable renderer fields.
- Unsupported accelerator behavior uses stable reason enums and explicit fallback/materialization traces rather than silent CPU replay.

### Integration Points
- Manifest generation should happen during prepare/lowering or backend selection, not during public tensor graph construction.
- Prepare/backend-selection trace should carry the structured manifest. Run trace should carry region id, backend, runtime execution path/outcome, and materialization/fallback evidence.
- Coverage reports and later Phase 20 gates should consume stable manifest fields instead of parsing ad hoc strings.

</code_context>

<specifics>
## Specific Ideas

- Tentative wrapper name: `GpuLoweredRegionManifest`.
- The manifest should be Java-side only and must not alter Metal/CUDA native ABI in Phase 15.
- Candidate shortening is first-class evidence for later hot-path coverage gates.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-GPU Region Internal Lowered DAG Contract*
*Context gathered: 2026-05-01T07:12:42.000Z*
