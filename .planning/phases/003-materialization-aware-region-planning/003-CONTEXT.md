# Phase 3: Materialization-Aware Region Planning - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 makes `PART` and backend selection prefer profitable long device-owned accelerator regions and avoid short accelerator islands that force expensive CPU materialization. It adds materialization-aware static cost modeling, planner/selection trace visibility, and CPU hot-path safeguards. It does not add native layout ABI, profile-derived cost tuning, CUDA native implementation, or public device tensor APIs.

</domain>

<decisions>
## Implementation Decisions

### Planner Bias

- **D-01:** Use a balanced default: prefer longer device-owned accelerator regions when they reduce CPU boundaries, but require traceable cost justification and keep CPU alternatives intact.
- **D-02:** When scores are equal or ambiguous, tie goes to the accelerator if legality and safety gates pass.
- **D-03:** Legal neutral ops should stay inside longer accelerator regions when doing so avoids materialization boundaries. The planner should optimize whole-flow profitability, not only per-op profitability.

### Cost Model Inputs

- **D-04:** Phase 3 uses static estimates first: tensor byte counts, estimated work, layout class, boundary count, fallback mode, and dispatch constants.
- **D-05:** The first model should be a full static model: boundary/materialization signals plus estimated compute work, dispatch overhead, layout fallback class, and avoided intermediate bytes.
- **D-06:** Expose named presets with internal constants. Do not expose raw penalties/credits as user-tunable profile fields in Phase 3.
- **D-07:** Profile/calibration-derived costs are explicitly deferred to Phase 4, after tuning/profile ownership is audited.

### Trace Detail

- **D-08:** Planner/backend-selection traces should expose summary cost fields by default: reason codes, final score, boundary count, estimated transfers, estimated compute, and selected preset.
- **D-09:** Include the selected candidate plus top rejected competitors/finalists with score summaries and rejection reasons. Do not dump every rejected candidate by default.
- **D-10:** Surface planner cost decisions in compile/selection traces and benchmark reports. Runtime step traces should stay focused on execution metadata.

### CPU Safeguards

- **D-11:** Add CPU-first safety rules: preserve CPU natural regions and fusion when accelerator benefit is not clearly better, with tests proving preservation.
- **D-12:** Phase 3 tests must protect both CPU natural regions and CPU fusion/BLAS behavior.
- **D-13:** Phase 3 should not use profile-derived speed evidence for selection. Keep static-model selection and require CPU-focused regression tests; Phase 4 handles profile-derived cost updates.

### the agent's Discretion

- Exact class names, record shapes, and trace field names are left to the planner/researcher, provided they remain backend-neutral where appropriate and align with existing trace/config patterns.
- The planner may decide whether to extend existing `AcceleratorPartitionScoreModel` and trace records directly or introduce small adjacent records, as long as public `Tensor` API remains untouched.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope And Requirements

- `.planning/ROADMAP.md` — Phase 3 goal, success criteria, and constraints.
- `.planning/REQUIREMENTS.md` — `PLAN-01` through `PLAN-04`, plus Phase 4 tuning deferral context.
- `.planning/PROJECT.md` — project constraints: logical public tensor API, CPU correctness/performance baseline, traceable fallback.
- `.planning/STATE.md` — current milestone state and prior decisions.

### Prior Phase Contracts

- `.planning/phases/001-accelerator-buffer-layout-abi/001-VERIFICATION.md` — validated backend-neutral buffer layout ABI and reason taxonomy.
- `.planning/phases/002-metal-layout-aware-device-flow/002-VERIFICATION.md` — validated Metal layout-aware device flow, fallback visibility, and materialization boundaries.
- `.planning/phases/002-metal-layout-aware-device-flow/002-03-SUMMARY.md` — latest runtime fixes and residual risk around native buffer parity fixture.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` — compile/optimizer/prepare/execute architecture and partition flow.
- `.planning/codebase/INTEGRATIONS.md` — native bridge/runtime integration context.
- `.planning/codebase/CONCERNS.md` — relevant concerns around profile parsing, native bridge visibility, benchmark hygiene, and CPU hot paths.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` — current backend selection gate; already records selected/rejected candidate decisions but only uses runtime enablement, availability, and minimum estimated work.
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` — current acceptance model; can be expanded or wrapped to account for materialization/transfer/static profitability.
- `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java` — existing structural/work scoring with `TransferMetrics` and `TransferPolicy`; likely starting point for Phase 3 static cost modeling.
- `src/main/java/config/optimizer/PartitionConfig.java` and `src/main/java/config/optimizer/MetalTransferModel.java` — existing named/static transfer-cost configuration; fits the decision to expose presets and keep raw weights internal.
- `src/main/java/graph/execution/trace/PartitionDecisionTrace.java` and `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` — existing trace records for planner and backend selection decisions; likely targets for summary cost fields and finalist reporting.

### Established Patterns

- Optimizer stages are ordered `AR -> CSE -> PART -> FUSE -> MEM`; Phase 3 should keep materialization-aware decisions in/around `PART` and backend selection, not in public tensor APIs.
- Accelerator partitions are region/anchor based; interior nodes are skipped during execution after a selected accelerator plan is prepared.
- CPU natural regions and CPU fusion already have separate planner/config paths; Phase 3 must preserve them when accelerator cost does not clearly win.
- Existing tuning candidate spaces already vary accelerator region policy and buffer binding mode, but Phase 4 owns deeper profile/calibration-derived cost updates.

### Integration Points

- `graph.optimizer.partition.PartitionIntentRule`, `GreedyMaxRegionPartitionPlanner`, and `ScoredCandidatePartitionPlanner` are the planning side of accelerator/CPU region selection.
- `backend.select.DefaultBackendSelectionPolicy` is the prepare-time acceptance side where runtime availability and cost gates are currently applied.
- `graph.execution.trace` records feed compile/selection diagnostics and should carry summary cost fields.
- Benchmark/reporting code should summarize planner cost decisions without duplicating all runtime step metadata.

</code_context>

<specifics>
## Specific Ideas

- Default behavior should lean toward longer device-owned accelerator flow when the model cannot distinguish candidates, but only after legality and CPU-safety gates pass.
- Neutral legal ops can be absorbed into accelerator regions if that avoids CPU materialization.
- Planner trace output should be actionable: selected candidate plus top rejected finalists, not a flood of all candidates.
- Phase 4 should explicitly plan profile/calibration-derived cost model updates after graph autotune versus platform calibration ownership is audited.

</specifics>

<deferred>
## Deferred Ideas

- Phase 4 should plan an update that wires reliable profile/calibration-derived costs into the accelerator cost model after tuning/profile ownership is audited.

</deferred>

---

*Phase: 003-materialization-aware-region-planning*
*Context gathered: 2026-04-30*
