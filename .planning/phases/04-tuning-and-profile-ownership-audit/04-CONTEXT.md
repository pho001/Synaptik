# Phase 4: Tuning And Profile Ownership Audit - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4 audits and hardens ownership between graph autotune, platform calibration, profile persistence, and benchmark flows. It must ensure graph/workload-specific policy lives in graph autotune, hardware/dtype runtime thresholds live in platform calibration, profile IO validates schema/version and accelerator buffer/layout fields, and benchmark flows remain observational/read-only. It also must plan the profile/calibration-derived accelerator cost model update deferred from Phase 3 after this ownership boundary is enforced.

This phase does not broaden native accelerator operation coverage, introduce a public device tensor API, make benchmark reports a runtime source of truth, or implement Phase 5 observability closure.

</domain>

<decisions>
## Implementation Decisions

### Knob Ownership Boundary

- **D-01:** `ACCELERATOR_BUFFER_MODE` belongs to graph autotune, not platform calibration. It affects workload-specific graph flow, accelerator region behavior, and materialization boundaries.
- **D-02:** `METAL_SELECTION` remains platform calibration, explicit accelerator opt-in only. Standard CPU calibration must not automatically mutate Metal enablement, runtime-availability requirements, or minimum-work thresholds.
- **D-03:** `MetalTransferModel` and static transfer-cost preset choice remain graph autotune/planner policy. Graph autotune may explore transfer-model or preset variants because they change graph/region planning decisions.
- **D-04:** Phase 4 must enforce a knob ownership matrix with tests. Each knob must be classified as `graph/workload`, `platform/dtype`, or `obsolete`; graph autotune and calibration candidate spaces must not mutate knobs outside their ownership without an explicit documented exception.

### Carry-Forward From Phase 3

- **D-05:** Profile/calibration-derived accelerator costs must be wired into the cost model in Phase 4 only after the graph-autotune versus platform-calibration ownership audit is complete.
- **D-06:** The cost-model update must preserve the Phase 3 rule that CPU natural regions, CPU fusion, and BLAS paths remain available and competitive when accelerator offload is not clearly profitable.

### the agent's Discretion

- The exact ownership matrix representation is left to the planner. It can be a code enum/table, test fixture, documentation table, or a combination, as long as tests enforce candidate-space ownership.
- The planner may decide whether ownership enforcement extends the existing `CalibrationCandidateOwnershipTest` directly or adds a parallel graph-autotune ownership test suite.
- Profile IO strictness, exact profile schema migration strategy, and benchmark write-boundary enforcement still need implementation research. Do not infer user decisions beyond the ownership boundary above.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope And Requirements

- `.planning/ROADMAP.md` - Phase 4 goal, success criteria, and the explicit note that profile/calibration-derived cost model updates belong in Phase 4.
- `.planning/REQUIREMENTS.md` - `TUNE-01` through `TUNE-04`, plus accelerator observability requirements that Phase 4 must not accidentally absorb from Phase 5.
- `.planning/PROJECT.md` - project constraints: keep public tensors logical, preserve CPU hot paths, keep fallback traceable, and avoid local profile artifact churn.
- `.planning/STATE.md` - current milestone state and operating note to include the Phase 3 cost-model deferral in Phase 4 planning.

### Prior Phase Contracts

- `.planning/phases/003-materialization-aware-region-planning/003-CONTEXT.md` - Phase 3 decisions deferring profile/calibration-derived costs to Phase 4.
- `.planning/phases/003-materialization-aware-region-planning/003-VERIFICATION.md` - validated static materialization-aware planning and CPU safeguard evidence.
- `.planning/phases/003-materialization-aware-region-planning/003-VALIDATION.md` - task verification map and Nyquist coverage for Phase 3 planner/cost behavior.

### Tuning Architecture And Persistence

- `src/main/java/tuning/ARCHITECTURE.md` - tuning ownership model: `PlatformRuntimeProfile`, `GraphExecutionPolicy`, `ExecutionProfile`, and persistence/explain artifacts.
- `src/main/java/tuning/PERSISTENCE.md` - profile layout, runtime source-of-truth rules, best-profile records, history, and benchmark/report artifact meaning.
- `src/main/java/tuning/README.md` - high-level tuning workflow split: benchmark, per-graph autotune, and platform calibration.

### Code Entry Points

- `src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java` - graph autotune candidate generation, including current `ACCELERATOR_BUFFER_MODE` variants.
- `src/main/java/tuning/candidate/graph/GraphPolicyMutators.java` - graph-policy mutation surface, including accelerator region and transfer-model variants.
- `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java` - calibration family ownership metadata and existing `METAL_SELECTION` opt-in family.
- `src/test/java/CalibrationCandidateOwnershipTest.java` - existing test pattern for enforcing calibration-owned knob changes.
- `src/main/java/config/profile/PlatformRuntimeProfile.java` - platform runtime profile ownership contract and conversion into runtime config.
- `src/main/java/config/profile/PlatformRuntimeProfileIO.java` - current profile persistence parser/loader that needs schema/version and accelerator field hardening.
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` - prepare-time accelerator cost gate that should consume audited profile/calibration-derived cost inputs.
- `src/main/java/synaptik/app/TuningCli.java` - CLI orchestration for calibration, autotune, benchmark, and profile-root write paths.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` - compile/optimizer/prepare/execute and tuning architecture overview.
- `.planning/codebase/STACK.md` - Java/Gradle/profile persistence stack context.
- `.planning/codebase/INTEGRATIONS.md` - native runtime and benchmark/calibration filesystem integration context.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `CalibrationFamilyRegistry.validateCandidateChanges(...)` and `CalibrationCandidateOwnershipTest` already enforce calibration family ownership; Phase 4 can extend this pattern for graph autotune and obsolete-knob checks.
- `GraphAutotuneCandidateSpace` already labels candidate metadata with `graphParameter`, `offloadPolicy`, `acceleratorRegionPolicy`, `metalTransferModel`, `cpuRegionPolicy`, `cpuFusionPolicy`, and `productionEligible`.
- `PlatformRuntimeProfile` already documents that graph autotune consumes platform runtime as frozen runtime input and varies graph policy.
- `BestProfileRecord.rebaseOnRuntime(...)` is used by `TuningCli` to reassemble persisted graph winners with the current calibrated runtime profile.

### Established Patterns

- Benchmark sessions validate and measure explicit `ExecutionProfile` entries; `BenchmarkRequest` documents that benchmark must not create, search, or persist profiles.
- Platform calibration owns runtime profile families such as scheduler, matmul, conv2d dispatch, fused dispatch, elementwise dispatch, reduction, attention thresholds, materialization, and opt-in Metal selection.
- Graph autotune owns workload graph-policy variants and production eligibility metadata, with research-only variants explicitly separated from standard production candidates.
- Profile persistence distinguishes platform runtime profiles, best profile records, tuning history, and benchmark/report artifacts; reports are not runtime sources of truth.

### Integration Points

- Ownership enforcement connects `tuning.candidate.graph`, `tuning.calibration.family`, `tuning.calibration.runtime`, `tuning.store`, and `config.profile`.
- Cost-model integration connects audited platform/runtime fields into `AcceleratorPlanCostModel` and the Phase 3 `AcceleratorPartitionScoreModel` summary path.
- CLI write-boundary checks connect `TuningCli`, `PersistencePolicy`, `JsonFileBestProfileStore`, calibration save helpers, and benchmark report stores.

</code_context>

<specifics>
## Specific Ideas

- Keep `ACCELERATOR_BUFFER_MODE` in graph autotune because it affects concrete graph execution flow rather than reusable platform capability alone.
- Keep `METAL_SELECTION` platform-owned and opt-in so default CPU calibration remains stable and does not silently choose accelerator enablement.
- Treat `MetalTransferModel` as graph/planner policy for now; measured constants may later come from calibration, but choosing the planner model is graph-side.
- Build Phase 4 around an explicit ownership matrix and tests, not a docs-only audit.

</specifics>

<deferred>
## Deferred Ideas

- Profile IO strictness, profile migration compatibility, cost-input subset selection, and benchmark write-boundary enforcement remain Phase 4 implementation areas for research/planning; they were not discussed with the user in this context pass.

</deferred>

---

*Phase: 04-tuning-and-profile-ownership-audit*
*Context gathered: 2026-04-30*
