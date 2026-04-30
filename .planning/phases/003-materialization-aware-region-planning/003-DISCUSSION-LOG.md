# Phase 3: Materialization-Aware Region Planning - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 003-materialization-aware-region-planning
**Areas discussed:** Planner Bias, Cost Model Inputs, Trace Detail, CPU Safeguards

---

## Planner Bias

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative wins only | Accept accelerator regions only when estimated benefit is clearly above CPU plus transfer/materialization cost. | |
| Balanced default | Prefer longer device-owned accelerator regions when they reduce CPU boundaries, but require traceable cost justification and keep CPU alternatives intact. | ✓ |
| Aggressive exploration | Bias strongly toward longer Metal regions to expose performance potential. | |
| Other | Freeform preference. | |

**User's choice:** Balanced default.
**Notes:** Phase 3 should prefer longer device-owned regions when they reduce CPU boundaries, but every acceptance needs cost justification and CPU alternatives stay available.

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit margin | Require a configurable minimum benefit margin over CPU. | |
| Tie goes to CPU | Equal or ambiguous scores choose CPU. | |
| Tie goes to accelerator | Equal or ambiguous scores choose longer device-owned flow when legality and safety gates pass. | ✓ |
| Other | Freeform preference. | |

**User's choice:** Tie goes to accelerator.
**Notes:** The default remains balanced, but ambiguity nudges toward reducing CPU/GPU boundaries.

| Option | Description | Selected |
|--------|-------------|----------|
| Keep neutral ops inside | Include legal neutral ops if they avoid materialization boundaries. | ✓ |
| Only include profitable ops | Split at neutral ops unless each node contributes positive value. | |
| Depends on op family | Allow layout/elementwise neutral ops inside, split around weak reductions/softmax/matmul. | |
| Other | Freeform preference. | |

**User's choice:** Keep neutral ops inside.
**Notes:** Optimize whole-flow profitability rather than per-op profitability.

---

## Cost Model Inputs

| Option | Description | Selected |
|--------|-------------|----------|
| Static estimates first | Use tensor byte counts, estimated work, layout class, boundary count, fallback mode, and dispatch constants. | ✓ |
| Profile-derived costs now | Wire existing profile/calibration values into planning immediately. | |
| Hybrid but static-default | Static estimates as default, optional hooks for reliable existing profile values. | |
| Other | Freeform preference. | |

**User's choice:** Static estimates first.
**Notes:** User explicitly asked not to forget Phase 4: profile/calibration-derived cost model updates should be planned there.

| Option | Description | Selected |
|--------|-------------|----------|
| Boundary-first | Boundary count, input/output bytes, CPU materialization count/reasons, tensor-array fallback risk. | |
| Full static model | Boundary signals plus estimated compute work, dispatch overhead, layout fallback class, avoided intermediate bytes. | ✓ |
| Minimal safe model | Estimated work plus transfer bytes plus runtime availability. | |
| Other | Freeform preference. | |

**User's choice:** Full static model.
**Notes:** Phase 3 model should include both boundary/materialization and compute/dispatch/layout signals.

| Option | Description | Selected |
|--------|-------------|----------|
| Named presets only | Conservative/balanced/aggressive style enums, no raw weights. | |
| Raw config weights | Expose individual penalties/credits in optimizer config. | |
| Presets plus internal constants | Expose named presets, keep raw weights internal until Phase 4. | ✓ |
| Other | Freeform preference. | |

**User's choice:** Presets plus internal constants.
**Notes:** Raw weights should wait for Phase 4 tuning/profile ownership decisions.

---

## Trace Detail

| Option | Description | Selected |
|--------|-------------|----------|
| Reason codes only | Compact accepted/rejected/split reasons; cost details hidden unless debug mode. | |
| Summary cost fields | Reason codes plus final score, boundary count, estimated transfers, estimated compute, selected preset. | ✓ |
| Full per-candidate breakdown | Every candidate gets all component costs and alternatives. | |
| Other | Freeform preference. | |

**User's choice:** Summary cost fields.
**Notes:** Default traces should be explanatory without becoming noisy.

| Option | Description | Selected |
|--------|-------------|----------|
| Selected only | Rejected candidates keep compact reasons. | |
| Rejected finalists too | Selected candidate plus top rejected competitors get score summaries and reason detail. | ✓ |
| All rejected candidates | Every rejected candidate includes cost fields. | |
| Other | Freeform preference. | |

**User's choice:** Rejected finalists too.
**Notes:** Include selected candidate plus top rejected competitors/finalists.

| Option | Description | Selected |
|--------|-------------|----------|
| Compile/selection traces only | Keep runtime traces focused on execution. | |
| Compile/selection plus benchmark reports | Planner decisions visible in traces and summarized in benchmark output. | ✓ |
| Every runtime step trace | Duplicate planner context into runtime step metadata. | |
| Other | Freeform preference. | |

**User's choice:** Compile/selection plus benchmark reports.
**Notes:** Runtime step traces should stay focused on execution metadata.

---

## CPU Safeguards

| Option | Description | Selected |
|--------|-------------|----------|
| Regression gate only | Rely on CPU tests/benchmarks after changes. | |
| CPU-first safety rules | Preserve CPU natural regions/fusion when accelerator benefit is not clearly better. | ✓ |
| Separate policy modes | Default CPU-biased and opt-in accelerator-biased mode. | |
| Other | Freeform preference. | |

**User's choice:** CPU-first safety rules.
**Notes:** Add tests for CPU-region preservation.

| Option | Description | Selected |
|--------|-------------|----------|
| CPU natural regions stay intact | Accelerator changes must not fragment CPU natural regions in CPU-only/default profiles. | |
| CPU fusion and BLAS stay competitive | Preserve fused elementwise and BLAS/matmul dispatch choices where CPU is better. | |
| Both | Protect CPU natural regions and CPU fusion/BLAS behavior. | ✓ |
| Other | Freeform preference. | |

**User's choice:** Both.
**Notes:** CPU hot paths remain first-class.

| Option | Description | Selected |
|--------|-------------|----------|
| CPU wins | Runtime/profile evidence overrides accelerator preference. | |
| Trace conflict, but still use static model | Record conflict; Phase 4 resolves profile-derived overrides. | |
| No profile evidence in Phase 3 | Ignore profile-derived speed evidence for selection, require CPU-focused regression tests. | ✓ |
| Other | Freeform preference. | |

**User's choice:** No profile evidence in Phase 3.
**Notes:** Phase 4 handles profile-derived cost updates.

---

## the agent's Discretion

- Exact class names, record shapes, and trace field names.
- Whether to extend existing scoring/trace records directly or add small adjacent records.

## Deferred Ideas

- Phase 4 should plan an update that wires reliable profile/calibration-derived costs into the accelerator cost model after tuning/profile ownership is audited.
