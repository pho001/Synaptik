# Phase 4: Tuning And Profile Ownership Audit - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 4-Tuning And Profile Ownership Audit
**Areas discussed:** Knob ownership boundary

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Knob ownership boundary | What belongs to graph autotune vs platform calibration? Includes `ACCELERATOR_BUFFER_MODE`, `METAL_SELECTION`, transfer model/presets, and ownership enforcement. | yes |
| Profile IO strictness | How strictly loader should validate schema/version and accelerator buffer/layout fields. | |
| Cost model inputs from profiles | Which calibrated values may feed the accelerator cost model. | |
| Persistence and benchmark write boundary | How hard to enforce benchmark read-only behavior and profile-writing flow boundaries. | |

**User's choice:** 1
**Notes:** User chose to discuss only the knob ownership boundary.

---

## ACCELERATOR_BUFFER_MODE Ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Graph autotune | Buffer mode is a workload/region-flow decision affecting materialization boundaries. | yes |
| Platform calibration | Buffer mode is a hardware/runtime capability and should be calibrated per platform/dtype. | |
| Hybrid | Calibration decides safe/default capability; graph autotune chooses among allowed variants. | |
| Agent decides | Planner chooses after codebase audit. | |

**User's choice:** 1
**Notes:** Locked as graph autotune ownership.

---

## METAL_SELECTION Ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Platform calibration, opt-in only | Runtime/hardware area; not included automatically in standard calibration. | yes |
| Graph autotune | Graph autotune tries Metal enablement and minimum-work thresholds as workload policy. | |
| Split | `enabled` and `requireRuntimeAvailability` stay platform-owned; `minimumEstimatedWork` moves to graph autotune. | |
| Agent decides | Planner chooses after audit. | |

**User's choice:** 1
**Notes:** Locked as platform calibration ownership with explicit accelerator opt-in only.

---

## MetalTransferModel Ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Graph autotune | Transfer model/preset changes graph/region planning policy. | yes |
| Platform calibration | Hardware transfer-cost profile should be calibrated per platform/dtype. | |
| Split | Calibration provides measured transfer/copy constants; graph autotune selects planner policy using those constants. | |
| Static only for now | Keep out of profile persistence and use code presets only. | |

**User's choice:** 1
**Notes:** Locked as graph autotune/planner policy. This does not prohibit calibration-derived constants later; it locks the policy selection side.

---

## Ownership Enforcement

| Option | Description | Selected |
|--------|-------------|----------|
| Test-enforced ownership matrix | Classify every knob and enforce candidate spaces do not mutate foreign knobs. | yes |
| Docs-first audit | Document classification and fix only obvious issues. | |
| Allowlist exceptions | Enforce ownership with temporary exceptions for unstable accelerator knobs. | |
| Agent decides | Planner chooses strictness based on risk and scope. | |

**User's choice:** 1
**Notes:** Locked as test-enforced ownership matrix. Classification categories: `graph/workload`, `platform/dtype`, and `obsolete`.

---

## Completion Gate

| Option | Description | Selected |
|--------|-------------|----------|
| Create context | Enough decisions captured for Phase 4 planning. | yes |
| More questions | Continue with additional details before writing context. | |

**User's choice:** 1
**Notes:** User chose to create `04-CONTEXT.md` after the ownership boundary discussion.

## the agent's Discretion

- Exact implementation form of the ownership matrix.
- Whether to extend `CalibrationCandidateOwnershipTest` or add a parallel graph autotune ownership test.
- Details of profile IO strictness, cost-input subset, and benchmark write-boundary enforcement remain for research/planning.

## Deferred Ideas

- None outside Phase 4 scope. Undiscussed Phase 4 areas remain open for planner/research rather than deferred to another phase.
