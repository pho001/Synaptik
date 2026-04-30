# Phase 5: Accelerator Verification And Documentation Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 05-accelerator-verification-and-documentation-closure
**Areas discussed:** Evidence bar

---

## Evidence Bar

### Trace Evidence Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Assert core fields in tests | Tests must check accelerator path, buffer mode, fallback reason, materialization count/reason, copy times presence, and residency where available. | yes |
| Render-focused evidence | Tests only prove reports/traces render the fields; exact values are inspected through benchmark/report output. | |
| Strict end-to-end evidence | Tests must prove field presence and meaningful values across an actual Metal-backed flow where capability gates allow it. | |
| Other | Freeform answer. | |

**User's choice:** Assert core fields in tests.
**Notes:** This locks evidence as a testable contract rather than a render-only documentation/reporting feature.

### Benchmark Evidence Treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Report contract only | Tests assert benchmark reports include accelerator evidence fields, but measured benchmark output is not committed. | yes |
| Committed sample report | Add a small canonical example report/fixture showing the expected accelerator evidence shape. | |
| Measured closure artifact | Commit a real benchmark result from this machine as milestone evidence. | |
| Other | Freeform answer. | |

**User's choice:** Report contract only.
**Notes:** The user initially selected measured closure artifact, then explicitly asked to go back and changed the decision to report-contract only.

### Evidence Field Set

| Option | Description | Selected |
|--------|-------------|----------|
| ROADMAP minimum | Accelerator path, buffer mode, fallback reason, materialization count/reason, copy times, and storage residency. | |
| Minimum + planner context | Minimum plus selected accelerator candidate, top rejected finalist, cost summary, and region boundary counts. | yes |
| Minimum + operation breakdown | Minimum plus per-op/per-step breakdown for each part of the accelerator region. | |
| Other | Freeform answer. | |

**User's choice:** Minimum + planner context.
**Notes:** This ties runtime/report evidence to Phase 3/4 planner and cost-model decisions without requiring noisy per-operation dumps.

### Optional Metal Runtime Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Capability-gated strict tests | Java-side tests assert report/trace contract always; Metal-specific meaningful values are verified through `metalTest` and skipped when the native shim is unavailable. | yes |
| No native dependency | Everything is verified only through Java-side/stubbed contracts, with no `metalTest` dependency. | |
| Require Metal locally | Phase 5 closure requires successful `metalTest`, otherwise it is not complete. | |
| Other | Freeform answer. | |

**User's choice:** Capability-gated strict tests.
**Notes:** This preserves portable default tests while still requiring real Metal path evidence where the native shim is available.

---

## the agent's Discretion

- The user chose to stop discussion after Evidence bar and proceed to context creation.
- Benchmark workload shape, Metal correctness gate details, documentation organization, and hygiene mechanism are left to planner discretion within the roadmap success criteria and the evidence decisions above.

## Deferred Ideas

None.
