# Phase 13: Coverage Benchmark And Regression Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md; this log preserves the alternatives considered.

**Date:** 2026-04-30T20:14:16Z
**Phase:** 13-coverage-benchmark-and-regression-gate
**Areas discussed:** Coverage evidence contract, Representative workloads, Regression gate behavior, Artifact hygiene
**Mode:** `$gsd-next` auto-routed to Phase 13 context capture; noninteractive defaults selected.

---

## Coverage Evidence Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Coverage-first report contract | Make GPU coverage ratio, region length, fallback reasons, materializations, copy timing, residency, and handoffs explicit report fields. | ✓ |
| Timing-first benchmark closure | Treat raw performance delta as primary, with coverage fields as secondary diagnostics. | |
| Minimal trace-only closure | Keep evidence only in traces and avoid benchmark report changes. | |

**User's choice:** Auto-selected coverage-first report contract.
**Notes:** This matches the roadmap statement that the milestone success metric is coverage and materialization behavior, not only raw speed.

---

## Representative Workloads

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing workload catalog | Use `StandardWorkloads` and existing transformer/MLP/conv/normalization specs before adding new workload infrastructure. | ✓ |
| Create new synthetic-only fixtures | Build isolated synthetic graphs only for report metrics. | |
| Depend on local debug benchmarks | Use debug benchmark classes as the primary evidence path. | |

**User's choice:** Auto-selected reuse existing workload catalog.
**Notes:** New deterministic tests may still use small fixtures, but planning should prefer existing workload abstractions and avoid committing local benchmark output.

---

## Regression Gate Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Fail on hidden exits for supported targets | Gate supported workloads against lost GPU coverage, unexpected CPU materialization, and hidden tensor-array fallback. | ✓ |
| Advisory-only warnings | Emit report warnings but never fail tests. | |
| Require native hardware everywhere | Make native Metal/CUDA execution mandatory for all gates. | |

**User's choice:** Auto-selected fail on hidden exits for supported targets.
**Notes:** Native Metal/CUDA checks remain capability-gated; portable Java gates must still prove fallback/report contracts.

---

## Artifact Hygiene

| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic fixtures only | Commit only stable tests/docs/fixtures; keep local `build/` and profile tuning output untracked. | ✓ |
| Commit latest local reports | Treat generated local benchmark reports as phase evidence. | |
| Persist calibration updates | Update platform profiles as part of coverage gate closure. | |

**User's choice:** Auto-selected deterministic fixtures only.
**Notes:** Existing local profile tuning files under `profiles/platform/.../tuning/abc/*` remain out of scope.

---

## the agent's Discretion

- Exact metric class and renderer field names are left to planning.
- Planning may choose whether to extend existing report records or add a dedicated coverage summary record.

## Deferred Ideas

- Broad new GPU operation support beyond Phase 11 matrix coverage.
- Larger fused GPU kernels beyond Phase 12's safe compound subset.
- Higher-rank native ABI expansion beyond current workload needs.
