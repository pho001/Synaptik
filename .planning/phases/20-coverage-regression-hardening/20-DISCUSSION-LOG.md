# Phase 20: Coverage Regression Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-01
**Phase:** 20-Coverage Regression Hardening
**Mode:** `--auto`
**Areas discussed:** Gate scope and strictness, evidence source, target-specific expectations, portable and native evidence, reporting and closure

---

## Gate Scope And Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Timing-oriented gate | Fail when benchmark medians regress. | |
| Coverage/materialization gate | Fail on structured trace/report regressions; timing is supporting context only. | ✓ |
| Advisory-only report | Render evidence but do not fail. | |

**User's choice:** Auto-selected coverage/materialization gate.
**Notes:** Phase 20 roadmap requires auditable "hot path stayed on GPU" evidence, not a performance impression.

---

## Evidence Source

| Option | Description | Selected |
|--------|-------------|----------|
| Structured trace/report fields | Use `GpuCoverageSummary`, run trace metadata, benchmark report fields, and stable reason maps. | ✓ |
| Prose/string scraping | Parse human report text and native bridge log strings. | |
| Local benchmark artifacts | Treat machine-local tuning/profile outputs as proof. | |

**User's choice:** Auto-selected structured trace/report fields.
**Notes:** This carries forward the Phase 13-19 evidence contract and keeps tensor-array bridge execution distinct from native buffer coverage.

---

## Target-Specific Expectations

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 14 target registry | Gate `transformer_block_hot_path`, `mlp_classifier_small`, `conv2d_resnet_3x3`, and `layer_norm_small`. | ✓ |
| Opportunistic workloads | Gate whichever workload has convenient tests. | |
| Universal operation promises | Require unsupported conv/norm/loss coverage to pass natively. | |

**User's choice:** Auto-selected Phase 14 target registry.
**Notes:** Partially blocked conv/norm targets should assert explicit blocker evidence instead of pretending universal native support exists.

---

## Portable And Native Evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Portable-first plus capability-gated native evidence | Portable Java gates are mandatory; native Metal/CUDA pass/skip evidence is explicit. | ✓ |
| Native-only proof | Require local Metal/CUDA hardware for all Phase 20 gates. | |
| Ignore native evidence | Only run synthetic portable checks. | |

**User's choice:** Auto-selected portable-first plus capability-gated native evidence.
**Notes:** CUDA hardware absence remains an environment risk; it should not weaken portable failure semantics.

---

## Reporting And Closure

| Option | Description | Selected |
|--------|-------------|----------|
| Report gate inputs and results | Text/JSON reports and docs show gate inputs, failure reasons, native skip/pass status, and artifact hygiene. | ✓ |
| Tests only | Gate behavior is tested but not visible in reports/docs. | |
| New persisted benchmark artifact | Commit local report/profile outputs as canonical proof. | |

**User's choice:** Auto-selected report gate inputs and results.
**Notes:** Local `profiles/platform/.../tuning/abc/*` artifacts remain unstaged unless explicitly promoted by a future plan.

---

## the agent's Discretion

- Exact Java type names for target expectations and gate policy extensions.
- Exact split of implementation plans and test classes.
- Exact documentation pages updated for closure, as long as `docs/testing.md`, coverage semantics, and artifact hygiene remain clear.

## Deferred Ideas

- Vendor library routing through cuBLAS, cuDNN, MPSGraph, or similar libraries.
- Backend-native primitive cost model.
- Universal native coverage for reductions, normalizations, convolution, dynamic shape, sparse, high-rank, and advanced indexing.
- Public GPU tensor/device API.
- Treating local tuning/profile artifacts as canonical proof.
