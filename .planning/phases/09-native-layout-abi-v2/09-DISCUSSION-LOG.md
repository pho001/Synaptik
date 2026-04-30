# Phase 9: Native Layout ABI v2 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 09-native-layout-abi-v2
**Mode:** `--auto`
**Areas discussed:** ABI compatibility and scope, layout metadata shape, capability and version handshake, fallback and required-mode semantics, test and native rollout

---

## ABI Compatibility And Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Additive ABI v2 beside existing dense ABI | Preserve current dense buffer execution and add v2 metadata/capability paths incrementally. | ✓ |
| Replace dense ABI with v2 immediately | Forces all Metal/CUDA bridge paths through the new contract in one phase. | |
| Backend-specific v2 contracts only | Let Metal and CUDA diverge completely at the Java contract boundary. | |

**User's choice:** Auto-selected additive ABI v2 beside existing dense ABI.
**Notes:** This preserves v1.1 behavior and keeps Phase 9 focused on bridge contract readiness.

---

## Layout Metadata Shape

| Option | Description | Selected |
|--------|-------------|----------|
| Full layout descriptor | Include rank, full shape/strides, storage offset, logical bytes, physical span, access, backend id, dtype, layout class, and handle identity. | ✓ |
| Minimal v2 fields | Add only rank and strides to the current dim0-dim3 style ABI. | |
| Native-only descriptor | Keep Java common records unchanged and build metadata only in backend-specific bridges. | |

**User's choice:** Auto-selected full layout descriptor.
**Notes:** Full metadata is needed so later phases can reason about view residency without changing public `Tensor`.

---

## Capability And Version Handshake

| Option | Description | Selected |
|--------|-------------|----------|
| Optional symbols with separate layout capability | Missing v2 symbols do not break bridge availability; they only disable layout ABI v2. | ✓ |
| Require v2 symbols for bridge availability | Treat any missing v2 symbol as full Metal/CUDA bridge unavailable. | |
| Java-only version flag | Infer v2 support from Java code without native symbol confirmation. | |

**User's choice:** Auto-selected optional symbols with separate layout capability.
**Notes:** This matches existing graceful native bridge behavior and supports portable tests.

---

## Fallback And Required-Mode Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| New stable ABI-specific reason codes | Distinguish ABI unavailable/version mismatch/layout unsupported from generic non-contiguous fallback. | ✓ |
| Reuse existing generic layout reasons | Keep enum churn lower but lose diagnostic specificity. | |
| Throw generic native exceptions | Let backend exceptions surface directly. | |

**User's choice:** Auto-selected new stable ABI-specific reason codes.
**Notes:** Phase 13 depends on reportable, stable reasons; Phase 9 should establish them.

---

## Test And Native Rollout

| Option | Description | Selected |
|--------|-------------|----------|
| Java-first portable tests plus optional native checks | Prove metadata and fallback behavior without requiring local hardware; run Metal/CUDA native checks when available. | ✓ |
| Native-first rollout | Implement native symbol changes first and validate primarily through hardware/tooling gates. | |
| Java-only planning stub | Add records and tests but defer all native symbol work. | |

**User's choice:** Auto-selected Java-first portable tests plus optional native checks.
**Notes:** This follows prior v1.1 CUDA and Metal portability rules.

---

## the agent's Discretion

- Exact Java type names and package placement.
- Exact native symbol names and binary layout, as long as they are versioned and optional.
- Whether metadata crosses FFM as structs, parallel arrays, or packed descriptors.

## Deferred Ideas

- GPU-side layout/view execution — Phase 10.
- Broader GPU lowering coverage — Phase 11.
- Fused GPU region execution — Phase 12.
- Full coverage benchmarks and regression gates — Phase 13.
