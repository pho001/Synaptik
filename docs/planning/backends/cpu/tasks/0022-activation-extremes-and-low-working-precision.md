# CPU 0022 — Correct activation extremes and low working precision

## Status

Draft

## Goal and class

Class C generated-code/numerical repair for project-review C1/C5. Implement after CPU 0021 is
integrated, with its final base and launch verification recorded before promotion to Ready.

## Scope

`CpuScalarEmitter`, `CpuVectorMath`, affected activation reference and generated compatibility
identity, plus focused generation/conformance/public-execution tests and Javadoc/explanation.

- Large finite ERF/GELU must not evaluate overflowing rational polynomials into NaN. Reproduced
  F64 +/-1e100 and F32 vector body 1e10 (scalar tail gives 1). Add a justified saturation guard
  before unsafe evaluation, preserving NaN, infinities, sign, and required zero behavior.
- Homogeneous BF16/F16 activation composites require named FLOAT32 working sites, not an entire
  DOUBLE expression followed by one conversion. BF16 GELU(-6) currently gives `0xb1cb` instead of
  the named-site zero. Round elementary results/constants at required F32 boundaries; do not
  weaken Model semantics to match a shared defective reference.
- Review all supported activation composite paths using that emitter. No unrelated family,
  capability expansion, new math library, or copied third-party source.

## Contracts and acceptance

Read AGENTS; root scope index/Core invariants/ownership/dependencies; foundational Model
profile-free numerical semantics and activation family Javadocs; backend-execution Concrete
backend modules, CPU backend routes, Profile-free backend capability and identity.
Use optimal clean Java for each specialized case as semantic/dataflow/hot-loop oracle.

- Independent exact/class/named-site tests cover large finite inputs, saturation boundary,
  NaN/Inf/zero, F32/F64, BF16/F16, heap/native, scalar/vector body/tail where supported.
- Change generated identity atomically and test stale rejection. Preserve supported strategies.
- Inspect emitted Class-File/decompilation for actual F32 sites and no hidden helper/allocation
  regression. Run small focused generated/direct diagnostic performance evidence for changed hot
  paths; no broad historical five-fork program or performance-policy change.
- Focused regressions then CPU test/Javadoc once; affected conformance/integration tests.
  Independent Class C review reuses evidence and checks numeric boundary derivation/code shape.
- Report files/commit/tests/doc impact/limitations and completion status; no push/shared plans.

## Dependencies and integration

- Depends on: CPU 0021 and current Model semantics; source-confirmed C1/C5.
- Conflicts with: all CPU generated schema, emitter, activation/reference and numeric writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS, after serial CPU predecessor.
- Common base revision: pending CPU 0021 integration.
- Integration order: CPU 0021 → 0022 → combined checkpoint.
- Integration validation: generated/semantic/targeted performance evidence, independent review,
  coordinator's final checkpoint and diff checks.
- Shared-document integration owner: coordinator.
