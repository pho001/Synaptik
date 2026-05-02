# Phase 29 Context: Metal DType ABI And Capability Truth

## Goal

Establish versioned Metal dtype ABI and capability contracts before widening native compute beyond the current FLOAT32 path.

## Requirements

- `METALDTYPE-01`: Metal capability probing distinguishes representable storage dtypes, legal compute dtypes, legal output dtypes, and operation-specific dtype support for `FLOAT32`, `BFLOAT16`, `BOOL`, `INT32`, and `FLOAT64`.
- `METALDTYPE-02`: A versioned Metal native ABI can carry dtype metadata beyond the current `_f32` path without letting older `.dylib` builds silently claim unsupported dtype execution.
- `METALDTYPE-03`: Coverage reports and planner diagnostics expose dtype-specific support, fallback, and capability reasons without conflating dtype residency with native dtype compute.

## Locked Decisions

- Phase 29 is contract, capability, ABI discovery, diagnostics, tests, and docs. It must not implement BF16, BOOL-producing, INT32, or FLOAT64 native compute.
- Current `_f32` compile and execute symbols remain the safe default path. Any wider dtype ABI must be optional-symbol and version gated.
- Missing dtype ABI v3 support means "unsupported for widened dtype execution", not "bridge unavailable for existing FLOAT32 execution".
- Dtype support must distinguish:
  - storage representability in runtime buffers,
  - external input legality by role,
  - compute dtype legality,
  - output dtype legality,
  - per-operation dtype support or rejection reason.
- `BOOL` remains legal only for current predicate input roles until Phase 31 adds BOOL-producing compute.
- `INT32` remains storage/index-residency only until Phase 32 adds index execution.
- `BFLOAT16` remains storage-residency only until Phase 30 adds compute/output support.
- `FLOAT64` must receive explicit unsupported capability decisions instead of accidental support via CPU dtype defaults.
- CUDA should keep capability-gated behavior and shared report vocabulary where useful, but this phase must not imply CUDA parity.

## Current Findings

- `MetalMpsCapabilities` is the Java source of truth today, but it only exposes boolean helpers for compute/output/external input and a short native dtype code mapper.
- `MetalMpsFfmBridge.compile(...)` sends external input dtype codes to the native `_f32` compile ABI, but node output dtypes and final output dtypes are currently assumed FLOAT32 on the Java side.
- The native shim exposes layout ABI v2 optional/version probes. That pattern should be reused for dtype ABI v3 discovery.
- Existing reports already have language around dtype residency versus native dtype compute, but Phase 29 needs stable reason codes and data that can be consumed by coverage gates and planner diagnostics.
- Local benchmark/profile artifacts under `profiles/platform/...` remain non-canonical and must stay unstaged.

## Phase Direction

The correct implementation should create a narrow but durable dtype truth layer:

- model dtype support by role and operation without widening execution;
- add native dtype ABI v3 discovery and validation symbols in a backward-compatible way;
- expose dtype support decisions in coverage/report diagnostics;
- update docs and tests so Phase 30-32 can add actual dtype execution without reopening ABI semantics.

## Canonical References

- `.planning/ROADMAP.md` - Phase 29 scope and success criteria.
- `.planning/REQUIREMENTS.md` - `METALDTYPE-01/02/03`.
- `src/main/java/backend/metal/MetalMpsCapabilities.java` - current Metal dtype legality source.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - current FFM ABI discovery and compile path.
- `src/main/java/backend/metal/bridge/MetalMpsBridgeCapabilities.java` - current native capability record.
- `src/main/native/apple/synaptik_apple_mps_stub.m` - native shim and layout ABI v2 probe pattern.
- `docs/metal-backend.md` - public Metal backend contract.
- `docs/gpu-lowering-coverage.md` - public lowering and dtype residency contract.

---

*Phase: 29-metal-dtype-abi-and-capability-truth*
*Context gathered: 2026-05-02*
