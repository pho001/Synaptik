---
phase: 17-normalization-reduction-and-loss-adjacent-lowering
plan: "01"
status: complete
requirements-completed: [GPUNORM-01, GPUNORM-02]
completed: 2026-05-01
---

# Phase 17 Plan 01: Coverage Matrix Contract Summary

Phase 17 coverage matrix contract now makes normalization, reduction, softmax-ish, conv, and loss-adjacent Metal/CUDA rows explicit and ties them to the Phase 14 hot-path targets.

## Phase 17 coverage matrix contract

- Added focused matrix tests for Phase 17 operation rows across `GPU_METAL` and `GPU_CUDA`.
- Added `entriesForFamily(...)` and `plannerUnsupportedDetail(...)` helpers to keep backend legality diagnostics sourced from the shared matrix.
- Added target evidence notes for `target=layer_norm_small`, `target=conv2d_resnet_3x3`, and `target=transformer_block_hot_path`.
- Documented the Phase 17 support/rejection contract in `docs/gpu-lowering-coverage.md`.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest` | Passed |
| `rg -n "Phase 17 normalization, reduction, and loss-adjacent contract|GPUNORM-01|GPUNORM-02|target=layer_norm_small|target=conv2d_resnet_3x3|target=transformer_block_hot_path|LOG_SOFTMAX remains lowered as SOFTMAX followed by LOG|unsupported loss-adjacent rows must remain visible fallback" docs/gpu-lowering-coverage.md` | Passed |

## Requirement Coverage

- `GPUNORM-01`: Shared Metal/CUDA matrix rows explicitly cover normalization, reduction, softmax-ish, and loss-adjacent families.
- `GPUNORM-02`: Matrix notes tie high-impact blockers to the Phase 14 target workloads.

## Commits

| Commit | Description |
|--------|-------------|
| `ce5de64` | Added failing-first Phase 17 lowering matrix contract tests. |
| `e4e40e2` | Added shared matrix helpers and target evidence notes. |
| `8b1ceee` | Documented the Phase 17 lowering contract. |

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
