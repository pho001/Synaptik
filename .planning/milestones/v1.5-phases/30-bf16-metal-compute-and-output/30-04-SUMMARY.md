# Summary 30-04: Docs And Regression Closure

**Status:** Complete
**Completed:** 2026-05-02

## What Changed

- Updated `docs/metal-backend.md` to describe scoped Metal BF16 compute/output, dtype ABI v3 requirements, BF16-supported operation families, and unsupported BF16 families.
- Updated `docs/gpu-lowering-coverage.md` with the Phase 30 BF16 contract, BF16 hot-path targets, dtype evidence expectations, and fallback boundaries.
- Updated `docs/troubleshooting.md` with BF16 native capability checks and trace diagnostics for unexpected BF16 fallback.
- Updated `docs/development.md` and `MetalMpsCapabilities` comments so source guidance no longer says Metal compute/output is FLOAT32-only.
- Verified local benchmark/profile artifacts remain unstaged.

## Verification

```bash
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuCoverageSummaryTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

All Gradle commands and `git diff --check` passed. `git status --short profiles/platform` still shows only local tuning/profile artifacts, which were intentionally not staged.

## Notes

- Docs explicitly avoid claiming universal BF16 Metal support.
- `METALBF16-01`, `METALBF16-02`, and `METALBF16-03` are ready for phase verification closure.
