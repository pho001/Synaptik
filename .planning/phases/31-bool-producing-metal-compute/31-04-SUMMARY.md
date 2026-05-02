# 31-04 Summary: Docs Parity And Report Closure

**Completed:** 2026-05-02
**Requirements:** METALBOOL-01, METALBOOL-02, METALBOOL-03

## Delivered

- Updated Metal backend docs to describe scoped BOOL compute/output support, internal BOOL mask residency, `WHERE` mask-chain expectations, and unsupported masked SDPA boundaries.
- Updated the GPU lowering coverage matrix so Metal compare/logical/BOOL-reduction rows are supported while CUDA BOOL-producing rows remain `UNSUPPORTED_DTYPE`.
- Added troubleshooting guidance for missing BOOL native evidence, missing BOOL `dtypeResidency`, tensor-array fallback, CPU fallback, and unexpected CPU materialization in mask chains.
- Added `31-VERIFICATION.md` mapping `METALBOOL-01/02/03` to implementation, tests, native parity evidence, and report gates.
- Marked Phase 31 requirements complete in `.planning/REQUIREMENTS.md`.

## Verification

```bash
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.lowering.MetalRegionLowererTest --tests GpuCoverageSummaryTest
./gradlew metalTest
git diff --check
git status --short profiles/platform
```

## Notes

- Direct masked/causal SDPA remains Phase 34 scope; Phase 31 only proves reusable BOOL mask residency.
- Local profile artifacts under `profiles/platform/...` remain unstaged.
