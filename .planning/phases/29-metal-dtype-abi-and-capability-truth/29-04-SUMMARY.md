# 29-04 Summary: Docs, Tests, And Migration Closure

## Completed

- Updated `docs/metal-backend.md` with dtype ABI v3 symbols and the current dtype truth table.
- Updated `docs/gpu-lowering-coverage.md` to distinguish dtype storage/residency, descriptor ABI support, native compute, and native output legality.
- Added tests for all public dtype role decisions, descriptor codes, execution ABI narrowness, and bridge dtype ABI capability reporting.
- Ran focused Java tests, `classes`, and native `metalTest`.
- Kept local profile/calibration artifacts unstaged.

## Verification

- `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest`
- `./gradlew classes`
- `./gradlew metalTest`

## Outcome

Phase 29 migration closure is in place. Phase 30-32 can add BF16, BOOL-output, and INT32 execution against a stable capability/ABI contract.
