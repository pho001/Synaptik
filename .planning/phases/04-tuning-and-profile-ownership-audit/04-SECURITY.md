---
phase: 04
slug: tuning-and-profile-ownership-audit
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 04 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Graph autotune -> executable profile assembly | Workload-specific graph policy candidates are assembled with calibrated runtime profiles. | Candidate knob assignments, optimizer policy, runtime overrides |
| Platform calibration -> runtime profile persistence | Hardware/dtype calibration writes reusable platform runtime defaults. | Platform profile JSON, schema versions, accelerator runtime fields |
| Profile persistence -> CLI runtime loading | CLI flows load persisted calibration and best-profile artifacts before autotune or benchmark. | PlatformRuntimeProfile and ExecutionProfile artifacts |
| Runtime profile -> accelerator cost model | Prepare-time backend selection consumes audited RuntimeConfig values. | Runtime thresholds and accelerator cost factors |
| Benchmark CLI -> local artifact tree | Benchmark commands read profiles and produce reports without mutating best-profile/history/calibration state. | Profile paths, report text/json, local generated artifacts |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-4-01 | Tampering | Graph autotune candidate space | mitigate | `GraphAutotuneCandidateSpace` validates declared knob assignments with `TuningKnobOwnership.validateGraphWorkload(...)`; graph candidate tests assert generated knobs are `GRAPH_WORKLOAD`. | closed |
| T-4-02 | Tampering | Platform calibration families | mitigate | `CalibrationFamilyRegistry.validateCandidateChanges(...)` checks family-owned knobs and calls `TuningKnobOwnership.validatePlatformDtype(...)`; calibration ownership tests assert platform candidates stay `PLATFORM_DTYPE`. | closed |
| T-4-03 | Tampering | Tuning knob registry | mitigate | `TuningKnobOwnership.ownerOf(...)` rejects unknown knobs; ownership tests cover unknown knob rejection and generated candidate knob ownership. | closed |
| T-4-04 | Tampering | Platform runtime profile IO | mitigate | `PlatformRuntimeProfileIO.fromJsonStrict(...)` validates `cudaBufferBindingMode`, `openclBufferBindingMode`, and `metalBufferBindingMode`, and tests assert invalid present values fail with the offending key. | closed |
| T-4-05 | Tampering | Platform runtime profile schema | mitigate | Strict IO validates `plannerSchemaVersion` and `persistenceSchemaVersion` before loading; tests assert unsupported schema versions fail loudly. | closed |
| T-4-06 | Availability | Legacy profile migration | mitigate | Strict loading allows absent legacy accelerator buffer keys to use fallback defaults while invalid present fields still fail; regression tests cover missing-key defaults. | closed |
| T-4-07 | Tampering | Accelerator prepare-time cost model | mitigate | `ProfileDerivedAcceleratorCostFactors.fromRuntimeConfig(...)` derives costs from supplied `RuntimeConfig`; backend cost packages have no direct profile store or local profile path references. | closed |
| T-4-08 | Denial of service | CPU hot-path preservation | mitigate | `AcceleratorPlanCostModel` keeps minimum-work and materialization-cost rejection gates; CPU natural, fusion, BLAS, and prepared-execution safeguard tests pass. | closed |
| T-4-09 | Repudiation | Backend selection diagnostics | mitigate | Runtime-derived summaries use preset `PROFILE_DERIVED` and preserve existing reason codes including `estimated-work-below-minimum` and `rejected-materialization-cost`. | closed |
| T-4-10 | Tampering | Benchmark CLI commands | mitigate | `TuningCli.CommandKind.writesProfileArtifacts()` marks benchmark commands read-only; parsing tests assert benchmark commands do not write profile artifacts. | closed |
| T-4-11 | Repudiation | Benchmark/report documentation | mitigate | `tuning/PERSISTENCE.md` states benchmark reports are explain artifacts, not runtime sources of truth, and architecture docs point to ownership boundaries. | closed |
| T-4-12 | Tampering | Final artifact hygiene | mitigate | `04-04-SUMMARY.md` records final `git status --short` hygiene; only intended planning files were staged, with local `profiles/platform/.../tuning/abc/*` and `.planning/tmp/` left unstaged. | closed |

*Status: open / closed*
*Disposition: mitigate (implementation required) / accept (documented risk) / transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 12 |
| Closed | 12 |
| Open | 0 |

## Evidence

| Evidence | Result |
|----------|--------|
| `rg -n "validateGraphWorkload|validatePlatformDtype|knownKnobs|ownerOf|acceleratorBufferModeIsGraphWorkloadOwned|metalSelectionIsPlatformDtypeOwned|metalTransferModelIsGraphWorkloadOwned|standardGraphAutotuneCandidatesOnlyMutateGraphOwnedKnobs|standardFamilyCandidatesOnlyReportOwnedKnobs" src/main/java/tuning src/test/java` | PASS |
| `rg -n "fromJsonStrict|loadStrict|validateSupportedSchemaVersions|validateBufferBindingMode|Unsupported plannerSchemaVersion|Unsupported persistenceSchemaVersion|strictPlatformProfileLoaderRejectsInvalidAcceleratorBufferMode" src/main/java/config/profile src/main/java/tuning/calibration/run src/main/java/synaptik/app src/test/java` | PASS |
| `rg -n "ProfileDerivedAcceleratorCostFactors|fromRuntimeConfig|PROFILE_DERIVED|minimumEstimatedWork|contiguousMaterializeThreshold|rejected-materialization-cost|estimated-work-below-minimum" src/main/java/backend/accelerator/select src/test/java/graph/optimizer/partition/cost src/test/java/PreparedExecutionBuildTest.java` | PASS |
| `rg -n "PlatformRuntimeProfileIO|JsonFileBestProfileStore|CalibrationArtifactLayout|profiles/platform" src/main/java/backend/accelerator/select src/main/java/graph/optimizer/partition/cost` | PASS - no direct profile IO references found |
| `rg -n "writesProfileArtifacts|benchmarkCommandsAreProfileReadOnly|autotuneAndCalibrationCommandsWriteProfileArtifacts|Benchmark commands are profile-read-only|Benchmark reports are explain artifacts|profiles/platform|\\.planning/tmp|git status --short" src/main/java/synaptik/app src/test/java/synaptik/app src/test/java/BenchmarkSessionTest.java src/main/java/tuning/PERSISTENCE.md .planning/phases/04-tuning-and-profile-ownership-audit/04-04-SUMMARY.md` | PASS |
| `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` | PASS |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 12 | 12 | 0 | Codex |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
