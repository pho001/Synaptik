# Tuning Reporting

## Contents

- [Purpose](#purpose)
- [Benchmark Reporting](#benchmark-reporting)
- [Autotune Reporting](#autotune-reporting)
- [Platform Calibration Reporting](#platform-calibration-reporting)
- [Calibration Progress Reporting](#calibration-progress-reporting)
- [Trace-Derived Reporting](#trace-derived-reporting)
- [Diff Reporting](#diff-reporting)
- [JSON Expectations](#json-expectations)

## Purpose

The reporting layer has two jobs:

1. make results readable for humans
2. make results exportable for tooling

Reporting is intentionally separate from:

- execution
- measurement
- persistence decisions

## Benchmark Reporting

Core types:

- [BenchmarkCandidateReport.java](./report/BenchmarkCandidateReport.java)
- [BenchmarkReport.java](./report/BenchmarkReport.java)
- [BenchmarkSuiteReport.java](./report/BenchmarkSuiteReport.java)

Benchmark reporting answers:

- which concrete candidate won
- whether validation passed
- what the measured timings were
- how candidates compare against a baseline

Typical benchmark report contains:

- validation result
- measurement result
- best candidate
- speedup helpers
- suite-level summaries
- suite hotspots

## Autotune Reporting

Core types:

- [TuningResult.java](./session/TuningResult.java)
- [TuningSummary.java](./report/TuningSummary.java)
- [TextTuningResultRenderer.java](./report/TextTuningResultRenderer.java)
- [JsonTuningResultRenderer.java](./report/JsonTuningResultRenderer.java)

Autotune reporting answers:

- which graph candidate won
- how many candidates were evaluated
- what search strategy was used
- how many finalists survived validation/measurement

## Platform Calibration Reporting

Core types:

- [PlatformCalibrationResult.java](./session/PlatformCalibrationResult.java)
- [PlatformCalibrationStepResult.java](./session/PlatformCalibrationStepResult.java)
- [PlatformCalibrationCandidateSummary.java](./session/PlatformCalibrationCandidateSummary.java)
- [PlatformCalibrationScore.java](./session/PlatformCalibrationScore.java)
- [TextPlatformCalibrationResultRenderer.java](./report/TextPlatformCalibrationResultRenderer.java)
- [JsonPlatformCalibrationResultRenderer.java](./report/JsonPlatformCalibrationResultRenderer.java)

Platform calibration reporting answers:

- which runtime family was being calibrated
- which candidate won in each family step
- what score model was used
- what the candidate-level score breakdown looked like
- which runtime profile became the new family winner

## Calibration Progress Reporting

Platform calibration should also expose a live progress-reporting channel.

This is different from final calibration reporting.

Final report answers:

- what happened

Progress reporting answers:

- what is happening right now

The intended live hierarchy is:

- family
- workload/scenario
- candidate
- phase

Minimum live fields:

- `platformId`
- `family`
- `familyStepIndex`
- `familyStepCount`
- `workloadName`
- `workloadIndex`
- `workloadCount`
- `candidateId`
- `candidateIndex`
- `candidateCount`
- `phase`
- `currentLeaderId`
- `currentLeaderScore`
- `message`

Expected use:

- long-running calibration sessions
- local terminal monitoring
- CI visibility
- debugging of stalls, invalid candidates, or unexpectedly large candidate spaces

Boundary rule:

- progress reporting belongs to orchestration
- not to score policy
- not to measurement engine
- not to persistence

## Trace-Derived Reporting

Reporting does not generate trace data.

Execution does.

Trace-based reporting consumes:

- compile trace
- prepare trace
- run trace
- execution-step trace

and exposes:

- compile time
- prepare time
- traced run time
- step hotspots
- step count

## Diff Reporting

Current diff families:

- benchmark report diffs
- benchmark suite diffs
- tuning result diffs

They compare already materialized report DTOs.

They do not rerun workloads.

## JSON Expectations

JSON renderers should be treated as:

- machine-readable explain artifacts
- not execute source of truth

Source-of-truth persistence remains:

- `PlatformRuntimeProfile` for platform runtime defaults
- `ExecutionProfile` for runnable graph winners
