# Tuning Reporting

The reporting layer turns measurement and search results into:

- human-readable explanations
- machine-readable artifacts

It does not execute workloads and does not search candidates.

## Reporting Surfaces

### Benchmark reporting

Important types:

- [report/BenchmarkCandidateReport.java](./report/BenchmarkCandidateReport.java)
- [report/BenchmarkReport.java](./report/BenchmarkReport.java)
- [report/BenchmarkSuiteReport.java](./report/BenchmarkSuiteReport.java)
- [report/TextBenchmarkReportRenderer.java](./report/TextBenchmarkReportRenderer.java)
- [report/JsonBenchmarkReportRenderer.java](./report/JsonBenchmarkReportRenderer.java)

Benchmark reporting answers:

- which candidate won
- how fast each candidate was in steady state
- how compile/prepare/traced execution compared
- what the hot traced steps were

Typical text fields today include:

- candidate name
- status
- compile time
- prepare time
- traced run time
- steady-state mean / median / p90
- baseline delta
- validation status
- top hot steps
- detailed step metadata

### Autotune reporting

Important types:

- [session/TuningResult.java](./session/TuningResult.java)
- [report/TuningSummary.java](./report/TuningSummary.java)
- [report/TextTuningResultRenderer.java](./report/TextTuningResultRenderer.java)
- [report/JsonTuningResultRenderer.java](./report/JsonTuningResultRenderer.java)

Autotune reporting answers:

- which profile won
- how many candidates were selected/evaluated
- how many were valid
- how many finalists survived
- whether persistence happened
- how many history entries were written

### Platform calibration reporting

Important types:

- [session/PlatformCalibrationResult.java](./session/PlatformCalibrationResult.java)
- [session/PlatformCalibrationStepResult.java](./session/PlatformCalibrationStepResult.java)
- [report/TextPlatformCalibrationResultRenderer.java](./report/TextPlatformCalibrationResultRenderer.java)
- [report/JsonPlatformCalibrationResultRenderer.java](./report/JsonPlatformCalibrationResultRenderer.java)

Calibration reporting answers:

- which step families ran
- which candidate won each step
- what score policy was used
- what the final runtime profile became

## Where Numbers Come From

A typical report may contain several kinds of timing:

- compile
  - compile-time graph work
- prepare
  - backend/runtime metadata preparation
- traced run
  - one fully traced execution run
- steady-state
  - repeated runs used for benchmark/autotune scoring

These should not be conflated.
For example:

- a candidate with slightly slower compile but much faster steady-state may still be the correct winner
- traced run is diagnostic and often more expensive than steady-state because it captures detailed step metadata

## Worked Example

Suppose benchmark compares:

- `baseline-no-opt`
- `best-profile`

and steady-state medians are:

- baseline = `10.2 ms`
- best profile = `7.9 ms`

Then reporting should make all of the following visible:

- winner = `best-profile`
- speedup vs baseline ≈ `1.29x`
- absolute delta = `-2.3 ms`
- whether compile/prepare also changed
- which traced hot steps dominate the runtime

That last point is important because the report is not just a leaderboard.
It is also a diagnosis artifact.

## Progress Events vs Final Reports

Progress listeners are for live orchestration visibility.
Final reports are for post-run evidence and inspection.

That boundary matters because:

- progress events can be partial
- final reports should be complete enough to understand the result without rerunning the workflow

## Persistence Relationship

Reports are explain artifacts.
They are not the runtime source of truth.

That means:

- do not execute "from a report"
- execute from stored `ExecutionProfile` / `PlatformRuntimeProfile`
- use reports to understand why a winner was chosen
