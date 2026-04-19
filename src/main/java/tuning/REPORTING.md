# Tuning Reporting

The reporting layer does two things:

- turns results into a human-readable form
- turns results into machine-readable explain artifacts

It does not do:

- execution
- candidate search
- persistence decisions

## Reading Guide

Use this document if you want to understand:

- what benchmark/tuning/calibration reports actually contain today
- where compile/prepare/run numbers come from
- where hot steps appear in text reports
- what belongs in progress events and what belongs in final reports

## Benchmark Reporting

Main types:

- [BenchmarkCandidateReport.java](./report/BenchmarkCandidateReport.java)
- [BenchmarkReport.java](./report/BenchmarkReport.java)
- [BenchmarkSuiteReport.java](./report/BenchmarkSuiteReport.java)
- [TextBenchmarkReportRenderer.java](./report/TextBenchmarkReportRenderer.java)

Today the benchmark report answers:

- which candidate won
- how many candidates succeeded/failed
- what the steady-state times were
- how candidates compare to baseline
- what the hot runtime steps were in the traced run

### What `TextBenchmarkReportRenderer` Actually Shows Today

A summary table with fields:

- `name`
- `status`
- `compileMs`
- `prepareMs`
- `traceMs`
- `medianMs`
- `p90Ms`
- `vsBaseline`

And then, for each candidate, a detail section:

- validation status
- optimizer stage order
- compile/prepare/traced run time
- step count
- `parallelUsed`
- `vectorUsed`
- steady-state mean/median/p90
- `speedupVsBaseline`
- top hot steps
- full step dump with trace metadata

That means benchmark reporting is not just "name and median". It is a useful performance diagnostic artifact.

## Autotune Reporting

Main types:

- [TuningResult.java](./session/TuningResult.java)
- [TuningSummary.java](./report/TuningSummary.java)
- [TextTuningResultRenderer.java](./report/TextTuningResultRenderer.java)

Autotune reporting answers:

- which `ExecutionProfile` won
- which search strategy was used
- how many candidates were selected/evaluated
- how many passed validation
- how many history entries were written

### What `TextTuningResultRenderer` Shows Today

- `bestProfile`
- `persisted`
- `summary`
- `strategy`
- `selected`
- `evaluated`
- `valid`
- `finalists`
- `historyEntriesWritten`
- `bestMedianMs`

and then a finalists table:

- name
- median
- mean
- validation status

## Platform Calibration Reporting

Main types:

- [PlatformCalibrationResult.java](./session/PlatformCalibrationResult.java)
- [PlatformCalibrationStepResult.java](./session/PlatformCalibrationStepResult.java)
- [PlatformCalibrationCandidateSummary.java](./session/PlatformCalibrationCandidateSummary.java)
- [PlatformCalibrationScore.java](./session/PlatformCalibrationScore.java)
- [TextPlatformCalibrationResultRenderer.java](./report/TextPlatformCalibrationResultRenderer.java)
- [JsonPlatformCalibrationResultRenderer.java](./report/JsonPlatformCalibrationResultRenderer.java)

The platform calibration report answers:

- what the seed runtime profile was
- which family steps were run
- which candidate won in each step
- which score metric was used
- which final runtime profile was produced

### What `TextPlatformCalibrationResultRenderer` Shows Today

- `platformId`
- `createdAt`
- `persisted`
- `outputProfilePath`
- `profileName`
- `dataType`
- `mode`
- `seedRuntimeProfile`
- `finalRuntimeProfile`
- hardware summary
- a step table:
  - `name`
  - `family`
  - `seedRuntime`
  - `selectedExec`
  - `score`
  - `metric`

## Trace-Derived Reporting

Reporting does not generate trace data on its own.

Trace comes from the execution layer through `MeasurementResult.trace()`.

It typically contains:

- compile trace
- prepare trace
- run trace
- step traces

The benchmark renderer can then extract:

- compile/prepare duration
- traced cold-run duration
- hot steps
- layout/dispatch/reduction/matmul/fused metadata on each step

## Why Reporting Is Separate

Reporting must not decide:

- whether a candidate is valid
- how measurement is done
- what gets persisted

It should only convert already-existing DTOs into output.

That guarantees:

- the text renderer does not change result semantics
- the JSON renderer is not the execute source of truth

## Progress Reporting

In addition to final reports, there are also live progress events.

### Autotune progress

Types:

- [AutotuneProgressEvent.java](./session/AutotuneProgressEvent.java)
- [AutotuneProgressPhase.java](./session/AutotuneProgressPhase.java)

Current phases:

- `STARTED`
- `SEARCH_BATCH`
- `CANDIDATE_VALIDATING`
- `CANDIDATE_INVALID`
- `CANDIDATE_MEASURING`
- `CANDIDATE_MEASURED`
- `CANDIDATE_FAILED`
- `ROUND_COMPLETED`
- `COMPLETED`

### Platform calibration progress

Types:

- [PlatformCalibrationProgressEvent.java](./session/PlatformCalibrationProgressEvent.java)
- [PlatformCalibrationProgressPhase.java](./session/PlatformCalibrationProgressPhase.java)

Current phases:

- `STARTED`
- `FAMILY_STARTED`
- `WORKLOAD_STARTED`
- `CANDIDATE_VALIDATING`
- `CANDIDATE_INVALID`
- `CANDIDATE_MEASURING`
- `CANDIDATE_MEASURED`
- `CANDIDATE_FAILED`
- `CANDIDATE_SCORED`
- `FAMILY_COMPLETED`
- `COMPLETED`
- `FAILED`

These events are not a replacement for the final report. They are meant for:

- long terminal runs
- CI log visibility
- debugging stalls and candidate explosions

## JSON Expectations

JSON renderers should be:

- machine-readable explain artifacts
- suitable for diffing or archiving

They should not be:

- execute source of truth
- the only persistence artifact

The source of truth remains:

- `PlatformRuntimeProfile` for platform defaults
- `ExecutionProfile` for graph winners

## Example: Benchmark Report Interpretation

If a benchmark report shows:

- `bestMedianMs` better than baseline
- but `traceMs` worse

it may mean:

- higher cold/traced overhead
- but better steady-state

So it is worth tracking:

- compile/prepare/traced run
- steady-state median
- hot steps

not just one number.

## Example: Calibration Report Interpretation

If in a platform calibration report you see:

- a good matmul score
- but a later fused family regression

that is expected in a sequential family flow:

- each later step already starts from the previous winner
- the report preserves the audit trail of what each step changed

## Common Mistakes

- using the text report as the only persistent artifact
- reading speedup without the context of validation status
- confusing traced cold run with steady-state median
- ignoring the hot-step dump during performance regression analysis

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- search: [SEARCH.md](./SEARCH.md)

## Reading Benchmark Reports In Practice

A useful mental model is:

- `compileMs`
  - graph compilation and optimizer cost
- `prepareMs`
  - backend-specific preparation cost
- `traceMs`
  - one traced cold run with step metadata
- `medianMs`
  - steady-state throughput signal

So a candidate can legitimately:

- lose on `traceMs`
- but still win on `medianMs`

That usually means:

- cold start got a bit heavier
- steady-state execution got faster

## Hot-Step Interpretation

When the benchmark report prints hot steps, treat them as:

- the first place to inspect real bottlenecks
- not a proof that optimizer shape alone is wrong

For example:

- a hot `FUSED` step suggests looking at fused profitability, family thresholds, or ASM width
- a hot `MATMUL` step suggests looking at tiles, microkernels, BLAS crossover, or shape gates
- a hot `CONV2D_GEMM` step suggests looking at conv2d lowering and GEMM crossover policy

## Why Reports And Persistence Stay Separate

Even a perfect JSON report should not become runtime config.
Reports are for:

- explanation
- audit
- CI artifacts
- regression comparison

Profiles are for execution.
