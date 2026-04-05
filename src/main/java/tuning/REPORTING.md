# Tuning Reporting

## Contents

- [Purpose](#purpose)
- [Benchmark Reports](#benchmark-reports)
- [Tuning Reports](#tuning-reports)
- [Baseline Reporting](#baseline-reporting)
- [Trace Reporting](#trace-reporting)
- [Search Tree Reporting](#search-tree-reporting)
- [Cross-Run Diff Reporting](#cross-run-diff-reporting)
- [JSON Contract Notes](#json-contract-notes)
- [Examples](#examples)

## Purpose

The reporting layer has two jobs:

1. make benchmark/autotune results readable for a human
2. make them exportable in machine-friendly form

That is why every major report family has:

- text renderer
- JSON renderer

## Benchmark Reports

Core types:

- [BenchmarkCandidateReport.java](./report/BenchmarkCandidateReport.java)
- [BenchmarkReport.java](./report/BenchmarkReport.java)
- [BenchmarkSuiteReport.java](./report/BenchmarkSuiteReport.java)

### Candidate report

Per candidate:

- candidate identity
- validation result
- measurement result
- success/failure
- failure reason
- baseline role

### Benchmark report

Per workload:

- workload name
- candidate list
- best candidate
- success/failure counts
- baseline references
- speedup helpers

### Suite report

Across workloads:

- total candidate count
- total success/failure count
- per-workload reports
- overall best measured candidate across the suite
- candidate summaries aggregated by candidate name
- suite-level hotspots collected from traced run steps

## Tuning Reports

Core types:

- [TuningSummary.java](./report/TuningSummary.java)
- [TuningResult.java](./session/TuningResult.java)
- [TextTuningResultRenderer.java](./report/TextTuningResultRenderer.java)
- [JsonTuningResultRenderer.java](./report/JsonTuningResultRenderer.java)

`TuningResult` now contains:

- best profile
- finalists
- summary string
- structured summary details
- persisted flag

### Structured summary

Current fields include:

- strategy name
- selected count
- evaluated count
- valid count
- finalist count
- history entries written
- best median time

## Baseline Reporting

Benchmarks now support two explicit baseline roles:

- `BASELINE_NO_OPT`
- `BASELINE_NO_OPT_CONSERVATIVE_RUNTIME`

Relevant types:

- [BenchmarkBaselineKind.java](./report/BenchmarkBaselineKind.java)
- [BaselinePolicy.java](./session/BaselinePolicy.java)
- [BenchmarkBaselineProfiles.java](./session/BenchmarkBaselineProfiles.java)

This allows reports to answer:

- how much did optimizer stages help?
- how much did tuned runtime help over no-opt + conservative runtime?

### Speedup fields

`BenchmarkReport` exposes:

- `speedupVsNoOpt(...)`
- `speedupVsNoOptConservativeRuntime(...)`

The renderers surface those values directly.

Interpretation:

- `> 1.0x` means the candidate is faster than the baseline
- `1.0x` means parity
- `< 1.0x` means the candidate is slower than the baseline
- `null` / `n/a` means no valid comparable baseline measurement existed

## Suite-Level Aggregation

The suite renderers now expose two additional views that matter in practice.

### Candidate summaries

Relevant type:

- [BenchmarkSuiteCandidateSummary.java](./report/BenchmarkSuiteCandidateSummary.java)

Purpose:

- answer how one candidate family behaves across many workloads

Current fields:

- candidate name
- baseline role
- number of workloads seen
- number of successful measurements
- average median time
- average speedup vs no-opt
- average speedup vs no-opt conservative runtime

This is the first useful cross-workload comparison layer.

### Suite hotspots

Relevant type:

- [BenchmarkSuiteHotspot.java](./report/BenchmarkSuiteHotspot.java)

Purpose:

- surface the slowest traced steps across the whole benchmark suite

Current fields:

- workload name
- candidate name
- op type
- trace label
- duration

This lets you answer:

- which exact step is dominating the suite
- whether the same candidate is consistently hot
- whether one workload family is the true bottleneck

## Trace Reporting

Benchmark reports now use execution trace data emitted by the core execution layer.

Important core trace types:

- [CompileTrace.java](../graph/execution/trace/CompileTrace.java)
- [PrepareTrace.java](../graph/execution/trace/PrepareTrace.java)
- [RunTrace.java](../graph/execution/trace/RunTrace.java)
- [ExecutionStepTrace.java](../graph/execution/trace/ExecutionStepTrace.java)

Current benchmark reporting uses:

- compile time
- prepare time
- traced representative run time
- hot steps
- step count

Important measurement detail:

- if steady-state measurement is enabled, `trace.run()` is taken from a traced run after warmup
- it is therefore intended to be representative of warm execution
- it is not just the very first cold-start execution snapshot

### Hot steps

Current benchmark report renderers show:

- top step durations from traced run

That is enough to identify major hotspots even before full family aggregation is added.

## Search Tree Reporting

Tree search debugging uses:

- [SearchTreeSnapshot.java](./search/SearchTreeSnapshot.java)
- [SearchTreeReport.java](./search/SearchTreeReport.java)
- [TextSearchTreeReportRenderer.java](./search/TextSearchTreeReportRenderer.java)
- [JsonSearchTreeReportRenderer.java](./search/JsonSearchTreeReportRenderer.java)

These are separate from benchmark reports because they describe:

- the search process
- not the execution result itself

### What tree reports show

- strategy name
- node count
- frontier size
- max depth
- node lineage
- frontier fingerprints

## Cross-Run Diff Reporting

The reporting layer now also supports comparing two already-produced runs.

This is intentionally a pure reporting concern:

- it does not re-execute workloads
- it does not require benchmark/autotune orchestration
- it compares already materialized report/result DTOs

Current diff types:

- [BenchmarkReportDiff.java](./report/BenchmarkReportDiff.java)
- [BenchmarkSuiteReportDiff.java](./report/BenchmarkSuiteReportDiff.java)
- [TuningResultDiff.java](./report/TuningResultDiff.java)

Current renderers:

- [TextBenchmarkSuiteReportDiffRenderer.java](./report/TextBenchmarkSuiteReportDiffRenderer.java)
- [JsonBenchmarkSuiteReportDiffRenderer.java](./report/JsonBenchmarkSuiteReportDiffRenderer.java)
- [TextTuningResultDiffRenderer.java](./report/TextTuningResultDiffRenderer.java)
- [JsonTuningResultDiffRenderer.java](./report/JsonTuningResultDiffRenderer.java)

Typical questions this answers:

- did the new run get faster or slower?
- did the winning candidate change?
- did one workload regress while the rest improved?
- did autotune converge on a different best profile?

## JSON Contract Notes

The JSON renderers intentionally use:

- numeric values for valid timings and speedups
- `null` for non-finite values such as missing speedups
- explicit `baselineKind` markers on benchmark candidates

Why this matters:

- downstream tools do not need to parse `NaN`/`Infinity`
- baseline rows stay machine-distinguishable from user candidates
- partial benchmark failure is still representable without breaking the whole report

## Examples

### Example: text benchmark report

```java
String text = TextBenchmarkReportRenderer.render(report);
```

Input:

- one `BenchmarkReport`

Output:

- human-readable summary
- candidate table
- per-candidate detail
- hot-step section

### Example: JSON benchmark suite report

```java
String json = JsonBenchmarkSuiteReportRenderer.render(report);
```

Output includes:

- suite summary
- overall best candidate
- candidate summaries
- suite hotspots
- workload reports
- per-candidate timing
- trace summary

Example fragment:

```json
{
  "name": "candidate-a",
  "baselineKind": "NONE",
  "success": true,
  "timing": {
    "compileMs": 2.104221,
    "prepareMs": 0.442991,
    "tracedRunMs": 1.334882,
    "meanMs": 0.991400,
    "medianMs": 0.982113,
    "p90Ms": 1.031287
  },
  "speedup": {
    "vsNoOpt": 1.317000,
    "vsNoOptConservativeRuntime": null
  }
}
```

Meaning:

- the candidate beat the no-opt baseline by `1.317x`
- no comparable conservative-runtime baseline measurement existed

### Example: tuning result

```java
String text = TextTuningResultRenderer.render(result);
String json = JsonTuningResultRenderer.render(result);
```

Output includes:

- best profile
- persistence flag
- strategy used
- finalist count
- finalist timing summary

### Example: suite diff

```java
BenchmarkSuiteReportDiff diff = BenchmarkSuiteReportDiff.compare(previousSuite, currentSuite);
String text = TextBenchmarkSuiteReportDiffRenderer.render(diff);
String json = JsonBenchmarkSuiteReportDiffRenderer.render(diff);
```

Output includes:

- previous/current overall best candidate
- workload-level best-candidate change summary
- current best timing and speedup vs previous run
