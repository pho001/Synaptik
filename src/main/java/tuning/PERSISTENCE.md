# Tuning Persistence

Persistence in the tuning layer has to distinguish what is:

- runtime source of truth
- search/history prior
- explain/report artifact

If those get mixed, the system drifts between:

- what was measured
- what gets executed later
- what the repository stores as the "winner"

## Main Artifact Types

### 1. Built-in defaults

These live in code and are not persisted tuning artifacts.

### 2. `PlatformRuntimeProfile`

Produced by platform calibration.

This is the machine-specific runtime-default artifact used later when assembling executable profiles.

### 3. Best `ExecutionProfile`

Produced by workload-specific autotune.

This is the runnable winner for one workload/hardware context.

### 4. Tuning history

Append-only candidate evidence:

- fingerprint
- validity
- scores
- measured timings

This is useful as a prior for history-aware search and for diagnosis.

### 5. Explain artifacts

Examples:

- text/json benchmark reports
- text/json autotune reports
- text/json calibration reports

These are for humans and tooling, not for direct execution.

## Preferred Layout Today

Preferred versioned layout:

```text
profiles/
  platform/
    <platform-id>/
      calibration/
      reports/
      tuning/
        abc/
```

### Platform calibration

Typical files:

- `profiles/platform/<platform-id>/calibration/<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.txt`

### Graph autotune

For workload `abc`:

- `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json`
- `profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl`

Best-profile and history records include structured autotune metadata:

- autotune kind
- graph autotune mode
- candidate kind
- candidate-space metadata
- runtime profile id
- production eligibility

Production best-profile lookup should treat research-only candidates as opt-in artifacts, not default production winners.

Executable-profile fingerprints and structured candidate identity are separate concepts:

- executable fingerprints identify the runnable `ExecutionProfile` used for deduplication and history lookup
- candidate identity additionally includes candidate kind and metadata for explanation and research/prod separation

## Canonical Calibration Layout

Platform calibration writes append-only schema-v2 artifacts under:

- `profiles/platform/<platform-id>/calibration/schema-v2/runs/<run-id>/...`
- `profiles/platform/<platform-id>/calibration/schema-v2/history/<dtype>/<mode>/<family-id>.jsonl`
- `profiles/platform/<platform-id>/calibration/schema-v2/latest/<dtype>/<mode>/profile.json`

Reports and history are diagnostic artifacts. Runtime loading uses only the latest profile.
Legacy calibration fallback reads under `build/...` have been removed.

## Generic Tensor Convenience Autotune Paths

There is one additional persistence area used by the `Tensor.compute(ComputeOptions)` convenience API when generic graph autotune is enabled:

```text
build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/...
```

This is intentionally separate from the main workload-specific versioned tuning tree because it is:

- graph-signature oriented
- convenience API driven
- generic rather than one named benchmark workload like `abc`

## Stores And IO Types

Important types:

- platform runtime profile:
  - [../config/profile/PlatformRuntimeProfileIO.java](../config/profile/PlatformRuntimeProfileIO.java)
  - [store/PlatformRuntimeProfileStore.java](./store/PlatformRuntimeProfileStore.java)
  - [store/JsonFilePlatformRuntimeProfileStore.java](./store/JsonFilePlatformRuntimeProfileStore.java)
- best profile:
  - [store/BestProfileStore.java](./store/BestProfileStore.java)
  - [store/JsonFileBestProfileStore.java](./store/JsonFileBestProfileStore.java)
- history:
  - [store/TuningHistoryStore.java](./store/TuningHistoryStore.java)
  - [store/JsonFileTuningHistoryStore.java](./store/JsonFileTuningHistoryStore.java)
- report stores:
  - [store/BenchmarkReportStore.java](./store/BenchmarkReportStore.java)
  - [store/PlatformCalibrationResultStore.java](./store/PlatformCalibrationResultStore.java)

## Practical Meaning Of Each Persisted Artifact

### Calibration profile

Meaning:

- platform runtime defaults
- reusable across workloads on the same machine

### Best profile

Meaning:

- one concrete workload winner
- immediately executable

### History

Meaning:

- search evidence
- not necessarily safe to execute as the chosen default

### Reports

Meaning:

- diagnostic and explain data
- not execution source of truth

## Benchmark commands are profile-read-only

CLI benchmark commands such as `benchmark-winner` and `benchmark-graph-space`
load already selected profiles, validate and measure explicit entries, and render
reports. They do not create best-profile records, append tuning history, update
calibration profiles, or persist benchmark reports as part of benchmark session
execution.

Autotune and calibration are the only profile-writing CLI flows.

Benchmark reports are explain artifacts, not runtime sources of truth. A report
can diagnose a selected candidate, backend path, cost summary, or fallback
reason, but later execution must use real `ExecutionProfile` /
`PlatformRuntimeProfile` artifacts rather than replaying report data.

## Worked Example

Suppose:

- platform id = `macos-aarch64-temurin-25`
- dtype = `f64`
- mode = `forward_backward`

Then the preferred calibration profile path is:

```text
profiles/platform/macos-aarch64-temurin-25/calibration/f64-forward_backward.json
```

and the `abc` best profile path is:

```text
profiles/platform/macos-aarch64-temurin-25/tuning/abc/f64-best-profile.json
```

Those two artifacts have different meanings:

- the first is reusable platform runtime policy
- the second is a workload-specific executable winner

## Persistence Rules

The intended rules are:

- execute from real profiles, not from reports
- store platform defaults separately from workload winners
- keep search history append-only
- keep legacy `build/...` paths readable for migration, but not as the preferred canonical layout
