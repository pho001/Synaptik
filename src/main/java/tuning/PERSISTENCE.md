# Tuning Persistence

Persistence in the tuning layer is not just about "saving some JSON". It must precisely distinguish what is:

- execute source of truth
- tuning prior
- explain artifact

If these layers get mixed, drift starts between what was benchmarked, what gets executed, and what is stored as the "winner".

## Reading Guide

This document describes:

- which persistent artifacts exist today
- where they should live
- what lifecycle they have
- which of them are used for execution and which are only used for explain/search priors

## Artifact Types

### 1. Built-In Defaults

They live in code.

They are not a persistent tuning artifact.

### 2. `PlatformRuntimeProfile`

The result of platform calibration.

This is a machine-specific runtime default artifact.

It is used as the actual input to `ExecutionProfileAssembler`.

### 3. Best `ExecutionProfile`

The result of graph autotune for a specific workload/hardware context.

This is a workload-specific runnable winner.

### 4. Tuning History

Append-only evidence about candidates:

- valid/invalid
- median/mean
- score
- summary

It is used as a prior for history-aware search.

### 5. Explain Artifacts

This group includes:

- text/json benchmark reports
- text/json autotune reports
- text/json platform calibration reports

They are not used as the runtime source of truth.

## Preferred Layout Today

The preferred layout is platform-versioned storage under:

- `profiles/platform/<platform-id>/...`

The concrete pattern used in [Main.java](../synaptik/app/Main.java):

### Platform calibration

- `profiles/platform/<platform-id>/calibration/<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.json`
- `profiles/platform/<platform-id>/reports/calibration-<dtype>-<mode>.txt`

### Graph autotune

For workload `abc`:

- `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json`
- `profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl`

## Compatibility Fallbacks

The repo can still read older fallback layouts under `build/...`:

- `build/platform-calibration/...`
- `build/tuning/best-profiles/...`
- `build/tuning/history/...`

But this is no longer the preferred long-term layout.

The documentation needs to say this explicitly:

- `build/...` is compatibility / temporary output space
- `profiles/platform/...` is the preferred place for versioned persisted tuning state

## Platform Runtime Profile Persistence

Main types:

- [PlatformRuntimeProfile.java](../config/profile/PlatformRuntimeProfile.java)
- [PlatformRuntimeProfileIO.java](../config/profile/PlatformRuntimeProfileIO.java)
- [PlatformRuntimeProfileStore.java](./store/PlatformRuntimeProfileStore.java)
- [JsonFilePlatformRuntimeProfileStore.java](./store/JsonFilePlatformRuntimeProfileStore.java)

Stored content:

- metadata
- matmul family
- fused family
- elementwise dispatch family
- reduction family
- scheduler family
- materialization family
- numerics family

Meaning:

- you can recalibrate only part of the families and keep the rest
- you can reuse the same runtime profile across multiple benchmark/autotune workloads

## Best Profile Persistence

Main types:

- [BestProfileRecord.java](./store/BestProfileRecord.java)
- [BestProfileStore.java](./store/BestProfileStore.java)
- [JsonFileBestProfileStore.java](./store/JsonFileBestProfileStore.java)

The JSON currently stores:

- `score`
- `updatedAt`
- `hardwareKey`
- `workloadKey`
- embedded `ExecutionProfile`

This is an important reality:

- a best profile record is not just a bare `ExecutionProfile`
- it also contains the identity of the context it applies to

## Tuning History Persistence

Main types:

- [TuningHistoryEntry.java](./store/TuningHistoryEntry.java)
- [TuningHistoryStore.java](./store/TuningHistoryStore.java)
- [JsonFileTuningHistoryStore.java](./store/JsonFileTuningHistoryStore.java)

Format:

- JSON Lines
- one candidate observation per line

Current fields:

- `fingerprint`
- `candidateName`
- `valid`
- `medianMs`
- `meanMs`
- `score`
- `failureReason`
- `summary`
- `timestamp`
- `hardwareKey`
- `workloadKey`

This makes sense for:

- history-aware ordering
- pruning invalid candidates
- preserving an audit trail without rewriting the past

## Explain Artifact Persistence

Platform calibration report persistence:

- [PlatformCalibrationSaveHelper.java](./store/PlatformCalibrationSaveHelper.java)
- [JsonFilePlatformCalibrationResultStore.java](./store/JsonFilePlatformCalibrationResultStore.java)

Rules:

- both JSON and text reports are explain artifacts
- the runtime source of truth is still the `PlatformRuntimeProfile` itself

## Fingerprints

### Hardware Fingerprint

Type:

- [HardwareFingerprint.java](./store/HardwareFingerprint.java)

Usage:

- platform runtime profile reuse
- best profile reuse
- tuning history filtering

### Workload Fingerprint

Type:

- [WorkloadFingerprint.java](./store/WorkloadFingerprint.java)

Usage:

- distinguishing workload-specific best profiles and history

The best profile resolver is intentionally strict:

- hardware key must match
- workload key must match

See:

- [FileBestProfileResolver.java](./store/FileBestProfileResolver.java)

## Invalidation Rules

A persistent tuning artifact is not eternal.

### Invalidate a platform runtime profile when:

- the hardware fingerprint changed
- the semantics of runtime knobs changed
- the schema format changed
- framework/runtime behavior changed enough that old winners no longer make sense

### Invalidate a best profile when:

- the hardware fingerprint does not match
- the workload fingerprint does not match
- the schema or field meaning in `ExecutionProfile` changed

### Invalidate or ignore history when:

- workload/hardware keys do not match
- candidate fingerprints no longer correspond to today's candidate space

## Source Of Truth Rules

### What is execute source of truth

- `PlatformRuntimeProfile`
- `ExecutionProfile`

### What is not execute source of truth

- tuning history
- benchmark report
- calibration report
- tuning summary text

The documentation must not blur this rule.

## Example: Save Platform Calibration

```java
PlatformCalibrationSaveHelper.saveAll(
        result,
        layout.profilePath(),
        layout.jsonReportPath(),
        layout.textReportPath()
);
```

This saves:

- the final runtime profile
- the JSON report
- the text report

## Example: Save Best Profile

```java
bestProfileStore.save(path, new BestProfileRecord(
        hardware,
        workload,
        bestProfile,
        score,
        java.time.OffsetDateTime.now()
));
```

## Example: Append History

```java
historyStore.append(path, new TuningHistoryEntry(
        fingerprint,
        candidateName,
        valid,
        medianMs,
        meanMs,
        score,
        failureReason,
        summary,
        java.time.OffsetDateTime.now(),
        hardware,
        workload
));
```

## Practical Recommendations

- keep `profiles/platform/...` in a repository-known location
- use `build/...` only as fallback or scratch space
- do not store reports and source-of-truth profiles in the same file
- keep history append-only
- overwrite the best profile only when there is a genuinely better winner

## Common Mistakes

- storing a report instead of a profile artifact
- using history JSONL as runtime config
- ignoring hardware/workload fingerprints when reloading
- mixing platform defaults and graph winners into one file without identity

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- reporting: [REPORTING.md](./REPORTING.md)
- search: [SEARCH.md](./SEARCH.md)
