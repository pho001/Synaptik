# Tuning Persistence

## Contents

- [Purpose](#purpose)
- [Artifact Types](#artifact-types)
- [Platform Runtime Profile Persistence](#platform-runtime-profile-persistence)
- [Graph Autotune Persistence](#graph-autotune-persistence)
- [Explain Artifact Persistence](#explain-artifact-persistence)
- [Fingerprints And Identity](#fingerprints-and-identity)
- [Invalidation Rules](#invalidation-rules)
- [Recommended Layout Split](#recommended-layout-split)
- [Examples](#examples)

## Purpose

Persistence exists so that tuning workflows can reuse results across runs without inventing a second execution model.

Important distinction:

- execution artifacts
- tuning priors
- explain artifacts

must not be mixed together.

## Artifact Types

### 1. Built-in defaults

These live in code.

They are:

- fallback values
- not tuned artifacts

### 2. Platform runtime profile

Persisted runtime-default artifact:

- `PlatformRuntimeProfile`

This is the output of platform calibration.

It is not a benchmark report and not a graph-tuned profile.

### 3. Best graph-tuned profile

Persisted graph-level winner:

- best `ExecutionProfile` for one hardware/workload context

This is the output of per-graph autotune.

### 4. Tuning history

Append-only evidence used to influence later graph autotune search.

This is not the same thing as a final winner.

### 5. Explain artifacts

Human/audit artifacts:

- benchmark reports
- autotune reports
- platform calibration reports

These are not source of truth for execute.

## Platform Runtime Profile Persistence

Current core types:

- [PlatformRuntimeProfile.java](../config/profile/PlatformRuntimeProfile.java)
- [PlatformRuntimeProfileIO.java](../config/profile/PlatformRuntimeProfileIO.java)
- [PlatformRuntimeProfileStore.java](./store/PlatformRuntimeProfileStore.java)
- [JsonFilePlatformRuntimeProfileStore.java](./store/JsonFilePlatformRuntimeProfileStore.java)

Persisted sections:

- metadata
- matmul family
- fused family
- element-wise dispatch family
- reduction family
- scheduler family
- materialization family
- numerics family

That means one persisted platform profile can be:

- loaded
- partially replaced by recalibration of one family
- re-used to assemble future `ExecutionProfile` instances

## Graph Autotune Persistence

Graph autotune still persists `ExecutionProfile` as the runnable winner.

Main types:

- [BestProfileRecord.java](./store/BestProfileRecord.java)
- [BestProfileStore.java](./store/BestProfileStore.java)
- [JsonFileBestProfileStore.java](./store/JsonFileBestProfileStore.java)

Stored fields:

- hardware fingerprint
- workload fingerprint
- best `ExecutionProfile`
- score
- timestamp

This persistence is intentionally separate from platform calibration persistence.

Reason:

- platform runtime profile is reusable across many graphs
- best graph profile is workload-specific

## Explain Artifact Persistence

Platform calibration explain persistence:

- [PlatformCalibrationSaveHelper.java](./store/PlatformCalibrationSaveHelper.java)
- [JsonFilePlatformCalibrationResultStore.java](./store/JsonFilePlatformCalibrationResultStore.java)

Benchmark and autotune explain artifacts are stored separately through their report renderers and stores.

Important rule:

- explain artifacts are never used as execute source of truth

## Fingerprints And Identity

### Hardware fingerprint

Core type:

- [HardwareFingerprint.java](./store/HardwareFingerprint.java)

Current captured properties include:

- OS
- architecture
- VM
- vendor
- CPU core count

This fingerprint identifies the platform context for:

- platform runtime profile reuse
- best-profile reuse
- tuning-history reuse

### Workload fingerprint

Core type:

- [WorkloadFingerprint.java](./store/WorkloadFingerprint.java)

Current workload identity includes:

- workload name
- workload kind
- dtype
- execution mode
- workload-specific attributes

This fingerprint identifies the graph/workload context for best-profile and history reuse.

## Invalidation Rules

Persisted runtime artifacts must not be treated as permanently valid.

Platform runtime profile invalidation should happen when:

- hardware fingerprint changes materially
- framework version changes
- runtime/planner schema changes
- persistence schema changes
- knob semantics change

That is why `PlatformRuntimeProfile` metadata carries:

- platform profile id
- hardware key
- framework version
- planner schema version
- persistence schema version
- timestamp
- dtype
- execution mode

Best graph profiles become invalid when:

- hardware fingerprint no longer matches
- workload fingerprint no longer matches
- the runtime/profile schema changes so that old profile values no longer mean the same thing

## Recommended Layout Split

Recommended practical split:

- platform runtime profiles
  - small, reusable, long-lived

- best graph profiles
  - workload-specific winners

- tuning history
  - append-only search priors

- explain reports
  - larger human-facing artifacts

This avoids a common failure mode:

- mixing machine-readable priors with large explain dumps in one directory and one lifecycle

## Examples

### Example: save a platform runtime profile

```java
config.profile.PlatformRuntimeProfile profile = result.finalRuntimeProfile();
config.profile.PlatformRuntimeProfileIO.save(path, profile);
```

Input:

- path
- runtime profile

Output:

- JSON runtime-default artifact

### Example: save a best graph profile

```java
BestProfileStore store = new JsonFileBestProfileStore();
store.save(path, new BestProfileRecord(
        HardwareFingerprint.capture(),
        WorkloadFingerprint.of(workload, metadata, profile),
        profile,
        1.23,
        java.time.OffsetDateTime.now()
));
```

Input:

- hardware context
- workload context
- best `ExecutionProfile`

Output:

- graph-specific best-profile record

### Example: history-aware graph autotune

```java
tuning.search.SearchStrategy strategy = new tuning.search.HistoryAwareSearchStrategy(
        baseStrategy,
        new tuning.store.FileBestProfileResolver(new JsonFileBestProfileStore()),
        new tuning.store.JsonFileTuningHistoryStore(),
        bestProfilePath,
        historyPath
);
```

Effect:

- historically good graph candidates are considered earlier
- historically bad ones may be pruned or delayed
- search reuses prior evidence without changing execution semantics
