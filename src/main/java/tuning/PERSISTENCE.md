# Tuning Persistence

## Contents

- [Purpose](#purpose)
- [Fingerprints](#fingerprints)
- [Stored Records](#stored-records)
- [History Workflow](#history-workflow)
- [Best Profile Workflow](#best-profile-workflow)
- [Current Limitations](#current-limitations)
- [Operational Notes](#operational-notes)
- [Examples](#examples)

## Purpose

Persistence in `tuning` exists for one reason:

- to let benchmark/autotune learn across runs without inventing a second execution model

The persisted result is still:

- an `ExecutionProfile`

not a benchmark-only shadow object.

## Fingerprints

### Hardware

- [HardwareFingerprint.java](./store/HardwareFingerprint.java)

Current key includes:

- OS
- architecture
- VM name
- vendor
- CPU core count

Example key:

```text
os=mac_os_x|arch=aarch64|vm=openjdk_64-bit_server_vm|vendor=oracle|cores=10
```

### Workload

- [WorkloadFingerprint.java](./store/WorkloadFingerprint.java)

Current key includes:

- workload name
- workload kind
- dtype
- mode
- workload attributes

Example key:

```text
name=conv2d_resnet_3x3|kind=CONV2D|dtype=FLOAT32|mode=FORWARD|batch=2|inChannels=64|outChannels=128|...
```

These two fingerprints form the identity context for best-profile reuse and history reuse.

## Stored Records

### Best profile

- [BestProfileRecord.java](./store/BestProfileRecord.java)
- [BestProfileStore.java](./store/BestProfileStore.java)
- [JsonFileBestProfileStore.java](./store/JsonFileBestProfileStore.java)

Stored fields:

- hardware fingerprint
- workload fingerprint
- execution profile
- score
- timestamp

### Tuning history

- [TuningHistoryEntry.java](./store/TuningHistoryEntry.java)
- [TuningHistoryStore.java](./store/TuningHistoryStore.java)
- [JsonFileTuningHistoryStore.java](./store/JsonFileTuningHistoryStore.java)

Stored fields:

- candidate fingerprint
- candidate name
- valid / invalid
- median / mean
- score
- failure reason
- summary
- timestamp
- hardware/workload context

This is more than report data.
It is used by search as a prior.

## History Workflow

The current history-aware flow is:

1. generate candidates
2. resolve hardware + workload fingerprint
3. load matching history entries
4. sort historically good candidates earlier
5. optionally drop historically invalid candidates if pruning is enabled

This is implemented in:

- [HistoryAwareSearchStrategy.java](./search/HistoryAwareSearchStrategy.java)

## Best Profile Workflow

The current best-profile flow is:

1. run autotune
2. choose best finalist
3. save best profile record
4. next run resolves the same hardware + workload key
5. best profile is moved to the front of candidate ordering

Relevant classes:

- [BestProfileResolver.java](./store/BestProfileResolver.java)
- [FileBestProfileResolver.java](./store/FileBestProfileResolver.java)

## Current Limitations

This persistence layer is already usable, but it is not the final lifecycle yet.

Current important limitations:

- no retention policy
- no multi-objective best-profile store
- no merge/update conflict policy for multiple sources
- no explicit “unsafe candidate” semantic separate from generic invalid
- no compaction/index format beyond current file layout

Those are workflow limitations, not architectural blockers.

## Operational Notes

Current persistence expectations:

- best-profile store keeps the current winner for one hardware/workload context
- tuning-history store keeps append-only evidence for search reuse
- benchmark report store and tuning result store are report artifacts, not search priors

Recommended practical split:

- keep best-profile/history files small and frequently reused
- keep full benchmark/tuning reports in a separate report directory

This prevents a common failure mode:

- mixing machine-readable priors with large human-facing report dumps in one place

## Examples

### Example: save a best profile

```java
BestProfileStore store = new JsonFileBestProfileStore();
store.save(path, new BestProfileRecord(
        HardwareFingerprint.capture(),
        WorkloadFingerprint.of(workload, metadata, profile),
        profile,
        1.23,
        OffsetDateTime.now()
));
```

Input:

- path
- hardware fingerprint
- workload fingerprint
- best profile
- score

Output:

- JSON file containing the profile and its context

### Example: history-aware search

```java
SearchStrategy strategy = new HistoryAwareSearchStrategy(
        baseStrategy,
        new FileBestProfileResolver(new JsonFileBestProfileStore()),
        new JsonFileTuningHistoryStore(),
        bestProfilePath,
        historyPath
);
```

Effect:

- preferred known-good profiles come first
- known-bad candidates may be skipped
- search reuses information from earlier runs
