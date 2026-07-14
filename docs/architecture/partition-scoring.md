# Partition Scoring

This document explains backend-neutral partition scoring as defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The operation-capability question and provider collaboration used before scoring are implemented.
Planning now also has one internal per-query hard-eligibility step: it validates the complete
provider/snapshot association, applies current availability and one exact hard requirement, and
retains supported backend identities in provider order. Configuration can separately record one
optional soft `DeviceClass` preference for later ranking after hard eligibility. Reusable or
public capability matrices, public planning orchestration, preference interpretation, score
calculation, ownership decisions, partitioning, and logical memory planning remain planned. This
page explains the accepted boundary and permitted inputs; it does not define a current scoring
formula, weights, or callable public scoring API.

## Purpose and pipeline position

Partition scoring answers this compile-time question:

```text
When a node or segment can run on more than one backend,
which backend should own it in the overall plan?
```

It runs after intent propagation and capability analysis and before maximal same-owner partitioning:

```text
backend intent
  -> current OperationCapabilityQuery
  -> current BackendCapabilityProvider boolean answer
  -> current internal per-query hard eligibility
  -> planned reusable/public capability matrix and orchestration
  -> current PartitionScoringConfig preference input
  -> backend-neutral partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
```

The current public capability contract asks whether one named backend can semantically own one
immutable operation occurrence. The query contains an `Operation` plus ordered input and output
`TensorDescriptor` snapshots; the explicitly supplied provider returns only a deterministic
boolean answer for its stable `BackendId`. It does not discover a backend, inspect availability,
evaluate a hard requirement, select a device or route, or explain a rejection.

The current package-private eligibility step combines those answers with caller-supplied
`BackendAvailabilitySnapshot` values and `BackendIntent`. It first validates a complete
one-provider/one-snapshot association by equal `BackendId`. It then skips empty snapshots and
exact hard-requirement mismatches before querying each remaining provider once. The result is an
immutable provider-ordered `BackendId` list, which may be empty. Exact-device and device-class
requirements prove only matching availability in the associated snapshot: capability remains
backend-level, and the step neither selects nor retains a device.

Later public planning orchestration may build ownership candidates from that internal fact and
must treat an empty list as terminal before scoring rather than weakening a hard requirement.
Current `PartitionScoringConfig` may supply one preferred coarse device class, but it neither
filters eligible backends nor guarantees that a backend of that class wins. Later scoring will
interpret that soft input together with other compile-time information, and the partitioner will
then group adjacent work with the same selected owner.

## Information scoring may use

Scoring may use immutable or estimated compile-time facts, including:

- graph metadata and graph phase;
- operation kind;
- data type, shape, estimated element count, and estimated byte size;
- backend capability and availability information represented for planning;
- configured backend intent;
- candidate ownership of producers and consumers;
- logical materialization requirements and estimates;
- transfer and partition-boundary estimates; and
- immutable backend-neutral planning cost profiles supplied as configuration after their
  consumer and cost classification are stable.

These inputs describe the graph and likely ownership costs. They do not require live runtime resources.

## Information scoring must not use

Planning scoring must not inspect or select runtime and implementation details, including:

- current runtime residency or current device buffers;
- physical buffer addresses or mutable `RunState`;
- prepared units or prepared executables;
- concrete kernel classes or calls;
- concrete OpenBLAS routes;
- concrete MPSGraph executables or routes;
- concrete CUDA kernels; or
- backend-specific executable DAGs;
- backend route names, vector species or lanes, unroll factors, thread counts, chunks, or tiles;
  or
- backend-local workload-cache entries, route configurations, or other model-autotuning values.

Concrete kernel or runtime scoring belongs to backend prepare, not planning.

## Scoring factors

The scoring model may account for these backend-neutral factors:

- **Backend intent** supplies an optional hard eligibility target.
- **Device-class preference** is a current optional soft input that later scoring may apply only
  after hard eligibility; an absent preference promises no default, fallback, or equal scores.
- **Capability** will remove unsupported ownership candidates using the current query/provider
  contract plus later matrix and eligibility work.
- **Transfer penalty** estimates the cost of moving values across backend ownership boundaries.
- **Materialization penalty** estimates logical layout or contiguity work needed by an ownership choice.
- **Boundary penalty** discourages plans fragmented into costly backend transitions.
- **Accelerator bonus** favors accelerator ownership when the region is large and suitable enough to benefit.
- **Small-region penalty** avoids offloading regions too small to justify accelerator and transfer overhead.
- **Planning cost profile** may supply compact backend-neutral estimates for the ownership
  comparison. It is separate from measurement evidence and from backend-local model-autotuning
  values.

The factors can support node-level or, after an explicit architecture evolution, more advanced segment-level and profile-guided policies. They do not change the meaning of the output.

No stable production operation-family or workload-bucket contract exists yet. Planning must not
invent one merely to accept profile data. A future cost-profile schema follows the stable cost
consumer and backend-neutral cost classification rather than preceding them.

## Ownership is not implementation selection

Planning produces ownership such as:

```text
owner = CPU
owner = Metal
owner = CUDA
```

It does not produce implementation choices such as:

```text
CPU scalar vs Vector API vs OpenBLAS
MPSGraph vs custom Metal kernel
specific CUDA kernel
fused backend executable
```

Those choices depend on backend-specific lowering, specialization, fusion, prepare configuration, and workspace constraints. They are made by the owning concrete backend during prepare.

The result of scoring is therefore an ownership decision used to build `PlannedPartition` values. It is part of the immutable compile recipe, not an executable implementation or physical schedule.

See [Lifecycle](lifecycle.md) for the full compile pipeline and [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for where implementation selection occurs.
See [Performance Evidence and Model Autotuning](performance-evidence-and-tuning.md) for the
separate benchmarking, model-autotuning, runtime-profiling, and planning-cost boundaries.
