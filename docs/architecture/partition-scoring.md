# Partition Scoring

This document explains backend-neutral partition scoring as defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The operation-capability question and provider collaboration used before scoring are implemented.
Planning also has two internal per-query steps. Hard eligibility validates the complete
provider/snapshot association, applies current availability and one exact hard requirement, and
retains supported backend identities in provider order. Baseline owner selection treats that list
as the complete candidate set and applies the current optional soft `DeviceClass` preference.
Planning now also exposes the immutable `PlannedPartition(owner, nodeIds)` recipe and contains an
internal generator for maximal consecutive same-owner runs. Reusable or public capability
matrices, public planning orchestration or owner selection, numeric or cost scoring, and logical
memory planning remain planned. This page explains the accepted boundary and current internal
baseline; it does not define a public scoring API, weights, or a general cost formula.

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
  -> current PartitionScoringConfig preference input
  -> current internal preferred-class/provider-order baseline
  -> current BackendId owner for that occurrence
  -> later orchestration assembles a complete Map<NodeId, BackendId>
  -> current internal maximal consecutive same-owner partitioning
  -> current immutable PlannedPartition(owner, nodeIds) recipes
  -> planned logical memory/materialization requirements
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

The current package-private selector uses the eligible identity list directly as its complete
candidate set. It validates every supplied snapshot for null and duplicate equal identities,
requires one equal-identity snapshot for every eligible backend, and permits extra unique
snapshots. An empty eligible list fails internally before any snapshot element is read. With no
preference, the first eligible identity wins. With a preference, the first eligible identity whose
snapshot reports that class wins; if none matches, the first eligible identity still wins. An
empty matching snapshot is only a preference nonmatch, and provider order resolves ties and
fallback. The selector returns the exact identity reference from hard eligibility and never
selects or retains a device.

This is one cost-free internal ownership baseline, not the public planning workflow or a general
numeric scoring model. Later cost-bearing scoring may use additional compile-time facts. Once a
complete owner map exists, the current package-private partitioner scans
`CompiledGraphModel.nodes()` in its stored validated topological order and starts a partition only
at the first node or an owner transition. Equality is `BackendId` value equality; graph edges,
phase changes, fan-out, merges, repeated inputs, graph boundaries, and multi-output values do not
redefine adjacency or split an equal-owner run.

The resulting public recipe retains only the first node's exact owner reference and the exact
graph `NodeId` references in order. Its outer and inner lists are immutable. A valid zero-node
pass-through graph yields no partitions because inputs and outputs are values, not synthetic
nodes. The generator remains internal, so there is still no public workflow that assembles the
owner map or invokes partitioning.

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
- **Device-class preference** is a current optional soft input that the internal baseline applies
  only after hard eligibility; an absent preference supplies no class match and the baseline uses
  provider order.
- **Capability** removes unsupported ownership candidates through the current query/provider and
  internal hard-eligibility contracts.
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

The current internal baseline returns one `BackendId` owner for one eligible occurrence. The
current internal generator can turn a complete per-node owner map into public immutable
`PlannedPartition` values. Building that map through public planning orchestration and assembling
the complete immutable compile recipe remain planned. Neither an owner identity nor a current
partition recipe is an executable implementation or physical schedule.

See [Lifecycle](lifecycle.md) for the full compile pipeline and [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for where implementation selection occurs.
See [Performance Evidence and Model Autotuning](performance-evidence-and-tuning.md) for the
separate benchmarking, model-autotuning, runtime-profiling, and planning-cost boundaries.
