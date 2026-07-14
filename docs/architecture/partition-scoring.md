# Partition Scoring

This document explains backend-neutral partition scoring as defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The operation-capability question and provider collaboration used before scoring are implemented.
Capability matrices, hard-eligibility evaluation, scoring, ownership decisions, partitioning, and
logical memory planning remain planned. This page explains the accepted boundary and permitted
inputs; it does not define a current scoring formula, weights, or callable scoring API.

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
  -> planned capability matrix and hard eligibility
  -> backend-neutral partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
```

The current capability contract asks whether one named backend can semantically own one immutable
operation occurrence. The query contains an `Operation` plus ordered input and output
`TensorDescriptor` snapshots; the explicitly supplied provider returns only a deterministic
boolean answer for its stable `BackendId`. It does not discover a backend, inspect availability,
evaluate a hard requirement, select a device or route, or explain a rejection.

Later capability-matrix and hard-eligibility work will turn those narrow answers into valid
ownership candidates. Scoring will compare the candidates using compile-time information, and the
partitioner will then group adjacent work with the same selected owner.

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
- immutable platform, backend, or tuning profiles supplied as configuration.

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
- backend-specific executable DAGs.

Concrete kernel or runtime scoring belongs to backend prepare, not planning.

## Scoring factors

The scoring model may account for these backend-neutral factors:

- **Backend intent** reflects explicit compile configuration and preferences.
- **Capability** will remove unsupported ownership candidates using the current query/provider
  contract plus later matrix and eligibility work.
- **Transfer penalty** estimates the cost of moving values across backend ownership boundaries.
- **Materialization penalty** estimates logical layout or contiguity work needed by an ownership choice.
- **Boundary penalty** discourages plans fragmented into costly backend transitions.
- **Accelerator bonus** favors accelerator ownership when the region is large and suitable enough to benefit.
- **Small-region penalty** avoids offloading regions too small to justify accelerator and transfer overhead.
- **Platform profile** supplies immutable tuning or profiling hints without exposing live runtime state.

The factors can support node-level or, after an explicit architecture evolution, more advanced segment-level and profile-guided policies. They do not change the meaning of the output.

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
