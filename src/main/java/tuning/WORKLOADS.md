# Tuning Workloads

## Purpose

`tuning.workload` defines what is being measured.

It is the abstraction boundary shared by:

- benchmark
- per-graph autotune
- platform calibration

A workload is **not**:

- a compiled runtime artifact
- a backend execution plan
- a benchmark script with hardcoded measurement logic

A workload is:

- a reproducible graph-construction contract
- a validation contract
- a small piece of metadata describing the scenario being measured

## Reading Guide

This file is easiest to read if you keep three levels separate:

1. `WorkloadSpec`
   - reusable recipe
2. `WorkloadInstance`
   - one fresh instantiated graph and its validation contract
3. catalogs
   - named collections such as `StandardWorkloads` and `CalibrationWorkloads`

If those levels get blurred, it becomes very easy to:

- accidentally reuse stale graph state
- compare different graphs under the same workload name
- hide synthetic probes inside scenario-level benchmark stories

## Why The Workload Layer Matters

Without a stable workload layer, every workflow would invent its own way to build graphs and validate results.
That quickly leads to:

- inconsistent graphs between benchmark and autotune
- accidental state reuse across candidates
- untraceable measurement differences

The workload layer prevents that by requiring:

- a fresh graph instance per candidate
- explicit validation target/reference
- stable scenario metadata

## Core Contracts

### `WorkloadSpec`

Defines:

- workload name
- workload kind
- how to instantiate a fresh workload instance

Think of `WorkloadSpec` as the recipe, not the baked graph.

It should be stable enough that:

- two different workflows can instantiate the same named workload
- the workload meaning stays the same even when the candidate profile changes

### `WorkloadEnvironment`

Provides the environment in which the graph is built.

Most importantly it exposes:

- `environment.profile()`

That is how workload builders read:

- dtype
- execution mode
- profile metadata that may influence graph construction

The important rule is:

- workloads read the profile
- workloads do not need to know which tuning workflow is driving them

### `WorkloadInstance`

Represents one instantiated runnable scenario and defines:

- root tensor to execute
- validation target
- validation reference
- workload metadata

One `WorkloadSpec` can produce many fresh `WorkloadInstance`s.

That is required because:

- different candidates must not share mutated graph state
- backward graphs and gradients are stateful
- runtime preparation may cache data tied to one candidate/profile

## Fresh-Graph Contract

Every candidate measurement must get a fresh workload instance.

This rule exists because tensor graphs are not immutable value objects in the strict functional sense.
They can accumulate:

- gradients
- prepared/runtime state
- backend-specific caches

Reusing one graph across candidates would poison the measurement.

This is not just a cleanliness rule.
It directly affects result validity because:

- autograd mutates tensors
- prepared execution can attach runtime caches
- some specialized paths keep auxiliary runtime state on tensors

## Interaction With Profiles

Workloads do not know whether they are being used by:

- benchmark
- autotune
- calibration

They only receive an environment with a profile.

In practice that means:

- benchmark passes a concrete candidate profile
- autotune passes candidate profiles produced by the search space
- calibration assembles temporary execution profiles from runtime-profile candidates

From the workload's point of view, the contract stays stable:

- "build the graph for this profile"

What workloads may depend on from the profile:

- dtype
- execution mode
- explicit workload metadata that genuinely changes graph construction

What workloads should usually **not** depend on directly:

- backend-private implementation details
- tuning-session internals
- persistence state

## Public Surface Expectations

Workloads should primarily build graphs through the public tensor surface:

- [tensor/Tensor.java](../tensor/Tensor.java)
- [tensor/TensorOps.java](../tensor/TensorOps.java)
- public semantic types from:
  - [tensor/options](../tensor/options)
  - [tensor/loss](../tensor/loss)

Examples of public semantic types used by workloads:

- `tensor.options.Conv2dOptions`
- `tensor.options.Pool2dOptions`
- `tensor.options.AttentionOptions`
- `tensor.loss.LossReduction`

This is important because tuning should measure the same modeling surface that normal code uses.

Calibration workloads are allowed to be more synthetic than end-user scenarios, but they should still stay inside the public tensor graph model whenever practical.

## Built-In Workload Families

The built-in families are not all the same kind of thing.
They roughly split into:

- isolated kernel-family probes
- composed scenario workloads
- application-like end-to-end-ish graphs

### MatMul

- [tuning/workload/MatMulWorkloadSpec.java](./workload/MatMulWorkloadSpec.java)

Parameters:

- batch
- `m`
- `k`
- `n`

Example:

```java
WorkloadSpec spec = StandardWorkloads.matmul(
        "matmul_small",
        1, 64, 64, 64
);
```

Typical purpose:

- isolate GEMM policy
- compare microkernels, tiles, BLAS thresholds, and scheduler choices

Typical output:

- scalarized `sum(matmul(left, right))`

Validation target:

- the matrix result before scalarization

### Conv2d

- [tuning/workload/Conv2dWorkloadSpec.java](./workload/Conv2dWorkloadSpec.java)

Parameters include:

- batch
- input/output channels
- spatial size
- kernel size
- `Conv2dOptions`
- bias on/off

Example:

```java
WorkloadSpec spec = StandardWorkloads.conv2d(
        "conv2d_resnet_3x3",
        2, 64, 128, 56, 56, 3, 3,
        tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
        true
);
```

Typical purpose:

- stress convolution lowering/runtime policy on a realistic CNN-like shape

### Transformer Hot Path

- [tuning/workload/TransformerHotPathWorkloadSpec.java](./workload/TransformerHotPathWorkloadSpec.java)

This is intentionally a composed graph family rather than a micro-op.
It exercises a mix of:

- attention-related matmuls
- layout traffic
- fused elementwise chains
- residual paths
- normalization and feed-forward style subgraphs

Use it when you want an end-to-end-ish hot path instead of a single kernel family.

It is the right choice when the question is not:

- "is this one kernel faster?"

but rather:

- "what happens on a realistic fused/layout-heavy subgraph?"

### ABC Sequence Matmul

- [tuning/workload/AbcSequenceMatmulWorkloadSpec.java](./workload/AbcSequenceMatmulWorkloadSpec.java)

Useful when you want a sequence of dependent matmuls rather than one isolated GEMM.

### Normalization

- [tuning/workload/NormalizationWorkloadSpec.java](./workload/NormalizationWorkloadSpec.java)

Supported kinds:

- `BATCH_NORM`
- `LAYER_NORM`
- `RMS_NORM`

Purpose:

- measure reduction-heavy and affine-normalization style graphs

### Pool2d

- [tuning/workload/Pool2dWorkloadSpec.java](./workload/Pool2dWorkloadSpec.java)

Supported kinds:

- `MAX`
- `AVG`

### Indexed Loss

- [tuning/workload/LossWorkloadSpec.java](./workload/LossWorkloadSpec.java)

Supported kinds:

- `CROSS_ENTROPY_FROM_INDICES`
- `NLL_FROM_INDICES`

Purpose:

- stress gather/index-target loss paths and their backward behavior

### MLP Classification

- [tuning/workload/MlpClassificationWorkloadSpec.java](./workload/MlpClassificationWorkloadSpec.java)

Purpose:

- combine linear layers, activations, and indexed loss into a more application-like classification scenario

## Calibration Workloads

Platform calibration uses a separate catalog:

- [tuning/workload/CalibrationWorkloads.java](./workload/CalibrationWorkloads.java)

These workloads are intentionally more representative probes than end-user benchmarks.

Examples:

- square matmul
- tall-skinny matmul
- attention-like batched matmul
- fused cheap elementwise chain
- fused transcendental chain
- strided elementwise probes
- reduction sum
- scheduler cheap-parallel chain
- materialization probe
- conv2d representative case

These are probe workloads.
They are allowed to be synthetic as long as they isolate a real execution family that occurs in real graphs.

This split exists because calibration and benchmark solve different problems:

- benchmark should answer "which candidate is faster on this real workload?"
- calibration should answer "which runtime defaults are best for this execution family on this machine?"

## Synthetic Probes vs Real Workloads

Calibration probes are allowed to be synthetic if they isolate a family that truly exists in real graphs.

That is the intended rule:

- synthetic is fine for calibration
- synthetic is not fine for pretending a benchmark win exists when the pattern does not appear in real graphs

Examples of good synthetic probes:

- dense fused cheap chain to calibrate fused thresholds
- transposed strided elementwise probe to calibrate materialization
- attention-like batched matmul to calibrate attention matmul tiles

## Validation Contracts

Each workload chooses a validation strategy explicitly.

Typical options:

- no validation for exploratory/probe workloads
- baseline-profile validation
- snapshot/reference validation

The validation target can differ from the execution root.

Example:

- a workload may execute `sum(out)` for stable scalar timing
- but validate the pre-scalarized tensor output

This is common and intentional.

That separation is often the right design for tuning:

- execution root should be timing-stable
- validation target should be semantically meaningful

## Metadata And Catalogs

Workloads carry metadata so reports can group and label scenarios coherently.

Catalog helpers live in:

- [tuning/workload/WorkloadCatalog.java](./workload/WorkloadCatalog.java)
- [tuning/workload/StandardWorkloads.java](./workload/StandardWorkloads.java)

The standard catalog gives a reusable named set of scenarios for:

- local benchmark runs
- autotune defaults
- regression comparisons

Catalogs should stay honest about what they represent:

- `StandardWorkloads` should read like reusable benchmark/autotune scenarios
- `CalibrationWorkloads` should read like family probes for platform calibration

## Adding A Custom Workload

When adding a new workload:

1. Decide whether it is:
   - a real application scenario
   - a microbenchmark
   - a calibration probe
2. Build the graph from the public tensor surface.
3. Choose a validation target/reference deliberately.
4. Keep construction deterministic enough for stable comparison.
5. Register it in a catalog only if it is broadly useful.

Minimal sketch:

```java
WorkloadSpec spec = new TensorRootWorkloadSpec(
        "my_workload",
        WorkloadKind.GENERIC,
        environment -> {
            Tensor x = ...;
            Tensor y = ...;
            return x.add(y).relu().sum();
        },
        environment -> ValidationReference.none(),
        environment -> WorkloadMetadata.of("my_workload", WorkloadKind.GENERIC)
);
```

## Choosing The Right Workload For The Job

Use:

- `StandardWorkloads.*` when you want reusable scenario-level comparisons
- `CalibrationWorkloads.*` when you want family-specific runtime calibration probes
- a custom `TensorRootWorkloadSpec` when you need a narrowly targeted local experiment

The most important thing is not the class name.
It is whether the workload is honest about what it measures.

## Built-In Catalog Today

`StandardWorkloads.defaultCatalog()` currently registers:

- `matmul_small`
- `matmul_batched_attention_like`
- `abc_sequence_matmul_small`
- `conv2d_resnet_3x3`
- `mlp_classifier_small`
- `mlp_classifier_blas_heavy`
- `layer_norm_small`
- `max_pool2d_small`
- `cross_entropy_small`
- `transformer_hot_path`

This mix is intentional.
It covers:

- isolated probes
- composed hot paths
- application-like training graphs

## Why Scalarized Roots Are Common

Many workloads end with `sum(...)` or another scalar reduction.
That is not because the workload only cares about reductions.
It is a practical way to:

- force the whole graph to run
- give the benchmark one clear sink
- keep forward/backward behavior stable across candidates

The validation target can still point at a richer intermediate tensor.

## Workload Design Guidelines

Good workloads are:

- deterministic
- built from the public tensor surface
- explicit about shapes and options
- focused on one clear performance question

Bad workloads are:

- backend-private shortcuts
- reused graph instances across candidates
- giant synthetic graphs with no clear diagnostic value

## Choosing The Right Workload Type

Use isolated probes when:

- calibrating thresholds
- comparing microkernels
- measuring one family in isolation

Use composed scenarios when:

- measuring optimizer/runtime interaction
- validating fused or lowered hot paths
- searching workload-specific best profiles
