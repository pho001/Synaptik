# Tuning Workloads

## Contents

- [Purpose](#purpose)
- [Workload Contracts](#workload-contracts)
- [How Workloads Interact With Profiles](#how-workloads-interact-with-profiles)
- [Built-In Workload Families](#built-in-workload-families)
- [Calibration Workloads](#calibration-workloads)
- [Materialization Calibration Workloads](#materialization-calibration-workloads)
- [Validation References](#validation-references)
- [Catalogs](#catalogs)
- [Selecting Scenarios](#selecting-scenarios)
- [Adding Custom Workloads](#adding-custom-workloads)

## Purpose

`tuning.workload` describes what is being measured or tuned.

It is **not**:

- a compiled runtime artifact
- a benchmark script
- a backend-specific execution plan

It is a graph-construction contract over the public tensor surface.

## Workload Contracts

### `WorkloadSpec`

Defines:

- workload name
- workload kind
- how to instantiate a fresh workload instance

### `WorkloadEnvironment`

Provides the execution context used while instantiating the graph.

Most importantly:

- `environment.profile()`

This is how workload builders read:

- dtype
- execution mode
- workload profile metadata

### `WorkloadInstance`

Defines:

- benchmark root
- validation target
- validation reference
- workload metadata

This separation is important:

- one spec can instantiate many fresh graphs
- every benchmark/autotune candidate gets a fresh graph
- workloads do not reuse mutated graph state between candidates

## How Workloads Interact With Profiles

Workloads do not know whether they are being used by:

- benchmark
- graph autotune
- platform calibration

They just receive:

- a profile in the environment

Current practical behavior:

- benchmark instantiates workloads with concrete candidate `ExecutionProfile`
- graph autotune also instantiates workloads with concrete candidate `ExecutionProfile`
- platform calibration generates runtime-profile candidates first, then assembles temporary `ExecutionProfile` values before workload execution

That means workload builders always stay on one stable abstraction:

- “I build a graph for the provided execution profile”

## Built-In Workload Families

### MatMul

- [MatMulWorkloadSpec.java](./workload/MatMulWorkloadSpec.java)

Parameters:

- batch
- `m`
- `k`
- `n`

Example:

```java
tuning.workload.WorkloadSpec spec = tuning.workload.StandardWorkloads.matmul(
        "matmul_small",
        1, 64, 64, 64
);
```

Input:

- left matrix
- right matrix

Output root:

- scalarized `sum(matmul(left, right))`

Validation target:

- labeled matrix output before scalarization

### Conv2d

- [Conv2dWorkloadSpec.java](./workload/Conv2dWorkloadSpec.java)

Parameters:

- batch
- in/out channels
- spatial size
- kernel size
- `Conv2dOptions`
- optional bias

Example:

```java
tuning.workload.WorkloadSpec spec = tuning.workload.StandardWorkloads.conv2d(
        "conv2d_resnet_3x3",
        2, 64, 128, 56, 56, 3, 3,
        tensor.Conv2dOptions.defaults().withPadding(1, 1),
        true
);
```

Output root:

- scalarized convolution result

Validation target:

- convolution tensor before scalarization

### Transformer Hot Path

- [TransformerHotPathWorkloadSpec.java](./workload/TransformerHotPathWorkloadSpec.java)

This is a composed graph family, not a micro-op:

- attention
- reshape / permute
- residual
- normalization
- feed-forward stack

It is meant to exercise:

- matmul policy
- fused/runtime dispatch
- shape/layout traffic

### Normalization

- [NormalizationWorkloadSpec.java](./workload/NormalizationWorkloadSpec.java)

Supported kinds:

- `BATCH_NORM`
- `LAYER_NORM`
- `RMS_NORM`

### MLP Classification

- [MlpClassificationWorkloadSpec.java](./workload/MlpClassificationWorkloadSpec.java)

This family stresses:

- linear layers
- loss path
- BLAS/runtime policy

### Pool2d

- [Pool2dWorkloadSpec.java](./workload/Pool2dWorkloadSpec.java)

Supported kinds:

- `MAX`
- `AVG`

### Indexed Loss

- [LossWorkloadSpec.java](./workload/LossWorkloadSpec.java)

Supported kinds:

- `CROSS_ENTROPY_FROM_INDICES`
- `NLL_FROM_INDICES`

## Calibration Workloads

Platform calibration uses a separate representative workload catalog:

- [CalibrationWorkloads.java](./workload/CalibrationWorkloads.java)

These workloads are not meant to be end-user scenarios.

They are representative probes for execution families.

Current examples:

- matmul square
- matmul tall-skinny
- attention-like batched matmul
- fused cheap elementwise chain
- fused transcendental chain
- reduction sum
- scheduler cheap parallel chain
- materialization strided elementwise chain

Why this split exists:

- graph autotune should optimize one real workload family
- platform calibration should optimize reusable runtime defaults over representative execution-family probes

## Materialization Calibration Workloads

`MATERIALIZATION` calibration now uses explicit non-contiguous workloads.

Current shape:

- create rank-2 tensors
- transpose them to obtain non-contiguous logical views
- run element-wise compute over the transposed views

Purpose:

- force the runtime to choose between:
  - strided execution
  - materialize-then-fast-path execution

This is the right signal for tuning:

- `cpu.contiguousMaterializeThreshold`

## Validation References

Workloads may choose different validation contracts:

- no validation
- tensor snapshot
- baseline profile comparison
- explicit labeled target

This allows:

- benchmark root to be scalarized for full execution
- validation target to remain semantic and stable

That is especially useful for:

- numerically sensitive reductions
- conv2d direct vs GEMM lowering
- transformer blocks where the semantic tensor matters more than a final sum

## Catalogs

Main entry point:

- [StandardWorkloads.java](./workload/StandardWorkloads.java)

Calibration catalog:

- [CalibrationWorkloads.java](./workload/CalibrationWorkloads.java)

You can also create local catalogs if you want a private scenario set.

## Selecting Scenarios

Use:

- benchmark
  - when you want to compare explicit runnable variants
- graph autotune
  - when you want to search graph candidates for one concrete workload
- platform calibration
  - when you want machine-level runtime defaults without one specific end-user model

Rule of thumb:

- if the question is “which graph policy is best for this model block?”:
  - graph autotune
- if the question is “which runtime thresholds are good on this machine?”:
  - platform calibration

## Adding Custom Workloads

Extension path:

1. create a `WorkloadSpec`
2. build a fresh graph from `WorkloadEnvironment`
3. choose root tensor
4. choose validation target/reference
5. attach stable metadata
6. register it in a catalog if needed

Minimal example:

```java
tuning.workload.WorkloadSpec spec = new tuning.workload.TensorRootWorkloadSpec(
        "custom_bias_gelu",
        tuning.workload.WorkloadKind.GENERIC,
        environment -> {
            tensor.Tensor x = tensor.Tensor.randn(new int[]{32, 128}, environment.profile().dataType(), "x");
            tensor.Tensor w = tensor.Tensor.randn(new int[]{128, 128}, environment.profile().dataType(), "w");
            tensor.Tensor b = tensor.Tensor.randn(new int[]{128}, environment.profile().dataType(), "b");
            tensor.Tensor y = x.matmul(w).add(b).gelu();
            y.setLabel("custom_bias_gelu_output");
            return y.sum();
        },
        environment -> tuning.validate.ValidationReference.none(),
        environment -> tuning.workload.WorkloadMetadata.of(
                "custom_bias_gelu",
                tuning.workload.WorkloadKind.GENERIC
        )
);
```

Design rules:

- workloads are graph builders
- not benchmark scripts
- not precompiled execution objects
- validation target may differ from benchmark root
