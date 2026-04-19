# Tuning Workloads

The workload layer defines what is being measured.

It is shared by:

- benchmark
- autotune
- platform calibration

A workload is not:

- a prepared execution artifact
- a benchmark renderer
- an executor-specific script

A workload is:

- a reproducible graph-construction contract
- a validation contract
- metadata describing the measured scenario

## Core Contracts

### `WorkloadSpec`

Recipe for how to create a fresh workload instance.

It defines:

- workload name
- workload kind
- how to instantiate a fresh graph

### `WorkloadEnvironment`

Provides profile/environment context during workload construction.

Most importantly, builders can read:

- dtype
- execution mode
- workload-related profile metadata

### `WorkloadInstance`

Represents one fresh instantiated scenario.

It includes:

- root tensor to execute
- validation target
- validation reference
- workload metadata

## Fresh-Graph Rule

Every candidate measurement must instantiate a fresh workload instance.

Why:

- tensor graphs are not immutable pure values
- gradients are stateful
- compile/prepare may attach run-specific artifacts
- reusing one mutated graph across candidates would poison measurements

This is one of the most important correctness rules in the tuning architecture.

## Built-In Workload Families

Current package examples include:

- [workload/MatMulWorkloadSpec.java](./workload/MatMulWorkloadSpec.java)
- [workload/Conv2dWorkloadSpec.java](./workload/Conv2dWorkloadSpec.java)
- [workload/NormalizationWorkloadSpec.java](./workload/NormalizationWorkloadSpec.java)
- [workload/LossWorkloadSpec.java](./workload/LossWorkloadSpec.java)
- [workload/Pool2dWorkloadSpec.java](./workload/Pool2dWorkloadSpec.java)
- [workload/TransformerHotPathWorkloadSpec.java](./workload/TransformerHotPathWorkloadSpec.java)
- [workload/AbcSequenceMatmulWorkloadSpec.java](./workload/AbcSequenceMatmulWorkloadSpec.java)
- [workload/TensorRootWorkloadSpec.java](./workload/TensorRootWorkloadSpec.java)

Catalog helpers:

- [workload/StandardWorkloads.java](./workload/StandardWorkloads.java)
- [workload/CalibrationWorkloads.java](./workload/CalibrationWorkloads.java)

## Why Workloads Need Metadata

Metadata is not decoration.
It helps explain:

- what scenario was measured
- what shapes and family characteristics were involved
- why one candidate may have won

Example metadata fields may include:

- shape
- dtype
- batch size
- sequence length
- graph signature
- whether backward is enabled

## Worked Examples

### Example 1: workload-specific autotune

For the main CLI, autotune uses:

- `StandardWorkloads.abcSequenceMatmulBlasBenchmark("abc_sequence_matmul_<dtype>")`

This means:

- autotune, reports, and persistence all talk about one named workload family
- the winner is meaningful for that workload, not for all graphs globally

### Example 2: calibration workloads

Platform calibration does not use one giant synthetic super-workload.
It uses family-specific representative workloads such as:

- square matmul
- tall-skinny matmul
- attention-like batched matmul
- conv2d pointwise projection
- conv2d resnet-style 3x3
- fused cheap elementwise
- fused transcendental

This is important because runtime knobs are often family-sensitive.

## Validation Contract

Workloads also define validation expectations:

- target tensors to compare
- reference generation strategy
- tolerance profile

So the workload layer is not only "what graph to build".
It is also "what correctness means for this scenario".
