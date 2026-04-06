# Tuning Workloads

## Contents

- [Purpose](#purpose)
- [Workload Contracts](#workload-contracts)
- [Built-in Workload Families](#built-in-workload-families)
- [Validation References](#validation-references)
- [Catalog](#catalog)
- [Adding Custom Workloads](#adding-custom-workloads)
- [Selecting Scenarios for Benchmark Runs](#selecting-scenarios-for-benchmark-runs)
- [Preset-Driven Scenario Selection](#preset-driven-scenario-selection)
- [Examples](#examples)

## Purpose

`tuning.workload` describes what is being measured or tuned.

It is **not**:

- a compiled execution artifact
- a benchmark script
- a backend-specific runtime plan

It is a clean graph-construction contract over the public tensor surface.

## Workload Contracts

### `WorkloadSpec`

Defines:

- name
- workload kind
- how to instantiate a fresh workload instance for a profile

### `WorkloadInstance`

Defines:

- benchmark root
- validation target
- validation reference
- workload metadata

This separation is important:

- one spec can instantiate many fresh graphs
- every candidate gets a fresh graph
- benchmark/autotune do not reuse mutated graph state between candidates
- a workload may scalarize the benchmark root to force full execution
- validation can still compare the semantic tensor output instead of a numerically sensitive reduction
- validation target is resolved against the current graph after rewrite/optimization

## Built-in Workload Families

### MatMul

- [MatMulWorkloadSpec.java](./workload/MatMulWorkloadSpec.java)

Parameters:

- batch
- `m`
- `k`
- `n`

Example:

```java
WorkloadSpec spec = StandardWorkloads.matmul("matmul_small", 1, 64, 64, 64);
```

Input:

- left shape `[64, 64]`
- right shape `[64, 64]`

Output root:

- `sum(matmul(left, right))`

Validation target:

- label of `matmul(left, right)`

Why summed:

- forward-only and forward-backward modes then both end in a scalar root
- this keeps comparison and timing shape-stable across modes
- validation still compares the matrix output, not the scalar reduction

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
WorkloadSpec spec = StandardWorkloads.conv2d(
        "conv2d_resnet_3x3",
        2, 64, 128, 56, 56, 3, 3,
        new Conv2dOptions(1, 1, 1, 1, 1, 1, 1),
        true
);
```

Input:

- input shape `[2, 64, 56, 56]`
- weight shape `[128, 64, 3, 3]`
- bias shape `[128]`

Output root:

- `sum(conv2d(...))`

Validation target:

- label of `conv2d(...)`

Why the split matters:

- the benchmark still forces materialization of the full convolution output
- validation avoids using a numerically sensitive scalar reduction as the semantic correctness target
- this prevents false mismatches between legitimate F32 implementations such as direct conv vs lowered GEMM conv

### Transformer Hot Path

- [TransformerHotPathWorkloadSpec.java](./workload/TransformerHotPathWorkloadSpec.java)

This is a composed workload family, not a micro-op:

- attention
- reshape/permute merge
- residual
- layer norm
- feed-forward linear stack
- rms norm

It depends on `WorkloadProfile.transformerHotPathDefaults()` or a custom transformer workload profile.

Example:

```java
ExecutionProfile profile = new ExecutionProfile(
        "tr",
        "tr",
        DataType.FLOAT32,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults(),
        StandardWorkloads.transformerHotPathDefaults()
);

WorkloadInstance instance = StandardWorkloads
        .transformerHotPath("transformer_hot_path")
        .instantiate(new WorkloadEnvironment(profile));
```

Input:

- Q/K/V tensors
- residual branch tensors
- normalization params
- feed-forward weights/biases

Output:

- scalar root from the whole block composition

Validation target:

- label of the final block tensor before scalarization

### Normalization

- [NormalizationWorkloadSpec.java](./workload/NormalizationWorkloadSpec.java)

Supported sub-kinds:

- `BATCH_NORM`
- `LAYER_NORM`
- `RMS_NORM`

Example:

```java
WorkloadSpec spec = StandardWorkloads.normalization(
        "layer_norm_small",
        NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
        4, 64, 8, 1, 1e-5
);
```

Input:

- normalization input tensor
- required parameter tensors (`gamma`, `beta` where relevant)

Output:

- scalar root via `sum(normalizedOutput)`

Validation target:

- label of the normalized tensor output before reduction

### MLP Classification

- [MlpClassificationWorkloadSpec.java](./workload/MlpClassificationWorkloadSpec.java)

This workload models a small feed-forward classifier:

- pre-encoded floating input
- linear encoder projection
- hidden linear block
- output linear classifier
- `crossEntropyLossFromIndices(...)` at the end

Example:

```java
WorkloadSpec spec = StandardWorkloads.mlpClassification(
        "mlp_classifier_small",
        16, 32, 48, 24, 6,
        LossReduction.MEAN
);
```

Heavier built-in variant intended to stress GEMM/runtime policy:

```java
WorkloadSpec spec = StandardWorkloads.mlpClassification(
        "mlp_classifier_blas_heavy",
        64, 256, 512, 256, 32,
        LossReduction.MEAN
);
```

Input:

- encoded feature tensor shape `[batch, inputFeatures]`
- three weight matrices
- three bias vectors
- target index tensor shape `[batch]`

Output:

- scalar classification loss

Why the heavier variant exists:

- `mlp_classifier_small` is useful for functional coverage and fast tuning loops
- `mlp_classifier_blas_heavy` gives the autotuner a much stronger signal for:
  - BLAS on/off
  - BLAS thread policy
  - matmul parallel threshold tuning

### Pool2d

- [Pool2dWorkloadSpec.java](./workload/Pool2dWorkloadSpec.java)

Supported sub-kinds:

- `MAX`
- `AVG`

Example:

```java
WorkloadSpec spec = StandardWorkloads.pool2d(
        "max_pool2d_small",
        Pool2dWorkloadSpec.PoolKind.MAX,
        2, 8, 16, 16,
        Pool2dOptions.square(2)
);
```

Input:

- rank-4 `NCHW` tensor

Output:

- scalar root via `sum(poolOutput)`

### Indexed Loss

- [LossWorkloadSpec.java](./workload/LossWorkloadSpec.java)

Supported sub-kinds:

- `CROSS_ENTROPY_FROM_INDICES`
- `NLL_FROM_INDICES`

Example:

```java
WorkloadSpec spec = StandardWorkloads.indexedLoss(
        "cross_entropy_small",
        LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES,
        8,
        16,
        LossReduction.MEAN
);
```

Input:

- logits shape `[batch, classes]`
- target indices shape `[batch]`

Output:

- scalar loss

## Validation References

Built-in workload families now provide validation references automatically.

For the main families they use:

- `ValidationReference.baselineProfile(...)`

That means:

- candidate execution is compared against a baseline execution profile
- not against a hand-written ad hoc tensor literal in benchmark code

The helper is:

- [WorkloadValidationProfiles.java](./workload/WorkloadValidationProfiles.java)

Current baseline policy:

- `OptimizerConfig.noOptimization()`
- approximation off
- exact transcendentals forced
- BLAS disabled

This gives a conservative and stable correctness reference.

## Catalog

The built-in catalog is:

- [StandardWorkloads.java](./workload/StandardWorkloads.java)

It currently registers:

- `matmul_small`
- `matmul_batched_attention_like`
- `conv2d_resnet_3x3`
- `layer_norm_small`
- `max_pool2d_small`
- `cross_entropy_small`
- `transformer_hot_path`

The catalog is intentionally lightweight:

- names are stable lookup keys
- specs stay as normal Java code
- no separate scenario DSL is required

## Adding Custom Workloads

The recommended pattern is:

1. define the graph root only from public tensor API
2. make the root scalar when possible
3. attach shape/workload metadata needed by reports and persistence
4. choose a validation reference that is deterministic and conservative

Minimal shape-stable example:

```java
WorkloadSpec spec = new TensorRootWorkloadSpec(
        "mlp_block",
        WorkloadKind.GENERIC,
        environment -> {
            Tensor x = Tensor.randn(new int[]{16, 128}, environment.profile().dataType(), "x");
            Tensor w1 = Tensor.randn(new int[]{128, 256}, environment.profile().dataType(), "w1");
            Tensor b1 = Tensor.randn(new int[]{256}, environment.profile().dataType(), "b1");
            Tensor w2 = Tensor.randn(new int[]{256, 128}, environment.profile().dataType(), "w2");
            Tensor b2 = Tensor.randn(new int[]{128}, environment.profile().dataType(), "b2");
            Tensor y = x.matmul(w1).add(b1).relu().matmul(w2).add(b2);
            y.setLabel("mlp_block_output");
            return y.sum();
        },
        environment -> ValidationTarget.label("mlp_block_output"),
        environment -> ValidationReference.baselineProfile(
                WorkloadValidationProfiles.baselineFor(environment.profile()),
                java.util.List.of()
        ),
        environment -> WorkloadMetadata.of(
                "mlp_block",
                WorkloadKind.GENERIC,
                java.util.Map.of("batch", 16, "in", 128, "hidden", 256, "out", 128)
        )
);
```

Input:

- profile-selected dtype/mode/workload knobs
- fresh tensors per candidate

Output:

- reproducible benchmark/autotune scenario

Recommended rule:

- if you scalarize only for benchmarking, expose the pre-reduction tensor through a stable validation target
- if the scalar is the real semantic result, use the same tensor for both roots
- for rewriteable workloads, prefer an explicit label-based target over object identity

## Selecting Scenarios for Benchmark Runs

There are two normal ways to choose what gets measured.

### 1. Direct spec call

Use this when the benchmark is hardcoded in one program:

```java
BenchmarkRequest request = TuningDefaults.quickBenchmark(
        StandardWorkloads.pool2d(
                "max_pool2d_small",
                Pool2dWorkloadSpec.PoolKind.MAX,
                2, 8, 16, 16,
                Pool2dOptions.square(2)
        ),
        candidates
);
```

### 2. Catalog lookup by name

Use this when one entry program can run many named scenarios:

```java
WorkloadCatalog catalog = StandardWorkloads.defaultCatalog();

BenchmarkSuiteRequest request = catalog.benchmarkSuiteRequest(
        java.util.List.of("matmul_small", "conv2d_resnet_3x3", "transformer_hot_path"),
        candidates,
        TuningDefaults.balancedMeasurement(),
        TuningDefaults.quickValidation(),
        TuningDefaults.defaultReportPolicy()
);
```

Input:

- selected scenario names
- one comparable candidate family

Output:

- suite request that benchmarks only those scenarios

## Preset-Driven Scenario Selection

For normal usage, the preferred entry point is now preset-driven request construction.

Example:

```java
BenchmarkRequest request = StandardWorkloads.benchmark(
        "matmul_small",
        candidates,
        TuningPreset.BALANCED
);
```

And for a suite:

```java
BenchmarkSuiteRequest request = StandardWorkloads.benchmarkSuite(
        java.util.List.of("matmul_small", "layer_norm_small", "transformer_hot_path"),
        candidates,
        TuningPreset.QUICK
);
```

This is preferred over manually spelling out:

- `MeasurementPolicy`
- `ValidationPolicy`
- `ReportPolicy`
- `BaselinePolicy`

unless you are intentionally overriding those defaults.

For the most common path, workload-aware recommendations are also available.

Example:

```java
BenchmarkRequest request = StandardWorkloads.benchmark(
        "cross_entropy_small",
        candidates
);
```

This uses the recommended benchmark preset for `LOSS` workloads through:

- [WorkloadPresetFamily.java](./session/WorkloadPresetFamily.java)

So the caller can start with:

- named workload
- candidates

and accept the workload-family default preset unless there is a reason to override it.

## Examples

### Example: register custom workload

```java
WorkloadCatalog catalog = StandardWorkloads.defaultCatalog()
        .register(new TensorRootWorkloadSpec(
                "custom_add",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(1.0).add(Tensor.scalar(2.0))
        ));
```

Input:

- no explicit tensors from outside

Output:

- one new workload entry callable by name
- the benchmark runner can now select it through catalog lookup

### Example: instantiate workload for a profile

```java
ExecutionProfile profile = new ExecutionProfile(
        "workload",
        "workload",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults(),
        WorkloadProfile.none()
);

WorkloadInstance instance = StandardWorkloads
        .matmul("matmul_small", 1, 64, 64, 64)
        .instantiate(new WorkloadEnvironment(profile));
```

Output:

- fresh tensor graph
- baseline validation reference
- workload metadata for reports
