<!-- generated-by: gsd-doc-writer -->
# Examples

Navigation: [Index](index.md#recommended-reading-paths) | [Tensor API](tensor-api.md#operation-catalog) | [Compute Flow](compute-flow.md#worked-example) | [Public API](public-api.md#tensor) | [Calibration & Autotune](calibration-autotune.md#ergonomic-fluent-api) | [Testing](testing.md#exact-commands)

Chapters: [Running Examples](#running-examples) | [Broadcast Add And ReLU](#broadcast-add-and-relu) | [Reverse-Mode Autodiff](#reverse-mode-autodiff) | [Matrix Multiplication](#matrix-multiplication) | [Sequence-Shaped Tensor](#sequence-shaped-tensor) | [Boolean Mask With `where`](#boolean-mask-with-where) | [Softmax](#softmax) | [Explicit Compile And Runtime Config](#explicit-compile-and-runtime-config) | [Reusing PreparedExecution](#reusing-preparedexecution) | [ComputeOptions With Explicit Defaults](#computeoptions-with-explicit-defaults) | [Programmatic Tuning API](#programmatic-tuning-api) | [CLI Examples](#cli-examples) | [Verification Notes](#verification-notes)

These examples are small Java snippets using the public `Tensor`, compile, and configuration APIs. They are written to be pasted into a small class in this repository or adapted into a test.

## Table Of Contents

- [Running Examples](#running-examples)
- [Broadcast Add And ReLU](#broadcast-add-and-relu)
- [Reverse-Mode Autodiff](#reverse-mode-autodiff)
- [Matrix Multiplication](#matrix-multiplication)
- [Sequence-Shaped Tensor](#sequence-shaped-tensor)
- [Boolean Mask With `where`](#boolean-mask-with-where)
- [Softmax](#softmax)
- [Explicit Compile And Runtime Config](#explicit-compile-and-runtime-config)
- [Reusing PreparedExecution](#reusing-preparedexecution)
- [ComputeOptions With Explicit Defaults](#computeoptions-with-explicit-defaults)
- [Programmatic Tuning API](#programmatic-tuning-api)
- [CLI Examples](#cli-examples)
- [Verification Notes](#verification-notes)

## Running Examples

Use the Gradle wrapper so the required JVM flags and dependencies are present:

```bash
./gradlew classes
```

For standalone Java execution outside Gradle, include:

- `build/classes/java/main`
- ASM runtime jars from Gradle's dependency cache
- JVM flags from `build.gradle`: `--add-modules=jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED`

The easiest executable pattern is to add a temporary class under `/tmp` or a scratch directory and run it with the Gradle runtime classpath. The examples below focus on the Java body and expected output.

## Broadcast Add And ReLU

```java
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class BroadcastReluExample {
    public static void main(String[] args) {
        Tensor a = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new int[]{2, 2},
                null,
                "a",
                DataType.FLOAT64
        );
        Tensor b = new Tensor(
                new double[]{10.0, 20.0},
                new int[]{2},
                null,
                "b",
                DataType.FLOAT64
        );

        Tensor y = a.add(b).relu().compute();

        System.out.println(Arrays.toString(y.getShape()));
        System.out.println(Arrays.toString(y.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[2, 2]
[11.0, 22.0, 13.0, 24.0]
```

## Reverse-Mode Autodiff

```java
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class AutodiffExample {
    public static void main(String[] args) {
        Tensor x = new Tensor(
                new double[]{1.0, -2.0, 3.0},
                new int[]{3},
                null,
                "x",
                DataType.FLOAT64
        );
        x.setRequiresGrad(true);

        Tensor loss = x.mul(x).sum();
        loss.compute(CompileMode.TRAINING);

        System.out.println(loss.scalarAsDouble());
        System.out.println(Arrays.toString(x.getGradient().toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
14.0
[2.0, -4.0, 6.0]
```

## Matrix Multiplication

```java
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class MatmulExample {
    public static void main(String[] args) {
        Tensor left = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT64
        );
        Tensor right = new Tensor(
                new double[]{7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                new int[]{3, 2},
                null,
                "right",
                DataType.FLOAT64
        );

        Tensor product = left.matmul(right).compute();

        System.out.println(Arrays.toString(product.getShape()));
        System.out.println(Arrays.toString(product.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[2, 2]
[58.0, 64.0, 139.0, 154.0]
```

## Sequence-Shaped Tensor

This example keeps a sequence as one tensor with shape `[batch, time, features]`. It demonstrates three public APIs together:

- `linear(weight, bias)` projects the last dimension and preserves `[batch, time]`.
- `mean(axis, mask)` ignores padded timesteps.
- `take(axis, int[])` selects explicit positions from the time axis.

```java
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class SequenceTensorExample {
    public static void main(String[] args) {
        Tensor x = new Tensor(
                new double[]{
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                        10, 11, 12
                },
                new int[]{2, 2, 3},
                null,
                "x",
                DataType.FLOAT64
        );
        Tensor weight = new Tensor(
                new double[]{
                        1, 10,
                        2, 20,
                        3, 30
                },
                new int[]{3, 2},
                null,
                "weight",
                DataType.FLOAT64
        );
        Tensor bias = new Tensor(new double[]{0.5, -0.5}, new int[]{1, 2}, null, "bias", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);

        Tensor projected = x.linear(weight, bias).compute();
        Tensor meanOverValidTime = projected.mean(1, mask).compute();
        Tensor endpoints = projected.take(1, new int[]{0, 1}).compute();

        System.out.println(Arrays.toString(projected.getShape()));
        System.out.println(Arrays.toString(projected.toDoubleArrayCopy()));
        System.out.println(Arrays.toString(meanOverValidTime.getShape()));
        System.out.println(Arrays.toString(meanOverValidTime.toDoubleArrayCopy()));
        System.out.println(Arrays.toString(endpoints.getShape()));
    }
}
```

Expected output:

```text
[2, 2, 2]
[14.5, 139.5, 32.5, 319.5, 50.5, 499.5, 68.5, 679.5]
[2, 2]
[14.5, 139.5, 59.5, 589.5]
[2, 2, 2]
```

Why the masked mean is `[14.5, 139.5, 59.5, 589.5]`:

```text
batch 0 mask = [true, false]
  mean = projected[0, 0, :] = [14.5, 139.5]

batch 1 mask = [true, true]
  mean = (projected[1, 0, :] + projected[1, 1, :]) / 2
       = ([50.5, 499.5] + [68.5, 679.5]) / 2
       = [59.5, 589.5]
```

See [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope) for the full shape and autograd contract.

## Boolean Mask With `where`

```java
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class WhereExample {
    public static void main(String[] args) {
        Tensor values = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new int[]{2, 2},
                null,
                "values",
                DataType.FLOAT64
        );
        Tensor mask = new Tensor(
                new byte[]{1, 0, 1, 0},
                new int[]{2, 2},
                null,
                "mask",
                DataType.BOOL
        );

        Tensor selected = Tensor.where(
                mask,
                values,
                Tensor.scalar(-1.0, DataType.FLOAT64)
        ).compute();

        System.out.println(Arrays.toString(selected.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[1.0, -1.0, 3.0, -1.0]
```

## Softmax

```java
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class SoftmaxExample {
    public static void main(String[] args) {
        Tensor logits = new Tensor(
                new double[]{1.0, 2.0, 3.0},
                new int[]{3},
                null,
                "logits",
                DataType.FLOAT64
        );

        Tensor probabilities = logits.softmax(0).compute();

        System.out.println(Arrays.toString(probabilities.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[0.09003057317038045, 0.2447284710547976, 0.6652409557748218]
```

## Explicit Compile And Runtime Config

```java
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class ExplicitCompileExample {
    public static void main(String[] args) {
        Tensor x = new Tensor(
                new double[]{1.0, 2.0},
                new int[]{2},
                null,
                "x",
                DataType.FLOAT64
        );
        Tensor y = x.mul(4.0);

        CompiledGraph graph = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
        graph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        System.out.println(Arrays.toString(y.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[4.0, 8.0]
```

## Reusing PreparedExecution

Use `PreparedExecution` when graph structure and runtime config are stable and only input storage changes between runs.

```java
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import runtime.execution.PreparedExecution;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;

public class PreparedExecutionExample {
    public static void main(String[] args) {
        Tensor x = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor y = x.mul(10.0);

        CompiledGraph graph = CompiledGraph.compile(y, CompileConfig.inference());
        PreparedExecution prepared = graph.prepare(RuntimeConfig.inferenceDefaults());

        prepared.execute(ExecutionMode.FORWARD);
        System.out.println(Arrays.toString(y.toDoubleArrayCopy()));

        x.setData(new double[]{3.0, 4.0});
        prepared.execute(ExecutionMode.FORWARD);
        System.out.println(Arrays.toString(y.toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[10.0, 20.0]
[30.0, 40.0]
```

Prepared executions create per-run execution state and forward leaf runtime tensors alias the current source tensor storage. Reusing a prepared execution is supported when shapes, dtypes, and graph structure stay unchanged; changing those structural properties requires recompiling.

## ComputeOptions With Explicit Defaults

```java
import tensor.AutotunePolicy;
import tensor.CompileMode;
import tensor.ComputeOptions;
import tensor.DataType;
import tensor.Tensor;

public class ComputeOptionsExample {
    public static void main(String[] args) {
        Tensor x = new Tensor(new double[]{2.0}, new int[]{1}, null, "x", DataType.FLOAT64);

        Tensor y = x.mul(3.0).compute(new ComputeOptions()
                .compileMode(CompileMode.INFERENCE_ONLY)
                .autotune(AutotunePolicy.NEVER));

        System.out.println(y.scalarAsDouble());
    }
}
```

Expected output:

```text
6.0
```

## Programmatic Tuning API

Use `synaptik.app.TuningCli` for shell-driven calibration and benchmark runs. Use
`tuning.api.Synaptik` when an application wants regular Java configuration with dot-style builders.
The same low-level objects are produced either way.

```java
List<PlatformCalibrationResult> results = Synaptik.tuning()
        .calibration()
        .dtypes().single(DataType.FLOAT64)
        .families().all()
        .quick()
        .mode().training()
        .measurement().iterations(1, 3, 1)
        .progress().lines()
        .color().auto()
        .outputRoot(Path.of("profiles"))
        .run();

PlatformRuntimeProfile calibratedRuntime = results.getLast().finalRuntimeProfile();

ExecutionProfile baseline = Synaptik.tuning()
        .profile()
        .name("example-baseline-no-opt-f64")
        .candidate("baseline-no-opt")
        .dtype(DataType.FLOAT64)
        .mode().training()
        .compile().noGraphOptimization()
        .runtime().noOptNoVecNoPar()
        .build();

ExecutionProfile calibrated = Synaptik.tuning()
        .profile()
        .name("example-calibrated-runtime-f64")
        .candidate("calibrated-runtime")
        .dtype(DataType.FLOAT64)
        .mode().training()
        .compile().trainingDefaults()
        .runtime().fromPlatformProfile(calibratedRuntime)
        .build();

WorkloadSpec workload = StandardWorkloads.abcSequenceMatmulBlasBenchmark(
        "example_abc_sequence_matmul_f64"
);

BenchmarkReport report = Synaptik.tuning()
        .benchmark()
        .workload(workload)
        .quick()
        .report()
                .hotStepLimit(5)
                .includeTrace()
                .includeFailedCandidates()
                .done()
        .compare()
        .baseline("baseline-no-opt", baseline)
        .candidate("calibrated-runtime", calibrated)
        .run();

System.out.println(TextBenchmarkReportRenderer.render(report));

// baseline.compile() = CompileConfig.noGraphOptimizationBaseline()
// baseline.runtime() = RuntimeConfig.noOptNoVecNoPar()
// calibrated.compile() = CompileConfig.training()
// calibrated.runtime() = calibratedRuntime.toRuntimeConfig()
```

## CLI Examples

The Gradle application entry point is `synaptik.app.TuningCli`. `synaptik.app.Main` is a
programmatic example that configures calibration and benchmarking through `tuning.api.Synaptik`.

Print usage for an unknown command:

```bash
./gradlew run --args="help"
```

Run the full local flow for `f64`:

```bash
./gradlew run --args="full f64"
```

Run calibration for all supported dtypes and all non-accelerator calibration families:

```bash
./gradlew run --args="calibrate --dtypes all --families all"
```

Run a single calibration family with explicit measurement counts:

```bash
./gradlew run --args="calibrate --dtype f32 --family matmul --measurement 30:100:2 --progress lines --color never"
```

Run autotune and then benchmark the stored winner:

```bash
./gradlew run --args="autotune f32"
./gradlew run --args="benchmark-winner f32"
```

Expected side effects:

- Calibration writes under `profiles/platform/<platform-id>/calibration/schema-v2/...` unless `--output-root` changes the root.
- CLI autotune writes `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json` and `<dtype>-history.jsonl`.
- `autotune` requires a calibration profile first.
- `benchmark-winner` requires a best-profile artifact first.

## Verification Notes

The first five Java examples were executed against `build/classes/java/main` after `./gradlew classes`. The targeted verification command also passed:

```bash
./gradlew test --tests BroadcastBinaryOpsTest --tests CompiledGraphIdempotencyTest --tests PreparedExecutionBuildTest --tests ExecutionProfileIoTest
```
