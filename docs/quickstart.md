<!-- generated-by: codex-docs-audit -->
# Quickstart

Navigation: [Index](index.md#recommended-reading-paths) | [README](../README.md#quickstart) | [Tensor API](tensor-api.md#api-surface-and-conventions) | [Compute Flow](compute-flow.md#lifecycle-map) | [Public API](public-api.md#stability-map) | [ONNX](onnx.md#onnx-import-and-export) | [Troubleshooting](troubleshooting.md#java-heap-space)

Chapters: [What Synaptik Is](#what-synaptik-is) | [Prerequisites](#prerequisites) | [Build And Verify](#build-and-verify) | [Mental Model](#mental-model) | [First Tensor Graph](#first-tensor-graph) | [Broadcasting](#broadcasting) | [Sequence-Shaped Tensors](#sequence-shaped-tensors) | [Reverse-Mode Autodiff](#reverse-mode-autodiff) | [Compile Prepare Execute](#compile-prepare-execute) | [Publication Policy](#publication-policy) | [Execution Profiles](#execution-profiles) | [ONNX Import And Export](#onnx-import-and-export) | [Accelerator Expectations](#accelerator-expectations) | [Autotune And Calibration](#autotune-and-calibration) | [Troubleshooting Checklist](#troubleshooting-checklist) | [What To Read Next](#what-to-read-next)

This guide is intentionally explicit. It explains the terms it uses, shows small value-level examples, and points to the deeper documentation when a topic grows beyond a quickstart.

## What Synaptik Is

Synaptik is an autograd engine and compiled tensor runtime written in Java.

That means:

- **Tensor**: a typed multidimensional value with shape, dtype, layout metadata, and storage.
- **Autograd**: automatic differentiation. Synaptik can build a backward graph from differentiable tensor operations and publish gradients to trainable leaf tensors.
- **Compiled tensor runtime**: tensor expressions are not executed immediately at every API call. User code first builds a graph, then Synaptik compiles, prepares, and executes it.
- **Runtime**: the part that actually runs kernels on CPU or an accelerator backend after compile-time planning is done.

Synaptik is not trying to be a full neural-network framework at the core layer. It does not own high-level concepts such as `Layer`, `Module`, `Optimizer` libraries for every architecture, dataloaders, checkpoint formats, or model zoos. Those can be built above Synaptik. The core repository should stay focused on:

- tensor graph construction;
- reverse-mode autodiff;
- compile-time graph optimization;
- backend planning;
- prepared CPU/accelerator execution;
- ONNX import/export for the supported static dense primitive subset;
- calibration, benchmark, and autotune infrastructure.

The practical consequence is that examples in this guide use primitive tensor operations directly. A higher-level neural-network layer such as batch normalization, attention, or an RNN can call these primitives, but the primitive engine should not become a full layer framework by accident.

## Prerequisites

Required:

- JDK 25.
- The bundled Gradle wrapper, `./gradlew`.
- The JDK Vector API module, `jdk.incubator.vector`.

The Gradle build already passes `--add-modules jdk.incubator.vector` for compile, test, and run tasks, so ordinary Gradle commands are the supported path.

Recommended local checks:

```bash
java -version
./gradlew --version
```

Expected shape of the output:

```text
openjdk version "25..."
Gradle 9.4.1...
```

If the Java version is older than 25, the build can fail before the Synaptik code is reached. If Gradle complains about the Vector API module, use the wrapper from the repository rather than a system Gradle with different JVM settings.

## Build And Verify

Start with the fastest useful verification:

```bash
./gradlew classes
```

This compiles the main source set. It is the first command to run after changing public API, operation descriptors, backend code, or docs that include source links.

Run the public-preview focused test set:

```bash
./gradlew test --tests 'onnx.*' --tests SourceTreeHygieneTest
```

Why this set:

- `onnx.*` verifies import/export, supported ONNX operator coverage, static parameter handling, checked-in fixtures, and round trips for many public graph forms.
- `SourceTreeHygieneTest` catches source-tree hygiene issues such as generated artifacts or unexpected files.

Run the full test suite when you need broader confidence:

```bash
./gradlew test
```

Hardware-specific checks are optional and depend on the local machine:

```bash
./gradlew metalTest
./gradlew cudaTest
```

These tests need the relevant native runtime pieces. A machine without Metal/CUDA support should not be treated as proving those paths broken. The source-level contract is that unsupported accelerator paths fail or fall back visibly, not silently.

## Mental Model

The core lifecycle has four stages:

```text
Tensor API
  -> semantic tensor DAG
  -> CompiledGraph
  -> PreparedExecution
  -> execute
```

Definitions:

- **Semantic tensor DAG**: the user-visible graph built by chaining `Tensor` operations. DAG means directed acyclic graph. A node points to its input nodes, and there are no cycles.
- **CompiledGraph**: an immutable-ish compile artifact rooted at one output tensor. It snapshots graph structure, applies compile-time rules, builds backward artifacts when requested, plans backend ownership, and records memory planning information.
- **PreparedExecution**: the runtime recipe for one compiled graph and one runtime policy. It decides concrete kernels, fused executables, backend metadata, and step order.
- **Execute**: the phase that allocates per-run state, runs prepared forward/backward steps, and publishes requested values back to public `Tensor` objects.

The most important boundary:

```text
compile decides what graph should be executed
prepare decides how this runtime should execute it
execute consumes the prepared recipe
```

Execution should not rediscover graph optimizer decisions. Runtime policy should not mutate the semantic graph. Tuning should compare real `ExecutionProfile` candidates rather than inventing another hidden execution path.

## First Tensor Graph

This example builds a two-operation graph:

```text
y = relu(a + b)
```

Java:

```java
import java.util.Arrays;

import tensor.DataType;
import tensor.Tensor;

public class SynaptikTinyForward {
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

What happened:

1. `new Tensor(...)` created two leaf tensors. A **leaf tensor** is an input-like tensor with no operation descriptor.
2. `a.add(b)` created a derived tensor node. It did not execute yet.
3. `.relu()` created another derived tensor node. It also did not execute yet.
4. `.compute()` compiled, prepared, and executed the graph with inference defaults.
5. `toDoubleArrayCopy()` copied the published result into an ordinary Java `double[]`.

Value-level calculation:

```text
a =
  [[1.0, 2.0],
   [3.0, 4.0]]

b =
  [10.0, 20.0]

a + b =
  [[1.0 + 10.0, 2.0 + 20.0],
   [3.0 + 10.0, 4.0 + 20.0]]

relu(a + b) =
  [[11.0, 22.0],
   [13.0, 24.0]]
```

Why the flat output is `[11.0, 22.0, 13.0, 24.0]`: Synaptik stores contiguous dense arrays in row-major order for these simple tensors. Row-major means the last dimension changes fastest.

## Broadcasting

Broadcasting means a smaller tensor is virtually expanded to match a larger tensor without physically copying the smaller data first.

In the previous example:

```text
a shape = [2, 2]
b shape = [2]
```

`b` is aligned to the trailing dimension:

```text
a logical rows:
  row 0 = [1.0, 2.0]
  row 1 = [3.0, 4.0]

b applies to each row:
  row 0 adds [10.0, 20.0]
  row 1 adds [10.0, 20.0]
```

Another example:

```java
Tensor left = new Tensor(
        new double[]{
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0
        },
        new int[]{2, 1, 4},
        null,
        "left",
        DataType.FLOAT64
);

Tensor right = new Tensor(
        new double[]{
                10.0, 20.0, 30.0, 40.0,
                50.0, 60.0, 70.0, 80.0,
                90.0, 100.0, 110.0, 120.0
        },
        new int[]{3, 4},
        null,
        "right",
        DataType.FLOAT64
);

Tensor out = left.add(right).compute();
```

Shape alignment:

```text
left shape  = [2, 1, 4]
right shape =    [3, 4]
aligned     = [2, 1, 4]
              [1, 3, 4]
out shape   = [2, 3, 4]
```

Broadcasting is also part of autodiff. If a broadcasted input receives a gradient with shape `[2, 3, 4]`, Synaptik reduces that gradient back to the original input shape. That is why a bias vector can receive a vector-shaped gradient even though it was used across many rows or batches.

## Sequence-Shaped Tensors

Synaptik does not provide RNN, LSTM, GRU, `Layer`, or `Model` abstractions. A neural-network framework built above Synaptik can own those concepts. The core tensor layer provides the N-D operations needed to represent a sequence as one tensor instead of as `Tensor[]`.

A common sequence layout is:

```text
[batch, time, features]
```

For example, two batches, three timesteps, and four input features is shape `[2, 3, 4]`. For the complete reference, including autograd contracts and implementation source mapping, see [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope).

### N-D Linear Over The Last Dimension

`linear(weight, bias)` projects only the last dimension and preserves every leading dimension:

```text
input  shape [..., inFeatures]
weight shape [inFeatures, outFeatures]
bias   shape [outFeatures] or [1, outFeatures]
output shape [..., outFeatures]
```

Concrete example:

```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
Tensor w = Tensor.randn(new int[]{4, 5}, 0.0, 0.02, DataType.FLOAT64, "w");
Tensor b = Tensor.zeros(new int[]{5}, DataType.FLOAT64, "b");

Tensor y = x.linear(w, b).compute();
```

Shapes:

```text
x = [2, 3, 4]
w = [4, 5]
b = [5]
y = [2, 3, 5]
```

### Stack And Unstack

`Tensor.stack(axis, ...)` inserts a new dimension. This is useful when legacy code still has one tensor per timestep:

```java
Tensor t0 = Tensor.randn(new int[]{2, 4}, 0.0, 1.0, DataType.FLOAT64, "t0");
Tensor t1 = Tensor.randn(new int[]{2, 4}, 0.0, 1.0, DataType.FLOAT64, "t1");
Tensor t2 = Tensor.randn(new int[]{2, 4}, 0.0, 1.0, DataType.FLOAT64, "t2");

Tensor byTime = Tensor.stack(1, t0, t1, t2);
```

Shape:

```text
t0, t1, t2 = [batch, features] = [2, 4]
byTime     = [batch, time, features] = [2, 3, 4]
```

`unstack(axis)` reverses that shape transformation and returns one tensor per position on the selected axis:

```java
Tensor[] timesteps = byTime.unstack(1);
```

Each entry has shape `[2, 4]`. Gradients flow through both `stack` and `unstack` because they are composed from existing differentiable layout/index primitives.

### Indexing A Time Axis

Use `sliceAxis` for a contiguous range and `take` for explicit positions:

```java
Tensor firstTwoSteps = byTime.sliceAxis(1, 0, 2); // shape [2, 2, 4]
Tensor endpoints = byTime.take(1, new int[]{0, 2}); // shape [2, 2, 4]
```

`take(axis, int[])` is a convenience wrapper over ONNX-style `gatherAxis`: the index list shape is inserted at the gathered axis.

### Masked Reductions And Masked Loss

Padded sequences usually need a BOOL validity mask. `true` means the timestep is valid, `false` means it is padding:

```java
Tensor values = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "values");
Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor meanOverTime = values.mean(1, mask); // shape [2, 4]
```

The mask `[2, 3]` is interpreted as `[2, 3, 1]` and broadcast over `features`. The denominator is the number of valid timesteps, not the padded sequence length.

For per-timestep classification:

```java
Tensor logits = Tensor.randn(new int[]{2, 3, 10}, 0.0, 1.0, DataType.FLOAT64, "logits");
Tensor targets = new Tensor(new double[]{
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,

        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0
}, new int[]{2, 3, 10}, null, "oneHotTargets", DataType.FLOAT64);
Tensor loss = logits.crossEntropyLoss(targets, 2, mask);
```

Here `classDimension = 2`, so the mask shape is the logits shape with the class axis removed: `[2, 3]`. The loss is normalized by valid mask positions, not by all padded `[batch * time]` positions.

## Reverse-Mode Autodiff

Reverse-mode autodiff computes gradients from one output back to many inputs. It is the usual mode for neural-network training because one scalar loss depends on many parameters.

Example:

```java
import java.util.Arrays;

import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

public class SynaptikTinyGradient {
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

        System.out.println(Arrays.toString(loss.toDoubleArrayCopy()));
        System.out.println(Arrays.toString(x.getGradient().toDoubleArrayCopy()));
    }
}
```

Expected output:

```text
[14.0]
[2.0, -4.0, 6.0]
```

Explanation:

```text
x = [1.0, -2.0, 3.0]
x * x = [1.0, 4.0, 9.0]
sum(x * x) = 14.0
d(x^2)/dx = 2x
gradient = [2.0, -4.0, 6.0]
```

Important terms:

- **Requires grad**: `x.setRequiresGrad(true)` marks a leaf tensor as trainable or gradient-observable.
- **Loss**: a scalar or small output used as the root of backward propagation.
- **Backward graph**: the graph Synaptik builds during training compile to compute gradients from the loss to trainable leaves.
- **Gradient publication**: the final step that attaches computed gradient tensors back to the public leaf tensors, for example through `x.getGradient()`.

`CompileMode.TRAINING` means "build and execute backward if the graph has trainable leaves." It does not fabricate gradients for graphs with no trainable inputs.

## Compile Prepare Execute

For examples and small tests, `compute()` is enough. For real application code, repeated execution, backend diagnostics, and benchmarks, use the explicit lifecycle.

```java
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

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
Tensor y = a.add(b).relu();

CompiledGraph compiled = CompiledGraph.compile(
        y,
        CompileConfig.inference(),
        CompileMode.INFERENCE_ONLY
);

PreparedExecution prepared = compiled.prepare(RuntimeConfig.inferenceDefaults());
prepared.execute(ExecutionMode.FORWARD);

double[] values = y.toDoubleArrayCopy();
```

What each line owns:

| Code | Stage | Meaning |
|---|---|---|
| `a.add(b).relu()` | graph build | Creates semantic tensor nodes. No kernels run. |
| `CompiledGraph.compile(...)` | compile | Builds compile artifacts, optimizer output, backend ownership, memory plan. |
| `compiled.prepare(...)` | prepare | Resolves runtime/backend policy into concrete executable steps. |
| `prepared.execute(...)` | execute | Runs the prepared steps and publishes requested outputs. |
| `y.toDoubleArrayCopy()` | user read | Copies public tensor storage into an ordinary Java array. |

Reuse rule:

- If graph topology, shapes, dtypes, layout intent, backend intent, and compile config are unchanged, reuse `CompiledGraph`.
- If runtime policy is unchanged, reuse `PreparedExecution`.
- If input values change but shapes/dtypes/layouts do not, update input tensor storage and execute the same prepared artifact again.
- If you mutate graph structure or dtype/layout contracts, compile and prepare again.

Example repeated execution:

```java
a.setData(new double[]{5.0, 6.0, 7.0, 8.0});
prepared.execute(ExecutionMode.FORWARD);
System.out.println(java.util.Arrays.toString(y.toDoubleArrayCopy()));
```

Expected second output:

```text
[15.0, 26.0, 17.0, 28.0]
```

## Publication Policy

Publication means copying values from run-scoped execution state back to user-visible `Tensor` objects after execution.

Why this exists:

- During execution, backends may keep temporary values in per-run buffers.
- Accelerators may keep values device-owned.
- Benchmarks may not need every intermediate tensor copied back to CPU-visible storage.
- Training loops often need gradients and output, but not every intermediate forward value.

`PublicationPolicy` controls what is published:

| Policy | Publishes intermediates | Publishes output | Publishes gradients | Typical use |
|---|---:|---:|---:|---|
| `ALL` | yes | yes | yes | Debugging intermediate values. |
| `OUTPUT_AND_GRADIENTS` | no | yes | yes | Default ordinary execution. |
| `OUTPUT_ONLY` | no | yes | no | Optimizer steps or inference where gradients are not read. |
| `NONE` | no | no | no | Benchmark paths that only need traces or device-residency behavior. |

Example:

```java
import backend.runtime.ExecutionMode;
import graph.execution.PublicationPolicy;

prepared.execute(ExecutionMode.FORWARD, PublicationPolicy.OUTPUT_ONLY);
```

Important boundary: publication is not graph optimization and not backend planning. It only controls which already-computed run values are synchronized to public tensors after execution.

## Execution Profiles

An `ExecutionProfile` is the complete runnable policy object. It combines:

- a profile name;
- a candidate name used in benchmarks/autotune;
- a dtype;
- an execution mode;
- a `CompileConfig`;
- a `RuntimeConfig`;
- optional workload metadata.

Example:

```java
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import tensor.DataType;

ExecutionProfile profile = new ExecutionProfile(
        "quickstart-inference-f64",
        "cpu-inference-defaults",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        CompileConfig.inference(),
        RuntimeConfig.inferenceDefaults()
);

y.compute(profile);
```

Built-in profile helpers exist for common cases:

```java
ExecutionProfile inferenceCpu = ExecutionProfile.inferenceCpu();
ExecutionProfile trainingCpu = ExecutionProfile.trainingCpu();
ExecutionProfile inferenceAutoAccelerator = ExecutionProfile.inferenceAutoAccelerator();
ExecutionProfile trainingAutoAccelerator = ExecutionProfile.trainingAutoAccelerator();
```

Profile naming tip:

- `profileName` should identify the workload or profile namespace.
- `candidateName` should identify the measured candidate inside that namespace.

Example:

```text
profileName   = "abc-sequence-matmul-f32"
candidateName = "cpu-default-runtime"
```

That split makes benchmark reports easier to read because many candidates can belong to one workload.

## ONNX Import And Export

ONNX support is a static dense inference subset. Static means shapes and shape-like parameters are known at import/export time. Dense means regular dense tensors, not sparse tensors. Inference subset means the supported interchange path focuses on forward graphs, not full ONNX training metadata or control flow.

Minimal export:

```java
import java.nio.file.Path;

import onnx.Onnx;
import onnx.OnnxExportOptions;
import onnx.OnnxLeafTensorPolicy;
import onnx.OnnxModel;
import tensor.DataType;
import tensor.Tensor;

Tensor a = new Tensor(
        new float[]{1f, 2f, 3f, 4f},
        new int[]{2, 2},
        null,
        "a",
        DataType.FLOAT32
);
Tensor b = new Tensor(
        new float[]{10f, 20f},
        new int[]{2},
        null,
        "b",
        DataType.FLOAT32
);
Tensor y = a.add(b).relu();
y.setLabel("y");

OnnxModel model = Onnx.exportModel(
        y,
        OnnxExportOptions.defaults()
                .withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
);
model.write(Path.of("build/quickstart/relu_add.onnx"));
```

Why `OnnxLeafTensorPolicy.INPUTS` matters: normal leaf tensors become ONNX model inputs instead of serialized initializers. That is usually what you want for an interchange model with runtime inputs.

Minimal import and execute:

```java
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import onnx.ImportedOnnxModel;
import onnx.Onnx;
import java.nio.file.Path;

ImportedOnnxModel imported = Onnx.read(Path.of("build/quickstart/relu_add.onnx"));

imported.input("a").setData(new float[]{1f, 2f, 3f, 4f});
imported.input("b").setData(new float[]{10f, 20f});

imported.compile("y", CompileConfig.inference())
        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

double[] output = imported.output("y").toDoubleArrayCopy();
```

Expected output:

```text
[11.0, 22.0, 13.0, 24.0]
```

Common ONNX boundaries:

- Dynamic shape execution is not general yet.
- `NonZero` is rejected because its output shape depends on input values.
- `If`, `Loop`, and `Scan` are not supported.
- Sparse tensors, quantized tensors, string tensors, sequence/map/optional values, external data files, and custom domains are outside the current core importer.
- Some ONNX ops are represented as composed Synaptik DAGs, for example `ReduceL2` as square, sum, and sqrt.
- `Split` is a narrow multi-output exception through static slice siblings, not proof of a general multi-output runtime architecture.

For the exact current matrix, read [ONNX Coverage](onnx-coverage.md).

## Accelerator Expectations

CPU is the correctness baseline. Metal and CUDA support real accelerator paths for scoped operation families, but support is backend-specific.

Terms:

- **Backend**: an execution target such as CPU, Metal, CUDA, or OpenCL.
- **Accelerator**: a non-CPU backend such as Metal or CUDA.
- **Backend planning**: compile-time ownership planning. It decides which graph regions are CPU-owned or accelerator-owned.
- **Runtime accelerator policy**: prepare/execute-time hardware policy, such as availability checks and buffer binding mode.
- **Fallback**: using CPU or another path because the requested accelerator path is unavailable, illegal, or unsupported.

Do not assume "ONNX supported" means "GPU native supported." ONNX import/export support says Synaptik can represent the graph. GPU support says a specific backend can lower and execute it natively.

Use accelerator profiles when you want accelerator region discovery:

```java
ExecutionProfile profile = ExecutionProfile.inferenceAutoAccelerator();
y.compute(profile);
```

Use required accelerator modes only for tests and diagnostics. Required modes should fail when an accelerator plan cannot be built, which is useful when proving coverage. Optional modes are better for ordinary experimentation because fallback is visible in traces.

## Autotune And Calibration

Synaptik has two related measurement systems:

| System | Owns | Example question |
|---|---|---|
| Platform calibration | Runtime/hardware defaults | "At what size should this CPU use vector or parallel kernels on this machine?" |
| Graph autotune | Compile/profile candidates for one workload | "For this graph and this frozen runtime profile, which backend planning or graph policy candidate is fastest?" |

Calibration should not invent graph ownership. Graph autotune should not invent hidden runtime semantics. Both systems eventually produce or compare real `ExecutionProfile` objects.

Common CLI commands:

```bash
./gradlew run --args="calibrate --dtype f32 --families all"
./gradlew run --args="autotune f32"
./gradlew run --args="benchmark-winner f32"
./gradlew run --args="benchmark-graph-space f32"
```

Local profile artifacts are normally written under:

```text
profiles/platform/<platform-id>/calibration/schema-v2/...
profiles/platform/<platform-id>/tuning/abc/...
```

`<platform-id>` is a short current-platform key such as `macos-arm64`. For ordinary `Tensor.compute(...)` calls,
Synaptik first tries to load a compatible calibrated runtime profile from the configured profile roots and then falls
back to built-in runtime defaults if none exists. Consumer projects can point at an external profile root with
`-Dsynaptik.profiles.root=/path/to/profiles` or `SYNAPTIK_PROFILES_ROOT=/path/to/profiles`.

Do not commit local benchmark/calibration artifacts unless the plan explicitly promotes them as canonical fixtures or release evidence.

## Troubleshooting Checklist

| Symptom | Likely cause | What to check |
|---|---|---|
| `UnsupportedClassVersionError` or JDK-related compile failure | Wrong Java version | Run `java -version`; use JDK 25. |
| Vector API module error | JVM not launched with `jdk.incubator.vector` | Use `./gradlew`; the build config adds the module. |
| Shape mismatch exception | Tensor operands are not broadcast-compatible | Print `Arrays.toString(tensor.getShape())` for every operand. |
| `getGradient()` is `null` | No backward execution or no trainable leaf | Call `setRequiresGrad(true)` on leaves and execute with `CompileMode.TRAINING` or `FORWARD_BACKWARD`. |
| Output tensor still has old values | Execution did not publish that value | Check `PublicationPolicy`; `NONE` intentionally publishes nothing. |
| ONNX import rejects a model | Model crosses the static dense subset boundary | Read the exception message and [ONNX Coverage](onnx-coverage.md). |
| GPU path did not run | Planner rejected the region or runtime unavailable | Inspect compile/prepare/run traces and backend coverage docs. |
| BF16 on CPU is not faster than F32 | BF16 storage often promotes to wider compute and pays conversion costs | Read [CPU BF16 Runtime](cpu-bf16.md). |
| Benchmark changed after calibration | Benchmark may be loading current platform profiles or local profile artifacts changed | Check `profiles/platform/...` and the benchmark profile resolution path. |

## What To Read Next

Use these paths based on what you are doing:

- Learn public tensor operations: [Tensor API](tensor-api.md).
- Understand the full lifecycle: [Compute Flow](compute-flow.md).
- Understand compile-time rewrite stages: [Graph Optimizer](graph-optimizer.md).
- Understand backend ownership and regions: [Backend Planning And Regions](backend-planning-and-regions.md).
- Understand execution profiles and runtime knobs: [Configuration](configuration.md).
- Add a new primitive operation: [Adding A Tensor Operation](adding-tensor-operation.md).
- Work on ONNX: [ONNX](onnx.md) and [ONNX Coverage](onnx-coverage.md).
- Debug Metal: [Metal Backend](metal-backend.md) and [Metal Operation Parity](metal-operation-parity.md).
- Debug BF16 performance: [CPU BF16 Runtime](cpu-bf16.md).
- Work on tuning: [Calibration And Autotune](calibration-autotune.md) and the package docs under `src/main/java/tuning/`.
