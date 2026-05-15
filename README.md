<!-- generated-by: gsd-doc-writer -->
# Synaptik

Documentation: [docs/index.md](docs/index.md) | [Quickstart](docs/quickstart.md) | [Tensor API](docs/tensor-api.md) | [Sequence Tensor Primitives](docs/sequence-tensor-primitives.md) | [Compute Flow](docs/compute-flow.md) | [Graph Optimizer](docs/graph-optimizer.md) | [Calibration & Autotune](docs/calibration-autotune.md) | [Public API](docs/public-api.md)

Current version: `0.1.0-alpha.3`

Synaptik is a Java autograd engine and compiled tensor runtime built around an explicit graph lifecycle:

1. the public `Tensor` API builds a semantic graph
2. `CompiledGraph` snapshots and optimizes that graph
3. `PreparedExecution` attaches runtime policy and backend metadata
4. the selected backend executes prepared node steps

The project is not designed as an eager-only numerical notebook library.
Its center of gravity is compiled graph execution, reverse-mode autodiff, separated compile/runtime policy, and platform/profile-driven execution.

Today the CPU backend is the broadest and most complete backend. Metal and CUDA also have real scoped accelerator paths with native buffer/residency plumbing and explicit fallback evidence in traces. OpenCL remains much thinner. Unsupported operations, dtypes, layouts, or unavailable native runtimes must fall back visibly rather than pretending that an accelerator path ran.

## Public Preview Status

`0.1.0-alpha.3` is a public technical preview baseline. It is suitable for
architecture review, local experimentation, ONNX static dense inference trials,
CPU correctness work, and accelerator coverage investigation.

It is not a production-stability claim:

- public Java APIs may still change before `1.0.0`;
- internal package names and configuration records may still be cleaned up;
- CPU is the correctness baseline;
- accelerator execution is scoped, capability-gated, and trace-visible;
- ONNX support is a static dense inference subset, not the full ONNX ecosystem;
- benchmark and calibration artifacts are local unless explicitly promoted.

Release notes live in [CHANGELOG.md](CHANGELOG.md). The release checklist is in
[docs/release.md](docs/release.md). The current source license status is in
[LICENSE.md](LICENSE.md).

## Quickstart

For the full guided path with terms, shapes, exact values, explicit compile/prepare/execute examples, publication policy, ONNX import/export, accelerator expectations, and troubleshooting, start with [docs/quickstart.md](docs/quickstart.md).

Use Synaptik from another Gradle project through JitPack:

```groovy
repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation "com.github.pho001:Synaptik:v0.1.0-alpha.3"
}
```

The main Synaptik artifact declares the macOS ARM64 Metal native runtime artifact as a runtime dependency. On Apple
Silicon, the Metal loader looks for `native/macos-arm64/libsynaptik_apple_mps.dylib` on the runtime classpath, extracts it
to `~/.synaptik/native/metal-mps/...`, and loads it through Java FFM. Explicit `-Dsynaptik.metal.mps.lib=<path>` or
`SYNAPTIK_METAL_MPS_LIB` still wins when testing a local native build.

The consuming project must run on JDK 25 and enable the incubating Vector API module for compilation and execution:

```groovy
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

tasks.withType(JavaCompile).configureEach {
    options.compilerArgs += ['--add-modules', 'jdk.incubator.vector']
}

tasks.withType(JavaExec).configureEach {
    jvmArgs '--add-modules=jdk.incubator.vector'
    jvmArgs '--enable-native-access=ALL-UNNAMED'
}

tasks.withType(Test).configureEach {
    jvmArgs '--add-modules=jdk.incubator.vector'
    jvmArgs '--enable-native-access=ALL-UNNAMED'
}
```

Build and run the focused public-preview verification set:

```bash
./gradlew classes
./gradlew test --tests 'onnx.*' --tests SourceTreeHygieneTest
```

Run a tiny tensor graph:

```java
import tensor.DataType;
import tensor.Tensor;

Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
Tensor b = new Tensor(new double[]{10.0, 20.0}, new int[]{2}, null, "b", DataType.FLOAT64);

Tensor y = a.add(b).relu().compute();
double[] values = y.toDoubleArrayCopy();
```

`values` is:

```text
[11.0, 22.0, 13.0, 24.0]
```

Run a tiny reverse-mode autodiff graph:

```java
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

Tensor x = new Tensor(new double[]{1.0, -2.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);

Tensor loss = x.mul(x).sum();
loss.compute(CompileMode.TRAINING);

double[] grad = x.getGradient().toDoubleArrayCopy();
```

`grad` is:

```text
[2.0, -4.0, 6.0]
```

## What Works In 0.1

- Dense tensor graph construction through the public `Tensor` API.
- N-D sequence-friendly tensor primitives such as last-dimension `linear`, `stack`/`unstack`, axis `take`, axis slicing, masked reductions, masked cross-entropy, and ergonomic factories.
- Forward compute and reverse-mode autodiff for the supported primitive set.
- CPU execution as the broad correctness backend.
- Compile-time graph optimization, backend planning, region optimization, and memory planning.
- CPU fused execution for selected fused elementwise families.
- Static dense ONNX import/export for the documented subset.
- Runtime `INT32` and `INT64` index tensors on CPU where ONNX/index semantics require them.
- Metal and CUDA capability/coverage truth with explicit unsupported rows.
- Benchmark, graph autotune, platform calibration, and profile persistence.

## Known Limits

- No general runtime dynamic-shape execution yet.
- No general core multi-output runtime operation model yet.
- No ONNX control flow (`If`, `Loop`, `Scan`).
- No sparse tensors, quantized tensors, string tensors, sequence/map/optional values, external ONNX data files, or custom ONNX domains.
- No layer-aware ONNX import/export yet.
- Accelerator coverage is backend-specific; a supported ONNX row does not imply native Metal or CUDA execution.
- Local tuning output under `profiles/platform/...` should not be committed unless intentionally promoted as canonical evidence.

## What This Repository Contains

The repository is organized around five main layers.

| Layer | Main package | Responsibility |
|---|---|---|
| Public modeling surface | `src/main/java/tensor` | Build tensor graphs, expose ergonomic API |
| Primitive descriptors | `src/main/java/operations` | Describe what each graph node means |
| Graph compile/prepare pipeline | `src/main/java/graph` | Canonicalize, optimize graph structure, plan backend ownership, prepare runtime artifacts |
| Runtime/backend execution | `src/main/java/backend` | Resolve kernels and execute prepared steps |
| Benchmark/autotune/calibration | `src/main/java/tuning` | Measure, compare, search, and persist execution profiles |

That split is intentional:

- `tensor` decides what graph to build
- `operations` decides what primitive a node represents
- `graph` decides how that graph can be rewritten, planned into backend regions, optimized inside those regions, and prepared
- `backend` decides how to execute the prepared region or node
- `tuning` decides which executable profile is faster on a real workload

## Reading Guide

Top-level docs:

1. [docs/index.md](docs/index.md) - the main documentation index and recommended reading paths.
2. [docs/quickstart.md](docs/quickstart.md) - detailed first path through build, tensors, broadcasting, autodiff, compile/prepare/execute, publication, profiles, ONNX, accelerators, tuning, and troubleshooting.
3. [docs/tensor-api.md](docs/tensor-api.md) - detailed operation-level Tensor API guide with signatures, edge cases, examples, and concrete calculations.
4. [docs/sequence-tensor-primitives.md](docs/sequence-tensor-primitives.md) - N-D sequence-friendly primitives for consumer frameworks: factories, shape helpers, `linear`, `stack`/`unstack`, axis indexing, masked reductions, and masked cross entropy.
5. [docs/compute-flow.md](docs/compute-flow.md) - deep walkthrough from graph construction through compile, prepare, execution, memory binding, and traces.
6. [docs/graph-optimizer.md](docs/graph-optimizer.md) - backend-neutral graph optimization: `AR`, `CF`, `CSE`, `DCE`, and optional `LOWER`.
7. [docs/backend-planning-and-regions.md](docs/backend-planning-and-regions.md) - backend ownership planning, CPU natural regions, accelerator regions, region optimization, memory planning, and publication policy.
8. [docs/cpu-bf16.md](docs/cpu-bf16.md) - current CPU BF16 storage/compute contract and why BF16 is not automatically faster than F32 on CPU.
9. [docs/calibration-autotune.md](docs/calibration-autotune.md) - calibration families, owned knobs, candidate values, graph autotune parameters, persistence, and progress.
10. [docs/architecture.md](docs/architecture.md) - implementation-grounded lifecycle, backend dispatch, module boundaries, tuning, and diagrams.
11. [docs/modules.md](docs/modules.md) - package-by-package map for tensor, operations, graph, optimizer, backend, CPU kernels, accelerators, config, tuning, CLI, numerics, and utilities.
12. [docs/documentation-audit.md](docs/documentation-audit.md) - documentation inventory, source-of-truth map, terminology baseline, stale-risk areas, and verification procedure.

If you want the shortest reliable path through the codebase:

1. [src/main/java/tensor/README.md](src/main/java/tensor/README.md)
2. [src/main/java/operations/README.md](src/main/java/operations/README.md)
3. [src/main/java/graph/README.md](src/main/java/graph/README.md)
4. [src/main/java/backend/README.md](src/main/java/backend/README.md)
5. [src/main/java/tuning/README.md](src/main/java/tuning/README.md)

If you are solving a specific problem:

- public tensor API: [docs/tensor-api.md](docs/tensor-api.md), [docs/sequence-tensor-primitives.md](docs/sequence-tensor-primitives.md), then [src/main/java/tensor/API.md](src/main/java/tensor/API.md)
- compile/prepare/execute behavior: [docs/compute-flow.md](docs/compute-flow.md)
- graph optimizer internals: [docs/graph-optimizer.md](docs/graph-optimizer.md)
- backend planning, regions, memory planning, and publication: [docs/backend-planning-and-regions.md](docs/backend-planning-and-regions.md)
- CPU BF16 behavior and performance interpretation: [docs/cpu-bf16.md](docs/cpu-bf16.md)
- calibration and graph autotune: [docs/calibration-autotune.md](docs/calibration-autotune.md)
- graph optimization and concrete rewrite behavior:
  - [src/main/java/graph/optimizer/README.md](src/main/java/graph/optimizer/README.md)
  - [src/main/java/graph/optimizer/AR.md](src/main/java/graph/optimizer/AR.md)
  - [src/main/java/graph/optimizer/CSE.md](src/main/java/graph/optimizer/CSE.md)
  - [src/main/java/graph/optimizer/FUSE.md](src/main/java/graph/optimizer/FUSE.md)
  - [src/main/java/graph/optimizer/MEM.md](src/main/java/graph/optimizer/MEM.md)
- runtime/tuning/persistence:
  - [src/main/java/tuning/ARCHITECTURE.md](src/main/java/tuning/ARCHITECTURE.md)
  - [src/main/java/tuning/KNOBS.md](src/main/java/tuning/KNOBS.md)
  - [src/main/java/tuning/PERSISTENCE.md](src/main/java/tuning/PERSISTENCE.md)
  - [src/main/java/tuning/WORKLOADS.md](src/main/java/tuning/WORKLOADS.md)
- numerics drift harness: [src/main/java/numerics/README.md](src/main/java/numerics/README.md)

## Current Capabilities

The current codebase includes:

- dense tensors with explicit shape, strides, dtype, and storage offset
- public tensor graph construction with autodiff support
- compile-time semantic canonicalization
- graph-level optimization:
  - `AR` algebraic rewrite
  - `CF` constant folding
  - `CSE` common subexpression elimination
  - `DCE` dead-code elimination
  - optional `LOWER` backend-neutral graph lowering
- compile-time backend planning:
  - CPU-only planning
  - explicit accelerator intent planning
  - automatic accelerator region discovery
  - CPU natural regions
- region optimization and memory planning as separate compile phases
- CPU kernel families for:
  - elementwise
  - broadcast/where
  - reductions
  - layout/view-like remaps
  - matmul / linear
  - conv2d / pool2d
  - softmax / log-softmax
  - cross-entropy from indices
  - scaled dot-product attention
- fused CPU execution with ASM-specialized backends for selected fused families
- benchmark, autotune, and platform calibration
- versionable platform/runtime profiles under `profiles/platform/...`

## Requirements

- JDK 25
- Gradle 9.4.1 compatible environment, or the bundled Gradle wrapper
- `jdk.incubator.vector` available at compile and runtime

The Gradle build already adds the Vector API module for compile, test, and run tasks.

## Build And Test

Typical local commands:

```bash
./gradlew classes
./gradlew test
./gradlew check
./gradlew run
```

On Windows use `gradlew.bat`.

Focused public-preview gates:

```bash
./gradlew test --tests 'onnx.*' --tests SourceTreeHygieneTest
```

Optional hardware-specific gates:

```bash
./gradlew metalTest
./gradlew cudaTest
```

## The Main Lifecycle

The most important concept in Synaptik is that the same root tensor can be observed at multiple lifecycle stages.

### Stage 1: build a semantic graph

```java
Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
Tensor b = new Tensor(new double[]{10.0, 20.0}, new int[]{2}, null, "b", DataType.FLOAT64);
Tensor y = a.add(b).relu();
```

Shapes:

- `a` = `[2, 2]`
- `b` = `[2]`
- `a.add(b)` broadcasts `b` across rows and produces `[2, 2]`
- `relu()` keeps `[2, 2]`

Value example:

- row 0: `[1.0, 2.0] + [10.0, 20.0] = [11.0, 22.0]`
- row 1: `[3.0, 4.0] + [10.0, 20.0] = [13.0, 24.0]`
- `relu()` leaves the same values unchanged because they are already non-negative

### Stage 2: compile the graph

```java
CompiledGraph compiled = y.compile(CompileMode.INFERENCE_ONLY);
```

Compile does not execute kernels.
It:

- snapshots the graph
- canonicalizes forward structure
- optionally builds backward structure
- runs graph optimization
- plans backend ownership and regions
- plans region optimization and memory reuse
- produces a stable compile artifact

### Stage 3: prepare runtime execution

```java
PreparedExecution prepared = y.prepare(
        new ExecutionProfile(
                "demo",
                "demo",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults()
        )
);
```

Prepare does not rebuild graph semantics.
It resolves backend/runtime policy into concrete prepared metadata:

- dispatch hints
- reduction hints
- matmul hints
- conv2d hints
- fused executable preparation
- per-node backend plans

### Stage 4: execute

```java
prepared.execute(ExecutionMode.FORWARD);
double[] out = y.toDoubleArrayCopy();
```

For the example above, `out` will contain:

```text
[11.0, 22.0, 13.0, 24.0]
```

## One-Shot Convenience API

For small scripts and tests you do not need to manually spell out compile and prepare every time.

### Forward-only convenience path

```java
Tensor probs = logits.softmax(-1).compute();
```

`compute()` means:

- compile with `CompileMode.INFERENCE_ONLY`
- use inference compile defaults
- use inference runtime defaults
- run `FORWARD`

### Explicit training intent

```java
loss.compute(CompileMode.TRAINING);
```

If the graph contains trainable leaf tensors, Synaptik will:

- compile a joint forward/backward artifact
- use training compile/runtime defaults
- run `FORWARD_BACKWARD`

If the graph has no trainable leaves, the runtime still falls back to a forward-only execution path.

### Configurable convenience path

```java
loss.compute(
        new ComputeOptions()
                .compileMode(CompileMode.TRAINING)
                .autotune(AutotunePolicy.IF_MISSING)
);
```

This convenience flow can reuse or create a generic best profile for the current graph signature.

Current generic tensor autotune behavior:

- it builds a generic tensor workload rooted at the current tensor
- it evaluates the standard `graphPolicy=current` candidate
- it persists winners under `build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/...`

That generic path is different from the main CLI autotune flow described below, which is workload-specific and versioned under `profiles/platform/...`.

## Reverse-Mode Autodiff Example

```java
Tensor x = new Tensor(new double[]{1.0, -2.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);

Tensor y = x.mul(x).sum();
y.compute(CompileMode.TRAINING);

double loss = y.toDoubleArrayCopy()[0];
double[] grad = x.getGradient().toDoubleArrayCopy();
```

Value intuition:

- `x * x` = `[1.0, 4.0, 9.0]`
- `sum` = `14.0`
- gradient of `sum(x^2)` with respect to `x` = `2x`
- so `grad` = `[2.0, -4.0, 6.0]`

## Compile Modes

The public convenience API currently uses three compile intents:

- `CompileMode.INFERENCE_ONLY`
  - compile forward only
  - use inference defaults
- `CompileMode.TRAINING`
  - compile backward when the graph actually has trainable leaves
  - use training defaults
- `CompileMode.AUTO`
  - choose between the above based on graph structure

The important subtlety is that `TRAINING` does not force fake gradients for graphs that do not need them.
If there are no trainable leaf tensors, execution still remains forward-only.

## Main CLI

The main CLI entry point is [src/main/java/synaptik/app/TuningCli.java](src/main/java/synaptik/app/TuningCli.java).

Supported commands:

```bash
./gradlew run --args="full f64"
./gradlew run --args="calibrate --dtype f64 --families all"
./gradlew run --args="calibrate --dtype f64 --family conv2d-gemm-dispatch --measurement 30:100:2"
./gradlew run --args="autotune f64"
./gradlew run --args="benchmark-winner f64"
./gradlew run --args="benchmark-graph-space f64"
```

Meaning:

- `full`
  - convenience flow `calibrate -> autotune -> benchmark-winner`
  - good for local iteration
  - not ideal for the cleanest measurements because JVM warmup carries across phases
- `calibrate`
  - calibrates platform runtime defaults
  - optionally for a single family
  - optionally with explicit measurement override `warmup measure repeats`
- `autotune`
  - runs standard graph autotune for the `abc_sequence_matmul_<dtype>` workload
  - standard graph autotune currently evaluates `graphPolicy=current`
- `benchmark-winner`
  - compares the stored winner against the no-opt baseline
- `benchmark-graph-space`
  - compares the standard graph candidate against the baseline

Supported CLI dtypes:

- `f64`
- `f32`
- `bf16`

Supported calibration family tokens are resolved from `CalibrationFamilyRegistry`.
Current public family names include:

- `matmul`
- `attention-matmul`
- `conv2d-gemm-dispatch`
- `fused-dispatch`
- `fused-cheap-contiguous-width`
- `fused-cheap-strided-width`
- `fused-noncheap-contiguous-width`
- `fused-noncheap-strided-width`
- `elementwise-dispatch`
- `reduction`
- `attention-thresholds`
- `scheduler`
- `materialization`
- `metal-selection` only when explicitly requested with `--include-accelerators`

## Persistence Layout

The preferred persisted layout for versioned tuning artifacts is:

```text
profiles/
  platform/
    <platform-id>/
      calibration/
        schema-v2/
          latest/
          history/
          runs/
      tuning/
        abc/
```

Typical files:

- calibration profile:
  - `profiles/platform/<platform-id>/calibration/schema-v2/latest/<dtype>/<mode>/profile.json`
- calibration history:
  - `profiles/platform/<platform-id>/calibration/schema-v2/history/<dtype>/<mode>/<family-id>.jsonl`
- calibration run artifacts:
  - `profiles/platform/<platform-id>/calibration/schema-v2/runs/<run-id>/...`
- workload-specific autotune winner:
  - `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json`
- workload-specific autotune history:
  - `profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl`

New platform ids use a short OS/architecture form such as `macos-arm64`; JVM vendor and CPU count remain in metadata
instead of the directory name. `Tensor.compute(...)` and `CompiledGraph.prepare()` try to load a compatible calibrated
runtime profile from `-Dsynaptik.profiles.root`, `SYNAPTIK_PROFILES_ROOT`, `./profiles`, `~/.synaptik/profiles`, or
bundled resources before falling back to hardcoded runtime defaults. Legacy `build/...` calibration locations are no
longer used as runtime fallbacks.

## Numerics Harness

For numerical drift checks use the standalone harness in [src/main/java/numerics/README.md](src/main/java/numerics/README.md).

Example:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR,CF,CSE,DCE,LOWER \
  -Dnumerics.size=200000 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

That harness is for numerical comparison, not for performance measurement.

## Where To Start If You Want To Change Performance

Use this rule of thumb:

- graph optimizer shape or backend-neutral pattern lowering:
  - start in `graph/optimizer`
- backend ownership, CPU natural regions, accelerator regions, or region optimization:
  - start in `graph/compile`, `graph/optimizer/partition`, `graph/optimizer/region`, and [docs/backend-planning-and-regions.md](docs/backend-planning-and-regions.md)
- CPU dispatch thresholds, tiles, microkernels, fused widths:
  - start in `config`, `backend/kernels/cpu`, and `tuning`
- new public API surface:
  - start in `tensor/ops/*`, then add/update `operations/*`
- runtime execution traces and benchmark reports:
  - start in `graph/execution/trace`, `tuning/benchmark/report`, `tuning/autotune/report`, `tuning/calibration/report`, and `tuning/reporting`

## Current Design Boundaries

These constraints are deliberate and repeatedly enforced in the current architecture:

- executors do not re-run compile-time optimizer logic
- runtime auxiliary caches do not belong on semantic `Tensor` nodes
- tuning does not invent a second execution model outside `ExecutionProfile`
- graph optimization transforms graph structure, not runtime dispatch knobs
- backend planning is compile-time execution planning, not execute-time offload
- backend `prepare(...)` resolves runtime policy; execution consumes the prepared recipe

If a proposed change violates one of those boundaries, it is probably pushing logic into the wrong layer.
