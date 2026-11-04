# Synaptik

Build tensor computations and neural-network models in Java, then run them through an explicitly
owned execution backend. Start with CPU; choose Metal or mixed execution only after checking the
actual program's supported operations, types and layouts.

## Start with your goal

| You want to… | Start here |
|---|---|
| Run your first model and learn the required concepts | [The complete learning journey](docs/tutorials/index.md) |
| Use existing ML knowledge with Java tensor/model APIs | [Task guides](docs/how-to/index.md) and [topic-sized reference](docs/reference/index.md) |
| Understand or contribute to the implementation | [One program through the framework](docs/explanation/framework.md), then the [package/native atlas](docs/explanation/packages/index.md) |

The learning project is one commented, runnable application: construct a model, understand shapes,
compute gradients, train, evaluate, save/close/load, and reuse execution. Its
[setup and canonical source](docs/examples/README.md) use supported public APIs and a separate
NN-only Maven consumer, not checkout classes or a hidden manual dependency list.

## Current implementation status

Tensor expressions describe meaning; compilation chooses supported ownership; preparation builds
reusable execution; each run uses current inputs and publishes results. The
[framework explanation](docs/explanation/framework.md) connects these stages without requiring you
to learn repository modules first.

The [production support matrix](docs/backend-guide/production-support.md) is the single entry point
for current operation/dtype/layout/derivative qualification. CPU and Metal have independent domains;
CPU support is not a Metal capability promise. CUDA, ONNX and CLI markers are not usable products.
Model persistence stores admitted source and current values, not prepared native executables or
optimizer/application continuation.

Distribution remains **private/internal**: no public project licence or redistribution grant has
been selected, and no public upload is authorized. The sealed
`0.1.0-public-audit.20261104` candidate has [source-matched audit evidence](docs/planning/public-release-audit-2026-11-04.md#result).
This documentation uses those artifacts; it does not silently replace or requalify them.

## Prerequisites

- JDK 26.
- Maven 3.9+ for the standalone learning project, plus the supplied private file Maven repository.
- The checked-in Gradle wrapper for framework development; no separate Gradle installation.

Follow [setup](docs/getting-started.md), then [first model](docs/tutorials/first-model.md).
The default CPU route needs no GPU or third-party native library. Engine consumers enable FFM
native access as shown in the runnable setup; explicit Metal requires its separately verified package.

## Gradle commands

From the framework source root:

```sh
./gradlew test
./gradlew build
./gradlew :modules:model:javadoc
```

These are development commands, not a substitute for the tutorial's external-consumer setup or
real-device qualification. Java preview features are not enabled globally; the learning project
needs no preview or incubating Vector API flag.

## Contributor and maintenance navigation

[Documentation home](docs/index.md) separates learning, tasks, explanation and reference.
[Contributor guides](docs/developer-guide/index.md) lead to source navigation, debugging and release
mechanics. [ARCHITECTURE.md](ARCHITECTURE.md) is the sole architecture authority index;
[AGENTS.md](AGENTS.md) defines automated-contributor workflow.
[Planning](docs/planning/README.md), [design history](docs/design/README.md) and audit receipts are
maintenance evidence, not prerequisites for a first prediction.
