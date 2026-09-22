# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs
across multiple backends. The project is under active development and currently provides a
runnable CPU-only public lifecycle, backend-independent Tensor and compiler capabilities, and
bounded Metal execution.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative architecture root and sole authority
index; it links the six incorporated scoped contracts. The contributor and agent workflow is
defined in [`AGENTS.md`](AGENTS.md).

Start with the [documentation index](docs/index.md). New contributors can follow
[Getting started](docs/getting-started.md); the
[implementation roadmap](docs/planning/roadmap.md) distinguishes current capabilities from
planned work.

## Current implementation status

The current public surface includes Tensor expressions, graph compilation, preparation, repeated
execution with isolated invocation state, detached host materialization, one-shot forward
computation, and a bounded scalar-objective backward convenience through one explicitly owned
CPU composition. A bounded Metal backend can execute supported static `FLOAT32` negation
partitions through MPSGraph or a single-operation custom route, but broader Metal coverage and
standard or mixed-owner Metal composition remain planned. CUDA, training orchestration,
persistence, and generic graph/plan tuning also remain planned. Focused documentation identifies
the exact current boundary for each area.

## Prerequisites

- JDK 26
- an IntelliJ IDEA version with Java 26 support, when using the IDE

The Gradle wrapper downloads the supported Gradle distribution. A separate Gradle installation is not required.

## Gradle commands

Use the repository wrapper:

```shell
./gradlew projects
./gradlew test
./gradlew build
./gradlew :modules:model:javadoc
```

Synaptik does not enable Java preview features globally. A focused implementation task may enable a required preview or incubator feature only for the owning module, with explicit build configuration, documentation, and validation.
