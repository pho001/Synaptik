# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs
across multiple backends. The project is under active development and currently provides a
runnable public lifecycle with fixed CPU convenience and explicit single- or bounded mixed-owner
CPU/Metal composition, plus backend-independent Tensor and compiler capabilities.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative architecture root and sole authority
index; it links the six incorporated scoped contracts. The contributor and agent workflow is
defined in [`AGENTS.md`](AGENTS.md).

Start with the [documentation index](docs/index.md). New contributors can follow
[Getting started](docs/getting-started.md); the
[implementation roadmap](docs/planning/roadmap.md) distinguishes current capabilities from
planned work.

## Current implementation status

The current public surface includes Tensor expressions, graph compilation, reusable inference
sessions that prepare once and run with isolated invocation state, detached host materialization,
one-shot forward computation, and a bounded scalar-objective backward convenience.
`Engine.standard()` owns a fixed
CPU composition; `Engine.builder()` explicitly owns opened CPU and/or Metal integrations and
executes non-empty single-owner plans or bounded mixed CPU/Metal plans through deterministic
owner-indexed representations and ordered transfer steps. Current cross-owner transfer supports
positive rank-1..16 fully static canonical contiguous `FLOAT32` values in both directions. Metal
executes supported static `FLOAT32` negation and terminal affine partitions through MPSGraph, with
a custom route for eligible singleton negation. Metal does not advertise binary arithmetic;
explicitly registered CPU may own it, while Metal-only binary graphs fail during ownership
selection. A standard-Metal convenience, generic plugin registration/discovery, broader Metal
coverage, CUDA, training orchestration, persistence, and generic graph/plan tuning remain planned.
Focused documentation identifies the exact current boundary for each area.

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
