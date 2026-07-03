# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs across multiple backends. The project is under active development and currently provides its initial model foundations and repository structure.

The authoritative architecture contract is defined in [`ARCHITECTURE.md`](ARCHITECTURE.md). The contributor and agent workflow is defined in [`AGENTS.md`](AGENTS.md).

Start with the [documentation index](docs/index.md). New contributors can follow [Getting started](docs/getting-started.md); the [implementation roadmap](docs/planning/roadmap.md) distinguishes implemented model contracts from planned lifecycle and backend work.

## Current implementation status

The implemented public surface is currently limited to backend-independent model value types: data types, static and symbolic shapes, broadcasting, resolved layouts, and typed tensor/graph identifiers. Public tensors, graph compilation, preparation, execution, backends, tracing, and training remain planned. Documentation for those areas describes the architecture contract and intended workflow and is labeled accordingly; it is not a runnable API promise.

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
