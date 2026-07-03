# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs across multiple backends. The project is under active development and currently provides its initial model foundations and repository structure.

The authoritative architecture contract is defined in [`ARCHITECTURE.md`](ARCHITECTURE.md). The contributor and agent workflow is defined in [`AGENTS.md`](AGENTS.md).

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
```

Synaptik does not enable Java preview features globally. A focused implementation task may enable a required preview or incubator feature only for the owning module, with explicit build configuration, documentation, and validation.
