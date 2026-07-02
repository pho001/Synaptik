# Synaptik

Synaptik is a modular Java foundation for compiling, preparing, and executing computational graphs across multiple backends. The repository currently contains only the initial project skeleton; framework behavior has not been implemented yet.

The authoritative architecture contract is defined in [`ARCHITECTURE.md`](ARCHITECTURE.md). The contributor and agent workflow is defined in [`AGENTS.md`](AGENTS.md).

## Gradle commands

When a Gradle installation is available, use:

```shell
gradle projects
gradle test
gradle build
```

If a Gradle wrapper is added later, use the equivalent `./gradlew` commands.
