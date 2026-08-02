# Architecture tests

## What you will learn

This guide explains which boundaries architecture tests must protect and how to run their Gradle module. The `testing/architecture-tests` project contains focused dependency checks for Config and Planning plus a conditional NN-to-Training direction check. Coverage remains incomplete: Runtime dependency and hot-path enforcement named by `ARCHITECTURE.md` is not yet present and remains owned by Draft Runtime 0014.

## Prerequisites and authority

Read [`ARCHITECTURE.md`](../../ARCHITECTURE.md) and [Dependency rules](../architecture/dependency-rules.md). Tests enforce those rules; they do not create new module ownership.

## Mental model

Architecture tests check relationships that ordinary unit tests may miss:

```text
module/package edges + forbidden API references + hot-path type usage
  -> compare with architecture contract
  -> fail on a prohibited dependency or semantic leak
```

Examples include keeping trace as a dependency leaf, preventing runtime from depending on concrete backends, ensuring `Operation` has no `supportedBackends()`, and keeping `Operation` and `CompiledNode` out of the runtime hot path.

## Run the current module

```bash
./gradlew :testing:architecture-tests:test
```

A successful run proves only the current focused Config, Planning, and conditional NN/Training assertions; it is not evidence that every contract rule is enforced. Runtime dependency and hot-path coverage remains absent until Runtime 0014. When a dependency or hot-path rule is implemented or changed, add a falsifiable test in this module and record what it protects.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| A test encodes a rule found only in a guide | Explanatory text was treated as authority. | Trace the assertion to `ARCHITECTURE.md` first. |
| A module dependency passes but a forbidden method appears | Only Gradle edges were checked. | Add source/bytecode checks for semantic API restrictions. |
| A new rule is added only to a test | The contract and enforcement drifted. | Update the coordinated architecture artifacts required by `AGENTS.md`. |

## Related documentation

- [Module boundaries](../architecture/module-boundaries.md)
- [Dependency rules](../architecture/dependency-rules.md)
- [Planning guide](../planning/planning-guide.md)
