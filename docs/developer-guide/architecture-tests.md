# Architecture tests

## What you will learn

This guide explains which boundaries architecture tests protect and how to run their Gradle module. The `testing/architecture-tests` project contains focused dependency checks for Config and Planning, a conditional NN-to-Training direction check, and Runtime dependency and direct-hot-path coverage from [Runtime 0014](../planning/modules/runtime/tasks/0014-runtime-architecture-enforcement.md). Coverage remains incomplete: these are focused assertions, not enforcement of every architecture rule.

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

A successful run proves only the implemented focused assertions; it is not evidence that every contract rule is enforced. The Runtime suite requires exactly the current Config, Backend Contract, and Trace project edges; rejects Engine, concrete-backend, and other project-edge drift; inventories every Runtime production Java source; and requires an explicit hot/non-hot classification. It scans the five-file direct execution/state subset for the exact `Operation` and `CompiledNode` source or binary identities forbidden by the Runtime hot-path contract. It does not prove Runtime behavior, bytecode mechanics, or every dependency and hot-path rule. When a dependency or hot-path rule is implemented or changed, add a falsifiable test in this module and record what it protects.

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
