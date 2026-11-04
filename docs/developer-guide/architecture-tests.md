# Architecture tests

## What you will learn

This guide explains which boundaries architecture tests protect and how to [run](../glossary.md#term-run) their [Gradle](../glossary.md#term-gradle) [module](../glossary.md#term-module).
The `testing/architecture-tests` project contains focused Config/[Planning](../glossary.md#term-planning) checks, exact generic
Training and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../glossary.md#term-dependency)/[admission](../glossary.md#term-admission) checks, and [Runtime](../glossary.md#term-runtime) dependency and direct-hot-path coverage
from [Runtime 0014](../planning/modules/runtime/tasks/0014-runtime-architecture-enforcement.md).
Coverage remains incomplete: these are focused assertions, not enforcement of every architecture rule.

An **[architecture test](../glossary.md#term-architecture-test)** checks how parts of the program may depend on one another, rather than
only checking a calculated answer. A dependency **edge** points from the part that uses something
to the part it uses. A **[dependency leaf](../glossary.md#term-dependency-leaf)** adds no further project dependencies; a **[hot path](../glossary.md#term-hot-path)** is
code executed repeatedly during actual computation. Here, NN means the neural-network extension.
Gradle is the build tool that selects and runs the test project.

## Prerequisites and authority

Read [`ARCHITECTURE.md`](../../ARCHITECTURE.md) and [Dependency rules](../architecture/dependency-rules.md). Tests enforce those rules; they do not create new [module](../glossary.md#term-module) [ownership](../glossary.md#term-ownership-meanings).

## Who runs the checks, and what comes back?

From the repository root, the contributor invokes the Gradle wrapper (`./gradlew`).
Gradle selects the architecture-test project and runs its assertions against project
relationships, source files and compiled identities. The output is a pass/fail result
for those assertions, not a new architecture rule or an execution benchmark.

When a check fails, identify the forbidden edge or reference, then compare it with
the owning [dependency rule](../architecture/dependency-rules.md). Correct the code
or coordinate an explicit contract change; weakening a test merely to make it pass
would erase the evidence of a boundary violation.

## [Mental model](../glossary.md#term-mental-model)

Architecture tests check relationships that ordinary unit tests may miss:

```text
module/package edges + forbidden API references + hot-path type usage
  -> compare with architecture contract
  -> fail on a prohibited dependency or semantic leak
```

Examples include keeping [trace](../glossary.md#trace) as a [dependency leaf](../glossary.md#term-dependency-leaf), preventing [runtime](../glossary.md#term-runtime) from depending on concrete [backends](../glossary.md#backend), ensuring [`Operation`](../glossary.md#operation) has no `supportedBackends()`, and keeping `Operation` and `CompiledNode` out of the runtime [hot path](../glossary.md#term-hot-path).

Read the flow as “collect relationships, compare them with the contract, then report a violation.”
For example, a [unit test](../glossary.md#term-unit-test) could produce the right answer while Runtime incorrectly imports a
[concrete backend](../glossary.md#concrete-backend). An [architecture test](../glossary.md#term-architecture-test) [targets](../glossary.md#term-target) that forbidden relationship. A backend-[conformance](../glossary.md#term-conformance)
test instead checks whether a backend follows the shared behavior required of it; neither kind
of test proves all possible inputs or all architecture rules.

## [Run](../glossary.md#term-run) the current [module](../glossary.md#term-module)

```bash
# Select the boundary-check project, rather than treating numerical tests as architecture evidence.
./gradlew :testing:architecture-tests:test
```

A successful run proves only the implemented focused assertions; it is not evidence that every contract rule is enforced. The [Runtime](../glossary.md#term-runtime) suite requires exactly the current Config, [Backend](../glossary.md#backend) Contract, and [Trace](../glossary.md#trace) project edges; rejects [Engine](../glossary.md#term-engine), concrete-backend, and other project-edge drift; inventories every Runtime production Java source; and requires an explicit hot/non-hot classification. It scans the five-file direct execution/state subset for the exact [`Operation`](../glossary.md#operation) and `CompiledNode` source or [binary](../glossary.md#term-binary-descriptor-and-metadata) [identities](../glossary.md#term-identity-meanings) forbidden by the Runtime hot-path contract. It does not prove Runtime behavior, [bytecode](../glossary.md#term-bytecode) mechanics, or every [dependency](../glossary.md#term-dependency) and hot-path rule. When a dependency or hot-path rule is implemented or changed, add a falsifiable test in this module and record what it protects.

## Typical mistakes

Training0007's checks require physical `modules/training` to retain `:extensions:training`, exact
public Training→[Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine) and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)→Model/Training edges, NN-free Training production/test sources,
generic open/copy [signatures](../glossary.md#term-signature), and the private live NN adapter. The full test task compiles isolated
Training-only and NN-only consumer source sets, so accidental transitive NN visibility cannot hide
a Training coupling. Source-reading tests declare Training and NN source trees as [Gradle](../glossary.md#term-gradle) inputs;
changing a scanned source must invalidate the test task. Engine's inward-edge test allows only
Training's explicit public Engine [lifecycle](../glossary.md#lifecycle) role, not all `modules/` projects.

| Symptom | Cause | Correction |
|---|---|---|
| A test encodes a rule found only in a guide | Explanatory text was treated as authority. | [Trace](../glossary.md#trace) the assertion to `ARCHITECTURE.md` first. |
| A [module](../glossary.md#term-module) [dependency](../glossary.md#term-dependency) passes but a forbidden method appears | Only Gradle edges were checked. | Add source/[bytecode](../glossary.md#term-bytecode) checks for semantic [API](../glossary.md#term-application-programming-interface-api) restrictions. |
| A new rule is added only to a test | The contract and enforcement drifted. | Update the coordinated architecture [artifacts](../glossary.md#term-artifact) required by `AGENTS.md`. |

## Related documentation

- [Module boundaries](../architecture/module-boundaries.md)
- [Dependency rules](../architecture/dependency-rules.md)
- [Planning guide](../planning/planning-guide.md)
