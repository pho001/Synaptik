# Synaptik documentation

Run one Java model first; open internals when you need to understand or change them. The
[learning project](examples/README.md) uses the audited private artifacts and a fixed CPU/FLOAT32
example throughout. It does not require a GPU, a dataset download or knowledge of repository modules.

## Start here

| Reader | Recommended path |
|---|---|
| Java developer new to tensors/ML | [Setup](getting-started.md) → [learning journey](tutorials/index.md); use [First terms](terms.md) as needed |
| ML/tensor user new to Synaptik | [First model](tutorials/first-model.md) → [task guides](how-to/index.md) → [reference](reference/index.md) |
| Framework contributor | [Whole-framework explanation](explanation/framework.md) → [package/native atlas](explanation/packages/index.md) → [contributor area](developer-guide/index.md) |

## Four kinds of documentation

| Kind | Use it for | Entry |
|---|---|---|
| Tutorial | Learn by running and modifying one connected project | [First model through train/save/load/reuse](tutorials/index.md) |
| How-to | Complete a particular task with known prerequisites | [Task index](how-to/index.md) |
| Explanation | Understand why the stages, geometry and ownership fit together | [Concepts and framework](explanation/index.md) |
| Reference | Find exact API details, constraints and generated Javadoc | [Topic-sized API reference](reference/index.md) |

The [complete glossary](glossary.md) is the terminology dictionary. The
[support matrix](backend-guide/production-support.md) qualifies actual operation, dtype, layout,
derivative and backend domains. A Tensor operation name, Java-public class or saved model alone
does not promise execution on every backend.

## Contributor guides

The [contributor index](developer-guide/index.md) connects repository layout, documentation/coding
rules, testing, diagnostics, benchmarking and release mechanics. For internals, the atlas covers
every production Java package, including internal packages and native components; test/product
markers are explicitly distinguished from usable execution.

[ARCHITECTURE.md](../ARCHITECTURE.md) is the sole authority index and incorporates six scoped
contracts. All tutorials, explanations, references, ADRs and plans remain subordinate. They do not
create a second numerical or ownership contract.

## Documentation areas

- **Learning and tasks:** [tutorials](tutorials/index.md), [canonical executable source](examples/README.md),
  [how-to](how-to/index.md), [concepts](explanation/index.md), [API/Javadoc](reference/index.md).
- **Implementation:** [package/native atlas](explanation/packages/index.md),
  [architecture navigation](architecture/current-architecture-plan.md),
  [backend authoring](backend-guide/writing-a-backend.md).
- **Maintenance and history:** [planning](planning/README.md), [design records](design/README.md),
  [release process](developer-guide/release-process.md),
  [completed public-readiness audit](planning/public-release-audit-2026-11-04.md).

Older `api/*.md` URLs retain incoming anchors and lead to their canonical reference topics.
Historical candidate receipts describe their original source and artifacts. Current distributions
remain private/internal; the owner has not selected a public licence or redistribution grant.
