# Synaptik documentation

[Run](glossary.md#term-run) one Java [model](glossary.md#term-model) first; open internals when you need to understand or change them. The
[learning project](examples/README.md) uses the audited private [artifacts](glossary.md#term-artifact) and a fixed [CPU](glossary.md#term-cpu)/[FLOAT32](glossary.md#term-float32)
example throughout. It does not require a [GPU](glossary.md#term-gpu), a [dataset](glossary.md#term-dataset) download or knowledge of repository [modules](glossary.md#term-module).

Unfamiliar terms link to the [central dictionary](glossary.md). You do not need prior machine-learning,
[compiler](glossary.md#term-compiler) or GPU knowledge to begin: [First terms](terms.md) gives a short reading order, and each
definition introduces its prerequisites before the implementation detail.

## Start here

| Reader | Recommended path |
|---|---|
| Java developer new to [tensors](glossary.md#tensor)/[ML](glossary.md#term-machine-learning-ml) | [Setup](getting-started.md) → [learning journey](tutorials/index.md); use [First terms](terms.md) as needed |
| ML/tensor user new to Synaptik | [First model](tutorials/first-model.md)<span data-glossary-navigation="true"> ([model glossary](glossary.md#term-model))</span> → [task guides](how-to/index.md) → [reference](reference/index.md) |
| Framework contributor | [Whole-framework explanation](explanation/framework.md) → [package/native atlas](explanation/packages/index.md) → [contributor area](developer-guide/index.md) |

## Four kinds of documentation

| Kind | Use it for | Entry |
|---|---|---|
| Tutorial | Learn by running and modifying one connected project | [First model through train/save/load/reuse](tutorials/index.md) |
| How-to | Complete a particular task with known prerequisites | [Task index](how-to/index.md) |
| Explanation | Understand why the stages, geometry and [ownership](glossary.md#term-ownership-meanings) fit together | [Concepts and framework](explanation/index.md) |
| Reference | Find exact [API](glossary.md#term-application-programming-interface-api) details, constraints and generated [Javadoc](glossary.md#term-javadoc) | [Topic-sized API reference](reference/index.md) |

The [complete glossary](glossary.md) is the terminology dictionary. The
[support matrix](backend-guide/production-support.md) qualifies actual [operation](glossary.md#operation), [dtype](glossary.md#data-type--datatype), [layout](glossary.md#layout),
[derivative](glossary.md#term-derivative) and [backend](glossary.md#backend) domains. A [Tensor](glossary.md#tensor) operation name, Java-public class or saved [model](glossary.md#term-model) alone
does not promise execution on every backend.

## Contributor guides

The [contributor index](developer-guide/index.md) connects repository layout, documentation/coding
rules, testing, diagnostics, [benchmarking](glossary.md#term-benchmarking) and [release](glossary.md#term-release) mechanics. For internals, the atlas covers
every production Java package, including internal packages and native components; test/product
markers are explicitly distinguished from usable execution.

[ARCHITECTURE.md](../ARCHITECTURE.md) is the sole authority index and incorporates six scoped
contracts. All tutorials, explanations, references, ADRs and plans remain subordinate. They do not
create a second numerical or [ownership](glossary.md#term-ownership-meanings) contract.

## Documentation areas

- **Learning and tasks:** [tutorials](tutorials/index.md), [canonical executable source](examples/README.md),
  [how-to](how-to/index.md), [concepts](explanation/index.md), [API/Javadoc](reference/index.md)<span data-glossary-navigation="true"> ([API glossary](glossary.md#term-application-programming-interface-api); [Javadoc glossary](glossary.md#term-javadoc))</span>.
- **Implementation:** [package/native atlas](explanation/packages/index.md),
  [architecture navigation](architecture/current-architecture-plan.md),
  [backend authoring](backend-guide/writing-a-backend.md)<span data-glossary-navigation="true"> ([backend glossary](glossary.md#backend))</span>.
- **Maintenance and history:** [planning](planning/README.md)<span data-glossary-navigation="true"> ([planning glossary](glossary.md#term-planning))</span>, [design records](design/README.md),
  [release process](developer-guide/release-process.md)<span data-glossary-navigation="true"> ([release glossary](glossary.md#term-release))</span>,
  [completed public-readiness audit](planning/public-release-audit-2026-11-04.md).

Older `api/*.md` URLs retain incoming anchors and lead to their [canonical](glossary.md#term-canonical-form) reference topics.
Historical candidate receipts describe their original source and [artifacts](glossary.md#term-artifact). Current [distributions](glossary.md#commit-release-distribution-log-and-seed-meanings)
remain private/internal; the [owner](glossary.md#term-owner) has not selected a public licence or redistribution grant.
