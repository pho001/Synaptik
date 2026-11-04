# Contribute to Synaptik

Application users should start with the [learning journey](../tutorials/index.md). This area is
for changing the framework, diagnosing its internals or maintaining its private [distribution](../glossary.md#term-distribution).
It is deliberately separate from the beginner path and historical [qualification](../glossary.md#term-qualification) records.

You do not need to know the contributor vocabulary before starting. A **[module](../glossary.md#term-module)** owns a part of
the framework; a **[dependency](../glossary.md#term-dependency)** means one part uses another; a **contract** states the behavior or
boundary that changes must preserve. The [glossary](../glossary.md) supplies the shared definitions.
The guides below explain the local meaning before asking you to use a tool or inspect evidence.

Keep two questions separate: “which component owns this change?” and “what evidence shows that
it works?” A passing test answers only the cases it checked. **Qualification** records acceptance
for a stated [source revision](../glossary.md#term-source-revision), environment and supported domain, not permission to publish.

## Find the responsibility before changing code

1. Follow [one program through the framework](../explanation/framework.md).
2. Find its [owner](../glossary.md#term-owner) in the [complete package/native atlas](../explanation/packages/index.md) and
   [repository layout](repository-layout.md). Java packages and [Gradle](../glossary.md#term-gradle) paths are not identical.
3. Read [ARCHITECTURE.md](../../ARCHITECTURE.md), the sole authority index, then the exact owning
   scoped headings. Explanatory pages and old task narratives cannot override them.
4. Apply [AGENTS.md](../../AGENTS.md) and the focused contribution workflow for the actual change.

## Working guides

- [Coding rules](coding-rules.md), [architecture tests](architecture-tests.md) and
  [documentation rules](documentation-rules.md).
- [Documentation profiles](documentation/README.md): learner tasks, explanations and reference
  have different purposes; do not turn the tutorial into an implementation log.<span data-glossary-navigation="true"> ([log glossary](../glossary.md#term-log))</span>
- [Diagnose preparation/execution](debugging-trace.md) and [bounded benchmarking](benchmarking.md)<span data-glossary-navigation="true"> ([benchmarking glossary](../glossary.md#term-benchmarking))</span>.
- [Backend authoring](../backend-guide/writing-a-backend.md)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend))</span> and
  [release mechanics](release-process.md)<span data-glossary-navigation="true"> ([release glossary](../glossary.md#term-release))</span>.
- [Generated API documentation](../reference/javadoc.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

## [Planning](../glossary.md#term-planning) and historical evidence

[Planning](../planning/README.md) [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) authorized work. [Design records](../design/README.md)
record decisions; [current architecture navigation](../architecture/current-architecture-plan.md)
connects explanations. The [audit report](../planning/public-release-audit-2026-11-04.md) and
[production support matrix](../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-review-matrix))</span> distinguish current
[qualification](../glossary.md#term-qualification) from older candidate evidence. Historical receipts bind their original source and
[artifacts](../glossary.md#term-artifact); rewriting learning material does not requalify new executable behavior.
The [historical documentation index](../history/index.md) separates superseded qualification
inventories from the current reference; their retained wording is not present capability guidance.

Public [publication](../glossary.md#term-publication) remains blocked by the [owner](../glossary.md#term-owner)'s unselected licence/grant. Local [Javadoc](../glossary.md#term-javadoc),
example consumers and source browsing do not authorize a [registry](../glossary.md#term-registry) upload or trusted publisher claim.
