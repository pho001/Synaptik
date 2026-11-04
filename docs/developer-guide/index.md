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

## Choose the evidence for the question

- To preserve responsibilities while changing code, read [architecture tests](architecture-tests.md).
  Their output checks implemented boundaries, not every possible rule.
- To diagnose a selected implementation, pass an observer as shown in
  [debugging traces](debugging-trace.md); events explain decisions without controlling execution.
- To compare performance, use [benchmarking](benchmarking.md). A report describes
  one workload and environment; it does not qualify a new default by itself.
- To change explanations, follow [documentation review](documentation-rules.md).
  A clean handoff names the diff and claims the reviewer must examine.

In those workflows, the contributor invokes Gradle or a diagnostic API. The named
framework component or harness then performs the work. “Owner” in a task assignment
means the person/component responsible for a change, not necessarily a closeable object.

## Working guides

- [Coding rules](coding-rules.md), [architecture tests](architecture-tests.md) and
  [documentation rules](documentation-rules.md).
- [Documentation profiles](documentation/README.md): learner tasks, explanations and reference
  have different purposes; do not turn the tutorial into an implementation log.
- [Diagnose preparation/execution](debugging-trace.md) and [bounded benchmarking](benchmarking.md).
- [Backend authoring](../backend-guide/writing-a-backend.md) and
  [release mechanics](release-process.md).
- [Generated API documentation](../reference/javadoc.md).

## [Planning](../glossary.md#term-planning) and historical evidence

[Planning](../planning/README.md) coordinates authorized work. [Design records](../design/README.md)
record decisions; [current architecture navigation](../architecture/current-architecture-plan.md)
connects explanations. The [audit report](../planning/public-release-audit-2026-11-04.md) and
[production support matrix](../backend-guide/production-support.md) distinguish current
[qualification](../glossary.md#term-qualification) from older candidate evidence. Historical receipts bind their original source and
[artifacts](../glossary.md#term-artifact); rewriting learning material does not requalify new executable behavior.
The [historical documentation index](../history/index.md) separates superseded qualification
inventories from the current reference; their retained wording is not present capability guidance.

Public [publication](../glossary.md#term-publication) remains blocked by the [owner](../glossary.md#term-owner)'s unselected licence/grant. Local [Javadoc](../glossary.md#term-javadoc),
example consumers and source browsing do not authorize a [registry](../glossary.md#term-registry) upload or trusted publisher claim.
