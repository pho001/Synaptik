# [Backend](../../glossary.md#backend) guide style

## Purpose

A [backend](../../glossary.md#backend) guide teaches a backend contributor how to integrate a [concrete backend](../../glossary.md#concrete-backend) while preserving shared [compile](../../glossary.md#compile), [planning](../../glossary.md#term-planning), [prepare](../../glossary.md#prepare), and [runtime](../../glossary.md#term-runtime) boundaries. It should connect the integration [lifecycle](../../glossary.md#lifecycle) to a focused [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) implementation.

Apply [General style](general-style.md), [Developer guide style](developer-guide-style.md), and the [example format](example-format.md) where relevant.

A **backend** supplies an implementation of supported computation for an execution [target](../../glossary.md#term-target).
**[Lowering](../../glossary.md#lowering)** converts the shared description into that backend's implementation steps.
**Native** code runs through libraries outside Java's managed execution environment.
**[Conformance](../../glossary.md#term-conformance)** means following the required shared behavior; **[qualification](../../glossary.md#term-qualification)** is the recorded,
bounded evidence used to accept a particular implementation. Neither a [native library](../../glossary.md#term-native-library)'s presence
nor one successful example establishes the backend's entire supported domain.

## Required content

- State the [backend](../../glossary.md#backend) integration goal and supported capability [scope](../../glossary.md#term-scope).
- List prerequisites, native/[toolchain](../../glossary.md#term-toolchain) requirements, and shared contracts the backend implements.
- Explain the [lifecycle](../../glossary.md#lifecycle) from capability reporting through [backend ownership](../../glossary.md#backend-ownership), [prepare](../../glossary.md#prepare)-time [lowering](../../glossary.md#lowering), executable construction, [runtime](../../glossary.md#term-runtime) execution, storage, and tracing.
- Identify which [module](../../glossary.md#term-module) owns each decision and resource.
- Provide a minimal [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) implementation or integration skeleton using current contracts; label conceptual code when contracts are not implemented yet.
- Explain registration and composition without implying runtime discovery.
- Show sample inputs, important intermediate [artifacts](../../glossary.md#term-artifact), expected result, and interpretation.
- Document failure handling, resource [ownership](../../glossary.md#term-ownership) and [release](../../glossary.md#term-release), [concurrency](../../glossary.md#term-concurrency) assumptions, and diagnostic evidence.
- List required unit, architecture, backend-[conformance](../../glossary.md#term-conformance), integration, and native [validation](../../glossary.md#term-validation) as applicable.

The sample should demonstrate one coherent path, such as capability declaration through preparation of one [partition](../../glossary.md#partition), rather than presenting a disconnected inventory of extension points.

## Avoid

- moving [backend](../../glossary.md#backend)-specific [lowering](../../glossary.md#lowering) or [kernel](../../glossary.md#kernel) selection into shared [planning](../../glossary.md#term-planning) or [runtime](../../glossary.md#term-runtime);
- treating [scalar](../../glossary.md#term-scalar), [Vector API](../../glossary.md#term-vector-api), or [OpenBLAS](../../glossary.md#term-openblas) routes as separate [CPU](../../glossary.md#term-cpu) backends;
- service-locator or reflective discovery examples presented as the core mechanism;
- native handles without [ownership](../../glossary.md#term-ownership-meanings) and [lifetime](../../glossary.md#term-lifetime) rules;
- a [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) that skips capability, [prepare](../../glossary.md#prepare), or error behavior; and
- claims of [conformance](../../glossary.md#term-conformance) without commands and results.

## [Validation](../../glossary.md#term-validation)

- [Trace](../../glossary.md#trace) the [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) through [compile](../../glossary.md#compile) [ownership](../../glossary.md#term-ownership-meanings), [prepare](../../glossary.md#prepare), and [run](../../glossary.md#term-run) boundaries.
- Verify [dependency direction](../../glossary.md#java-platform-and-build-tools) against `ARCHITECTURE.md` and architecture tests.
- Build and run the sample or label it conceptual with implementation prerequisites.
- Run applicable [backend](../../glossary.md#backend)-[conformance](../../glossary.md#term-conformance) and integration tests.
- Check native setup, cleanup, failure paths, links, and glossary terms.

## Template

~~~markdown
# Integrating <backend or backend feature>

## Outcome and supported scope

## Prerequisites

## Contracts and ownership

## Integration lifecycle

```text
capability -> ownership -> backend prepare -> executable -> runtime
```

## Minimal sample implementation

### Inputs and configuration

### Meaningful implementation steps

### Intermediate artifacts

### Result and interpretation

## Registration and composition

## Storage, resources, and concurrency

## Failures and diagnostics

## Conformance and validation

## Limitations and related documentation
~~~
