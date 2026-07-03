# Backend guide style

## Purpose

A backend guide teaches a backend contributor how to integrate a concrete backend while preserving shared compile, planning, prepare, and runtime boundaries. It should connect the integration lifecycle to a focused sample implementation.

Apply [General style](general-style.md), [Developer guide style](developer-guide-style.md), and the [example format](example-format.md) where relevant.

## Required content

- State the backend integration goal and supported capability scope.
- List prerequisites, native/toolchain requirements, and shared contracts the backend implements.
- Explain the lifecycle from capability reporting through backend ownership, prepare-time lowering, executable construction, runtime execution, storage, and tracing.
- Identify which module owns each decision and resource.
- Provide a minimal sample implementation or integration skeleton using current contracts; label conceptual code when contracts are not implemented yet.
- Explain registration and composition without implying runtime discovery.
- Show sample inputs, important intermediate artifacts, expected result, and interpretation.
- Document failure handling, resource ownership and release, concurrency assumptions, and diagnostic evidence.
- List required unit, architecture, backend-conformance, integration, and native validation as applicable.

The sample should demonstrate one coherent path, such as capability declaration through preparation of one partition, rather than presenting a disconnected inventory of extension points.

## Avoid

- moving backend-specific lowering or kernel selection into shared planning or runtime;
- treating scalar, Vector API, or OpenBLAS routes as separate CPU backends;
- service-locator or reflective discovery examples presented as the core mechanism;
- native handles without ownership and lifetime rules;
- a sample that skips capability, prepare, or error behavior; and
- claims of conformance without commands and results.

## Validation

- Trace the sample through compile ownership, prepare, and run boundaries.
- Verify dependency direction against `ARCHITECTURE.md` and architecture tests.
- Build and run the sample or label it conceptual with implementation prerequisites.
- Run applicable backend-conformance and integration tests.
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
