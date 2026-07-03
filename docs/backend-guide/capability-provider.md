# Report backend capabilities (planned contract)

## Outcome and status

This guide explains how a concrete backend will report which planned graph work it can accept. Planning, backend-contract, prepare, and backend modules are not implemented, so the sample is conceptual.

A capability is a declarative answer to “can this backend own this work?” It is not a live executable, a kernel registry, or a route selection.

## Lifecycle position

```text
operation + data type + shape + layout + availability
  -> capability provider
  -> valid ownership candidates
  -> backend-neutral scoring
```

The backend implements capability evaluation. Planning calls the shared contract and compares supported backend identities. Compile-time plans retain `BackendId`, never the provider object.

## Conceptual example

Assume a CPU capability provider receives a `FLOAT32` matrix multiplication with shapes `[2, 3]` and `[3, 4]`. The output shape is `[2, 4]`, containing `2 × 4 = 8` values. The provider may report CPU ownership as supported based on semantic facts. It must not select OpenBLAS or a scalar loop; CPU prepare makes that route decision later.

A rejection should carry typed or structured diagnostic evidence explaining the unsupported fact, such as data type or layout. Exact DTOs remain to be defined.

## Failures and diagnostics

- Invalid graph semantics belong to compiler validation, not capability fallback.
- Backend unavailability should remove or reject that ownership candidate before prepare.
- If no candidate supports required work, compilation must fail rather than defer discovery to runtime.
- Capability evaluation must be deterministic for the supplied immutable compile-time facts.

## Validation expectations

Future implementations require unit tests for supported and rejected combinations, architecture tests for dependency direction, and backend-conformance tests comparing declared support with actual preparation.

See [Partition scoring](../architecture/partition-scoring.md), [backend selection](../user-guide/backend-selection.md), and the [backend guide style](../developer-guide/documentation/backend-guide-style.md).
