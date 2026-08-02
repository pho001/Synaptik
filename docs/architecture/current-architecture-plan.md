# Current architecture documentation

This document is the navigation page for explanations of the current Synaptik architecture.

The authoritative architecture contract is:

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)

This file and the other documents in this directory are explanatory documentation. They do not replace the contract or report implementation completion.

Focused architecture documentation:

- [Overview](overview.md)
- [Lifecycle](lifecycle.md)
- [Module boundaries](module-boundaries.md)
- [Dependency rules](dependency-rules.md)
- [Partition scoring](partition-scoring.md)
- [Performance evidence and model autotuning](performance-evidence-and-tuning.md)
- [Training graph](training-graph.md)
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [ADR 0007: Neural-network module and training boundary](../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Tracing](tracing.md)
- [Runtime / Prepare / Backend boundary](runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../design/decisions/0011-per-run-runtime-resource-ownership.md)

## Status

This index is current. The architecture describes the intended complete system. The repository now has substantive Model, Backend Contract, Planning, Compiler, Runtime, and Prepare implementations, plus partial Config and Trace contracts; Engine, concrete backends, and higher-layer extensions and tools are not yet complete. The [implementation roadmap](../planning/roadmap.md) records delivery status.

## Decisions and strategies

- [Architecture decision records and design notes](../design/README.md)
- [Backend integration guides](../index.md#documentation-areas)
- [Developer guides](../index.md#documentation-areas)
