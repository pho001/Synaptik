# Current architecture documentation

This document is the navigation page for explanations of the current Synaptik architecture.

The authoritative architecture root and sole authority index is:

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)

The root explicitly incorporates exactly six scoped normative contracts:

- [Foundational modules](contracts/foundational-modules.md)
- [Fixed recurrent scan](contracts/recurrent-scan.md)
- [Compiler and automatic differentiation](contracts/compiler-autograd.md)
- [Runtime, Prepare, and Engine](contracts/runtime-prepare-engine.md)
- [Backend execution](contracts/backend-execution.md)
- [Extensions and training](contracts/extensions-training.md)

Those files are normative only within their stated scopes. This file and all other architecture
documents outside `contracts/` are explanatory. They do not replace the contract or report
implementation completion.

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
- [ADR 0012: Fixed recurrent scan without graph regions](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [ADR 0014: Scope-indexed normative architecture contracts](../design/decisions/0014-scope-indexed-normative-architecture-contracts.md)

## Status

This index is current. The architecture describes the intended complete system. The repository
now has substantive Model, Backend Contract, Planning, Compiler, Runtime, Prepare, Engine, and CPU
implementations, plus partial Config and Trace contracts. The current public lifecycle is
runnable through one explicitly owned CPU composition. Metal has bounded MPSGraph and custom
execution routes for supported static `FLOAT32` negation partitions, while broader Metal
coverage, standard or mixed-owner Metal composition, CUDA, generic graph/plan tuning,
persistence, and training orchestration remain planned. The [implementation
roadmap](../planning/roadmap.md) records the exact delivery frontier.

## Decisions and strategies

- [Architecture decision records and design notes](../design/README.md)
- [Backend integration guides](../index.md#documentation-areas)
- [Developer guides](../index.md#documentation-areas)
