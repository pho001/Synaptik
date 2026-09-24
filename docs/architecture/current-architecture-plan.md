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
- [ADR 0015: Explicit Engine backend composition](../design/decisions/0015-explicit-engine-backend-composition.md)
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md)
- [ADR 0017: Reusable inference session facade](../design/decisions/0017-reusable-inference-session-facade.md)
- [ADR 0018: Public Training Session and SGD lifecycle](../design/decisions/0018-public-training-session-and-sgd-lifecycle.md)
- [ADR 0019: Explicit numerical profiles](../design/decisions/0019-explicit-numerical-profiles.md)

## Status

This index is current. The architecture describes the intended complete system. The repository
now has substantive Model, Backend Contract, Planning, Compiler, Runtime, Prepare, Engine, CPU,
and Metal implementations, plus partial Config and Trace contracts. The public lifecycle is
runnable through fixed CPU `Engine.standard()` or explicit `Engine.builder()` registration of
opened CPU and Metal integrations. `Engine.session(...)` prepares one compiled graph once and
exposes repeated or concurrent runs through the same prepared execution and existing result
lifecycle. Ordinary preparation composes non-empty plans across registered owners, with
deterministic owner-indexed representations and explicit direct CPU-to-Metal and Metal-to-CPU
transfers for fully static canonical contiguous `FLOAT32` values.
CPU realizes both numerical profiles through identical exact behavior and routes. Metal execution
is profile-qualified: strict admits static `FLOAT32` NEG, locally composed affine layouts, and
explicit `CONTIGUOUS`; accelerator admits only canonical tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and
`DIV` under bounded DAZ/FTZ. The custom Metal route remains reserved for an eligible strict
singleton NEG.

The Training extension now owns a public reusable
Engine-backed scalar session with persistent SGD, accumulation, and detached in-memory state over
its bounded native-storage domain. A standard-Metal convenience, generic plugin
registration/discovery, broader Metal and transfer coverage, CUDA, generic graph/plan tuning,
broader optimizers, and durable persistence remain planned. The
[implementation roadmap](../planning/roadmap.md) records the exact delivery frontier.

## Decisions and strategies

- [Architecture decision records and design notes](../design/README.md)
- [Backend integration guides](../index.md#documentation-areas)
- [Developer guides](../index.md#documentation-areas)
