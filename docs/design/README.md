# Design records and notes

## Purpose

This index separates accepted architecture decisions from pre-implementation strategy notes. Neither category overrides [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

Decision records are retrospective summaries of rules present in the architecture contract. The repository does not preserve the original dates or full deliberation history, and the records say so rather than inventing it.

## Architecture decision records

- [ADR 0001: Layered architecture](decisions/0001-layered-architecture.md)
- [ADR 0002: Backend-owned lowering](decisions/0002-backend-owned-lowering.md)
- [ADR 0003: Typed trace DTOs](decisions/0003-typed-trace-dtos.md)
- [ADR 0004: Backend-neutral partition scoring](decisions/0004-partition-scoring.md)
- [ADR 0005: Combined forward and backward training graph](decisions/0005-training-combined-forward-backward-graph.md)
- [ADR 0006: No runtime service locator](decisions/0006-no-runtime-service-locator.md)
- [ADR 0007: Neural-network module and training boundary](decisions/0007-neural-network-module-and-training-boundary.md)
- [ADR 0008: Separate performance evidence and model-autotuning concerns](decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)

## Design notes

- [Autograd strategy](notes/autograd-strategy.md)
- [CPU kernel strategy](notes/cpu-kernel-strategy.md)
- [Memory planning strategy](notes/memory-planning-strategy.md)
- [Metal backend strategy](notes/metal-backend-strategy.md)
