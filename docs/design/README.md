# Design records and notes

## Purpose

This index separates accepted architecture decisions from pre-implementation strategy notes.
Neither category overrides the authoritative root or the scoped contracts it incorporates through
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).

Decision records summarize rules present in the [architecture contract](../glossary.md#architecture-contract). Retrospective records state
when their original date or deliberation history is unavailable; current decisions record their
acceptance date and considered alternatives.

## Architecture decision records

- [ADR 0001: Layered architecture](decisions/0001-layered-architecture.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0002: Backend-owned lowering](decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([Backend-owned lowering glossary](../glossary.md#backend-owned-lowering))</span>
- [ADR 0003: Typed trace DTOs](decisions/0003-typed-trace-dtos.md)<span data-glossary-navigation="true"> ([trace glossary](../glossary.md#trace))</span>
- [ADR 0004: Backend-neutral partition scoring](decisions/0004-partition-scoring.md)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend); [partition glossary](../glossary.md#partition))</span>
- [ADR 0005: Combined forward and backward training graph](decisions/0005-training-combined-forward-backward-graph.md)<span data-glossary-navigation="true"> ([training graph glossary](../glossary.md#training-graph))</span>
- [ADR 0006: No runtime service locator](decisions/0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([runtime glossary](../glossary.md#term-runtime); [service locator glossary](../glossary.md#term-service-locator))</span>
- [ADR 0007: Neural-network module and training boundary](decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([module glossary](../glossary.md#term-module))</span>
- [ADR 0008: Separate performance evidence and model-autotuning concerns](decisions/0008-performance-evidence-and-tuning-boundaries.md)<span data-glossary-navigation="true"> ([model glossary](../glossary.md#term-model); [autotuning glossary](../glossary.md#autotuning--model-autotuning))</span>
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler); [pre-capture Tensor-expression autograd glossary](../glossary.md#pre-capture-tensor-expression-autograd))</span>
- [ADR 0010: Staged backend preparation](decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run); [ownership glossary](../glossary.md#term-ownership); [binding glossary](../glossary.md#term-binding))</span>
- [ADR 0012: Fixed recurrent scan without graph regions](decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([scan glossary](../glossary.md#term-cumulative-scan); [graph glossary](../glossary.md#graph))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>
- [ADR 0014: Scope-indexed normative architecture contracts](decisions/0014-scope-indexed-normative-architecture-contracts.md)<span data-glossary-navigation="true"> ([Scope glossary](../glossary.md#term-scope); [normative glossary](../glossary.md#term-normative))</span>
- [ADR 0015: Explicit Engine backend composition](decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine); [backend composition glossary](../glossary.md#backend-composition))</span>
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](decisions/0016-cpu-metal-mixed-owner-schedule.md)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [Metal glossary](../glossary.md#term-metal); [owner glossary](../glossary.md#term-owner); [prepared schedule glossary](../glossary.md#prepared-schedule--preparedschedule))</span>
- [ADR 0017: Reusable inference session facade](decisions/0017-reusable-inference-session-facade.md)<span data-glossary-navigation="true"> ([inference glossary](../glossary.md#term-inference); [facade glossary](../glossary.md#term-facade))</span>
- [ADR 0018: Public Training Session and SGD lifecycle](decisions/0018-public-training-session-and-sgd-lifecycle.md)<span data-glossary-navigation="true"> ([Training Session glossary](../glossary.md#training-session--trainingsession); [SGD glossary](../glossary.md#term-stochastic-gradient-descent-sgd))</span>
- [ADR 0019: Explicit graph numerical profiles](decisions/0019-explicit-numerical-profiles.md)
- [ADR 0020: Synchronous single-default-device Metal execution](decisions/0020-synchronous-single-default-device-metal-execution.md)<span data-glossary-navigation="true"> ([Synchronous glossary](../glossary.md#term-synchronous); [device glossary](../glossary.md#term-device))</span>
- [ADR 0021: Total Recursive ACCELERATOR Numerical Floor (superseded by 0026)](decisions/0021-total-recursive-accelerator-numerical-floor.md)<span data-glossary-navigation="true"> ([ACCELERATOR glossary](../glossary.md#term-accelerator))</span>
- [ADR 0022: Auditable Custom Metal Route Cost Evidence](decisions/0022-auditable-custom-metal-route-cost-evidence.md)
- [ADR 0023: Low-precision ACCELERATOR parity and P0 evidence (profile rule superseded by 0026)](decisions/0023-low-precision-accelerator-parity.md)<span data-glossary-navigation="true"> ([precision glossary](../glossary.md#term-precision))</span>
- [ADR 0024: Environment-bound Metal raw-route certificates (superseded)](decisions/0024-environment-bound-metal-raw-certificates.md)
- [ADR 0025: Custom-only Metal low precision](decisions/0025-custom-only-metal-low-precision.md)
- [ADR 0026: Profile-free numerical semantics (accepted)](decisions/0026-profile-free-numerical-semantics.md)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../glossary.md#profile-free-numerical-semantics))</span>

## Design notes

- [Autograd strategy](notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../glossary.md#autograd))</span>
- [CPU kernel strategy](notes/cpu-kernel-strategy.md)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [kernel glossary](../glossary.md#kernel))</span>
- [Memory planning strategy](notes/memory-planning-strategy.md)<span data-glossary-navigation="true"> ([planning glossary](../glossary.md#term-planning))</span>
- [Metal backend strategy](notes/metal-backend-strategy.md)<span data-glossary-navigation="true"> ([Metal glossary](../glossary.md#term-metal); [backend glossary](../glossary.md#backend))</span>
