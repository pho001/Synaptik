# ADR 0005: Combined forward and backward training graph

## Status

Superseded — 2026-07-26, by
[ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](0009-compiler-owned-pre-capture-tensor-expression-autograd.md).
The combined immutable graph decision remains, but capture timing and construction ownership are
replaced.

## Context

Backward computation depends on forward values and may create optimization, accumulation, lifetime, and ownership relationships across the autograd boundary. Treating both phases as unrelated programs would hide those graph-level opportunities.

## Decision drivers

- keep global autograd in the compiler;
- allow post-autograd optimization over all training computation;
- expose complete logical lifetimes to planning; and
- keep training and concrete backends independent.

## Options considered

No historical comparison is available. The contract describes a combined compile-time graph and allows prepare to produce separate forward/backward schedules or one training-step schedule. A backend-owned global backward graph is explicitly excluded.

## Decision

This historical decision selected a combined forward/backward compile-time graph but did not
decide whether autograd ran before or after graph capture. ADR 0009 now requires compiler-owned
Tensor-expression autograd before one combined capture.

## Rationale

Compiler and planning passes can remove dead gradient work, simplify accumulation, preserve required forward values, and reason about lifetimes and ownership across both phases. Backends still receive only assigned partitions.

## Consequences

The compiler needs explicit graph phases and derivative semantics. Logical memory planning becomes more global. Preparation may choose schedule structure, while training owns optimizer algorithms and backends own backend-specific optimizer routes. Higher-order gradients and compiled optimizer updates remain future decisions.

## Related documentation

- [Training graph](../../architecture/training-graph.md)
- [Autograd strategy](../notes/autograd-strategy.md)
- [Training API status](../../api/training-api.md)
- [ADR 0009](0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
