# ADR 0005: Combined forward and backward training graph

## Status

Accepted — reflected in the current architecture contract. The original decision date and historical alternatives are not documented.

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

For backward-capable compile modes, the compiler may expand the forward graph into a combined forward and backward graph before post-autograd optimization, publication binding, and planning. The combined graph is immutable compile-time state and does not force one runtime schedule.

## Rationale

Compiler and planning passes can remove dead gradient work, simplify accumulation, preserve required forward values, and reason about lifetimes and ownership across both phases. Backends still receive only assigned partitions.

## Consequences

The compiler needs explicit graph phases and derivative semantics. Logical memory planning becomes more global. Preparation may choose schedule structure, while training owns optimizer algorithms and backends own backend-specific optimizer routes. Higher-order gradients and compiled optimizer updates remain future decisions.

## Related documentation

- [Training graph](../../architecture/training-graph.md)
- [Autograd strategy](../notes/autograd-strategy.md)
- [Training API status](../../api/training-api.md)
