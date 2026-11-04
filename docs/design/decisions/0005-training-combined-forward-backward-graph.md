# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0005: Combined forward and backward [training graph](../../glossary.md#training-graph)

## Status

Superseded — 2026-07-26, by
[ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr); [Compiler glossary](../../glossary.md#term-compiler); [pre-capture Tensor-expression autograd glossary](../../glossary.md#pre-capture-tensor-expression-autograd))</span>.
The combined [immutable](../../glossary.md#term-immutable) [graph](../../glossary.md#graph) decision remains, but [capture](../../glossary.md#term-capture-meanings) timing and construction [ownership](../../glossary.md#term-ownership-meanings) are
replaced.

## Context

Backward computation depends on forward values and may create optimization, accumulation, [lifetime](../../glossary.md#term-lifetime), and [ownership](../../glossary.md#term-ownership-meanings) relationships across the [autograd](../../glossary.md#autograd) boundary. Treating both phases as unrelated programs would hide those [graph](../../glossary.md#graph)-level opportunities.

## Decision drivers

- keep global [autograd](../../glossary.md#autograd) in the [compiler](../../glossary.md#term-compiler);
- allow post-autograd optimization over all training computation;
- expose complete logical lifetimes to [planning](../../glossary.md#term-planning); and
- keep training and concrete [backends](../../glossary.md#backend) independent.

## Options considered

No historical comparison is available. The contract describes a combined [compile](../../glossary.md#compile)-time [graph](../../glossary.md#graph) and allows [prepare](../../glossary.md#prepare) to produce separate forward/backward schedules or one training-step schedule. A [backend](../../glossary.md#backend)-owned global [backward graph](../../glossary.md#backward-graph) is explicitly excluded.

## Decision

This historical decision selected a combined forward/backward [compile](../../glossary.md#compile)-time [graph](../../glossary.md#graph) but did not
decide whether [autograd](../../glossary.md#autograd) ran before or after [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction). [ADR](../../glossary.md#term-architecture-decision-record-adr) 0009 now requires [compiler](../../glossary.md#term-compiler)-owned
[Tensor](../../glossary.md#tensor)-expression autograd before one combined [capture](../../glossary.md#term-capture-meanings).

## Rationale

[Compiler](../../glossary.md#term-compiler) and [planning](../../glossary.md#term-planning) passes can remove dead [gradient](../../glossary.md#term-gradient) work, simplify accumulation, preserve required forward values, and reason about lifetimes and [ownership](../../glossary.md#term-ownership-meanings) across both phases. [Backends](../../glossary.md#backend) still receive only assigned partitions.

## Consequences

The [compiler](../../glossary.md#term-compiler) needs explicit [graph](../../glossary.md#graph) phases and [derivative](../../glossary.md#term-derivative) semantics. Logical memory [planning](../../glossary.md#term-planning) becomes more global. Preparation may choose schedule structure, while training owns [optimizer](../../glossary.md#term-optimizer) algorithms and [backends](../../glossary.md#backend) own backend-specific optimizer routes. Higher-order [gradients](../../glossary.md#term-gradient) and compiled optimizer updates remain future decisions.

## Related documentation

- [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Autograd strategy](../notes/autograd-strategy.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../glossary.md#autograd))</span>
- [Training API status](../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [ADR 0009](0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
