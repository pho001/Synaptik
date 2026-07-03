# Autograd strategy

## Purpose and status

This note explains the architecture-approved direction for automatic differentiation (autograd). It is a pre-implementation design note, not a compiler specification. Exact derivative registries, intermediate types, diagnostics, and pass boundaries remain for compiler tasks.

## Strategy

When a compile mode requests gradients, the compiler expands the captured forward graph into backward computations before post-autograd optimization and planning:

```text
validated forward graph
  -> backward graph construction
  -> combined forward + backward graph
  -> optimize -> publication -> ownership planning
```

The combined graph lets compiler passes remove dead work, simplify gradient accumulation, reuse forward values, and plan lifetimes across both phases. It remains immutable compile-time state.

## Ownership boundaries

- `modules:model` represents backend-independent operation semantics, including compiler-generated operations when required.
- `modules/compiler` owns derivative rules, backward traversal, accumulation, validation, and combined-graph construction.
- `modules/planning` assigns backend ownership after expansion.
- Concrete backends prepare only their assigned regions; they do not perform global autograd.
- Runtime executes prepared schedules and does not derive gradients.

## Example

For `y = x × x`, the backward computation needs the forward value `x` and computes `dy/dx = 2 × x`. At `x = 3`, the gradient is `6`. In a graph with multiple uses of `x`, the compiler must represent and combine all contributions. The example explains semantic intent, not an implemented operation API.

## Open implementation detail

The architecture does not yet fix derivative-rule registration, saved-value representation, checkpointing, higher-order gradients, mutation interaction, or diagnostic payloads. Focused tasks must decide these without moving global graph logic into backends.

See [Training graph](../../architecture/training-graph.md), [ADR 0005](../decisions/0005-training-combined-forward-backward-graph.md), and the [compiler master plan](../../planning/modules/compiler/master-plan.md).
