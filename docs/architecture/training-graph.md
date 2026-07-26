# Training Graph

This document explains the training graph model established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

The three compile modes are implemented as standalone declarative configuration values. No
current compile aggregate or compiler entry point consumes them, and autograd, training, and
gradient publication remain planned. This page explains the intended graph boundary rather than
a runnable training API.

## Compile modes

Synaptik distinguishes three compile modes:

- **Forward-only mode** (`FORWARD_ONLY`) compiles only the forward computation and its requested publications.
- **Forward-and-backward mode** (`FORWARD_AND_BACKWARD`) constructs gradient Tensor expressions
  from the original forward expression and captures both phases together.
- **Training-step mode** (`TRAINING_STEP`) initially uses that same combined forward/backward
  construction. Optimizer-update graph work remains future.

The compile mode changes the compile-time graph and publication needs. It does not allow a backend to own global autograd.

## Combined forward and backward graph

When a compile mode requires gradients, the compiler builds backward expressions before the only
capture:

```text
original forward Tensor expression DAG
  -> fail-closed compiler autograd preflight
  -> gradient formulas through ordinary public Tensor operations
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined forward + backward graph
  -> inference, validation, combined optimization, and final validation
  -> publication binding
  -> planning and partitioning
```

The compiler uses exact Tensor identity only for temporary reverse-accumulation bookkeeping and
ordinary `Tensor.add` for contribution accumulation. This is not a second graph representation.
Named compiler rules own dispatch; model operations remain derivative-agnostic. Seeds and other
derivative constants are storage-free leaves registered explicitly as compile-time splats.

Phase-aware capture receives the original forward-producer identity set, forward outputs,
gradient roots and target roles, and exact constant facts. It assigns graph-local IDs once and
retains `FORWARD` or `BACKWARD` on every node. Two targets may share one captured gradient value;
the role mapping remains target-specific while the graph output boundary lists that value once.

The combined graph is immutable compile-time graph state. Backends receive only their planned
regions during prepare and execute only those prepared regions at runtime.

Compile-time combination does not require a single runtime schedule. Prepare may expose separate forward and backward schedules or one training-step schedule, depending on prepare-time decisions.

## Why combine the graph

Seeing forward and backward work together allows compiler and planning passes to reason across the autograd boundary. This can enable:

- dead-code elimination and common subexpression elimination across both phases;
- cleanup of redundant transposes and no-op unbroadcast operations;
- simpler gradient accumulation;
- reuse of forward values required by backward computation;
- more accurate lifetime and logical memory planning; and
- ownership and partition decisions informed by the full training computation.

These are global graph concerns, so they belong to compiler and backend-neutral planning rather than concrete backends.

The first implementation reuses only already-proved exact rewrites and folding rules. Common
subexpression elimination stays phase-local, and every changed immutable candidate is revalidated.
No new algebraic identity follows merely from combining the phases.

## Failures and future derivative order

Preflight rejects any unsupported backward-reachable operation, exact attributes, or required
derivative policy before it constructs backward expressions. Full inference follows capture, so a
later failure can consume temporary Tensor IDs; callers must already treat those IDs as opaque.

Generated gradients remain ordinary differentiable Tensor expressions. Higher derivatives are
deliberately absent from the first implementation. A later lifecycle decision must define an
explicit create-graph or derivative-order request, rules for every operation used inside gradient
formulas, and a graph representation that distinguishes derivative order from forward/backward
phase. No such extension may add mutable gradient state to Tensor.

## Optimizer as a backend-agnostic step

`extensions/nn` owns the model-module side of a training step: modules declare their trainable
`Parameter` values, persistent `Buffer` values, and train/eval forward behavior. `extensions/training`
is downstream of that declaration and owns optimizer algorithms and training-step orchestration.
This distinction lets a module be used for inference without importing an optimizer.

The initial training lifecycle keeps the optimizer algorithm in `extensions/training` and runs its step after gradients are produced:

```text
prepared forward/backward execution
  -> publish gradients
  -> optimizer.step()
```

Training owns the mathematical update represented by optimizers such as SGD, Adam, and AdamW.
It consumes parameters declared by `extensions/nn`; it does not own `Parameter`, `Buffer`, layer
behavior, or train/eval mode. It does not select a Metal, CUDA, CPU, or other backend-specific
execution route.

## Future compiled optimizer graph

A future architecture version may generate optimizer update operations as part of a larger graph:

```text
forward + backward + optimizer update graph
  -> optimize and partition
  -> backend prepare
  -> training-step schedule
```

In that model, backend prepare may fuse or specialize optimizer updates while the training extension remains backend-agnostic. This direction requires the explicit architecture updates required by the contract; it is not permission to move backend implementation into training now.

## Why `MetalOptimizerBridge` does not belong in training

A `MetalOptimizerBridge` would make the training extension depend on a concrete backend and give training responsibility for backend execution. That reverses the intended dependency direction.

The correct ownership is:

```text
extensions/training  -> optimizer algorithm and mathematical intent
compiler/model       -> graph operations or backend-neutral step representation
backends/metal       -> MPSGraph or custom fused Metal implementation
runtime              -> execution of the prepared executable
```

The same rule excludes CPU- and CUDA-specific optimizer bridges from training. See [Module Boundaries](module-boundaries.md) and [Dependency Rules](dependency-rules.md).
