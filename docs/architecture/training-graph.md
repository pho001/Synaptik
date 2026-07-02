# Training Graph

This document explains the training graph model established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

## Compile modes

Synaptik distinguishes three compile modes:

- **Forward-only mode** (`FORWARD_ONLY`) compiles only the forward computation and its requested publications.
- **Forward-and-backward mode** (`FORWARD_AND_BACKWARD`) applies autograd to the forward graph and compiles the resulting forward and backward computation.
- **Training-step mode** (`TRAINING_STEP`) represents a training step in which forward, backward, and eventually optimizer-update work may participate in one compiled plan.

The compile mode changes the compile-time graph and publication needs. It does not allow a backend to own global autograd.

## Combined forward and backward graph

When a compile mode requires gradients, the compiler may expand the captured forward graph before post-autograd optimization:

```text
forward graph
  -> autograd expansion
  -> combined forward + backward graph
  -> post-autograd optimization
  -> publication binding
  -> planning and partitioning
```

The combined graph is still immutable compile-time graph state. Backends receive only their planned regions during prepare and execute only those prepared regions at runtime.

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

## Optimizer as a backend-agnostic step

The initial training lifecycle keeps the optimizer algorithm in `extensions/training` and runs its step after gradients are produced:

```text
prepared forward/backward execution
  -> publish gradients
  -> optimizer.step()
```

Training owns the mathematical update represented by optimizers such as SGD, Adam, and AdamW. It does not select a Metal, CUDA, CPU, or other backend-specific execution route.

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
