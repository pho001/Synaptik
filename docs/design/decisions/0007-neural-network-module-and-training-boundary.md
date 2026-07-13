# ADR 0007: Neural-network module and training boundary

## Status

Accepted — 2026-07-13.

## Context

The architecture previously placed `Parameter` in `extensions/training`. A parameter, however,
is declared by a layer alongside its persistent state and governs what an optimizer may update.
Layer forward behavior also needs an explicit training/evaluation mode before any optimizer is
chosen. Keeping these concerns in training would make a neural-network layer depend on optimizer
orchestration and would conflate inference behavior with the act of updating parameters.

Synaptik must preserve the existing boundary in which `modules/model` owns generic Tensor and
operation semantics, while optimizers remain backend-agnostic and concrete backends own execution.

## Decision drivers

- let layers declare trainable state without choosing an optimizer;
- keep training/evaluation forward behavior with the modules it changes;
- let optimizers work over a common parameter contract without knowing layer types;
- preserve `modules/model` as a generic, backend-independent computation model; and
- retain a one-way, testable extension dependency direction.

## Options considered

### Keep `Parameter` in `extensions/training`

This retains the former contract but makes `nn` layers depend on training to declare their own
state. It couples inference composition to optimizer-oriented APIs and gives training ownership
of layer data it does not interpret.

### Put all training behavior in `extensions/nn`

This would keep layers and optimizer updates together, but would make module composition own
optimizer algorithms, sessions, and orchestration. It would also prevent a non-neural Tensor graph
from using training facilities without importing layer abstractions.

### Create `extensions/nn` and keep optimizer orchestration in training

`extensions/nn` owns modules, parameters, buffers, train/eval mode, forward context, layers, and
functional conveniences. `extensions/training` owns optimizers, parameter groups, sessions, and
training steps over nn-declared parameters.

## Decision

Adopt the third option. Add the planned `extensions/nn` extension with dependency direction:

```text
modules/model
  -> extensions/nn
  -> extensions/training
```

`Parameter` and `Buffer` are owned by `extensions/nn`. `Module` declares and traverses them;
`train()` and `eval()` plus forward context are also nn concerns. Training consumes the declared
parameter contract for optimizer algorithms and training orchestration. Neither extension owns
autograd construction, concrete backend storage, kernel selection, or backend-specific optimizer
execution.

This ADR changes the architecture contract and plans only. It does not create a Gradle module,
Java production API, or executable layer implementation.

## Rationale

A layer can be evaluated without being optimized, but it cannot define its own state if
`Parameter` belongs to a downstream optimizer extension. The selected boundary lets a module own
the values and mode that shape its forward computation, while a generic optimizer iterates over
those declarations. For example, a batch-normalization module owns train/eval behavior and
running-statistic buffers; an optimizer may update its trainable scale and bias but has no reason
to choose that forward behavior.

Keeping generic `Tensor` operations in `modules/model` avoids a second semantic algebra inside
nn. Keeping optimization in training avoids coupling every layer to Adam, SGD, sessions, or
checkpoint orchestration.

## Consequences

### Positive

- Neural-network modules can be composed for inference without importing optimizer APIs.
- A common parameter traversal supports optimizer algorithms without layer-specific knowledge.
- Train/eval behavior is co-located with the layers whose forward semantics it changes.
- The model remains reusable by compiler, backends, and non-nn clients.

### Negative and risks

- A future nn API must define ownership, naming, nesting, mutation, and checkpoint contracts for
  parameters and buffers before implementation.
- Training now has an explicit extension dependency, so its master plan and build configuration
  must change when `extensions/nn` is implemented.
- Train/eval mode is a forward-context concern; it must not silently become mutable backend or
  runtime execution state.

### Migration, testing, and follow-up

- Create the `extensions/nn` Gradle module and Java contracts only at the permitted planning
  frontier; add the declared `training -> nn` build dependency at the same time.
- Architecture tests must enforce the one-way extension direction when both Gradle projects
  exist; the current test records the conditional contract without pretending the unimplemented
  module exists.
- Plan concrete `Module`, `Parameter`, `Buffer`, and forward-context APIs before training tasks.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Module boundaries](../../architecture/module-boundaries.md)
- [Dependency rules](../../architecture/dependency-rules.md)
- [Lifecycle](../../architecture/lifecycle.md)
- [Training graph](../../architecture/training-graph.md)
- [NN master plan](../../planning/extensions/nn/master-plan.md)
- [Training master plan](../../planning/extensions/training/master-plan.md)
