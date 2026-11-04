# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0007: Neural-network [module](../../glossary.md#term-module) and training boundary


## Editorial reading context

A neural-network layer declares persistent parameter and buffer state; an optimizer computes updates to trainable parameters. Those are different jobs even when both participate in training. The dependency reconciliation in Status matters: the original arrows below are historical. Current Training receives a generic live parameter adapter rather than importing NN layer implementations.

Use the [parameter admission and optimizer roles](../../architecture/training-graph.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-07-13.

2026-10-16 boundary reconciliation: [ADR0039](0039-portable-graph-model-persistence.md#necessary-boundary-types-not-a-new-model-hierarchy)
and Training0007 supersede only the [dependency direction](../../glossary.md#java-platform-and-build-tools) and Training's physical placement below.
Training now lives at `modules/training` (unchanged project/package [identities](../../glossary.md#term-identity-meanings)), has exactly public
[Model](../../glossary.md#term-model)/[Engine](../../glossary.md#term-engine) edges, and accepts generic live [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters). [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) has exactly public
Model/Training edges for `Module.trainingParameters()`. [Parameter](../../glossary.md#term-parameter)/[Buffer](../../glossary.md#term-buffer) and mode [ownership](../../glossary.md#term-ownership-meanings) remain
NN concerns; [optimizer](../../glossary.md#term-optimizer) algorithms/orchestration remain Training concerns. The historical rationale
below is retained, not current [dependency](../../glossary.md#term-dependency) permission. The
[owning contract](../../architecture/contracts/extensions-training.md#modulestraining) governs.

## Context

The architecture previously placed [`Parameter`](../../glossary.md#term-parameter) in `extensions/training`. A parameter, however,
is declared by a layer alongside its persistent state and governs what an [optimizer](../../glossary.md#term-optimizer) may update.
Layer forward behavior also needs an explicit training/[evaluation](../../glossary.md#term-evaluation) mode before any optimizer is
chosen. Keeping these concerns in training would make a neural-network layer depend on optimizer
orchestration and would conflate [inference](../../glossary.md#term-inference) behavior with the act of updating parameters.

Synaptik must preserve the existing boundary in which `modules/model` owns generic [Tensor](../../glossary.md#tensor) and
[operation](../../glossary.md#operation) semantics, while optimizers remain [backend](../../glossary.md#backend)-agnostic and concrete backends own execution.

## Decision drivers

- let layers declare trainable state without choosing an [optimizer](../../glossary.md#term-optimizer);
- keep training/[evaluation](../../glossary.md#term-evaluation) forward behavior with the [modules](../../glossary.md#term-module) it changes;
- let optimizers work over a common [parameter](../../glossary.md#term-parameter) contract without knowing layer types;
- preserve `modules/model` as a generic, [backend](../../glossary.md#backend)-independent computation [model](../../glossary.md#term-model); and
- retain a one-way, testable extension [dependency direction](../../glossary.md#java-platform-and-build-tools).

## Options considered

### Keep [`Parameter`](../../glossary.md#term-parameter) in `extensions/training`

This retains the former contract but makes `nn` layers depend on training to declare their own
state. It couples [inference](../../glossary.md#term-inference) composition to [optimizer](../../glossary.md#term-optimizer)-oriented APIs and gives training [ownership](../../glossary.md#term-ownership-meanings)
of layer data it does not interpret.

### Put all training behavior in `extensions/nn`

This would keep layers and [optimizer](../../glossary.md#term-optimizer) updates together, but would make [module](../../glossary.md#term-module) composition own
optimizer algorithms, sessions, and orchestration. It would also prevent a non-neural [Tensor](../../glossary.md#tensor) [graph](../../glossary.md#graph)
from using training facilities without importing layer abstractions.

### Create `extensions/nn` and keep [optimizer](../../glossary.md#term-optimizer) orchestration in training

`extensions/nn` owns [modules](../../glossary.md#term-module), [parameters](../../glossary.md#term-parameter), [buffers](../../glossary.md#term-buffer), train/[eval mode](../../glossary.md#inference-and-evaluation), forward context, layers, and
functional conveniences. `extensions/training` owns optimizers, parameter groups, sessions, and
training steps over nn-declared parameters.

## Decision

Adopt the third option. Add the planned `extensions/nn` extension with [dependency direction](../../glossary.md#java-platform-and-build-tools):

```text
modules/model
  -> extensions/nn
  -> extensions/training
```

[`Parameter`](../../glossary.md#term-parameter) and [`Buffer`](../../glossary.md#term-buffer) are owned by `extensions/nn`. [`Module`](../../glossary.md#term-module) declares and traverses them;
`train()` and `eval()` plus forward context are also nn concerns. Training consumes the declared
parameter contract for [optimizer](../../glossary.md#term-optimizer) algorithms and training orchestration. Neither extension owns
[autograd](../../glossary.md#autograd) construction, [concrete backend](../../glossary.md#concrete-backend) storage, [kernel](../../glossary.md#kernel) selection, or [backend](../../glossary.md#backend)-specific optimizer
execution.

This [ADR](../../glossary.md#term-architecture-decision-record-adr) changes the [architecture contract](../../glossary.md#architecture-contract) and plans only. It does not create a [Gradle](../../glossary.md#term-gradle) module,
Java production [API](../../glossary.md#term-application-programming-interface-api), or executable layer implementation.

## Rationale

A layer can be evaluated without being optimized, but it cannot define its own state if
[`Parameter`](../../glossary.md#term-parameter) belongs to a downstream [optimizer](../../glossary.md#term-optimizer) extension. The selected boundary lets a [module](../../glossary.md#term-module) own
the values and mode that shape its forward computation, while a generic optimizer iterates over
those declarations. For example, a [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) module owns train/eval behavior and
running-[statistic](../../glossary.md#term-statistic) [buffers](../../glossary.md#term-buffer); an optimizer may update its trainable scale and [bias](../../glossary.md#term-bias) but has no reason
to choose that forward behavior.

Keeping generic [`Tensor`](../../glossary.md#tensor) operations in `modules/model` avoids a second semantic algebra inside
nn. Keeping optimization in training avoids coupling every layer to [Adam](../../glossary.md#adaptive-optimizer--adam-adamw), [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd), sessions, or
[checkpoint](../../glossary.md#term-checkpoint) orchestration.

## Consequences

### Positive

- Neural-network [modules](../../glossary.md#term-module) can be composed for [inference](../../glossary.md#term-inference) without importing [optimizer](../../glossary.md#term-optimizer) APIs.
- A common [parameter](../../glossary.md#term-parameter) traversal supports optimizer algorithms without layer-specific knowledge.
- Train/eval behavior is co-located with the layers whose forward semantics it changes.
- The [model](../../glossary.md#term-model) remains reusable by [compiler](../../glossary.md#term-compiler), [backends](../../glossary.md#backend), and non-nn clients.

### Negative and risks

- A future nn API must define ownership, naming, nesting, mutation, and checkpoint contracts for
  parameters and buffers before implementation.<span data-glossary-navigation="true"> ([API glossary](../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../glossary.md#term-ownership-meanings); [checkpoint glossary](../../glossary.md#term-checkpoint); [parameters glossary](../../glossary.md#term-parameter); [buffers glossary](../../glossary.md#term-buffer))</span>
- Training now has an explicit extension dependency, so its master plan and build configuration
  must change when <span data-glossary-navigation="true"> ([dependency glossary](../../glossary.md#term-dependency))</span>`extensions/nn` is implemented.
- Train/eval mode is a forward-context concern; it must not silently become mutable backend or
  runtime execution state.<span data-glossary-navigation="true"> ([eval mode glossary](../../glossary.md#inference-and-evaluation); [mutable glossary](../../glossary.md#term-mutable); [backend glossary](../../glossary.md#backend); [runtime glossary](../../glossary.md#term-runtime))</span>

### Migration, testing, and follow-up

- Create the `extensions/nn` Gradle module and Java contracts only at the permitted planning
  frontier; add the declared <span data-glossary-navigation="true"> ([Gradle glossary](../../glossary.md#term-gradle); [planning glossary](../../glossary.md#term-planning))</span>`training -> nn` build [dependency](../../glossary.md#term-dependency) at the same time.
- Architecture tests must enforce the one-way extension direction when both Gradle projects
  exist; the current test records the conditional contract without pretending the unimplemented
  module exists.
- Plan concrete [`Module`](../../glossary.md#term-module), [`Parameter`](../../glossary.md#term-parameter), [`Buffer`](../../glossary.md#term-buffer), and forward-context APIs before training tasks.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Module boundaries](../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Dependency rules](../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../glossary.md#term-dependency))</span>
- [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [NN master plan](../../planning/extensions/nn/master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Training master plan](../../planning/extensions/training/master-plan.md)
