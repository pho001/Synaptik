# Training API

## Purpose and implementation status

This reference records the planned public training concepts without inventing callable APIs. `extensions/training`, compiler autograd, gradient publication, and prepared execution are not implemented.

Training will own backend-independent optimizer algorithms and session concepts. The compiler will own global automatic differentiation (autograd), while concrete backends will own any backend-specific lowering or fused optimizer route.

## Planned concepts

- `Parameter` and `ParameterGroup` will describe trainable public tensor state and group-level optimizer settings.
- `Optimizer` implementations such as SGD, Adam, and AdamW will define mathematical updates without importing CPU, Metal, or CUDA modules.
- `TrainingSession` and `TrainingStep` will coordinate forward/backward execution, gradient publication, and optimizer updates through shared lifecycle contracts.

No signatures, default hyperparameters, mutation rules, serialization format, or exception types are stable yet. They will be defined by focused extension tasks after model, compiler, runtime, and publication contracts exist.

## Planned initial flow

```text
compile forward + backward graph
  -> prepare owned partitions
  -> run forward/backward
  -> publish gradients
  -> optimizer.step()
```

The initial optimizer step is backend-agnostic. A later architecture version may compile optimizer updates into the graph, but that direction requires the architecture update described by the contract.

## Boundary example

Adam computes moment estimates and a parameter update from gradients. That mathematical algorithm belongs to training. Choosing a fused Metal kernel for the update belongs to Metal prepare. A `MetalOptimizerBridge` inside training would reverse the required dependency direction and is not a supported design.

## Related contracts

- [Training graph](../architecture/training-graph.md)
- [Autograd guide](../user-guide/autograd.md)
- [Training guide](../user-guide/training.md)
- [Training master plan](../planning/extensions/training/master-plan.md)
