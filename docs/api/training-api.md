# Training API

## Purpose and implementation status

This reference records the planned public training concepts without inventing callable APIs. `extensions/training`, compiler autograd, gradient publication, and prepared execution are not implemented.

Training will own backend-independent optimizer algorithms and session concepts. The compiler will own global automatic differentiation (autograd), while concrete backends will own any backend-specific lowering or fused optimizer route.

The accepted compiler design builds first-order gradients with existing public Tensor operations
before one combined forward/backward capture. Model task 0025 is Complete; Compiler task 0004
remains Draft and awaits its dedicated planning pass. Public Tensors gain no gradient/backward
lifecycle state. The initial `TRAINING_STEP` mode uses combined forward/backward construction but
adds no optimizer updates. Publication, explicit objectives/targets/seeds, higher-order
differentiation, optimizer behavior, preparation, and execution remain planned.

## Planned concepts

- `extensions/nn` will own `Parameter` and `Buffer` declarations as module state. Training will
  consume those parameters, while `ParameterGroup` will describe optimizer-group settings.
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
