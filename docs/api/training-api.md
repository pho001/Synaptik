# Training API

## Purpose and implementation status

This reference records the planned public training concepts without inventing callable APIs.
`extensions/training`, public gradient requests and publication, optimizer behavior, and prepared
execution are not implemented. A bounded package-private compiler-owned first-order autograd path
is current internal graph-stage behavior.

Training will own backend-independent optimizer algorithms and session concepts. The compiler will own global automatic differentiation (autograd), while concrete backends will own any backend-specific lowering or fused optimizer route.

The accepted compiler design builds first-order gradients with existing public Tensor operations
before one combined forward/backward capture. Model task 0025 and Compiler task 0004 are Complete.
Public Tensors gain no gradient/backward lifecycle state. The current internal `TRAINING_STEP`
mode uses the same scalar-objective/implicit-unit-seed first-order construction as
`FORWARD_AND_BACKWARD`; it adds no optimizer update. Publication, public
objectives/targets/seeds, higher-order differentiation, optimizer behavior, preparation, and
execution remain planned.

Package-private `GraphCompiler` currently returns mode-neutral `GraphCompilation`. A
`TRAINING_STEP` result may carry the same combined forward/backward graph as
`FORWARD_AND_BACKWARD`; a `FORWARD_ONLY` result has no BACKWARD nodes and empty gradient results.
This internal graph-stage result is not the later `CompileArtifacts` aggregate or a training
session/result type.

The current internal first-order request supports one exact scalar floating objective among the
forward outputs and a non-empty ordered identity-unique target list in its differentiable
ancestry. Its only seed is an implicit exact typed positive one. The compiler preflights the
complete selected slice before creating formulas, uses explicit storage-free typed logical splats
for derivative zero/one values, captures forward and gradient roots once with per-node phase, and
optimizes that complete immutable graph once. These facts do not expose a public training
workflow, publish a gradient, mutate a parameter, choose a backend, prepare a schedule, or execute
training.

## Planned concepts

- `extensions/nn` will own `Parameter` and `Buffer` declarations as module state. Training will
  consume those parameters, while `ParameterGroup` will describe optimizer-group settings.
- `Optimizer` implementations such as SGD, Adam, and AdamW will define mathematical updates without importing CPU, Metal, or CUDA modules.
- `TrainingSession` and `TrainingStep` will coordinate forward/backward execution, gradient publication, and optimizer updates through shared lifecycle contracts.

No signatures, default hyperparameters, mutation rules, serialization format, or exception types are stable yet. They will be defined by focused extension tasks after model, compiler, runtime, and publication contracts exist.

## Planned public initial flow

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
