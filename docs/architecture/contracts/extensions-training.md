# Extensions and training contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns NN, Training, and ONNX responsibilities plus the optimizer/training lifecycle.
It does not own Model semantics, Compiler autograd, or concrete backend optimizer routes.

## `extensions/nn`

`extensions/nn` owns the stateful neural-network composition layer. It defines model modules,
their trainable parameters and persistent buffers, training/evaluation mode, and the forward
context needed to apply that mode consistently through a module tree.

Allowed:

- `Module`
- `Parameter`
- `Buffer`
- module-owned parameter and buffer traversal
- training/evaluation mode propagation
- forward-context contracts
- neural-network layers, blocks, and functional conveniences composed from model semantics

Forbidden:

- optimizer algorithms
- optimizer update orchestration
- autograd construction
- backend storage access
- backend kernel selection
- concrete backend dependencies

`extensions/nn` depends on `modules/model` for tensor and operation semantics. It must not make
the model depend on neural-network layers or stateful module ownership.

## `extensions/training`

`extensions/training` owns optimizer algorithms and training orchestration over parameters
declared by `extensions/nn` modules.

Allowed:

- `Optimizer`
- `Sgd`
- `Adam`
- `AdamW`
- `ParameterGroup`
- `TrainingSession`
- `TrainingStep`

Forbidden:

- dependency on concrete backends
- `MetalOptimizerBridge`
- `CudaOptimizerBridge`
- `CpuOptimizerBridge`
- backend-specific optimizer execution
- backend storage access
- backend kernel selection

Training depends on `extensions/nn`, not the reverse. `train()` and `eval()` mode are module
forward-behavior concerns owned by `extensions/nn`; an optimizer neither selects nor changes
that mode.

Training owns optimizer algorithms, not backend-specific optimizer execution.

Backend-specific optimizer routes, such as fused Adam on Metal, belong to backend prepare/kernels.

## `extensions/onnx`

`extensions/onnx` owns ONNX import/export and mapping.

It must not be part of runtime hot path.

Allowed:

- ONNX import
- ONNX export
- ONNX-to-model mapping
- model-to-ONNX mapping

Forbidden:

- runtime execution
- backend-specific lowering
- kernel selection
- runtime residency


## Optimizer/training lifecycle

Initial version:

```text
compile:
  forward Tensor expression DAG
  -> compiler-owned autograd as Tensor expressions
  -> one capture of the combined forward + backward DAG
  -> infer, validate, optimize, and revalidate the immutable combined graph
  -> CompileArtifacts

run:
  forward/backward prepared execution
  -> publish gradients
  -> optimizer.step()
```

Later version:

```text
compile:
  forward + backward + optimizer update graph
  -> optimize
  -> partition scoring
  -> CompileArtifacts

prepare:
  backend prepare may fuse optimizer update routes

run:
  trainingStep schedule
```

Rules:

- `extensions/nn` owns module-declared trainable parameters, persistent buffers, and train/eval forward behavior.
- `FORWARD_ONLY` performs no autograd.
- `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` use the same combined pre-capture
  forward/backward construction; the initial `TRAINING_STEP` adds no optimizer-update graph work.
- Training owns optimizer algorithms.
- Training consumes the parameters declared by `extensions/nn`; it does not own layer behavior or train/eval mode.
- Training does not own backend-specific optimizer execution.
- Concrete backend optimizer routes belong to backend prepare/kernels.
- No training module may depend on backend-metal, backend-cpu, or backend-cuda.
