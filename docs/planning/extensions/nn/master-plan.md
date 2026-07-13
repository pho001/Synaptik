# NN Extension Master Plan

## Goal

Define the stateful neural-network composition layer: module ownership, trainable parameters,
persistent buffers, train/eval behavior, forward context, layers, blocks, and functional
conveniences built from `modules/model` Tensor semantics.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [ADR 0007: Neural-network module and training boundary](../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Training graph](../../../architecture/training-graph.md)

## Scope

- `Module`, `Parameter`, and `Buffer` contracts
- module-tree parameter and buffer discovery
- train/eval mode propagation and forward context
- neural-network layers, blocks, and functional conveniences that compose model operations

## Out of scope

- generic Tensor or operation semantics
- compiler autograd construction
- optimizer algorithms, parameter updates, and training sessions
- backend storage, lowering, kernel selection, and backend-specific optimizer execution

## Module invariants

- NN owns module-declared `Parameter` and `Buffer` values.
- NN owns train/eval forward behavior and forward context.
- NN composes `modules/model`; it does not make model depend on nn.
- NN must not depend on `extensions/training`, compiler, runtime, prepare, engine, or concrete backends.
- Optimizers and training-step orchestration belong to `extensions/training` downstream of nn.

## Allowed dependencies

- modules/model

## Forbidden dependencies

- extensions/training
- modules/compiler
- modules/runtime
- modules/prepare
- modules/engine
- backends/cpu
- backends/metal
- backends/cuda

## Package structure

```text
io.github.pho001.synaptik.nn/
  module/       module ownership, traversal, and forward-context contracts
  layers/       public stateful neural-network layers
  functional/   stateless NN-oriented conveniences over model operations
```

The package map is provisional until this extension reaches the current or immediately following
planning frontier. A detailed task must confirm exact public types and package placement before it
becomes Ready.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|

## Milestones

- Module, parameter, buffer, and forward-mode contracts
- Initial stateful layers and functional conveniences
- Training-extension integration over the stable parameter contract

## Current status

Draft. This extension is architecture-planned only; no Gradle module, Java API, or detailed task
specification exists yet. Create detailed work only when this area becomes the current or
immediately following planning frontier.

## Open questions

- Define parameter and buffer naming, ownership, mutation, nested-module, and checkpoint contracts at the implementation frontier.
- Define how forward context represents train/eval mode without introducing runtime execution state.

## Decisions made

- `extensions/nn` owns `Module`, `Parameter`, `Buffer`, train/eval behavior, and forward context.
- `extensions/training` depends on nn and owns optimizer algorithms and training orchestration.
- Generic Tensor and operation semantics remain in `modules/model`.

## Risks

- Coupling layer composition to optimizer APIs would reverse the architecture dependency.
- Letting train/eval mode become backend residency or per-run execution state would blur the lifecycle boundary.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and
follow [the planning guide](../../planning-guide.md).
